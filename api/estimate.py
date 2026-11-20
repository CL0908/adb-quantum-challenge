"""Estimation API for the web workbench: runs RDPO v2 on a regional pool and returns the estimate with its
95% interval. Deployed as a Vercel Python function (POST /api/estimate, JSON body), and served locally by
web/serve.py. Exact answers are attached whenever they are cheap: pair targets (any size, exact reduction)
and every target for pools of up to 5 countries (dense enumeration)."""
import json
import math
import os
import sys
import time
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from src.tensor.synthetic import SyntheticModel, topology  # noqa: E402
from src.rdpo.problem import PortfolioProblem, TARGETS, UNIT, ADB_SCENARIOS, adb_model  # noqa: E402
from src.baselines.stronger import _simulate  # noqa: E402
from src.rdpo.train import rdpo_estimate  # noqa: E402
from src.rdpo.experiments_v2 import final_config  # noqa: E402

LIMITS = {"n": (2, 28), "budget": (1024, 65536), "m": (0.0, 4.0), "rainMult": (0.5, 3.0), "rainShared": (0.0, 1.0)}
TOPOLOGIES = {"chain": "neighbours in a line", "ring": "neighbours in a ring", "star": "one hub country",
              "complete": "every pair"}


class BadRequest(ValueError):
    pass


def _num(body, key, default, cast=float):
    v = body.get(key, default)
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise BadRequest(f"'{key}' must be a number.")
    lo, hi = LIMITS.get(key, (-math.inf, math.inf))
    if not lo <= v <= hi:
        raise BadRequest(f"'{key}' must be between {lo} and {hi}.")
    return v


def _plain_mc(P, budget, rng, scale):
    """Plain Monte Carlo with the same number of paths: mean, standard error and 95% interval (Wilson for 0/1 targets)."""
    f = _simulate(P, None, budget, rng)["f"]
    mean, se = float(f.mean()), float(f.std(ddof=1) / math.sqrt(budget))
    if P.target in ("joint_attachment", "joint_exhaustion", "exceedance"):
        z = 1.96
        den = 1 + z * z / budget
        centre = (mean + z * z / (2 * budget)) / den
        half = z * math.sqrt(mean * (1 - mean) / budget + z * z / (4 * budget ** 2)) / den
        ci = [max(0.0, centre - half), centre + half]
    else:
        ci = [mean - 1.96 * se, mean + 1.96 * se]
    return {"estimate": mean * scale, "ci": [c * scale for c in ci], "se": se * scale, "hits": int((f > 0).sum())}


def run(body):
    scenario = body.get("scenario")
    if scenario is not None and scenario not in ADB_SCENARIOS:
        raise BadRequest(f"'scenario' must be one of {', '.join(ADB_SCENARIOS)}.")
    n = 2 if scenario else _num(body, "n", 8, int)
    topo = "chain" if scenario else body.get("topology", "ring")
    if topo not in TOPOLOGIES:
        raise BadRequest(f"'topology' must be one of {', '.join(TOPOLOGIES)}.")
    if topo == "complete" and n > 7:
        raise BadRequest("'every pair' dependence is limited to 7 countries in this version.")
    target = body.get("target", "exceedance")
    if target not in TARGETS:
        raise BadRequest(f"'target' must be one of {', '.join(TARGETS)}.")
    budget = _num(body, "budget", 16384, int)
    m, rain_mult, rain_shared = _num(body, "m", 1.0), _num(body, "rainMult", 1.0), _num(body, "rainShared", 0.25)
    threshold = float(body.get("threshold_musd", 40 * n)) * 1e6
    if not 0 < threshold <= 400e6:
        raise BadRequest("'threshold_musd' must be between 0 and 400.")
    seed = int(body.get("seed", 0)) % (2 ** 32)
    try:
        if scenario:
            model = adb_model(scenario)
        else:
            model = SyntheticModel(n, topology(topo, n), m=m, rain_mult=rain_mult, rain_shared=rain_shared)
    except AssertionError:
        raise BadRequest("Shared-event rates exceed a country's total rate; lower the shared-shock intensity.")
    P = PortfolioProblem(model, target, threshold=threshold)
    t0 = time.perf_counter()
    x = rdpo_estimate(P, budget, np.random.default_rng(seed), final_config(target))
    sec = time.perf_counter() - t0
    scale = 1e-6 if target in ("expected_loss", "stop_loss") else 1.0     # money in US$m
    out = {"scenario": scenario, "n": n, "topology": topo, "target": target, "budget": budget, "threshold_musd": threshold / 1e6,
           "estimate": x["estimate"] * scale, "ci": [c * scale for c in x["ci"]], "se": x["se"] * scale,
           "seconds": sec, "paths_used": x["paths_used"], "parameters": x["n_params"], "sources": P.K,
           "ce_iterations": x["ce_iters"], "pg_steps": x["pg_steps"], "reached_tail": x["reached_level"],
           "money": target in ("expected_loss", "stop_loss"), "exact": None}
    out["mc"] = _plain_mc(P, budget, np.random.default_rng(seed + 1), scale)
    if target in ("joint_attachment", "joint_exhaustion"):
        out["exact"] = P.exact_pair()
    elif n <= 5:
        out["exact"] = P.exact_dense() * scale
    mu = out["exact"] if out["exact"] else out["estimate"]
    if not out["money"] and mu and 0 < mu < 1:
        out["mc_rel_error"] = math.sqrt((1 - mu) / (mu * budget))
        out["mc_zero_prob"] = math.exp(budget * math.log1p(-mu))
    return out


class handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, run(body))
        except BadRequest as e:
            self._send(400, {"error": str(e)})
        except json.JSONDecodeError:
            self._send(400, {"error": "The request body must be JSON."})
        except Exception as e:  # pragma: no cover
            self._send(500, {"error": f"Estimation failed: {type(e).__name__}."})

    def do_GET(self):
        self._send(200, {"service": "RDPO estimate", "method": "POST JSON", "targets": TARGETS,
                         "topologies": TOPOLOGIES, "limits": LIMITS})
