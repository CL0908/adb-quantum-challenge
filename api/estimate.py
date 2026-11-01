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
from src.rdpo.problem import PortfolioProblem, TARGETS, UNIT  # noqa: E402
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


def run(body):
    n = _num(body, "n", 8, int)
    topo = body.get("topology", "ring")
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
    seed = int(body.get("seed", 0)) % (2 ** 32)
    try:
        model = SyntheticModel(n, topology(topo, n), m=m, rain_mult=rain_mult, rain_shared=rain_shared)
    except AssertionError:
        raise BadRequest("Shared-event rates exceed a country's total rate; lower the shared-shock intensity.")
    P = PortfolioProblem(model, target, threshold=threshold)
    t0 = time.perf_counter()
    x = rdpo_estimate(P, budget, np.random.default_rng(seed), final_config(target))
    sec = time.perf_counter() - t0
    scale = 1e-6 if target in ("expected_loss", "stop_loss") else 1.0     # money in US$m
    out = {"n": n, "topology": topo, "target": target, "budget": budget, "threshold_musd": threshold / 1e6,
           "estimate": x["estimate"] * scale, "ci": [c * scale for c in x["ci"]], "se": x["se"] * scale,
           "seconds": sec, "paths_used": x["paths_used"], "parameters": x["n_params"], "sources": P.K,
           "ce_iterations": x["ce_iters"], "pg_steps": x["pg_steps"], "reached_tail": x["reached_level"],
           "money": target in ("expected_loss", "stop_loss"), "exact": None}
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
