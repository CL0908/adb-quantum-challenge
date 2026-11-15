"""Equal wall-clock study (v3): every method gets the same time budget per estimate, one core, everything charged
(training, pilots, selection, policy construction inside the estimator). Problem construction (source laws) is given
to all methods and is not charged. Resumable jobs in results/raw/v3_*.jsonl.

Methods: twist (fast product-form sampler, theta = single optimal exponential twist), tilt (source-tilt, fitted),
rdpo_cold (RDPO v2, path budget calibrated so that the fit takes the budget), rdpo_transfer (RDPO v2 started from a
policy trained beforehand on a neighbouring model, rates/dependence m x 1.2; amortised: that training is NOT charged),
selector (no transfer), selector_tr (selector with the transferred policy available; training not charged).

Usage: python -m src.rdpo.experiments_v3 prepare|run|report
"""
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.tensor.synthetic import SyntheticModel, topology
from .problem import PortfolioProblem
from .train import rdpo_estimate
from .experiments import summarise
from .experiments_v2 import run_jobs, final_config, _scale_refs, SCALE_CASES, RAW
from .experiments_dense import model_of, exact_refs, RUN_CASES
from .tilt import tilt_estimate
from .select import select_estimate

R = Path("results")
SHORT, LONG_CAP = 2.0, 5.0
METHODS = ["twist", "tilt", "rdpo_cold", "rdpo_transfer", "selector", "selector_tr"]
REPS = {"short": 8, "long": 3}
TRAIN_B = 16384          # budget (paths) of the amortised transfer-policy training


def cells():
    """[(cell_id, family, kind, n, target, mu, v2_seconds)]"""
    tens = {(r["topology"], r["n"]): r for r in json.loads((R / "tensor.json").read_text())["B_scaling_n"]
            if "targets" in r and r["tol"] == (1e-6 if (r["topology"] == "complete" and r["n"] == 7) else 1e-9)}
    refs = {(k, n): {"exceedance": tens[(k, n)]["targets"]["exceedance_prob"],
                     "stop_loss": tens[(k, n)]["targets"]["stop_loss"]} for k, n in SCALE_CASES}
    v2 = json.loads((R / "rdpo_v2_scale.json").read_text())["rows"]
    sec = {(r["topology"], r["n"], r["target"]): r["seconds_per_fit"] for r in v2 if r["variant"] == "v2-final"}
    out = []
    for kind, n in SCALE_CASES:
        for t in ("exceedance", "stop_loss"):
            out.append((f"{kind}{n}-{t}", "scaling", kind, n, t, refs[(kind, n)][t], sec[(kind, n, t)]))
    ex = exact_refs()
    dj = {(r["graph"], r["n"], r["target"]): r for r in json.loads((R / "dense_exact.json").read_text())["rows"]}
    for kind, n in RUN_CASES:
        for t in ("exceedance", "stop_loss"):
            out.append((f"{kind}{n}-{t}", "dense", kind, n, t, ex[(kind, n)][t],
                        dj[(kind, n, t)]["rdpo_sec_per_estimate_median"]))
    return out


_CACHE = {}


def problem(family, kind, n, target, m=1.0):
    key = (family, kind, n, target, m)
    if key not in _CACHE:
        if family == "scaling":
            mod = SyntheticModel(n, topology(kind, n), m=m)
        else:
            base = model_of(kind, n)
            mod = SyntheticModel(n, base.edges, m=m)
        _CACHE[key] = PortfolioProblem(mod, target)
    return _CACHE[key]


def budgets(v2_seconds):
    return {"short": SHORT, "long": float(min(v2_seconds, LONG_CAP))}


# ------------------------------------------------------------------------------------------------
def _prep_job(args):
    """Per cell: RDPO time model (t0 + c·B from two short fits) and the amortised transfer policy."""
    jid, family, kind, n, target = args
    P = problem(family, kind, n, target)
    cfg = final_config(target)
    ts = {}
    for B in (2048, 8192):
        t0 = time.perf_counter(); rdpo_estimate(P, B, np.random.default_rng(5), cfg); ts[B] = time.perf_counter() - t0
    c = max((ts[8192] - ts[2048]) / (8192 - 2048), 0.3 * ts[8192] / 8192); t_fix = max(ts[2048] - c * 2048, 0.0)
    Pn = problem(family, kind, n, target, m=1.2)
    t0 = time.perf_counter()
    th = rdpo_estimate(Pn, TRAIN_B, np.random.default_rng(777), cfg)["theta"]
    return {"id": jid, "c": c, "t_fix": t_fix, "theta": th, "train_seconds": time.perf_counter() - t0}


def prepare(procs=2):
    jobs = [(c[0], c[1], c[2], c[3], c[4]) for c in cells()]
    recs = run_jobs("v3_prep", jobs, _prep_job, procs)
    return {r["id"]: r for r in recs}


def _load_prep():
    path = RAW / "v3_prep.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in path.read_text().splitlines() if l.strip()}


def _calib_job(args):
    """Path budget B such that one RDPO v2 fit takes ~0.97 of the budget (two secant rounds, one seed each)."""
    jid, cid, family, kind, n, target, seconds = args
    prep = _load_prep()[cid]
    P = problem(family, kind, n, target)
    cfg = final_config(target)
    B = int(max(512, (0.95 * seconds - prep["t_fix"]) / prep["c"]))
    hist = []
    for rnd in range(2):
        t0 = time.perf_counter(); rdpo_estimate(P, B, np.random.default_rng(100 + rnd), cfg)
        T = time.perf_counter() - t0
        hist.append((B, T))
        tf = min(prep["t_fix"], 0.5 * T)
        B = int(np.clip(B * (0.97 * seconds - tf) / max(T - tf, 1e-3), 0.3 * B, 3 * B))
        B = max(B, 256)
    return {"id": jid, "B": B, "hist": hist}


def calibrate(procs=2):
    jobs = []
    for cid, family, kind, n, target, mu, v2s in cells():
        for bname, secs in budgets(v2s).items():
            jobs.append((f"{cid}-{bname}", cid, family, kind, n, target, secs))
    recs = run_jobs("v3_calB", jobs, _calib_job, procs)
    return {r["id"]: r for r in recs}


_PREP = {}
_CALB = {}


def _job(args):
    jid, cid, family, kind, n, target, bname, seconds, method, rep, seed = args
    if not _PREP:
        _PREP.update(_load_prep())
        _CALB.update({json.loads(l)["id"]: json.loads(l) for l in (RAW / "v3_calB.jsonl").read_text().splitlines() if l.strip()})
    prep = _PREP[cid]
    P = problem(family, kind, n, target)
    rng = np.random.default_rng(seed)
    load0 = os.getloadavg()[0]
    cfg = final_config(target)
    t0 = time.perf_counter()
    extra = {}
    if method in ("twist", "tilt"):
        x = tilt_estimate(P, rng, seconds=seconds, fit=(method == "tilt"))
        extra = {"rho2": x["rho2"]}
    elif method in ("rdpo_cold", "rdpo_transfer"):
        B = _CALB[f"{cid}-{bname}"]["B"]
        if method == "rdpo_transfer":
            cfg = replace(cfg, theta0=prep["theta"])
        x = rdpo_estimate(P, B, rng, cfg); extra = {"B": B}
    else:
        x = select_estimate(P, seconds, rng, theta_transfer=prep["theta"] if method == "selector_tr" else None,
                            rdpo_calibration=(prep["t_fix"], max((seconds - prep["t_fix"]) / _CALB[f"{cid}-{bname}"]["B"], 1e-7)))
        extra = {"choice": x["choice"]}
    sec = time.perf_counter() - t0
    return {"id": jid, "key": [cid, bname, method], "est": float(x["estimate"]), "ci": [float(v) for v in x["ci"]],
            "sec": sec, "paths": int(x["paths_used"]), "load": load0, **extra}


def run(procs=2, seed=31000):
    prepare(procs); calibrate(procs)
    jobs = []
    allc = cells()
    for bname in ("short", "long"):
        for rep in range(REPS[bname]):
            for ci, (cid, family, kind, n, target, mu, v2s) in enumerate(allc):
                secs = budgets(v2s)[bname]
                for method in METHODS:
                    jobs.append((f"{cid}-{bname}-{method}-{rep}", cid, family, kind, n, target, bname, secs, method, rep,
                                 seed + 1000 * ci + (500 if bname == "long" else 0) + rep))
    run_jobs("v3_runs", jobs, _job, procs)


def report():
    allc = cells()
    prep = _load_prep()
    res = {}
    for l in (RAW / "v3_runs.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l); res.setdefault(tuple(r["key"]), []).append(r)
    rows = []
    for cid, family, kind, n, target, mu, v2s in allc:
        for bname in ("short", "long"):
            secs = budgets(v2s)[bname]
            for m in METHODS:
                xs = res.get((cid, bname, m))
                if not xs:
                    continue
                s = summarise([x["est"] for x in xs], mu, [x["ci"] for x in xs])
                ch = {}
                for x in xs:
                    if "choice" in x:
                        ch[x["choice"]] = ch.get(x["choice"], 0) + 1
                rows.append({"cell": cid, "family": family, "target": target, "budget": bname, "budget_s": secs,
                             "v2_seconds": v2s, "method": m, "mu": mu, **s,
                             "paths_mean": float(np.mean([x["paths"] for x in xs])),
                             "sec_mean": float(np.mean([x["sec"] for x in xs])),
                             "sec_max": float(np.max([x["sec"] for x in xs])),
                             "load_max": float(np.max([x["load"] for x in xs])), "choices": ch or None,
                             "amortised_training_not_charged": m in ("rdpo_transfer", "selector_tr")})
    # win / tie / loss versus twisting (same cell, same budget): win if rel_rmse < 0.8 x twist, loss if > 1.25 x
    by = {(r["cell"], r["budget"], r["method"]): r for r in rows}
    summ = {}
    for fam in ("scaling", "dense"):
        for b in ("short", "long"):
            for m in METHODS:
                if m == "twist":
                    continue
                ratios = []
                for r in rows:
                    if r["family"] == fam and r["budget"] == b and r["method"] == m:
                        tw = by.get((r["cell"], b, "twist"))
                        if tw:
                            ratios.append((r["rel_rmse"] / tw["rel_rmse"], r["cell"]))
                if not ratios:
                    continue
                v = np.array([x for x, _ in ratios])
                summ[f"{fam}|{b}|{m}"] = {
                    "cells": len(v), "wins": int((v < 0.8).sum()), "ties": int(((v >= 0.8) & (v <= 1.25)).sum()),
                    "losses": int((v > 1.25).sum()), "median_ratio": float(np.median(v)),
                    "best": min(ratios)[1] + f" ({min(ratios)[0]:.2f})", "worst": max(ratios)[1] + f" ({max(ratios)[0]:.2f})"}
    meta = {"short_budget_s": SHORT, "long_budget_rule": f"min(RDPO-v2 seconds, {LONG_CAP} s)", "reps": REPS,
            "tie_rule": "ratio of rel_rmse (method / twist) in [0.8, 1.25]; win < 0.8; loss > 1.25",
            "contention": "2 worker processes, 1 BLAS thread each, on a 4-core machine; load average recorded per job",
            "transfer_policy": f"trained on the same graph with dependence m x 1.2, B={TRAIN_B} paths, not charged",
            "problem_construction_charged": False, "methods": METHODS,
            "train_seconds_transfer": {k: v["train_seconds"] for k, v in prep.items()}}
    (R / "rdpo_v3_equal_time.json").write_text(json.dumps({"meta": meta, "summary": summ, "rows": rows}, indent=2))
    for k, v in summ.items():
        print(k, {a: v[a] for a in ("cells", "wins", "ties", "losses", "median_ratio")})


if __name__ == "__main__":
    {"prepare": prepare, "calibrate": calibrate, "run": run, "report": report}[sys.argv[1]]()
