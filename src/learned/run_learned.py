"""Phase 2 driver: learned Doob proposal (CE) vs. the brief's baselines."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.baselines.common import make_problem
from src.model.scenarios import SCENARIOS, TARGETS
from .cross_entropy import fit

BUDGETS = [2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16]
REPS = 100
SEED = 20261101
KINDS = ["factorized", "pairwise"]


def run(out=Path("results/learned.json")):
    res = {"budgets": BUDGETS, "reps": REPS, "seed": SEED, "scenarios": {}}
    for si, name in enumerate(SCENARIOS):
        prob, sc = make_problem(name), SCENARIOS[name]
        res["scenarios"][name] = {}
        for ti, t in enumerate(TARGETS):
            res["scenarios"][name][t] = {}
            for ki, kind in enumerate(KINDS):
                seed = SEED + 1000 * si + 10 * ti + ki
                t0 = time.perf_counter()
                prop, info = fit(prob, t, sc, kind=kind, seed=seed)
                rng = np.random.default_rng(seed + 7)
                mu = prob.exact[t]
                curve = {}
                for N in BUDGETS:
                    x = np.array([prop.estimate(prob.f[t], N, rng) for _ in range(REPS)])
                    rmse = float(np.sqrt(np.mean((x - mu) ** 2)))
                    curve[N] = {"rmse": rmse, "rel_rmse": rmse / mu, "bias": float(x.mean() - mu)}
                res["scenarios"][name][t][kind] = {
                    "training_paths": info["training_paths"], "n_params": info["n_params"],
                    "ce_iterations": len(info["history"]), "theta": prop.theta.tolist(), "curve": curve}
                print(f"{name:17s} {t[5:40]:36s} {kind:10s} relRMSE@{BUDGETS[1]}="
                      f"{curve[BUDGETS[1]]['rel_rmse']:.4f} train={info['training_paths']} "
                      f"{time.perf_counter() - t0:.1f}s", file=sys.stderr, flush=True)
    out.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    run()
