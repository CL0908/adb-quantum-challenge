"""Phase 1 driver: RMSE of all baselines at fixed path budgets, all scenarios."""
import json
import sys
import time
from pathlib import Path

from src.model.scenarios import SCENARIOS
from .common import make_problem, rmse_study
from . import estimators as E

BUDGETS = [2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16]
REPS = 200
ZV_REPS = 10  # zero-variance estimator: every path returns mu, few reps suffice
SEED = 20261031

METHODS = {
    "mc": E.mc,
    "sobol": E.sobol,
    "stratified": E.stratified,
    "importance": E.importance,
    "zero_variance": E.zero_variance,
}


def run(out=Path("results/baselines.json"), reps=REPS, budgets=BUDGETS):
    res = {"budgets": budgets, "reps": reps, "seed": SEED, "query_unit": "one 3-year path", "scenarios": {}}
    for si, name in enumerate(SCENARIOS):
        prob = make_problem(name)
        res["scenarios"][name] = {"exact": prob.exact, "methods": {}}
        for mi, (m, est) in enumerate(METHODS.items()):
            t0 = time.perf_counter()
            r = rmse_study(est, prob, budgets, ZV_REPS if m == "zero_variance" else reps,
                           seed=SEED + 100 * si + mi)
            res["scenarios"][name]["methods"][m] = r
            print(f"{name:18s} {m:14s} {time.perf_counter() - t0:6.1f}s", file=sys.stderr, flush=True)
    out.write_text(json.dumps(res, indent=2))
    return res


def summary_table(res, N=None):
    N = N or res["budgets"][-1]
    lines = []
    for name, d in res["scenarios"].items():
        lines.append(f"\n### {name}  (N = {N} paths, relative RMSE over {res['reps']} reps)\n")
        targets = list(d["exact"])
        lines.append("| method | " + " | ".join(t.replace("term_", "") for t in targets) + " |")
        lines.append("|---" * (len(targets) + 1) + "|")
        for m, r in d["methods"].items():
            lines.append(f"| {m} | " + " | ".join(f"{r[t][str(N)]['rel_rmse']:.3g}" if isinstance(next(iter(r[t])), str)
                                                  else f"{r[t][N]['rel_rmse']:.3g}" for t in targets) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    r = run()
    print(summary_table(r))
