"""Scalable RDPO sweep: source-level Rényi-Doob policy vs plain MC, against exact tilted-MPS
references, as the portfolio grows (ring up to 28 countries, complete graph up to 7)."""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from src.tensor.synthetic import SyntheticModel, topology
from .source_rdpo import SourceProblem, fit_rdpo, mc_estimate

OUT = Path("results/scale_rdpo.json")
SEEDS = [101, 202, 303]
N_EVAL, R_EVAL = 4096, 30
CASES = [("ring", n, "exceedance") for n in [2, 4, 6, 8, 12, 16, 20, 28]] + \
        [("complete", n, "exceedance") for n in [4, 5, 6, 7]] + \
        [("ring", n, "stop_loss") for n in [4, 8, 16]] + [("complete", 6, "stop_loss")]


def exact_refs():
    t = json.loads(Path("results/tensor.json").read_text())["B_scaling_n"]
    ref = {}
    for r in t:
        if "targets" not in r:
            continue
        key = (r["topology"], r["n"])
        if key not in ref or r["tol"] < ref[key]["tol"]:
            ref[key] = {"tol": r["tol"], "exceedance": r["targets"]["exceedance_prob"],
                        "stop_loss": r["targets"]["stop_loss"], "chi": r["max_bond"], "mps_s": r["runtime_s"]}
    return ref


def run():
    ref = exact_refs()
    rows = []
    for kind, n, target in CASES:
        m = SyntheticModel(n, topology(kind, n))
        prob = SourceProblem(m, 40e6 * n, target)
        mu = ref[(kind, n)][target]
        # plain MC at the same evaluation budget
        rng = np.random.default_rng(7)
        x = np.array([mc_estimate(prob, N_EVAL, rng) for _ in range(R_EVAL)])
        mc = {"rel_rmse": float(np.sqrt(np.mean((x - mu) ** 2)) / mu), "frac_zero": float(np.mean(x == 0))}
        for s in SEEDS:
            t0 = time.perf_counter()
            pol, info = fit_rdpo(prob, seed=s, max_iter=40)
            rng = np.random.default_rng(s + 1)
            y = np.array([pol.estimate(N_EVAL, rng) for _ in range(R_EVAL)])
            # also the CE-only policy (before Rényi-2 refinement)
            th = pol.get().copy(); pol.set(np.array(info["ce_theta"]))
            z = np.array([pol.estimate(N_EVAL, rng) for _ in range(R_EVAL)]); pol.set(th)
            row = {"topology": kind, "n": n, "edges": len(m.edges), "target": target, "seed": s, "mu_exact": mu,
                   "mps_tol": ref[(kind, n)]["tol"], "mps_chi": ref[(kind, n)]["chi"], "mps_runtime_s": ref[(kind, n)]["mps_s"],
                   "K_sources": info["K_sources"], "n_params": info["n_params"],
                   "ce_paths": info["ce_paths"], "renyi_paths": info["renyi_paths"], "ce_iters": len(info["ce_history"]),
                   "reached_level": bool(info["ce_history"][-1]["gamma"] >= prob.level()),
                   "rel_rmse_ce": float(np.sqrt(np.mean((z - mu) ** 2)) / mu),
                   "rel_rmse_rdpo": float(np.sqrt(np.mean((y - mu) ** 2)) / mu),
                   "rel_bias_rdpo": float((y.mean() - mu) / mu),
                   "mc_rel_rmse": mc["rel_rmse"], "mc_frac_zero": mc["frac_zero"],
                   "mc_theory_rel_rmse": math.sqrt((1 - mu) / (mu * N_EVAL)) if target == "exceedance" else None,
                   "fit_seconds": time.perf_counter() - t0}
            rows.append(row)
            print(f"{kind:8s} n={n:2d} {target:10s} seed={s} mu={mu:.2e} CE={row['rel_rmse_ce']:.4f} "
                  f"RDPO={row['rel_rmse_rdpo']:.4f} MC={mc['rel_rmse']:.3f} (zero {mc['frac_zero']:.2f}) "
                  f"K={info['K_sources']} params={info['n_params']} {row['fit_seconds']:.0f}s",
                  file=sys.stderr, flush=True)
        OUT.write_text(json.dumps({"seeds": SEEDS, "n_eval": N_EVAL, "reps": R_EVAL, "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    run()
