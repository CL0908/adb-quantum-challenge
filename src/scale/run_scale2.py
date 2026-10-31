"""Second scaling sweep: classical exponential twisting vs RDPO warm-started from the twist."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.tensor.synthetic import SyntheticModel, topology
from .source_rdpo import SourceProblem, fit_rdpo, twisted_estimate, warm_start_theta
from .run_scale import exact_refs

OUT = Path("results/scale_rdpo_warm.json")
SEEDS = [101, 202, 303]
N_EVAL, R_EVAL = 4096, 30
CASES = [("ring", n) for n in [2, 8, 16, 20, 28]] + [("complete", n) for n in [6, 7]]


def rel(x, mu):
    return float(np.sqrt(np.mean((np.asarray(x) - mu) ** 2)) / mu)


def run():
    ref = exact_refs()
    rows = []
    for kind, n in CASES:
        m = SyntheticModel(n, topology(kind, n))
        prob = SourceProblem(m, 40e6 * n, "exceedance")
        mu = ref[(kind, n)]["exceedance"]
        th0, eta = warm_start_theta(prob)
        rng = np.random.default_rng(5)
        tw = rel([twisted_estimate(prob, eta, N_EVAL, rng) for _ in range(R_EVAL)], mu)
        for s in SEEDS:
            t0 = time.perf_counter()
            pol, info = fit_rdpo(prob, seed=s, max_iter=40, init_theta=th0)
            rng = np.random.default_rng(s + 1)
            y = rel([pol.estimate(N_EVAL, rng) for _ in range(R_EVAL)], mu)
            rows.append({"topology": kind, "n": n, "seed": s, "mu_exact": mu, "eta": eta, "rel_rmse_twisted": tw,
                         "rel_rmse_rdpo_warm": y, "ce_paths": info["ce_paths"], "renyi_paths": info["renyi_paths"],
                         "n_params": info["n_params"], "K_sources": info["K_sources"], "fit_seconds": time.perf_counter() - t0})
            print(f"{kind:8s} n={n:2d} seed={s} mu={mu:.2e} twisted={tw:.4f} RDPO-warm={y:.4f} "
                  f"train={info['ce_paths']}+{info['renyi_paths']} {rows[-1]['fit_seconds']:.0f}s", file=sys.stderr, flush=True)
        OUT.write_text(json.dumps({"seeds": SEEDS, "n_eval": N_EVAL, "reps": R_EVAL, "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    run()
