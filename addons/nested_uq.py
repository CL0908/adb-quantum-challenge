"""E2: parameter uncertainty for a large pool, with RDPO inside and the friend's policy transfer between draws.
Additive; rdpo_estimate is called unchanged (theta0 is the friend's own transfer option).

rates_unc showed parameter uncertainty is the largest uncertainty in this problem. For pools where exact methods
do not scale, the posterior of a tail probability needs one tail estimate per posterior draw. Bed: ring-16 pool
exceedance (the friend's own transfer bed); uncertain parameters = the friend's data posteriors of the two shared
shares (param_uncertainty: earthquake share -> m = s_eq / 0.0745 / 2, rain share). In a ring every country has two
neighbours; the friend's model already splits the rain share over the degree (cr = share·μ / maxdeg), and the
earthquake share is split the same way here (without it the posterior's upper half is infeasible: shared rates
would exceed the totals). The exact value per draw comes from
the friend's tilted MPS, so every estimator can be scored.
  refit      rdpo_estimate from scratch per draw, B = 16384
  cold       rdpo_estimate from scratch per draw, B = 2048            (same small budget, no transfer)
  transfer   one fit at the posterior median (B = 65536), then per draw B = 2048 with theta0 = that policy
Scored on the posterior summaries a risk report needs: mean over draws (= posterior predictive probability) and
the 2.5 / 97.5% quantiles, plus per-draw rel RMSE and interval coverage.
Pre-set rule: "strong" if transfer's posterior summaries all have rel RMSE <= 10%, it uses <= 1/5 of refit's
paths, and it beats cold at the same per-draw budget.
"""
import time
from dataclasses import replace

import numpy as np

from _src import in_repo, save, seed
from src.realdata import param_uncertainty as PU
from src.rdpo.experiments_v2 import _ring_target_exact, final_config
from src.rdpo.problem import PortfolioProblem
from src.rdpo.train import rdpo_estimate
from src.tensor.synthetic import SyntheticModel, topology

N_SITES, D, REPS, SEED = 16, 100, 5, seed(20261111)
BUDGETS = {"refit": 16384, "cold": 2048, "transfer": 2048}
B_SOURCE = 65536


def draws():
    with in_repo():
        eqp, rnp = PU.eq_posterior(), PU.rain_posterior()
    g = np.random.default_rng(SEED)
    return np.column_stack([g.beta(eqp["a"], eqp["b"], D) / PU.BRIEF_EQ / 2, g.beta(rnp["a"], rnp["b"], D)]), eqp, rnp


def problem(m, rs):
    M = SyntheticModel(N_SITES, topology("ring", N_SITES), m=float(m), rain_shared=float(rs))
    return M, PortfolioProblem(M, "exceedance")


def exact_job(th):
    M, P = problem(*th)
    return _ring_target_exact(M, P)


def fit_job(args):
    method, rep, d, th, theta0 = args
    _, P = problem(*th)
    cfg = final_config("exceedance")
    if theta0 is not None:
        cfg = replace(cfg, theta0=theta0)
    t0 = time.perf_counter()
    r = rdpo_estimate(P, BUDGETS[method], np.random.default_rng(SEED + 1_000_000 * rep + 1000 * d + list(BUDGETS).index(method)), cfg)
    return method, rep, d, r["estimate"], list(r["ci"]), r["paths_used"], time.perf_counter() - t0


def summary(mu):
    return {"mean": float(np.mean(mu)), "q025": float(np.quantile(mu, 0.025)), "q975": float(np.quantile(mu, 0.975))}


def run(procs=14):
    from multiprocessing import Pool
    TH, eqp, rnp = draws()
    med = np.median(TH, 0)
    with Pool(procs) as pool:
        mu = np.array(pool.map(exact_job, list(TH)))
        src = pool.map(_source_fit, [(rep, med) for rep in range(REPS)])     # one overnight fit per rep at the median
        jobs = [(m, rep, d, TH[d], src[rep][0] if m == "transfer" else None) for rep in range(REPS) for d in range(D)
                for m in BUDGETS]
        rows = pool.map(fit_job, jobs, chunksize=4)
    ex = summary(mu)
    out = {"bed": f"ring-{N_SITES} exceedance", "draws": D, "reps": REPS, "posteriors": {"eq_share": eqp, "rain_share": rnp},
           "exact_mu": mu.tolist(), "exact_summary": ex, "methods": {}}
    for m in BUDGETS:
        est = np.full((REPS, D), np.nan); cov = np.zeros((REPS, D), bool); paths = np.zeros(REPS); sec = np.zeros(REPS)
        for mm, rep, d, e, ci, pu, s in rows:
            if mm == m:
                est[rep, d] = e; cov[rep, d] = ci[0] <= mu[d] <= ci[1]; paths[rep] += pu; sec[rep] += s
        if m == "transfer":
            paths += np.array([s[1] for s in src])
        sm = [summary(est[r]) for r in range(REPS)]
        err = {k: float(np.sqrt(np.mean([(s[k] / ex[k] - 1) ** 2 for s in sm]))) for k in ex}
        out["methods"][m] = {"per_draw_budget": BUDGETS[m], "summary_rel_rmse": err,
                             "per_draw_rel_rmse": float(np.sqrt(np.nanmean((est / mu[None, :] - 1) ** 2))),
                             "coverage": float(cov.mean()), "paths_per_rep": float(paths.mean()), "cpu_sec_per_rep": float(sec.mean())}
    t, r, c = out["methods"]["transfer"], out["methods"]["refit"], out["methods"]["cold"]
    ok = (max(t["summary_rel_rmse"].values()) <= 0.10 and t["paths_per_rep"] <= r["paths_per_rep"] / 5
          and max(t["summary_rel_rmse"].values()) < max(c["summary_rel_rmse"].values()))
    out["verdict"] = "strong" if ok else "not strong"
    save("nested_uq.json", out)
    print(f"exact posterior: mean {ex['mean']:.3g}, 95% of draws [{ex['q025']:.3g}, {ex['q975']:.3g}]")
    for m, v in out["methods"].items():
        e = v["summary_rel_rmse"]
        print(f"{m:9s} paths/rep {v['paths_per_rep']:.3g}  cpu {v['cpu_sec_per_rep']:.0f}s  summary rel RMSE mean {e['mean']:.3f} "
              f"q025 {e['q025']:.3f} q975 {e['q975']:.3f}  per-draw {v['per_draw_rel_rmse']:.3f}  coverage {v['coverage']:.3f}")
    print("verdict:", out["verdict"])
    return out


def _source_fit(args):
    rep, med = args
    _, P = problem(*med)
    r = rdpo_estimate(P, B_SOURCE, np.random.default_rng(SEED + 7 + rep), final_config("exceedance"))
    return r["theta"], r["paths_used"]


def selftest():
    TH, _, _ = draws()
    assert TH.shape == (D, 2) and (TH[:, 0] > 0).all() and (TH[:, 1] < 1).all()
    M, P = problem(1.0, 0.25)                                    # the brief point equals the friend's default model
    M0 = SyntheticModel(N_SITES, topology("ring", N_SITES))
    assert abs(_ring_target_exact(M, P) / _ring_target_exact(M0, PortfolioProblem(M0, "exceedance")) - 1) < 1e-12
    print("selftest ok: posterior draws valid; (m, rain share) = (1, 0.25) reproduces the friend's ring model")


if __name__ == "__main__":
    selftest()
    run()
