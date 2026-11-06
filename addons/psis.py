"""C1: Pareto-smoothed importance sampling (Vehtari, Simpson, Gelman, Yao, Gabry, JMLR 25 (2024)) for RDPO.

RDPO's error at n = 28 is driven by rare outlier fits (repro/i25). PSIS fits a generalized Pareto distribution to the
largest values of g = w·f and replaces them by the fitted quantiles; k̂ (the fitted shape) diagnoses whether the
sample mean is trustworthy (k̂ < 0.7 rule of thumb). Implementation follows the reference psislw / gpdfit
(Zhang & Stephens 2009 estimator with the weak prior used by PSIS).

Experiment (ring-28 exceedance, E2 config RDPOConfig()): F independent fits of the friend's rdpo_estimate; for
each fit the trained θ is loaded into the friend's RDPOPolicy and N fresh iid paths are drawn. Per fit we record
the friend's own estimate, the raw iid estimate, the PSIS estimate, and k̂; errors are against the exact MPS value.
"""
import math
import time

import numpy as np

from _src import load, save

N_EST, FITS, SEED0 = 32768, 28, 1_100_000


def gpdfit(x):
    """x: sorted ascending positive exceedances. Returns (k, sigma) (Zhang & Stephens 2009, PSIS prior)."""
    n = len(x)
    m = 30 + int(math.sqrt(n))
    b = 1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b = b / (3 * x[int(n / 4 + 0.5) - 1]) + 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    ll = n * (np.log(-(b / k)) - k - 1)
    wts = 1 / np.exp(ll - ll[:, None]).sum(axis=1)
    ok = wts >= 10 * np.finfo(float).eps
    wts, b = wts[ok] / wts[ok].sum(), b[ok]
    b_post = (b * wts).sum()
    k_post = np.log1p(-b_post * x).mean()
    sigma = -k_post / b_post
    k_post = (n * k_post + 10 * 0.5) / (n + 10)     # weak prior toward 0.5, as in the reference code
    return k_post, sigma                             # k_post is the GPD shape ξ (ξ > 0: heavy tail)


def gpinv(p, k, sigma):
    return sigma * np.expm1(-k * np.log1p(-p)) / k if abs(k) > 1e-12 else -sigma * np.log1p(-p)


def psis_smooth(g):
    """Smooth the upper tail of a nonnegative sample g. Returns (g_smoothed, khat)."""
    g = np.asarray(g, float).copy()
    pos = np.nonzero(g > 0)[0]
    S = len(pos)
    M = int(math.ceil(min(0.2 * S, 3 * math.sqrt(S))))
    if M < 5:
        return g, float("inf")
    order = pos[np.argsort(g[pos], kind="stable")]
    cut = g[order[-M - 1]]
    tail = order[-M:][g[order[-M:]] > cut]          # ties at the cut carry no tail information
    if len(tail) < 5:
        return g, 0.0                               # no spread above the cut: nothing heavy to smooth
    M = len(tail)
    xi, sigma = gpdfit(g[tail] - cut)               # tail is already sorted ascending
    if np.isfinite(xi):
        q = gpinv((np.arange(M) + 0.5) / M, xi, sigma) + cut
        g[tail] = np.minimum(q, g[tail].max())
    return g, float(xi)


def _fit(seed):
    import sys
    from src.rdpo.problem import PortfolioProblem
    from src.rdpo.policy import RDPOPolicy
    from src.rdpo.train import rdpo_estimate, RDPOConfig
    from src.tensor.synthetic import SyntheticModel, topology
    cfg = RDPOConfig()
    P = PortfolioProblem(SyntheticModel(28, topology("ring", 28)), "exceedance")
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    res = rdpo_estimate(P, 65536, rng, cfg)
    pol = RDPOPolicy(P, cfg.alpha, cfg.edge_features, cfg.relevance)
    pol.theta = np.array(res["theta"], float)
    out = pol.rollout(N_EST, np.random.default_rng(seed + 7))
    g = out["w"] * out["f"]
    gs, xi = psis_smooth(g)
    return {"seed": seed, "friend_est": res["estimate"], "raw": float(g.mean()), "psis": float(gs.mean()),
            "khat": xi, "hits": int((g > 0).sum()), "max_share": float(g.max() / g.sum()) if g.sum() > 0 else None,
            "sec": time.perf_counter() - t0}


def run(procs=14, fits=FITS):
    from multiprocessing import Pool
    mu = next(r["mu"] for r in load("rdpo_e2.json")["rows"]
              if r["topology"] == "ring" and r["n"] == 28 and r["target"] == "exceedance")
    with Pool(procs) as pool:
        rows = []
        for r in pool.imap_unordered(_fit, [SEED0 + i for i in range(fits)]):
            for k in ("friend_est", "raw", "psis"):
                r[k + "_err"] = r[k] / mu - 1
            rows.append(r)
            print(f"seed {r['seed']}: friend {r['friend_est_err']:+.2%}  raw {r['raw_err']:+.2%}  "
                  f"psis {r['psis_err']:+.2%}  k̂={r['khat']:.2f}  max share {r['max_share']:.1%}  {r['sec']:.0f}s", flush=True)
    summ = {}
    for k in ("friend_est", "raw", "psis"):
        e = np.array([r[k + "_err"] for r in rows])
        summ[k] = {"rel_rmse": float(np.sqrt((e ** 2).mean())), "rel_bias": float(e.mean()), "max_abs": float(abs(e).max())}
    kh = np.array([r["khat"] for r in rows]); ae = np.array([abs(r["raw_err"]) for r in rows])
    summ["khat_ge_0.7"] = int((kh >= 0.7).sum())
    summ["corr_khat_abs_raw_err"] = float(np.corrcoef(kh, ae)[0, 1]) if len(rows) > 2 else None
    print({k: v for k, v in summ.items()}, flush=True)
    save("c1_psis.json", {"mu_exact": mu, "n_est": N_EST, "fits": fits, "config": "RDPOConfig() (E2)",
                          "summary": summ, "rows": rows})
    return summ


def selftest():
    for xi_true in (0.3, 0.6, 0.9):           # GPD(ξ, 1) samples by inverse CDF; mean k̂ over 10 seeds
        xi = np.mean([psis_smooth(gpinv(np.random.default_rng(s).random(20000), xi_true, 1.0))[1] for s in range(10)])
        assert abs(xi - xi_true) < 0.06, (xi_true, xi)
    g = np.zeros(1000); g[:50] = 1.0
    gs, _ = psis_smooth(g)
    assert abs(gs.mean() - g.mean()) < 1e-12        # a flat tail is left unchanged
    print("selftest ok: k̂ recovers ξ ∈ {0.3, 0.6, 0.9} within 0.06 (mean of 10); flat tail unchanged")


if __name__ == "__main__":
    import sys
    selftest()
    if "--selftest" not in sys.argv:
        run()
