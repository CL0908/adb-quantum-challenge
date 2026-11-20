"""Can the error bars be trusted? Coverage against exact values and the tail of the weights. Additive, read-only.

A low relative variance is worth little if the confidence interval misses the truth or a few paths carry the estimate.
Pools with exact values from the main author's junction solver: the 28-country ring at US$40m per country (1.3e-13) and
the ap_risk pool at US$450m (1.8e-3). Samplers: the twist, the uncapped small world, the small world with modules.
200 independent runs of 20,000 paths each; one run of 200,000 paths for the tail.
Pre-set rules, for the small world with modules, in both pools:
 R1 the 95% normal interval contains the exact value in 92% to 98% of the runs (200 runs: about +-3% of noise);
 R2 the mean of the 200 estimates is within 3 standard errors of the exact value;
 R3 the Pareto tail index k-hat of the weighted payoffs is below 0.7;
 R4 no single path carries more than 1% of the 200,000-path estimate.
"""
import numpy as np

import cap_modules as C
import dreamwake as D
import multi_country as M
import rdpo_engine as E
import small_world as W
from _src import save, seed
from src.rdpo.problem import PortfolioProblem
from src.rdpo.tilt import TiltSampler
from src.rdpo.train import _best_twist_sources
from src.tensor.junction import junction_solve


def measure(draw, p, runs=200, n=20_000, n_tail=200_000, s=0):
    est = np.empty(runs); cover = 0
    for r in range(runs):
        g = draw(n, np.random.default_rng(seed(5000 + s) + r))
        est[r] = g.mean()
        cover += abs(g.mean() - p) <= 1.96 * g.std() / np.sqrt(n)
    g = draw(n_tail, np.random.default_rng(seed(9000 + s)))
    nz = g[g > 0]
    return {"coverage_95": cover / runs, "mean_over_exact": float(est.mean() / p), "z_of_mean": float(abs(est.mean() - p) / (est.std(ddof=1) / np.sqrt(runs))),
            "khat": D.khat(nz), "max_path_share": float(g.max() / g.sum()), "ess_fraction": float(g.sum() ** 2 / (g ** 2).sum() / n_tail),
            "relative_variance": float(g.var() / p ** 2)}


def pool(P):
    p = junction_solve(P)["exceedance"]
    S = TiltSampler(P); S.set_theta(S.twist_theta(_best_twist_sources(P)))
    pol0, polm = W.teacher(P)[0], C.build(P, 1)[0]
    roll = lambda pol: (lambda n, rng: (lambda o: o["w"] * o["f"])(E.rollout(pol, n, rng)))
    samplers = {"twist": lambda n, rng: (lambda o: o["w"] * o["f"])(S.sample(n, rng)), "uncapped": roll(pol0), "modules": roll(polm)}
    return {"p_exact": p, **{k: measure(f, p, s=i) for i, (k, f) in enumerate(samplers.items())}}


def run():
    import ap_risk
    res = {"ring 28 @ 40m": pool(M.pool(28, 40e6)),
           "ap_risk 28 @ 450m": pool(PortfolioProblem(ap_risk.inputs()["model"], "exceedance", threshold=ap_risk.THRESH[0] * 1e6))}
    m = [r["modules"] for r in res.values()]
    res["checks"] = {"R1_coverage_92_to_98": bool(all(0.92 <= x["coverage_95"] <= 0.98 for x in m)), "R2_mean_unbiased": bool(all(x["z_of_mean"] <= 3 for x in m)),
                     "R3_khat_below_0.7": bool(all(x["khat"] < 0.7 for x in m)), "R4_no_path_above_1pct": bool(all(x["max_path_share"] <= 0.01 for x in m))}
    save("reliability.json", res)
    for k, r in res.items():
        if k == "checks":
            continue
        print(f"{k} (exact {r['p_exact']:.4g})")
        for name in ("twist", "uncapped", "modules"):
            x = r[name]
            print(f"   {name:9s}: 95% interval covers the exact value in {x['coverage_95']:.1%} of 200 runs | mean / exact {x['mean_over_exact']:.4f} (|z| {x['z_of_mean']:.1f}) | k-hat {x['khat']:.2f} | "
                  f"largest path {x['max_path_share']:.2%} of the estimate | effective sample {x['ess_fraction']:.1%} | relative variance {x['relative_variance']:.3g}")
    print("checks:", res["checks"])
    return res


def selftest():
    """Plain sampling of a fair coin through the same code: coverage near 95% and no heavy tail."""
    r = measure(lambda n, rng: (rng.random(n) < 0.5).astype(float), 0.5, runs=200, n=2000, n_tail=20_000)
    assert 0.90 <= r["coverage_95"] <= 0.99 and r["z_of_mean"] < 4 and r["max_path_share"] < 0.001, r
    print(f"selftest ok: fair coin, coverage {r['coverage_95']:.1%}, mean / exact {r['mean_over_exact']:.4f}")


if __name__ == "__main__":
    selftest()
    run()
