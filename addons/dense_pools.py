"""Saturation modules on dense pools, where exact contraction is out of reach. Additive, read-only.

cap_modules.py was tested on rings and on the ap_risk pool. On a ring the main author's tensor-network engine is exact anyway, so
the test that matters is a dense pool. The main author's model caps density (shared rates may not exceed a country's total:
a complete graph is valid up to about 13 countries at m = 1), so the pools are:
  complete graph, 6 countries, US$40m each   exact by enumeration of 10^6 states (the truth, not the twist)
  complete graph, 10 countries, US$40m each  a cut through the middle crosses 25 shared-source edges
  circulant, 28 countries, degree 8, US$30m  every country shares sources with its 4 neighbours on each side
  circulant, 28 countries, degree 12, US$30m
Thresholds were picked from the uncapped small world's own answer (no sampling) to land between 1e-3 and 1e-7.
Pre-set rules:
 D1 every estimate is within 3 standard errors of the reference (exact for 6 countries, the twist's 2,000,000 paths otherwise);
 D2 with modules, equal-CPU efficiency against the twist > 1 on every pool (build charged);
 D3 with modules, the small world's own answer is within a factor 2 of the truth on every pool;
 D4 with modules, the relative variance is below the uncapped teacher's on every pool.
"""
import numpy as np

import cap_modules as C
import dreamwake as D
import rdpo_engine as E
import small_world as W
from _src import save
from src.rdpo.problem import PortfolioProblem
from src.tensor.synthetic import SyntheticModel, dense_term_distribution, topology


def circulant(n, k):
    return sorted({(min(i, (i + d) % n), max(i, (i + d) % n)) for i in range(n) for d in range(1, k + 1)})


POOLS = {"complete 6": (6, topology("complete", 6), 40e6), "complete 10": (10, topology("complete", 10), 40e6),
         "circulant 28, degree 8": (28, circulant(28, 4), 30e6), "circulant 28, degree 12": (28, circulant(28, 6), 30e6)}


def pool(name):
    n, edges, t = POOLS[name]
    return PortfolioProblem(SyntheticModel(n, edges, **D.BASE), "exceedance", threshold=t * n)


def exact(P):
    dist = dense_term_distribution(P.model)
    codes = np.indices(dist.shape).reshape(P.n, -1)
    return float(dist.reshape(-1)[(P.a_u * (codes // 2) + P.r_u * (codes % 2)).sum(0) > P.T].sum())


def run():
    res = {}
    for name in POOLS:
        P = pool(name)
        r = C.cell(P, rounds=(1,))
        r["edges"] = len(POOLS[name][1])
        if P.n <= 6:                                                           # the truth replaces the twist as reference
            p = exact(P)
            r["p_exact"] = p
            for m in ("uncapped", "modules_1"):
                r[m]["answer_over_truth"] *= r["p_twist"] / p
            pol = C.build(P, 1)[0]
            o = E.rollout(pol, 400_000, np.random.default_rng(9))
            g = o["w"] * o["f"]
            r["modules_1"]["z_max"] = float(abs(g.mean() - p) / (g.std() / np.sqrt(len(g))))
            r["twist_z_vs_exact"] = float(abs(r["p_twist"] - p) / r["p_twist_se"])
        res[name] = r
    res["checks"] = {"D1_unbiased": bool(max(r[m]["z_max"] for r in res.values() for m in ("uncapped", "modules_1")) <= 3),
                     "D2_equal_cpu_gt_1": {k: bool(r["modules_1"]["equal_cpu_gain"] > 1) for k, r in res.items()},
                     "D3_own_answer_within_2x": {k: bool(0.5 <= r["modules_1"]["answer_over_truth"] <= 2) for k, r in res.items()},
                     "D4_below_uncapped": {k: bool(r["modules_1"]["rho2"] < r["uncapped"]["rho2"]) for k, r in res.items()}}
    save("dense_pools.json", res)
    for k, r in res.items():
        if k == "checks":
            continue
        print(f"{k} ({r['edges']} edges, {r['sources']} sources, p {r.get('p_exact', r['p_twist']):.3g}{' exact' if 'p_exact' in r else ''}, twist relative variance {r['rho2_twist']:.3g})")
        for m in ("uncapped", "modules_1"):
            x = r[m]
            print(f"   {m:10s}: relative variance {x['rho2']:.3g} | own answer {x['answer_over_truth']:.2f}x the truth | build {x['cpu_build']:.2f} s | equal-CPU x{x['equal_cpu_gain']:.2f} | max |z| {x['z_max']:.1f}")
    print("checks:", res["checks"])
    return res


def selftest():
    """A complete graph of 4 countries, exact by enumeration: modules keep the policy unbiased and bring the own answer to the truth."""
    P = PortfolioProblem(SyntheticModel(4, topology("complete", 4), **D.BASE), "exceedance", threshold=30e6 * 4)
    p = exact(P)
    pol, a, _, _ = C.build(P, 1)
    o = E.rollout(pol, 300_000, np.random.default_rng(3))
    g = o["w"] * o["f"]
    a0 = W.value_tables(P)[1]
    assert abs(g.mean() - p) < 4 * g.std() / np.sqrt(len(g)) and abs(a / p - 1) < abs(a0 / p - 1), (g.mean(), a, a0, p)
    print(f"selftest ok: complete graph of 4, exact {p:.5g}, estimate {g.mean():.5g}, own answer {a:.5g} (uncapped {a0:.5g})")


if __name__ == "__main__":
    selftest()
    run()
