"""Which sources of the 28-country pool need a state-dependent law? Additive, read-only.

rdpo_engine.py: on the 28-country pool (100 sources) the trained RDPO policy has 5.6x lower variance than the twist
but costs 20x more per path, so it loses at equal CPU (x0.28). Here the same trained policy keeps its state-dependent
law on a subset of the sources only; every other source is drawn from one fixed law (the path average of the policy's
own law at that source, or the twist's law). The weights are exact, so the estimate stays unbiased for every subset.
Pre-set rules:
 R1 the 20 sources with the largest state dependence (mean KL from the path-averaged law) carry >= 80% of the total;
 R2 that ranking agrees with the measured cost of fixing one source at a time (Spearman >= 0.7);
 R3 for every subset size the estimate is within 3 standard errors of the twist's;
 R4 with all sources state-dependent the relative variance reproduces rdpo_engine.json (within 10%);
 R5 some subset size has equal-CPU efficiency against the twist > 1 (training not charged);
 R6 the same with the training time charged, at the path count of this run.
Three seeds; the worst is reported. Limit: the policy is not retrained for the hybrid structure.
"""
import json

import numpy as np
from scipy.stats import spearmanr

import rdpo_engine as E
from _src import OUT, save, seed
from src.rdpo.problem import PortfolioProblem
from src.rdpo.tilt import TiltSampler
from src.rdpo.train import _best_twist_sources

SIZES = (0, 5, 10, 20, 50)


def setup(b_fit=65_536, n_tw=2_000_000, s=20261128):
    import ap_risk
    rng = np.random.default_rng(seed(s))                                    # same seed and order as rdpo_engine.pool_28
    P = PortfolioProblem(ap_risk.inputs()["model"], "exceedance", threshold=ap_risk.THRESH[0] * 1e6)
    S = TiltSampler(P); S.set_theta(S.twist_theta(_best_twist_sources(P)))
    t_tw, o = E.cpu(lambda: S.sample(n_tw, rng), reps=1)
    g = o["w"] * o["f"]
    pol, t_train = E.trained(P, b_fit, rng)
    return {"P": P, "S": S, "pol": pol, "t_train": t_train, "p": float(g.mean()), "p_se": float(g.std() / np.sqrt(n_tw)),
            "rho2_twist": float(g.var() / g.mean() ** 2), "cost_twist": t_tw / n_tw}


def run_one(c, fixed, n, s):
    t, o = E.cpu(lambda: E.rollout(c["pol"], n, np.random.default_rng(s), fixed=fixed), reps=1)
    g = o["w"] * o["f"]
    return {"p": float(g.mean()), "se": float(g.std() / np.sqrt(n)), "rho2": float(g.var() / c["p"] ** 2), "cost": t / n}


def dependence(c, n=200_000, s=11):
    stats = {}
    E.rollout(c["pol"], n, np.random.default_rng(s), stats=stats)
    return stats


def one_at_a_time(c, stats, n=100_000, s=12):
    base = run_one(c, None, n, s)["rho2"]                                    # common random numbers across the runs
    return {k: run_one(c, {k: stats[k]["law"]}, n, s)["rho2"] / base for k in stats}


def sweep(c, stats, order, kind, n=300_000, seeds=(21, 22, 23)):
    law = (lambda k: stats[k]["law"]) if kind == "average" else (lambda k: np.asarray(c["S"].q[k], float))
    out = {}
    for m in SIZES + (len(order),):
        fixed = {k: law(k) for k in order[m:]}
        rs = [run_one(c, fixed, n, s) for s in seeds]
        rho2, cost = max(r["rho2"] for r in rs), float(np.median([r["cost"] for r in rs]))
        ratio = cost / c["cost_twist"]
        out[m] = {"p": [r["p"] for r in rs], "z_max": max(abs(r["p"] - c["p"]) / np.hypot(r["se"], c["p_se"]) for r in rs),
                  "rho2_worst": rho2, "rho2_seeds": [r["rho2"] for r in rs], "cost_ratio": ratio,
                  "equal_cpu_gain": c["rho2_twist"] / rho2 / ratio,
                  "equal_cpu_gain_training_charged": c["rho2_twist"] * c["cost_twist"] / (rho2 * (cost + c["t_train"] / n))}
    return out


def run():
    c = setup()
    stats = dependence(c)
    order = sorted(stats, key=lambda k: -stats[k]["kl"])
    kl = np.array([stats[k]["kl"] for k in order])
    loo = one_at_a_time(c, stats)
    rank = float(spearmanr(kl, [loo[k] for k in order])[0])
    res = {"sources": c["P"].K, "state_dependent_sources": len(order), "p_twist": c["p"], "p_twist_se": c["p_se"], "rho2_twist": c["rho2_twist"],
           "cpu_training": c["t_train"], "order": [int(k) for k in order], "kl": kl.tolist(), "states": [stats[k]["states"] for k in order],
           "top20_share": float(kl[:20].sum() / kl.sum()), "sources_for_80pct": int(np.searchsorted(np.cumsum(kl) / kl.sum(), 0.8) + 1),
           "one_at_a_time_rho2_ratio": [loo[k] for k in order], "rank_agreement": rank,
           "sweep": {kind: sweep(c, stats, order, kind) for kind in ("average", "twist")}}
    ref = json.loads((OUT / "rdpo_engine.json").read_text())["pool_28"]["rho2_rdpo"]
    allm = [r for sw in res["sweep"].values() for r in sw.values()]
    full = res["sweep"]["average"][len(order)]
    res["checks"] = {"R1_top20_carry_80pct": bool(res["top20_share"] >= 0.8), "R2_rank_agreement_ge_0.7": bool(rank >= 0.7),
                     "R3_unbiased": bool(max(r["z_max"] for r in allm) <= 3),
                     "R4_reproduces_engine": bool(abs(float(np.median(full["rho2_seeds"])) / ref - 1) <= 0.10),
                     "R5_equal_cpu_gt_1": bool(max(r["equal_cpu_gain"] for r in allm) > 1),
                     "R6_with_training_gt_1": bool(max(r["equal_cpu_gain_training_charged"] for r in allm) > 1)}
    save("rdpo_hybrid.json", res)
    print(f"{res['state_dependent_sources']} of {res['sources']} sources are state-dependent in the trained policy; top 20 carry {res['top20_share']:.0%} of the state dependence "
          f"({res['sources_for_80pct']} sources for 80%); rank agreement with one-at-a-time cost {rank:+.2f}")
    for kind, sw in res["sweep"].items():
        for m, r in sw.items():
            print(f"  fixed law = {kind:7s} | {m:3d} state-dependent: relative variance {r['rho2_worst']:.3g} (twist {res['rho2_twist']:.3g}), cost x{r['cost_ratio']:.1f}, "
                  f"equal-CPU x{r['equal_cpu_gain']:.2f} (x{r['equal_cpu_gain_training_charged']:.2f} with training), max |z| {r['z_max']:.1f}")
    print("checks:", res["checks"])
    return res


def selftest():
    """Two-country exceedance, arbitrary policy: fixing sources at any positive law keeps the estimate unbiased (exact p from enumeration)."""
    import rdpo_gap as G
    from src.rdpo.experiments_v2 import final_config
    from src.rdpo.fast import FastPolicy
    from src.rdpo.problem import adb_model
    P = PortfolioProblem(adb_model(), "exceedance")
    cfg = final_config("exceedance")
    pol = FastPolicy(P, cfg.alpha, cfg.edge_features, cfg.relevance, value=cfg.value, stages=cfg.stages)
    pol.theta = np.random.default_rng(3).normal(0, 0.5, pol.d)
    _, ps, f = G.enumerate_outcomes(P)
    exact = float((G.outer(ps) * f).sum())
    stats = {}
    E.rollout(pol, 50_000, np.random.default_rng(1), stats=stats)
    for fixed in ({}, {k: stats[k]["law"] for k in list(stats)[::2]}, {k: stats[k]["law"] for k in stats}):
        o = E.rollout(pol, 400_000, np.random.default_rng(2), fixed=fixed)
        g = o["w"] * o["f"]
        assert abs(g.mean() - exact) < 4 * g.std() / np.sqrt(len(g)), (len(fixed), g.mean(), exact)
    print("selftest ok: the hybrid estimate matches the exact probability for no, half and all sources fixed")


if __name__ == "__main__":
    selftest()
    run()
