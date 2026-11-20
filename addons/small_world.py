"""An exact small world as RDPO's teacher: the pool without per-country caps. Additive, read-only.

RDPO learns its progress function h_k(L) (the log of "probability of ending above the threshold, given the running
loss L after source k") by training, about 13 s on the 28-country pool. If the per-country payout caps are dropped,
the losses of the sources simply add, and that probability is exact by one backward pass over the sources
(K x Lmax x V operations, no sampling, no training). Here that table is given to the main author's FastPolicy as its
progress table, one per source, with every other feature at zero. The policy still runs on the true capped model and
its weights are exact, so the estimate is unbiased; the small world only decides where to look.
Scope: the pool-exceedance target. (The pair targets depend on two countries only and are exact on 100 states; see
ceiling.py.) Pre-set rules:
 W1 the estimate is within 3 standard errors of the exact value (two-country) and of the twist's (28-country pool);
 W2 two-country: relative variance <= 2x the trained RDPO's (rdpo_gap.json);
 W3 28-country pool: relative variance <= 2x the trained RDPO's (rdpo_engine.json) and building takes < 1 s;
 W4 28-country pool: equal-CPU efficiency against the twist > 1 with the building time charged;
 W5 the small world's own answer (no sampling) is within a factor 2 of the true probability.
Three seeds; the worst relative variance is reported.
"""
import json
import time

import numpy as np

import rdpo_engine as E
from _src import OUT, save, seed
from src.rdpo.experiments_v2 import final_config
from src.rdpo.fast import FastPolicy
from src.rdpo.problem import PortfolioProblem, adb_model
from src.rdpo.tilt import TiltSampler
from src.rdpo.train import _best_twist_sources


def value_tables(P):
    """V[k][L] = Pr(final loss > T | running loss L before source k+1 ... K) in the uncapped world; and its answer V_0(0)."""
    L = np.arange(P.Lmax + 1)
    V = P.payoff_from(L, 0 * L, 0 * L).astype(float)
    out = [None] * P.K
    for k in reversed(range(P.K)):
        out[k] = V
        src = P.src[k]
        d = sum(P.a_u * np.asarray(inc) for _, inc in src["da"]) + sum(P.r_u * np.asarray(inc) for _, inc in src["dr"])
        V = sum(p * V[np.minimum(L + dv, P.Lmax)] for p, dv in zip(src["p"], np.broadcast_to(d, len(src["p"]))))
    return out, float(V[0])


def teacher(P, tables=None):
    """The main author's FastPolicy with the small world's tables as progress tables (one per source) and nothing learned.
    tables: (V, answer) from another small world (cap_modules.py) instead of the uncapped one."""
    t0 = time.process_time()
    V, answer = tables or value_tables(P)
    pol = FastPolicy(P, final_config(P.target).alpha, edge_features=False, stages=P.K)
    psi_t, chi_t, _ = pol._tables()                                           # theta = 0: country and edge tables are zero
    h = [np.log(np.maximum(v, 1e-300)) for v in V]
    pol._tables = lambda: (psi_t, chi_t, h)
    return pol, answer, time.process_time() - t0


def measure(P, p_ref, p_se, n, seeds=(31, 32, 33)):
    pol, answer, t_build = teacher(P)
    rs = []
    for s in seeds:
        t, o = E.cpu(lambda: E.rollout(pol, n, np.random.default_rng(s)), reps=1)
        g = o["w"] * o["f"]
        rs.append((float(g.mean()), float(g.std() / np.sqrt(n)), float(g.var() / p_ref ** 2), t / n))
    return {"small_world_answer": answer, "answer_over_truth": answer / p_ref, "cpu_build": t_build, "p": [r[0] for r in rs],
            "z_max": max(abs(r[0] - p_ref) / np.hypot(r[1], p_se) for r in rs), "rho2_worst": max(r[2] for r in rs),
            "rho2_seeds": [r[2] for r in rs], "cost_per_path": float(np.median([r[3] for r in rs])), "paths": n}, pol


def two_country(n=1_000_000):
    P = PortfolioProblem(adb_model(), "exceedance")
    ref = json.loads((OUT / "rdpo_gap.json").read_text())["exceedance"]
    r, pol = measure(P, ref["p"], 0.0, n)
    S = TiltSampler(P); S.set_theta(S.twist_theta(_best_twist_sources(P)))
    t_prod, _ = E.cpu(lambda: S.sample(n, np.random.default_rng(1)))
    r.update(p_exact=ref["p"], rho2_trained_rdpo=ref["rho2_rdpo_measured"], rho2_twist=ref["rho2_twist_exact"],
             cost_ratio=r["cost_per_path"] / (t_prod / n), same_paths_as_reference_engine=E.same_paths(pol, 5000)["paths_equal"])
    r["equal_cpu_gain_vs_twist"] = r["rho2_twist"] / r["rho2_worst"] / r["cost_ratio"]
    return r


def pool_28(n=300_000, n_tw=2_000_000):
    import ap_risk
    rng = np.random.default_rng(seed(20261128))
    P = PortfolioProblem(ap_risk.inputs()["model"], "exceedance", threshold=ap_risk.THRESH[0] * 1e6)
    S = TiltSampler(P); S.set_theta(S.twist_theta(_best_twist_sources(P)))
    t_tw, o = E.cpu(lambda: S.sample(n_tw, rng), reps=1)
    g = o["w"] * o["f"]
    p, p_se, rho2_tw, c_tw = float(g.mean()), float(g.std() / np.sqrt(n_tw)), float(g.var() / g.mean() ** 2), t_tw / n_tw
    r, pol = measure(P, p, p_se, n)
    # the division of labour of rdpo_hybrid.py on the teacher's policy: state-dependent law on the m most state-dependent sources only
    import rdpo_hybrid as H
    c = {"P": P, "S": S, "pol": pol, "t_train": r["cpu_build"], "p": p, "p_se": p_se, "rho2_twist": rho2_tw, "cost_twist": c_tw}
    stats = H.dependence(c)
    order = sorted(stats, key=lambda k: -stats[k]["kl"])
    kl = np.array([stats[k]["kl"] for k in order])
    r["hybrid"] = {"top20_share": float(kl[:20].sum() / kl.sum()), "sweep": H.sweep(c, stats, order, "average", n=n)}
    ref = json.loads((OUT / "rdpo_engine.json").read_text())["pool_28"]
    r.update(sources=P.K, p_twist=p, p_twist_se=p_se, rho2_twist=rho2_tw, rho2_trained_rdpo=ref["rho2_rdpo"], cpu_training_rdpo=ref["cpu_training"],
             cost_ratio=r["cost_per_path"] / c_tw, cost_ratio_trained_rdpo=ref["cost_ratio_this_engine"])
    r["equal_cpu_gain_vs_twist"] = rho2_tw / r["rho2_worst"] / r["cost_ratio"]
    r["equal_cpu_gain_build_charged"] = rho2_tw * c_tw / (r["rho2_worst"] * (r["cost_per_path"] + r["cpu_build"] / n))
    return r


def run():
    res = {"two_country": two_country(), "pool_28": pool_28()}
    two, big = res["two_country"], res["pool_28"]
    res["checks"] = {"W1_unbiased": bool(two["z_max"] <= 3 and big["z_max"] <= 3),
                     "W2_two_country_variance": bool(two["rho2_worst"] <= 2 * two["rho2_trained_rdpo"]),
                     "W3_pool_variance_and_build": bool(big["rho2_worst"] <= 2 * big["rho2_trained_rdpo"] and big["cpu_build"] < 1),
                     "W4_pool_equal_cpu_gt_1": bool(big["equal_cpu_gain_build_charged"] > 1),
                     "W5_own_answer_within_2x": bool(0.5 <= two["answer_over_truth"] <= 2 and 0.5 <= big["answer_over_truth"] <= 2)}
    save("small_world.json", res)
    for name, r in res.items():
        if name == "checks":
            continue
        r = {**r}
        print(f"{name}: small world alone says {r['small_world_answer']:.4g} ({r['answer_over_truth']:.2f}x the truth), built in {r['cpu_build']:.3f} s | as RDPO's teacher: relative variance {r['rho2_worst']:.3g} "
              f"(trained RDPO {r['rho2_trained_rdpo']:.3g}, twist {r['rho2_twist']:.3g}), max |z| {r['z_max']:.1f}, cost x{r['cost_ratio']:.1f} the twist's, equal-CPU x{r['equal_cpu_gain_vs_twist']:.2f}")
    hy = big["hybrid"]
    print(f"division of labour on the teacher's policy: top 20 sources carry {hy['top20_share']:.0%} of the state dependence")
    for m, x in hy["sweep"].items():
        print(f"  {m:3d} state-dependent: relative variance {x['rho2_worst']:.3g}, cost x{x['cost_ratio']:.1f}, equal-CPU x{x['equal_cpu_gain']:.2f}, max |z| {x['z_max']:.1f}")
    print("checks:", res["checks"])
    return res


def selftest():
    """Without caps binding the small world is the real world: with a threshold nothing can reach, and by brute force on the two-country model."""
    import rdpo_gap as G
    P = PortfolioProblem(adb_model(), "exceedance")
    V, answer = value_tables(P)
    assert all(((v >= 0) & (v <= 1 + 1e-12)).all() and (np.diff(v) >= -1e-12).all() for v in V)      # probabilities, increasing in the running loss
    _, ps, f = G.enumerate_outcomes(P)
    exact = float((G.outer(ps) * f).sum())
    assert answer >= exact - 1e-15, (answer, exact)                               # dropping the caps can only add loss
    pol, _, _ = teacher(P)
    o = E.rollout(pol, 400_000, np.random.default_rng(5))
    g = o["w"] * o["f"]
    assert abs(g.mean() - exact) < 4 * g.std() / np.sqrt(len(g)), (g.mean(), exact)
    assert E.same_paths(pol, 4000)["paths_equal"] == 1.0
    print(f"selftest ok: uncapped answer {answer:.5f} >= exact {exact:.5f}; teacher policy unbiased; same paths as the main author's engine")


if __name__ == "__main__":
    selftest()
    run()
