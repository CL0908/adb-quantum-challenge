"""A faster rollout for the friend's RDPO policy: same policy, same parameters, same paths. Additive, read-only.

rdpo_gap.py: a trained RDPO policy has 11-33x lower variance than the best product law but costs about 31x more per
path, so it only ties at equal CPU. The friend's FastPolicy.rollout builds the conditional law of every path for every
candidate outcome (N x V logits, exp, normalise, cumulative sum) at each source. But that law depends only on the
path's local state (codes of the touched countries and of the other ends of their edges, plus the target progress),
and few local states occur. Here, per source: group the paths by local state (the key the friend's own
FastPolicy.compile uses), compute the law once per distinct state by the friend's own table lookups, and
draw every path by one binary search in a stacked table of cumulative laws.
Pre-set rules: (1) with the same uniforms the paths, weights and payoffs equal the friend's rollout (two-country
model, three targets, and the 28-country pool); (2) cost per path <= 15x the product sampler's on the two-country
model; (3) equal-CPU gain over the best product law >= 2 on joint exhaustion (variance gain from rdpo_gap.json);
(4) on the 28-country pool, equal-CPU efficiency against the twist for the pool-exceedance probability >= 1.
Limit: training still uses the friend's engine and is timed separately, not charged.
"""
import json
import time

import numpy as np

import rdpo_gap as G
from _src import OUT, save, seed
from src.rdpo.experiments_v2 import final_config
from src.rdpo.fast import FastPolicy
from src.rdpo.problem import PortfolioProblem, adb_model
from src.rdpo.tilt import TiltSampler
from src.rdpo.train import _best_twist_sources, rdpo_estimate


def law(pol, k, tabs, S, Lt, Li, Lj):
    """q_k(. | state) for the given states, by the same table lookups as FastPolicy.rollout (lines for a non-frozen source)."""
    P, plan, src = pol.P, pol.plan[k], pol.P.src[k]
    psi_t, chi_t, h_t = tabs
    pi, pj = P.pair
    newS = {i: pol.next[k][i][S[:, i]] for i in plan["sites"]}
    lg = src["logp"][None, :].repeat(len(S), 0)
    for i in plan["fsites"]:
        lg += psi_t[i][newS[i]] - psi_t[i][S[:, i]][:, None]
    for e in plan["edges"]:
        i, j = P.edges[e]
        si = newS[i] if i in newS else S[:, i][:, None]
        sj = newS[j] if j in newS else S[:, j][:, None]
        lg += chi_t[e][si, sj] - chi_t[e][S[:, i], S[:, j]][:, None]
    ht = h_t[pol.blk[k]]
    if pol.pair_target:
        Lin = pol.loss_tab[newS[pi]] if pi in newS else Li[:, None]
        Ljn = pol.loss_tab[newS[pj]] if pj in newS else Lj[:, None]
        lg += ht[Lin, Ljn] - ht[Li, Lj][:, None]
    else:
        dL = sum(pol.loss_tab[newS[i]] - pol.loss_tab[S[:, i]][:, None] for i in plan["sites"])
        lg += ht[Lt[:, None] + dL] - ht[Lt][:, None]
    lg -= lg.max(1, keepdims=True)
    qt = np.exp(lg); qt /= qt.sum(1, keepdims=True)
    return (1 - pol.alpha) * qt + pol.alpha * src["p"][None, :]


def rollout(pol, N, rng, record=False, U=None, fixed=None, stats=None, trace=None):
    """Same contract and same random stream as FastPolicy.rollout (without the grad option).
    fixed: {source: law} draws those sources from a fixed law instead of the state-dependent one (rdpo_hybrid.py).
    stats: a dict that receives, per state-dependent source, its path-averaged law, the mean KL from it, and the number of states.
    trace: called as trace(k, S, Lt, logw) before each source is drawn (cap_modules.py)."""
    P = pol.P
    S, Lt, Li, Lj = pol._init_state(N)
    logw = np.zeros(N)
    acts = np.zeros((N, P.K), dtype=np.int64) if record else None
    pi, pj = P.pair
    tabs = pol._tables()
    for k in range(P.K):
        plan, src = pol.plan[k], P.src[k]
        V = len(src["p"])
        u = rng.random((N, 1))[:, 0] if U is None else U[:, k]
        if trace is not None:
            trace(k, S, Lt, logw)
        if plan["frozen"]:
            v = np.minimum(np.searchsorted(np.cumsum(src["p"]), u), V - 1)
        elif fixed is not None and k in fixed:
            v = np.minimum(np.searchsorted(np.cumsum(fixed[k]), u), V - 1)
            logw += src["logp"][v] - np.log(fixed[k][v])
        else:
            ends = sorted({i for e in plan["edges"] for i in P.edges[e]} | set(plan["sites"]))
            key = np.zeros(N, dtype=np.int64)
            for i in ends:
                key = key * 10 + S[:, i]
            key = (key * (P.F + 1) + Li) * (P.F + 1) + Lj if pol.pair_target else key * (P.Lmax + 1) + Lt
            _, rep, inv = np.unique(key, return_index=True, return_inverse=True)
            inv = inv.reshape(-1)
            q = law(pol, k, tabs, S[rep], Lt[rep], Li[rep], Lj[rep])
            cdf = q.cumsum(1)
            cdf[:, -1] = np.maximum(cdf[:, -1], 1.0)                       # the friend clips at the last outcome instead
            v = np.minimum(np.searchsorted((cdf + np.arange(len(rep))[:, None]).ravel(), inv + u) - inv * V, V - 1)
            logw += src["logp"][v] - np.log(q[inv, v])
            if stats is not None:
                c = np.bincount(inv, minlength=len(rep)) / N
                qb = c @ q
                stats[k] = {"law": qb, "kl": float(c @ (q * np.log(np.where(q > 0, q / np.maximum(qb, 1e-300), 1.0))).sum(1)), "states": len(rep)}
        for i in plan["sites"]:
            new = pol.next[k][i][S[:, i], v]
            Lt += pol.loss_tab[new] - pol.loss_tab[S[:, i]]
            S[:, i] = new
        if pi in plan["sites"]:
            Li = pol.loss_tab[S[:, pi]]
        if pj in plan["sites"]:
            Lj = pol.loss_tab[S[:, pj]]
        if record:
            acts[:, k] = v
    return {"f": P.payoff_from(Lt, Li, Lj), "w": np.exp(logw), "score": P.score_from(Lt, Li, Lj), "acts": acts}


def trained(P, b_fit, rng):
    cfg = final_config(P.target)
    t0 = time.process_time()
    r = rdpo_estimate(P, b_fit, rng, cfg)
    pol = FastPolicy(P, cfg.alpha, cfg.edge_features, cfg.relevance, value=cfg.value, stages=cfg.stages)
    pol.theta = np.array(r["theta"], float)
    return pol, time.process_time() - t0


def same_paths(pol, n=20000, s=7):
    a = pol.rollout(n, np.random.default_rng(s), record=True)
    b = rollout(pol, n, np.random.default_rng(s), record=True)
    return {"paths_equal": float((a["acts"] == b["acts"]).all(1).mean()), "max_weight_rel_diff": float(np.max(np.abs(b["w"] / a["w"] - 1))),
            "payoffs_equal": bool((a["f"] == b["f"]).all())}


def cpu(f, reps=3):
    best = np.inf
    for _ in range(reps):
        t0 = time.process_time(); out = f(); best = min(best, time.process_time() - t0)
    return best, out


def two_country(target, n=1_000_000, b_fit=65_536):
    rng = np.random.default_rng(seed(20261127))
    P = PortfolioProblem(adb_model(), target)
    pol, t_train = trained(P, b_fit, rng)
    S = TiltSampler(P)
    if target == "exceedance":
        S.set_theta(S.twist_theta(_best_twist_sources(P)))
    t_prod, _ = cpu(lambda: S.sample(n, rng))
    t_new, o = cpu(lambda: rollout(pol, n, rng))
    t_old, _ = cpu(lambda: pol.rollout(n // 8, rng), reps=1)
    g = o["w"] * o["f"]
    ref = json.loads((OUT / "rdpo_gap.json").read_text())[target]
    rho2 = float(g.var() / ref["p"] ** 2)
    return {"check": same_paths(pol), "estimate_z": float((g.mean() - ref["p"]) / (g.std() / np.sqrt(n))), "rho2_rdpo": rho2,
            "rho2_best_product": ref["rho2_best_product"], "cost_ratio_friend_engine": t_old * 8 / t_prod, "cost_ratio_this_engine": t_new / t_prod,
            "equal_cpu_gain": ref["rho2_best_product"] / rho2 / (t_new / t_prod), "cpu_training": t_train}


def pool_28(n_tw=2_000_000, n_rd=500_000, b_fit=65_536):
    import ap_risk
    rng = np.random.default_rng(seed(20261128))
    inp = ap_risk.inputs()
    P = PortfolioProblem(inp["model"], "exceedance", threshold=ap_risk.THRESH[0] * 1e6)
    S = TiltSampler(P); S.set_theta(S.twist_theta(_best_twist_sources(P)))
    t_tw, o = cpu(lambda: S.sample(n_tw, rng), reps=1)
    g = o["w"] * o["f"]
    pol, t_train = trained(P, b_fit, rng)
    t_rd, o2 = cpu(lambda: rollout(pol, n_rd, rng), reps=1)
    t_old, _ = cpu(lambda: pol.rollout(n_rd // 8, rng), reps=1)
    g2 = o2["w"] * o2["f"]
    p = float(g.mean())
    var_tw, var_rd = g.var() / p ** 2, g2.var() / p ** 2
    ct, cr = t_tw / n_tw, t_rd / n_rd
    return {"check": same_paths(pol, 5000), "sources": P.K, "p_twist": p, "p_twist_se": float(g.std() / np.sqrt(n_tw)), "p_rdpo": float(g2.mean()), "p_rdpo_se": float(g2.std() / np.sqrt(n_rd)),
            "rho2_twist": float(var_tw), "rho2_rdpo": float(var_rd), "variance_gain": float(var_tw / var_rd), "cost_ratio_friend_engine": t_old * 8 / n_rd / ct,
            "cost_ratio_this_engine": cr / ct, "equal_cpu_gain": float(var_tw / var_rd / (cr / ct)), "cpu_training": t_train,
            "equal_cpu_gain_training_charged": float(var_tw * t_tw / n_tw / (var_rd * (t_rd + t_train) / n_rd))}


def run():
    res = {"two_country": {t: two_country(t) for t in G.TARGETS}, "pool_28": pool_28()}
    two, big = res["two_country"], res["pool_28"]
    same = lambda c: c["paths_equal"] == 1.0 and c["max_weight_rel_diff"] < 1e-9 and c["payoffs_equal"]
    res["checks"] = {"same_paths": bool(all(same(r["check"]) for r in two.values()) and same(big["check"])),
                     "cost_ratio_le_15": bool(max(r["cost_ratio_this_engine"] for r in two.values()) <= 15),
                     "joint_exhaustion_gain_ge_2": bool(two["joint_exhaustion"]["equal_cpu_gain"] >= 2),
                     "pool_28_gain_ge_1": bool(big["equal_cpu_gain"] >= 1)}
    res["verdict"] = "strong" if all(res["checks"].values()) else "not strong"
    save("rdpo_engine.json", res)
    for t, r in two.items():
        print(f"{t}: same paths {r['check']['paths_equal']:.4f}, max weight diff {r['check']['max_weight_rel_diff']:.1e} | cost per path vs product sampler: friend's engine x{r['cost_ratio_friend_engine']:.1f}, "
              f"this engine x{r['cost_ratio_this_engine']:.1f} | relative variance {r['rho2_rdpo']:.3g} (best product law {r['rho2_best_product']:.3g}; estimate z {r['estimate_z']:+.1f}) | equal-CPU gain x{r['equal_cpu_gain']:.2f}")
    print(f"28-country pool ({big['sources']} sources): p twist {big['p_twist']:.4g} +- {big['p_twist_se']:.1g}, RDPO {big['p_rdpo']:.4g} +- {big['p_rdpo_se']:.1g} | relative variance twist {big['rho2_twist']:.3g}, RDPO {big['rho2_rdpo']:.3g} "
          f"(x{big['variance_gain']:.1f}) | cost per path: friend's engine x{big['cost_ratio_friend_engine']:.1f}, this engine x{big['cost_ratio_this_engine']:.1f} | equal-CPU gain x{big['equal_cpu_gain']:.2f} "
          f"(x{big['equal_cpu_gain_training_charged']:.2f} with {big['cpu_training']:.0f} s of training charged) | same paths {big['check']['paths_equal']:.4f}")
    print("checks:", res["checks"], "| verdict:", res["verdict"])
    return res


def selftest():
    for t in G.TARGETS:
        P = PortfolioProblem(adb_model(), t)
        cfg = final_config(t)
        pol = FastPolicy(P, cfg.alpha, cfg.edge_features, cfg.relevance, value=cfg.value, stages=cfg.stages)
        pol.theta = np.random.default_rng(3).normal(0, 0.5, pol.d)                 # an arbitrary policy, not a trained one
        c = same_paths(pol, 4000)
        assert c["paths_equal"] == 1.0 and c["max_weight_rel_diff"] < 1e-9 and c["payoffs_equal"], (t, c)
    print("selftest ok: with the same uniforms, paths, weights and payoffs equal the friend's rollout for an arbitrary policy (three targets)")


if __name__ == "__main__":
    selftest()
    run()
