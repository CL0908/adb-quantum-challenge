"""Dream and wake: does RDPO (which imagines the crisis) and the twin (which checks the world) each hand the other
something it lacks? Six exchanges, each tested against exact answers. Additive, read-only; the brief's two-country
model in the friend's source form (5,000 outcomes, enumerated), targets pool exceedance and joint exhaustion.

Design. RDPO conditions on the future: in a given world it learns how a crisis unfolds. The twin conditions on the
past: it holds a distribution over worlds and revises it. A "world" here is the friend's own three scenario inputs
(dependence multiplier m, rain multiplier, shared rain share); the prior over them is an assumption: uniform on the
range the brief's scenario table spans, m in [0, 2], rain multiplier in [1, 2], shared rain share in [0, 0.5].
The policy is trained once, in the brief's base world ("the dream"), with the friend's code; paths are drawn with
rdpo_engine.rollout (same paths as the friend's rollout).

RDPO gives the twin
  R1 what a crisis looks like   crisis-conditional frequency of every source outcome from dream paths
  R2 how close we are           the learned value function, read as the factor by which each source outcome multiplies
                                the crisis probability (the early-warning multiplier), against the exact Doob function
  R3 which input is lethal      d log P(crisis) / d(world input) from the same dream paths (score function)
The twin gives RDPO
  T1 where to dream             split a fixed path budget over worlds in proportion to the crisis probability the twin
                                predicts for each world (from R3's gradient), instead of evenly
  T2 how clear is enough        stop when the sampling error is a tenth of the spread of the answer over the twin's
                                rate posterior (quality_rates_unc.json), instead of the friend's fixed budget of 65,536
  T3 when the dream is stale    the twin's PSIS k-hat on the dream's weights in a new world flags the worlds where the
                                base-world policy must be retrained
Pre-set rules (all on pool exceedance unless said):
  R1 every frequency within 3 standard errors of exact, and equal-CPU efficiency against twist paths >= 1 (median)
  R2 weighted R^2 >= 0.9 between learned and exact log multipliers, for both targets
  R3 the three derivatives within 3 standard errors of exact and the same ranking by size
  T1 variance of the world-averaged crisis probability: even split / twin-guided split >= 1.5
  T2 paths needed <= 1/10 of 65,536
  T3 flagged worlds are at most half of all worlds, and no unflagged world has relative variance above twice the
     freshly trained policy's in that world (checked on the worlds where a fresh policy is trained)
Limits: two countries; one base policy; the prior over worlds is assumed, not estimated; training time is reported, not
charged; T2 uses the twin's rate posterior while T1 and T3 use the scenario-input prior.
"""
import json
import math
import time

import numpy as np

import rdpo_engine as E
import rdpo_gap as G
import twin
from _src import OUT, save, seed
from src.rdpo.experiments_v2 import final_config
from src.rdpo.fast import FastPolicy
from src.rdpo.problem import PortfolioProblem
from src.rdpo.tilt import TiltSampler
from src.rdpo.train import _best_twist_sources
from src.tensor.synthetic import SyntheticModel, topology

BASE = {"m": 1.0, "rain_mult": 1.0, "rain_shared": 0.25}
INPUTS = ("m", "rain_mult", "rain_shared")
RANGE = {"m": (0.0, 2.0), "rain_mult": (1.0, 2.0), "rain_shared": (0.0, 0.5)}
EPS = {"m": 0.02, "rain_mult": 0.02, "rain_shared": 0.005}


def problem(target, **kw):
    return PortfolioProblem(SyntheticModel(2, topology("chain", 2), **{**BASE, **kw}), target)


def exact(P):
    _, p, f = G.enumerate_outcomes(P)
    return p, f, float((G.outer(p) * f).sum())


def policy_in(P, theta):
    cfg = final_config(P.target)
    pol = FastPolicy(P, cfg.alpha, cfg.edge_features, cfg.relevance, value=cfg.value, stages=cfg.stages)
    pol.theta = np.array(theta, float)
    return pol


def batch_se(num, den, batches=20):
    """Standard error of ratio estimates sum(num) / sum(den) (num may have several columns), from batch means."""
    idx = np.array_split(np.arange(len(den)), batches)
    r = np.array([num[i].sum(0) / den[i].sum() for i in idx])
    return r.std(0, ddof=1) / math.sqrt(batches)


# ------------------------------------------------------------------------------------------- R1: what a crisis looks like
def looks(P, pol, p, f, prob, n, rng):
    K = P.K
    def freq(acts, g):
        cols, num = [], []
        for k in range(K):
            for v in range(len(p[k])):
                num.append(g * (acts[:, k] == v)); cols.append((k, v))
        num = np.stack(num, 1)
        return cols, num.sum(0) / g.sum(), batch_se(num, g)
    t0 = time.process_time(); o = E.rollout(pol, n, rng, record=True); t_rd = time.process_time() - t0
    cols, fr, se = freq(o["acts"], o["w"] * o["f"])
    S = TiltSampler(P); S.set_theta(S.twist_theta(_best_twist_sources(P)))
    t0 = time.process_time(); o2 = S.sample(n, rng, record=True); t_tw = time.process_time() - t0
    _, fr2, se2 = freq(o2["acts"].T.astype(np.int64), o2["w"] * o2["f"])
    joint = G.outer(p) * f
    ex = np.array([joint.sum(tuple(j for j in range(K) if j != k))[v] / prob for k, v in cols])
    keep = (ex > 1e-6) & (se > 0) & (se2 > 0)
    z = (fr[keep] - ex[keep]) / se[keep]
    eff = (se2[keep] ** 2 * t_tw) / (se[keep] ** 2 * t_rd)
    lift = np.array([ex[j] / p[k][v] if p[k][v] > 0 else 0.0 for j, (k, v) in enumerate(cols)])
    top = sorted(zip(lift, cols), reverse=True)[:3]
    return {"outcomes_checked": int(keep.sum()), "max_abs_z": float(np.abs(z).max()), "equal_cpu_efficiency_vs_twist_median": float(np.median(eff)),
            "most_over_represented": [{"source": P.src[k]["kind"] + str(P.src[k]["sites"]), "outcome": int(v), "lift": float(l)} for l, (k, v) in top]}, o


# ------------------------------------------------------------------------------------------- R2: how close we are
def closeness(P, pol):
    """log(q*/p) exact against log(q~/p) learned, for every (source step, state, outcome), weighted by how often a
    crisis passes through it. Codes: state of each country 0..9; 100 joint states."""
    K = P.K
    lt = pol.loss_tab
    s0, s1 = np.divmod(np.arange(100), 10)
    V = [None] * (K + 1)
    V[K] = P.payoff_from(lt[s0] + lt[s1], lt[s0], lt[s1]).astype(float)
    nxt = []
    for k in range(K):
        a = pol.next[k][0][s0] if 0 in pol.next[k] else np.repeat(s0[:, None], len(P.src[k]["p"]), 1)
        b = pol.next[k][1][s1] if 1 in pol.next[k] else np.repeat(s1[:, None], len(P.src[k]["p"]), 1)
        nxt.append(a * 10 + b)
    for k in reversed(range(K)):
        V[k] = (P.src[k]["p"][None, :] * V[k + 1][nxt[k]]).sum(1)
    tabs = pol._tables()
    S = np.stack([s0, s1], 1)
    occ = np.zeros(100); occ[0] = 1.0
    xs, ys, ws = [], [], []
    for k in range(K):
        p = P.src[k]["p"]
        with np.errstate(divide="ignore", invalid="ignore"):
            qstar = np.where(V[k][:, None] > 0, p[None, :] * V[k + 1][nxt[k]] / V[k][:, None], 0.0)
        if pol.plan[k]["frozen"]:
            q = np.repeat(p[None, :], 100, 0)
        else:
            q = (E.law(pol, k, tabs, S, lt[s0] + lt[s1], lt[s0], lt[s1]) - pol.alpha * p[None, :]) / (1 - pol.alpha)     # the unmixed learned law
        w = occ[:, None] * qstar
        m = (w > 0) & (p[None, :] > 0)
        xs.append(np.log(qstar[m] / np.broadcast_to(p[None, :], q.shape)[m])); ys.append(np.log(np.maximum(q[m], 1e-300) / np.broadcast_to(p[None, :], q.shape)[m])); ws.append(w[m])
        new = np.zeros(100); np.add.at(new, nxt[k].ravel(), w.ravel()); occ = new
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    w = w / w.sum()
    mx, my = w @ x, w @ y
    cov, vx, vy = w @ ((x - mx) * (y - my)), w @ (x - mx) ** 2, w @ (y - my) ** 2
    k0 = next(k for k in range(K) if P.src[k]["kind"] == "eq1" and P.src[k]["sites"][0] == 0)
    q0 = (E.law(pol, k0, tabs, S[:1], np.zeros(1, int), np.zeros(1, int), np.zeros(1, int))[0] - pol.alpha * P.src[k0]["p"]) / (1 - pol.alpha)
    return {"r2": float(cov ** 2 / (vx * vy)), "slope": float(cov / vx), "mean_abs_log_error": float(w @ np.abs(y - x)),
            "multiplier_after_25pct_eq_payout_KG_exact": float(V[k0 + 1][nxt[k0][0, 1]] / V[k0][0]), "multiplier_after_25pct_eq_payout_KG_learned": float(q0[1] / P.src[k0]["p"][1])}


# ------------------------------------------------------------------------------------------- R3: which input is lethal
def scores(target, acts):
    """sum over sources of d log p_k(outcome) / d input, for each path and each world input (central differences of the laws)."""
    out = []
    for name in INPUTS:
        hi, lo = problem(target, **{name: BASE[name] + EPS[name]}), problem(target, **{name: BASE[name] - EPS[name]})
        s = np.zeros(acts.shape[0])
        for k in range(hi.K):
            d = np.where(hi.src[k]["p"] > 0, hi.src[k]["logp"] - lo.src[k]["logp"], 0.0) / (2 * EPS[name])
            s += d[acts[:, k]]
        out.append(s)
    return np.stack(out, 1)


def lethal(target, o):
    g = o["w"] * o["f"]
    sc = scores(target, o["acts"])
    est, se = (g[:, None] * sc).sum(0) / g.sum(), batch_se(g[:, None] * sc, g)
    ex = np.array([(math.log(exact(problem(target, **{n: BASE[n] + EPS[n]}))[2]) - math.log(exact(problem(target, **{n: BASE[n] - EPS[n]}))[2])) / (2 * EPS[n]) for n in INPUTS])
    width = np.array([RANGE[n][1] - RANGE[n][0] for n in INPUTS])                 # effect of moving an input across its whole range
    return {"inputs": list(INPUTS), "dlogp_exact": ex.tolist(), "dlogp_from_dream": est.tolist(), "z": ((est - ex) / se).tolist(),
            "effect_over_range_exact": (ex * width).tolist(), "effect_over_range_from_dream": (est * width).tolist(),
            "same_ranking": bool((np.argsort(-np.abs(ex * width)) == np.argsort(-np.abs(est * width))).all())}, est


# ------------------------------------------------------------------------------------------- T1-T3: the twin steers the dream
def worlds(rng, M):
    return [{n: float(rng.uniform(*RANGE[n])) for n in INPUTS} for _ in range(M)]


def khat(hit):
    if len(hit) <= 50:
        return 1.0
    with np.errstate(all="ignore"):
        return twin.khat(hit)


def steer(target, theta, grad, p_base, rho2_base, rng, M=40, n=100_000, n_fresh=8, b_fit=65_536):
    from src.rdpo.train import rdpo_estimate
    W = worlds(rng, M)
    rows = []
    for kw in W:
        P = problem(target, **kw)
        pe = exact(P)[2]
        o = E.rollout(policy_in(P, theta), n, rng)
        g = o["w"] * o["f"]
        hit = g[g > 0]
        rows.append({**kw, "p_exact": pe, "p_dream": float(g.mean()), "z": float((g.mean() - pe) / (g.std() / math.sqrt(n))), "rho2_stale": float(g.var() / pe ** 2),
                     "khat": khat(hit), "p_predicted": float(p_base * math.exp(sum(grad[j] * (kw[nm] - BASE[nm]) for j, nm in enumerate(INPUTS))))})
    p, rho2, pred = (np.array([r[k] for r in rows]) for k in ("p_exact", "rho2_stale", "p_predicted"))
    var = lambda a: float((p ** 2 * rho2 / a).sum())                                # times 1 / (M^2 N), the same for every split
    even, guided, oracle = np.full(M, 1 / M), pred / pred.sum(), p * np.sqrt(rho2) / (p * np.sqrt(rho2)).sum()
    t1 = {"worlds": M, "p_over_worlds_mean": float(p.mean()), "p_over_worlds_cv": float(p.std() / p.mean()), "prediction_log_error_median": float(np.median(np.abs(np.log(pred / p)))),
          "variance_even_over_guided": var(even) / var(guided), "variance_even_over_best_possible": var(even) / var(oracle), "max_abs_z_dream_in_other_worlds": float(max(abs(r["z"]) for r in rows))}
    # T3: fresh policies on worlds spread over the k-hat range
    order = np.argsort([r["khat"] for r in rows])
    pick = order[np.linspace(0, M - 1, n_fresh).round().astype(int)]
    for i in pick:
        P = problem(target, **W[i])
        r = rdpo_estimate(P, b_fit, rng, final_config(target))
        o = E.rollout(policy_in(P, r["theta"]), n, rng)
        g = o["w"] * o["f"]
        rows[i]["rho2_fresh"] = float(g.var() / rows[i]["p_exact"] ** 2)
    flagged = np.array([r["khat"] > 0.7 for r in rows])
    ratios = [(bool(flagged[i]), rows[i]["rho2_stale"] / rows[i]["rho2_fresh"]) for i in pick]
    t3 = {"khat_threshold": 0.7, "flagged_share": float(flagged.mean()), "khat_range": [float(min(r["khat"] for r in rows)), float(max(r["khat"] for r in rows))],
          "stale_over_fresh_checked": [{"flagged": f, "ratio": float(x)} for f, x in ratios], "worst_unflagged_ratio": float(max([x for f, x in ratios if not f], default=0.0)),
          "worst_ratio": float(max(x for _, x in ratios)), "rho2_stale_range": [float(rho2.min()), float(rho2.max())], "rho2_base_world": rho2_base}
    return t1, t3, rows


def enough(rho2, key):
    s = json.loads((OUT / "quality_rates_unc.json").read_text())["sweeps"]["rates_only"]["summary"][key]
    spread = s["std"] / s["mean"]
    need = rho2 / (0.1 * spread) ** 2
    return {"spread_of_answer_over_rate_posterior": spread, "paths_needed": float(need), "friend_fixed_budget": 65536, "share_of_fixed_budget": float(need / 65536)}


def run(n=1_000_000):
    rng = np.random.default_rng(seed(20261129))
    res = {}
    for target in ("exceedance", "joint_exhaustion"):
        P = problem(target)
        p, f, prob = exact(P)
        pol, t_train = E.trained(P, 65_536, rng)
        r = {"p": prob, "cpu_training": t_train, "R2_closeness": closeness(P, pol)}
        if target == "exceedance":
            r["R1_looks"], o = looks(P, pol, p, f, prob, n, rng)
            g = o["w"] * o["f"]
            rho2 = float(g.var() / prob ** 2)
            r["R3_lethal"], grad = lethal(target, o)
            r["T1_where"], r["T3_stale"], r["worlds"] = steer(target, pol.theta, grad, prob, rho2, rng)
            r["T2_enough"] = enough(rho2, "exceedance")
        res[target] = r
    x = res["exceedance"]
    res["checks"] = {"R1_looks": bool(x["R1_looks"]["max_abs_z"] <= 3 and x["R1_looks"]["equal_cpu_efficiency_vs_twist_median"] >= 1),
                     "R2_closeness": bool(min(res[t]["R2_closeness"]["r2"] for t in ("exceedance", "joint_exhaustion")) >= 0.9),
                     "R3_lethal": bool(max(abs(z) for z in x["R3_lethal"]["z"]) <= 3 and x["R3_lethal"]["same_ranking"]),
                     "T1_where": bool(x["T1_where"]["variance_even_over_guided"] >= 1.5),
                     "T2_enough": bool(x["T2_enough"]["share_of_fixed_budget"] <= 0.1),
                     "T3_stale": bool(x["T3_stale"]["flagged_share"] <= 0.5 and x["T3_stale"]["worst_unflagged_ratio"] <= 2)}
    res["passed"] = int(sum(res["checks"].values()))
    res["verdict"] = "strong" if res["passed"] == 6 else f"not strong ({res['passed']}/6)"
    save("dreamwake.json", res)
    a, b, c = x["R1_looks"], x["R2_closeness"], x["R3_lethal"]
    print(f"R1 what a crisis looks like: {a['outcomes_checked']} outcome frequencies, max |z| {a['max_abs_z']:.2f}; equal-CPU efficiency vs twist paths x{a['equal_cpu_efficiency_vs_twist_median']:.2f}")
    for t in ("exceedance", "joint_exhaustion"):
        q = res[t]["R2_closeness"]
        print(f"R2 how close ({t}): R^2 {q['r2']:.3f}, slope {q['slope']:.2f}, mean |log error| {q['mean_abs_log_error']:.2f}; after a 25% earthquake payout in KG the crisis probability is multiplied by "
              f"{q['multiplier_after_25pct_eq_payout_KG_exact']:.2f} (exact), {q['multiplier_after_25pct_eq_payout_KG_learned']:.2f} (learned)")
    print("R3 which input is lethal: " + "; ".join(f"{n}: exact {e:+.3f}, dream {d:+.3f} (z {z:+.1f})" for n, e, d, z in zip(c["inputs"], c["effect_over_range_exact"], c["effect_over_range_from_dream"], c["z"])) + f" | same ranking {c['same_ranking']}")
    d, e, s = x["T1_where"], x["T2_enough"], x["T3_stale"]
    print(f"T1 where to dream: {d['worlds']} worlds, crisis probability varies with CV {d['p_over_worlds_cv']:.2f}; twin's prediction off by a median factor {math.exp(d['prediction_log_error_median']):.2f}; "
          f"variance even / guided x{d['variance_even_over_guided']:.2f} (best possible x{d['variance_even_over_best_possible']:.2f}); base-world dream in other worlds: max |z| {d['max_abs_z_dream_in_other_worlds']:.1f}")
    print(f"T2 how clear is enough: answer varies by {100 * e['spread_of_answer_over_rate_posterior']:.0f}% over the rate posterior -> {e['paths_needed']:.0f} paths, {100 * e['share_of_fixed_budget']:.2f}% of 65,536")
    print(f"T3 when the dream is stale: k-hat {s['khat_range'][0]:.2f}-{s['khat_range'][1]:.2f}, flagged {100 * s['flagged_share']:.0f}%; relative variance of the base-world policy in other worlds {s['rho2_stale_range'][0]:.2f}-{s['rho2_stale_range'][1]:.2f} "
          f"(base world {s['rho2_base_world']:.2f}); stale / fresh on {len(s['stale_over_fresh_checked'])} worlds: worst {s['worst_ratio']:.2f}, worst unflagged {s['worst_unflagged_ratio']:.2f}")
    print("checks:", res["checks"], "| verdict:", res["verdict"])
    return res


def selftest():
    P = problem("exceedance")
    p, f, prob = exact(P)
    assert abs(prob / G.C.prob("exceedance") - 1) < 1e-9                             # base world = the brief's model
    pol = policy_in(P, np.zeros(FastPolicy(P, stages=4).d))                           # an untrained policy samples the model itself
    c = closeness(P, pol)
    assert abs(c["multiplier_after_25pct_eq_payout_KG_learned"] - 1) < 1e-9 and c["multiplier_after_25pct_eq_payout_KG_exact"] > 1
    o = E.rollout(pol, 200_000, np.random.default_rng(2), record=True)
    sc = scores("exceedance", o["acts"])
    assert np.all(np.abs(sc.mean(0)) < 4 * sc.std(0) / math.sqrt(len(sc)))             # a score has mean zero under its own law
    print("selftest ok: base world = the brief's model; an untrained policy has multiplier 1; scores have mean zero under the model")


if __name__ == "__main__":
    selftest()
    run()
