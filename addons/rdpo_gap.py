"""Why does RDPO not beat twisting at equal CPU when ceiling.py says a state-dependent sampler could gain 48-375x?
Additive, read-only; the brief's two-country model in the friend's own source form (6 sources, one draw each).

ceiling.py worked on event paths in time. The friend's samplers draw each source's whole-term outcome once, so the fair
baseline is the best product law over the 6 source outcomes. Here, per crisis event:
  exact      the 5 x 5 x 25 x 2 x 2 x 2 = 5,000 outcomes are enumerated; relative variance rho2 of (a) plain sampling,
             (b) the single exponential twist, (c) the best product law (every source law free; fixed-point iteration
             q_k(v) ^2 proportional to sum over the other sources of p^2 f / prod q_j, which lowers the second moment each sweep)
  measured   rho2 and CPU per path of the friend's TiltSampler at the twist, and of a trained RDPO policy
             (rdpo_estimate + FastPolicy, the friend's final configuration)
RDPO wins at equal CPU iff  rho2(best product) / rho2(RDPO)  >  CPU per path (RDPO) / CPU per path (product).
Pre-set rule, on joint exhaustion: that ratio of ratios >= 2 -> RDPO is worth it for this event (phase 3);
otherwise the cause is named: (1) the product law is already near zero variance, (2) the trained policy is far from
the ideal, (3) cost per path.
Limit: two countries only; training cost is reported separately and not charged.
"""
import itertools
import time

import numpy as np

import crisis as C
from _src import save, seed
from src.rdpo.experiments_v2 import final_config
from src.rdpo.fast import FastPolicy
from src.rdpo.problem import PortfolioProblem, adb_model
from src.rdpo.tilt import TiltSampler
from src.rdpo.train import _best_twist_sources, rdpo_estimate

TARGETS = ("exceedance", "joint_exhaustion", "joint_attachment")


def enumerate_outcomes(P):
    """All joint source outcomes: probability tensor factors, payoff tensor f of shape (V_1, ..., V_K)."""
    S = TiltSampler(P)
    shape = tuple(S.Vk)
    grid = np.indices(shape).reshape(P.K, -1)
    A = np.zeros((P.n, grid.shape[1]), int); R = np.zeros_like(A)
    for k in range(P.K):
        for i, inc in S.inc_a[k]:
            A[i] += np.asarray(inc)[grid[k]]
        for i, inc in S.inc_r[k]:
            R[i] += np.asarray(inc)[grid[k]]
    Ls = P.a_u * np.minimum(A, 4) + P.r_u * np.minimum(R, 1)
    i, j = P.pair
    return S, [np.asarray(p, float) for p in S.p], P.payoff_from(Ls.sum(0), Ls[i], Ls[j]).reshape(shape)


def outer(vs):
    out = np.ones(())
    for v in vs:
        out = np.multiply.outer(out, v)
    return out


def rho2_product(p, q, f, prob):
    with np.errstate(divide="ignore", invalid="ignore"):
        num = outer(p) ** 2 * f ** 2
        m2 = np.where(num > 0, num / outer(q), 0.0).sum()          # outcomes of probability 0 contribute nothing
    return float(m2 / prob ** 2 - 1)


def best_product(p, f, q0, sweeps=400):
    W = outer(p) ** 2 * f ** 2
    q = [x.copy() for x in q0]
    K = len(p)
    for _ in range(sweeps):
        for k in range(K):
            others = outer([q[j] if j != k else np.ones_like(q[j]) for j in range(K)])
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.where(W > 0, W / others, 0.0).sum(tuple(j for j in range(K) if j != k))
            q[k] = np.sqrt(s) / np.sqrt(s).sum()
    return q


def measured(P, n_tw, n_rd, b_fit, rng):
    S = TiltSampler(P)
    eta = _best_twist_sources(P) if P.target == "exceedance" else None
    if eta is not None:
        S.set_theta(S.twist_theta(eta))
    t0 = time.process_time(); o = S.sample(n_tw, rng); cpu_tw = (time.process_time() - t0) / n_tw
    g = o["w"] * o["f"]
    cfg = final_config(P.target)
    t0 = time.process_time(); r = rdpo_estimate(P, b_fit, rng, cfg); cpu_train = time.process_time() - t0
    pol = FastPolicy(P, cfg.alpha, cfg.edge_features, cfg.relevance, value=cfg.value, stages=cfg.stages)
    pol.theta = np.array(r["theta"], float)
    t0 = time.process_time(); o2 = pol.rollout(n_rd, rng); cpu_rd = (time.process_time() - t0) / n_rd
    g2 = o2["w"] * o2["f"]
    return {"eta": eta, "tilt_mean": float(g.mean()), "tilt_var": float(g.var()), "cpu_per_path_product": cpu_tw,
            "rdpo_mean": float(g2.mean()), "rdpo_se": float(g2.std() / np.sqrt(n_rd)), "rdpo_var": float(g2.var()), "cpu_per_path_rdpo": cpu_rd, "cpu_rdpo_training": cpu_train}


def measure(target, n_tw=2_000_000, n_rd=262_144, b_fit=65_536):
    rng = np.random.default_rng(seed(20261126))
    P = PortfolioProblem(adb_model(), target)
    S, p, f = enumerate_outcomes(P)
    prob = float((outer(p) * f).sum())
    m = measured(P, n_tw, n_rd, b_fit, rng)
    out = {"p": prob, "p_chain": C.prob(target), "rho2_plain": 1 / prob - 1}
    if m["eta"] is not None:
        S.set_theta(S.twist_theta(m["eta"]))
        out["rho2_twist_exact"] = rho2_product(p, S.q, f, prob)
        out["rho2_twist_measured"] = m["tilt_var"] / prob ** 2
        start = S.q
    else:
        start = p
    q = best_product(p, f, start)
    out["rho2_best_product"] = rho2_product(p, q, f, prob)
    out["rho2_rdpo_measured"] = m["rdpo_var"] / prob ** 2
    out["rdpo_estimate_z"] = (m["rdpo_mean"] - prob) / m["rdpo_se"]
    out["variance_gain_rdpo"] = out["rho2_best_product"] / out["rho2_rdpo_measured"]
    out["cost_ratio_rdpo"] = m["cpu_per_path_rdpo"] / m["cpu_per_path_product"]
    out["equal_cpu_gain_rdpo"] = out["variance_gain_rdpo"] / out["cost_ratio_rdpo"]
    out["cpu"] = {k: m[k] for k in ("cpu_per_path_product", "cpu_per_path_rdpo", "cpu_rdpo_training")}
    return out


def run():
    res = {t: measure(t) for t in TARGETS}
    try:
        import json
        from _src import OUT
        ce = json.loads((OUT / "ceiling.json").read_text())
        for t in TARGETS:
            res[t]["path_space"] = {k: ce[t][k] for k in ("rho2_rich_static", "rho2_projected_doob")}
    except FileNotFoundError:
        pass
    r = res["joint_exhaustion"]
    cause = ("product law already near zero variance" if r["rho2_best_product"] < 1 else
             "trained policy far from the ideal" if r["variance_gain_rdpo"] < 2 else "cost per path")
    res["verdict"] = "RDPO worth it for joint exhaustion" if r["equal_cpu_gain_rdpo"] >= 2 else f"not worth it: {cause}"
    save("rdpo_gap.json", res)
    for t in TARGETS:
        r = res[t]
        tw = f"twist {r['rho2_twist_exact']:.3g} (measured {r['rho2_twist_measured']:.3g}), " if "rho2_twist_exact" in r else ""
        print(f"{t}: p {r['p']:.3g} | relative variance: plain {r['rho2_plain']:.3g}, {tw}best product law {r['rho2_best_product']:.3g}, "
              f"RDPO measured {r['rho2_rdpo_measured']:.3g} (estimate z {r['rdpo_estimate_z']:+.1f}) | variance gain x{r['variance_gain_rdpo']:.2f}, "
              f"cost per path x{r['cost_ratio_rdpo']:.1f}, equal-CPU gain x{r['equal_cpu_gain_rdpo']:.3f} | training {r['cpu']['cpu_rdpo_training']:.1f} s")
    print("verdict:", res["verdict"])
    return res


def selftest():
    P = PortfolioProblem(adb_model(), "exceedance")
    S, p, f = enumerate_outcomes(P)
    prob = float((outer(p) * f).sum())
    assert abs(prob / C.prob("exceedance") - 1) < 1e-9                               # source form = crisis.py's chain
    assert abs(rho2_product(p, p, f, prob) / (1 / prob - 1) - 1) < 1e-9                  # no tilt = plain Monte Carlo
    S.set_theta(S.twist_theta(_best_twist_sources(P)))
    a, b = rho2_product(p, S.q, f, prob), rho2_product(p, best_product(p, f, S.q, 50), f, prob)
    assert b < a                                                                      # the iteration improves on the twist
    o = S.sample(400_000, np.random.default_rng(1))
    g = o["w"] * o["f"]
    assert abs(g.var() / prob ** 2 / a - 1) < 0.1                                     # enumeration = the friend's sampler
    print("selftest ok: source form = chain; no tilt = plain Monte Carlo; best product law beats the twist; enumeration = the friend's sampler")


if __name__ == "__main__":
    selftest()
    run()
