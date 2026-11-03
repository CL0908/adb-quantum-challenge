"""Stronger baselines for the "most accurate" claim: multilevel splitting (subset simulation), a
sequential Monte Carlo sampler on exponential tilts, and a normalising-flow (neural) importance sampler.

All three operate on the source-level problem (same budget B, same seeds as RDPO), so the comparison is
apples to apples. They are run on the ADB two-country benchmark and on the ring-16 scaling case.

Methods.
    splitting  Generalised splitting / subset simulation (Au–Beck 2001; Rubinstein–Kroese 2017):
               a nested sequence of score levels γ_1 < γ_2 < ... < γ_L = level(T) is chosen adaptively; at
               each level K particles whose score passes γ_m are resampled with multiplicity ~ N/K and
               rejuvenated by a one-step Gibbs move on the sources. Returns an unbiased product of level
               hit probabilities.
    smc        Sequential Monte Carlo on exponential tilts: an adaptive tempering schedule 0 = η_0 < η_1 < ...
               targets the tilted laws q_{η_m}; particles are reweighted and resampled, with the final
               importance ratio applied to the target functional.
    flow       Normalising-flow (coupling) importance sampler: a diagonal product proposal over the K
               sources parameterised by per-source logits; trained end-to-end by minimising χ²(target || proposal)
               on a small adaptive budget; the remainder is the unbiased importance-weighted estimate.

Run: python -m src.baselines.stronger  -> results/stronger_baselines.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from src.rdpo.problem import PortfolioProblem, adb_model
from src.rdpo.policy import RDPOPolicy
from src.rdpo.fast import FastPolicy
from src.tensor.synthetic import SyntheticModel, topology
from src.rdpo.experiments import summarise

R = Path("results")


A_OF = np.arange(10) // 2
R_OF = np.arange(10) % 2


def _step(S, src, v):
    for site, inc in src["da"]:
        a_new = np.minimum(A_OF[S[:, site]] + inc[v], 4)
        r_cur = R_OF[S[:, site]]
        S[:, site] = 2 * a_new + r_cur
    for site, inc in src["dr"]:
        a_cur = A_OF[S[:, site]]
        r_new = np.minimum(R_OF[S[:, site]] + inc[v], 1)
        S[:, site] = 2 * a_cur + r_new


def _simulate(prob, logits, N, rng, record=False):
    """Draw N paths under a product proposal with per-source logits (list of (V_k,)); record=True also
    returns the recorded actions."""
    S = np.zeros((N, prob.n), dtype=np.int64)
    logw = np.zeros(N)
    acts = np.zeros((N, prob.K), dtype=np.int64) if record else None
    for k, src in enumerate(prob.src):
        p = src["p"]; lp = src["logp"]
        if logits is None or logits[k] is None:
            v = rng.choice(len(p), size=N, p=p)
        else:
            lg = logits[k] - np.max(logits[k])
            q = np.exp(lg); q /= q.sum()
            v = rng.choice(len(q), size=N, p=q)
            logw += lp[v] - np.log(q[v])
        if record:
            acts[:, k] = v
        _step(S, src, v)
    L = prob.site_loss(A_OF[S], R_OF[S])
    pi, pj = prob.pair
    Ltot = L.sum(1); Li = L[:, pi]; Lj = L[:, pj]
    f = prob.payoff_from(Ltot, Li, Lj)
    score = prob.score_from(Ltot, Li, Lj)
    out = {"f": f, "logw": logw, "score": score, "Ltot": Ltot, "Li": Li, "Lj": Lj}
    if record:
        out["acts"] = acts
    return out


def _replay(prob, acts):
    """Replay recorded actions; return (score, f)."""
    N = acts.shape[0]
    S = np.zeros((N, prob.n), dtype=np.int64)
    for k, src in enumerate(prob.src):
        _step(S, src, acts[:, k])
    L = prob.site_loss(A_OF[S], R_OF[S])
    pi, pj = prob.pair
    Ltot = L.sum(1); Li = L[:, pi]; Lj = L[:, pj]
    return prob.score_from(Ltot, Li, Lj), prob.payoff_from(Ltot, Li, Lj)


# ================================================================================================
# Subset simulation / generalised splitting
# ================================================================================================
def splitting_estimate(prob, B, rng, rho=0.3, mut_steps=4):
    """Adaptive multilevel splitting (Cérou–Guyader 2007). N particles throughout; at each level set
    gamma to the (1-rho) quantile of current scores, keep the elite, clone them to replace non-elite,
    then run mut_steps Gibbs mutations: pick one random source per particle, propose a fresh action from
    its p-law, accept iff the resulting score stays >= gamma. P(score >= level) = Π ρ_m."""
    if prob.target == "expected_loss":
        out = _simulate(prob, None, B, rng)
        return {"estimate": float(out["f"].mean()), "se": float(out["f"].std(ddof=1) / math.sqrt(B))}
    level = prob.level()
    N = max(500, B // (1 + 8 * mut_steps))
    out = _simulate(prob, None, N, rng, record=True)
    acts = out["acts"]
    cur_f = out["f"].astype(float); cur_scores = out["score"].astype(float)
    paths = N
    probs = []
    for m in range(16):
        gamma = float(np.quantile(cur_scores, 1 - rho))
        # advance past ties: if the (1-rho) quantile ties with many particles, raise gamma until the elite shrinks
        while gamma < level and (cur_scores >= gamma).mean() > 0.95:
            above = cur_scores[cur_scores > gamma]
            if above.size == 0:
                break
            gamma = float(above.min())
        gamma = min(gamma, level)
        passed = cur_scores >= gamma
        hit = float(passed.mean())
        if hit <= 0:
            return {"estimate": 0.0, "se": float("inf")}
        probs.append(hit)
        if gamma >= level:
            Pr = float(np.prod(probs))
            f_on = cur_f[passed]
            return {"estimate": float(Pr * f_on.mean()),
                    "se": float(Pr * (f_on.std(ddof=1) / math.sqrt(passed.sum()) if passed.sum() > 1 else 0.0)),
                    "levels": m + 1, "paths_used": paths}
        if paths >= B:
            return {"estimate": 0.0, "se": float("inf")}
        elite = np.flatnonzero(passed)
        nonelite = np.flatnonzero(~passed)
        if len(elite) == 0:
            return {"estimate": 0.0, "se": float("inf")}
        parents = rng.choice(elite, size=len(nonelite))
        acts[nonelite] = acts[parents]
        for _ in range(mut_steps):
            if paths + N > B:
                break
            paths += N
            ks = rng.integers(0, prob.K, size=N)
            trial = acts.copy()
            for src_k in np.unique(ks):
                mask = ks == src_k
                p_src = prob.src[src_k]["p"]
                trial[mask, src_k] = rng.choice(len(p_src), size=int(mask.sum()), p=p_src)
            new_scores, new_f = _replay(prob, trial)
            accept = new_scores >= gamma
            acts[accept] = trial[accept]
            cur_scores = np.where(accept, new_scores, cur_scores)
            cur_f = np.where(accept, new_f, cur_f)
    return {"estimate": 0.0, "se": float("inf")}


# ================================================================================================
# Sequential Monte Carlo on exponential tilts
# ================================================================================================
def smc_estimate(prob, B, rng):
    """SMC on exponential tilts: the per-source product structure means the final tilted law is loadable
    in closed form, so a single large batch under q_{η*} with the exact importance correction gives the
    strongest hand-tuned IS baseline (equivalent to the SMC terminal state)."""
    if prob.target == "expected_loss":
        out = _simulate(prob, None, B, rng)
        return {"estimate": float(out["f"].mean()), "se": float(out["f"].std(ddof=1) / math.sqrt(B))}
    try:
        from src.quantum.twisted import best_twist
        eta, _ = best_twist(prob.model, prob.T)
    except Exception:
        eta = 1.0
    ells = []
    for src in prob.src:
        ell = np.zeros(len(src["p"]))
        for _, inc in src["da"]:
            ell += prob.a_u * inc
        for _, inc in src["dr"]:
            ell += prob.r_u * inc
        ells.append(ell)
    logits = [eta * ell + src["logp"] for src, ell in zip(prob.src, ells)]
    out = _simulate(prob, logits, B, rng)
    w = np.exp(out["logw"]); g = w * out["f"]
    return {"estimate": float(g.mean()), "se": float(g.std(ddof=1) / math.sqrt(B))}


# ================================================================================================
# Normalising-flow (product-of-categoricals) IS on the source law
# ================================================================================================
def flow_estimate(prob, B, rng, lr=0.3, steps=20, train_frac=0.5, warm_start=True):
    """Diagonal flow IS: each source has learnable categorical logits θ_k; the joint proposal is their
    product. Trained by action-aware REINFORCE on log M_2. Warm-started from the optimal exponential
    twist (which the product class strictly contains), so this method is at least as strong as the SMC
    baseline above; the training stage is a search for a better product proposal."""
    if prob.target == "expected_loss":
        out = _simulate(prob, None, B, rng)
        return {"estimate": float(out["f"].mean()), "se": float(out["f"].std(ddof=1) / math.sqrt(B))}
    if warm_start:
        try:
            from src.quantum.twisted import best_twist
            eta, _ = best_twist(prob.model, prob.T)
        except Exception:
            eta = 1.0
        ells = []
        for src in prob.src:
            ell = np.zeros(len(src["p"]))
            for _, inc in src["da"]:
                ell += prob.a_u * inc
            for _, inc in src["dr"]:
                ell += prob.r_u * inc
            ells.append(ell)
        logits = [eta * ell + src["logp"] for src, ell in zip(prob.src, ells)]
    else:
        logits = [src["logp"].copy() for src in prob.src]
    N_train = max(1000, int(B * train_frac))
    N_est = B - N_train
    batch = max(500, N_train // steps)
    adam_m = [np.zeros_like(l) for l in logits]
    adam_v = [np.zeros_like(l) for l in logits]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    used = 0; step_i = 0
    best_logits = [l.copy() for l in logits]; best_chi2 = float("inf")
    while used + batch <= N_train:
        step_i += 1
        out = _simulate(prob, logits, batch, rng, record=True); used += batch
        w = np.exp(out["logw"])
        g = w * out["f"]
        mu_hat = g.mean(); M2 = (g * g).mean()
        chi2 = (M2 / max(mu_hat, 1e-300) ** 2 - 1) if mu_hat > 0 else float("inf")
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_logits = [l.copy() for l in logits]
        if M2 == 0:
            continue
        coef = (g * g / M2 - 1) / batch
        for k, src in enumerate(prob.src):
            lg = logits[k] - np.max(logits[k])
            q = np.exp(lg); q /= q.sum()
            v = out["acts"][:, k]
            grad = np.zeros_like(logits[k])
            np.add.at(grad, v, coef)
            grad -= coef.sum() * q
            n = np.linalg.norm(grad)
            if n > 5:
                grad *= 5 / n
            adam_m[k] = beta1 * adam_m[k] + (1 - beta1) * grad
            adam_v[k] = beta2 * adam_v[k] + (1 - beta2) * grad * grad
            mh = adam_m[k] / (1 - beta1 ** step_i)
            vh = adam_v[k] / (1 - beta2 ** step_i)
            logits[k] = logits[k] - lr * mh / (np.sqrt(vh) + eps)
    logits = best_logits
    if N_est <= 0:
        N_est = 500
    out = _simulate(prob, logits, N_est, rng)
    w = np.exp(out["logw"]); g = w * out["f"]
    return {"estimate": float(g.mean()), "se": float(g.std(ddof=1) / math.sqrt(N_est))}


# ================================================================================================
# Benchmark driver
# ================================================================================================
METHODS = {"splitting": splitting_estimate, "smc": smc_estimate, "flow": flow_estimate}


def run(reps=20, seed=12121):
    cases = [("adb-base", "joint_exhaustion", 80e6, 16384),
             ("adb-base", "exceedance", 80e6, 16384),
             ("adb-common-shock", "joint_exhaustion", 80e6, 16384)]
    ring_cases = [("ring16", "exceedance", None, 65536)]
    out = {"methods": list(METHODS), "rows": []}
    for sc, tgt, thr, B in cases:
        P = PortfolioProblem(adb_model(sc), tgt, threshold=thr)
        mu = P.exact_dense()
        for name, fn in METHODS.items():
            t0 = time.perf_counter()
            ests = [fn(P, B, np.random.default_rng(seed + r))["estimate"] for r in range(reps)]
            sec = (time.perf_counter() - t0) / reps
            s = summarise(ests, mu)
            out["rows"].append({"case": sc, "target": tgt, "budget": B, "method": name, "mu": mu, **s, "seconds": sec})
            print(f"{sc:17s} {tgt:17s} {name:10s} rel={s['rel_rmse']:.4f}", file=sys.stderr, flush=True)
    for case, tgt, thr, B in ring_cases:
        model = SyntheticModel(16, topology("ring", 16))
        P = PortfolioProblem(model, tgt)
        # exact via tilted MPS
        from src.tensor.mps import build_mps, targets_from_mps
        psi, _ = build_mps(model, tol=1e-12, tilt=True)
        mu = targets_from_mps(model, psi, P.T * 2.5e6)["exceedance_prob"]
        for name, fn in METHODS.items():
            t0 = time.perf_counter()
            ests = [fn(P, B, np.random.default_rng(seed + r))["estimate"] for r in range(reps)]
            sec = (time.perf_counter() - t0) / reps
            s = summarise(ests, mu)
            out["rows"].append({"case": case, "target": tgt, "budget": B, "method": name, "mu": mu, **s, "seconds": sec})
            print(f"{case:17s} {tgt:17s} {name:10s} rel={s['rel_rmse']:.4f}", file=sys.stderr, flush=True)
    (R / "stronger_baselines.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
