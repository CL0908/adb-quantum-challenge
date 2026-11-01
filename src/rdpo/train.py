"""RDPO: budgeted training + estimation (Algorithm 1 in the report).

Given a total path budget B (training *and* estimation):

  0. Warm start (exceedance / stop-loss): θ_h ← η*·T, the optimal exponential twist of the
     uncapped loss (large-deviation Doob transform), so the initial policy already reaches the tail.
  1. Cross-entropy policy search (KL stage). Batches of size b; elite level
     γ = min(level, (1−ρ)-quantile of the batch scores), advanced past ties; elite weights
     W = w·1{score >= γ} (or W = w·f once γ = level). θ ← argmax Σ W log q̃_θ − κ|θ|²/2 (convex),
     smoothed with factor β.
  2. Rényi-2 policy gradient (χ² stage). REINFORCE on log E_q[(wf)²] with baseline:
        ∇ log M₂ = −E_q[(g²/M₂ − 1) ∇ log q_θ(path)],  g = w f,
     Adam steps; the iterate with the lowest *estimated* χ² is kept; early stop after `patience`
     steps without improvement.
  3. Estimation. The remaining budget is spent with the selected policy. With recycling, every
     batch whose policy was already level-reaching *before* the batch was drawn enters the estimate
     with weight ∝ batch size. The weights and the eligibility are fixed before each batch is
     sampled and each batch is unbiased given its policy, so the combined estimator is exactly
     unbiased. Standard error from batch-wise sample variances; 95% CI = ±1.96·se.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .problem import PortfolioProblem
from .policy import RDPOPolicy


@dataclass
class RDPOConfig:
    alpha: float = 0.05
    edge_features: bool = True
    relevance: bool = True           # freeze sources that cannot affect the payoff at q = p (exact Doob property)
    warm_start: bool = True
    ce: bool = True
    pg: bool = True
    recycle: bool = True
    qmc: bool = True                 # stage 3 uses randomized QMC (scrambled Sobol) inputs
    n_scrambles: int = 8
    batch_frac: float = 1 / 16       # b = max(batch_min, B * batch_frac)
    batch_min: int = 500
    train_frac: float = 0.5          # training may use at most this fraction of B
    rho: float = 0.1
    kappa: float = 1e-3
    beta: float = 0.7
    ce_extra: int = 1                # CE iterations after reaching the level
    pg_lr: float = 0.05
    pg_max_steps: int = 12
    pg_patience: int = 3
    clip: float = 5.0
    # ---- v2 options (roadmap WP1 / WP3) ----
    engine: str = "v1"               # "fast": table-driven engine (identical policy, ~10x faster rollouts)
    value: str = "linear"            # "spline": piecewise-linear progress basis (fixed-knot ReLU network)
    ci: str = "z"                    # "t": Student-t with Welch–Satterthwaite effective degrees of freedom
    value_of_training: bool = False  # stop training when one more batch is not expected to pay for itself
    theta0: object = None            # transfer warm start (e.g. yesterday's / a neighbouring scenario's policy)
    stages: int = 1                  # stage-dependent progress parameters (fast engine only)
    select: bool = False             # model selection: keep the initial (twist / transferred) policy if its
                                     # estimated χ² beats every trained iterate


def v2_config(**kw):
    """RDPO v2 defaults: fast engine, t-intervals, value-of-training stopping."""
    base = dict(engine="fast", ci="t", select=True)
    base.update(kw)
    return RDPOConfig(**base)


def _warm_start(pol: RDPOPolicy):
    P = pol.P
    if P.target in ("exceedance", "stop_loss"):
        from src.quantum.twisted import best_twist
        eta, _ = best_twist(P.model, P.T)
        for b in range(getattr(pol, "stages", 1)):
            pol.theta[pol.off_h + b * pol.G] = eta * P.T
        return eta
    return None


def rdpo_estimate(prob: PortfolioProblem, budget: int, rng, cfg: RDPOConfig = RDPOConfig()):
    if cfg.engine == "fast":
        from .fast import FastPolicy
        pol = FastPolicy(prob, cfg.alpha, cfg.edge_features, cfg.relevance, value=cfg.value, stages=cfg.stages)
    else:
        pol = RDPOPolicy(prob, cfg.alpha, cfg.edge_features, cfg.relevance)
    b = int(max(cfg.batch_min, budget * cfg.batch_frac))
    train_cap = int(cfg.train_frac * budget)
    used, batches, log = 0, [], {"ce": [], "pg": []}
    level = prob.level()
    eligible = False                     # is the *current* policy already level-reaching?
    eta = _warm_start(pol) if cfg.warm_start else None
    if cfg.theta0 is not None and len(cfg.theta0) == pol.d:
        pol.theta = np.array(cfg.theta0, dtype=float)

    def draw(n, record):
        nonlocal used
        out = pol.rollout(n, rng, record=record)
        used += n
        g = out["w"] * out["f"]
        batches.append({"g": g, "eligible": eligible})
        return out, g

    # ---- stage 1: cross-entropy policy search ---------------------------------------------
    after = 0
    init_theta, init_chi2 = pol.theta.copy(), np.inf
    if cfg.ce:
        while used + b <= train_cap:
            out, _ = draw(b, True)
            sc, w = out["score"], out["w"]
            gam = min(level, np.quantile(sc, 1 - cfg.rho))
            if gam < level and (sc >= gam).sum() > 2 * cfg.rho * b:
                hi = sc[sc > gam]
                if hi.size:
                    gam = min(level, hi.min())
            if gam >= level:
                W = w * out["f"]; after += 1
                if len(log["ce"]) == 0:          # the initial policy already reaches the target: score it
                    g0 = w * out["f"]
                    init_chi2 = (g0 ** 2).mean() / g0.mean() ** 2 - 1 if g0.mean() > 0 else np.inf
            else:
                W = w * (sc >= max(gam, 1e-12))
            log["ce"].append({"gamma": float(gam), "elite": int((W > 0).sum())})
            keep = W > 0
            if keep.any():
                c = W[keep] / W[keep].sum(); acts = out["acts"][keep]
                if hasattr(pol, "compile"):          # v2: compress to unique local states once per fit
                    comp = pol.compile(acts); Cs = pol.aggregate(comp, c)

                    def obj(th):
                        pol.theta = th
                        v, gr = pol.ce_from(comp, Cs)
                        return v + 0.5 * cfg.kappa * th @ th, gr + cfg.kappa * th
                else:
                    def obj(th):
                        pol.theta = th
                        v, gr = pol.replay(acts, c, "ce")
                        return v + 0.5 * cfg.kappa * th @ th, gr + cfg.kappa * th

                th0 = pol.theta.copy()
                r = minimize(obj, th0, jac=True, method="L-BFGS-B", options={"maxiter": 100})
                pol.theta = cfg.beta * r.x + (1 - cfg.beta) * th0
            if gam >= level:
                eligible = True          # policy now trained on f-weighted elites
            if after >= cfg.ce_extra:
                break
    else:
        eligible = True                  # warm start / zero init used as is

    # ---- stage 2: Rényi-2 (χ²) policy gradient ----------------------------------------------
    if cfg.pg and eligible:
        m = np.zeros(pol.d); s2 = np.zeros(pol.d)
        best_chi2, best_th, since = np.inf, pol.theta.copy(), 0
        hist = []
        for k in range(1, cfg.pg_max_steps + 1):
            if used + b > train_cap:
                break
            if cfg.value_of_training and len(hist) >= 3:
                # predicted per-step improvement factor of χ² (geometric mean of the last two steps); one more
                # batch pays off only if  χ·r/(R − b) < χ/R  with R the remaining budget
                r_hat = float(np.clip(np.sqrt(hist[-1] / hist[-3]), 0.0, 1.0))
                R_left = budget - used
                if r_hat > (R_left - b) / R_left:
                    break
            out, g = draw(b, True)
            mu, M2 = g.mean(), (g ** 2).mean()
            chi2 = M2 / mu ** 2 - 1 if mu > 0 else np.inf
            log["pg"].append({"chi2_hat": float(chi2)})
            hist.append(max(min(chi2, best_chi2), 1e-12) if np.isfinite(chi2) else (best_chi2 if np.isfinite(best_chi2) else 1e12))
            if chi2 < best_chi2 * 0.97:
                best_chi2, best_th, since = chi2, pol.theta.copy(), 0
            else:
                since += 1
                if since >= cfg.pg_patience:
                    break
            if M2 == 0:
                continue
            coef = (g ** 2 / M2 - 1) / b
            if hasattr(pol, "compile"):
                comp = pol.compile(out["acts"]); grad = -pol.mix_from(comp, pol.aggregate(comp, coef))
            else:
                grad = -pol.replay(out["acts"], coef, "mix")     # ∇ log M₂ (baseline 1)
            nrm = np.linalg.norm(grad)
            if nrm > cfg.clip:
                grad *= cfg.clip / nrm
            m = 0.9 * m + 0.1 * grad; s2 = 0.999 * s2 + 0.001 * grad ** 2
            pol.theta = pol.theta - cfg.pg_lr * (m / (1 - 0.9 ** k)) / (np.sqrt(s2 / (1 - 0.999 ** k)) + 1e-8)
        pol.theta = best_th
        if cfg.select and init_chi2 < best_chi2:
            pol.theta = init_theta; log["selected"] = "initial"

    # ---- stage 3: estimation ------------------------------------------------------------------
    final_policy_is_eligible = eligible
    eligible = True                       # stage-3 batches always count (selected policy)
    if cfg.qmc and budget - used >= cfg.n_scrambles * 64:
        from scipy.stats import qmc as _qmc
        m = int(np.floor(np.log2((budget - used) / cfg.n_scrambles)))
        means = []
        for _ in range(cfg.n_scrambles):
            U = _qmc.Sobol(d=prob.K, scramble=True, seed=rng).random(2 ** m)
            out = pol.rollout(2 ** m, rng, U=U); used += 2 ** m
            means.append((out["w"] * out["f"]).mean())
        # one RQMC block: its mean is unbiased; its variance is estimated between scrambles
        batches.append({"g": None, "rqmc_means": np.array(means), "n": cfg.n_scrambles * 2 ** m,
                        "eligible": True, "final": True})
    while used < budget:
        n = min(b * 4, budget - used)
        draw(n, False)
        batches[-1]["final"] = True
    if cfg.recycle:
        use = [bt for bt in batches if bt["eligible"]]
    else:
        use = [bt for bt in batches if bt.get("final")]
    if not use:
        use = batches[-1:]
    size = lambda bt: bt["n"] if bt.get("g") is None else len(bt["g"])
    total = lambda bt: bt["n"] * bt["rqmc_means"].mean() if bt.get("g") is None else bt["g"].sum()
    def var_of_sum(bt):            # Var(Σ g) for an iid batch; n² Var(mean) for an RQMC block
        if bt.get("g") is None:
            r = bt["rqmc_means"]
            return bt["n"] ** 2 * r.var(ddof=1) / len(r)
        return len(bt["g"]) * bt["g"].var(ddof=1) if len(bt["g"]) > 1 else 0.0
    n_tot = sum(size(bt) for bt in use)
    est = sum(total(bt) for bt in use) / n_tot
    var = sum(var_of_sum(bt) for bt in use) / n_tot ** 2
    se = float(np.sqrt(var))
    zq, dof = 1.96, float("inf")
    if cfg.ci == "t" and var > 0:
        from scipy.stats import t as _t
        def dof_of(bt):
            return (len(bt["rqmc_means"]) - 1) if bt.get("g") is None else max(len(bt["g"]) - 1, 1)
        contrib = [var_of_sum(bt) / n_tot ** 2 for bt in use]
        den = sum(c ** 2 / dof_of(bt) for c, bt in zip(contrib, use))
        dof = var ** 2 / den if den > 0 else float("inf")
        zq = float(_t.ppf(0.975, dof))
    return {"estimate": float(est), "se": se, "ci": (float(est - zq * se), float(est + zq * se)), "dof": dof,
            "paths_used": int(used), "paths_in_estimate": int(n_tot), "n_params": pol.n_params,
            "ce_iters": len(log["ce"]), "pg_steps": len(log["pg"]), "eta_warm": eta,
            "reached_level": bool(final_policy_is_eligible), "log": log, "theta": pol.theta.tolist()}
