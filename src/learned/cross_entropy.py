"""Multilevel cross-entropy fitting of the log-linear Doob proposal.

Each CE iteration:
  1. sample N_ce paths from the current proposal (likelihood ratio w);
  2. choose a level γ = min(final level, (1-ρ)-quantile of the path score);
  3. elite weights W = w * H_γ(final state), with H = f once the final level is reached;
  4. per year t, solve the convex problem
       max_θt  Σ_j W_j [ θt·φ(s_{j,t} ⊕ y_{j,t}) − log Σ_y p(y) exp(θt·φ(s_{j,t} ⊕ y)) ] − ½κ|θt|²
  5. smooth θ ← β θ_new + (1−β) θ_old.
No exact enumeration of V is used: the method only touches sampled paths and the
annual distribution p (needed by any IS method to compute likelihood ratios).
"""
import numpy as np
from scipy.optimize import minimize

from src.model.term import ADD, YEARS
from .value_approx import DoobProposal

FULL = None  # filled per problem


def target_spec(prob, target, sc):
    """(score over term states, final level, f). Score drives the multilevel schedule."""
    from src.model.targets import losses
    L1, L2 = losses(sc)
    f = prob.f[target]
    if target in ("term_joint_attachment_probability", "term_both_bonds_exhausted_probability"):
        score = np.minimum(L1, L2)
        level = score[f > 0].min()
    elif target in ("term_portfolio_loss_exceedance_probability", "term_portfolio_stop_loss_usd"):
        score = L1 + L2
        level = score[f > 0].min()
    else:  # expected loss: f itself is not rare
        score = L1 + L2
        level = 0.0
    return score, level, f


def _fit_year(prop, t, S_t, Y_t, W, kappa, theta0):
    Phi = prop.Phi
    keep = W > 0
    if keep.sum() == 0:
        return theta0
    s, y, W = S_t[keep], Y_t[keep], W[keep] / W[keep].sum()
    Fall = Phi[ADD[s]]                 # (n, 100, d)
    Fy = Phi[ADD[s, y]]                # (n, d)
    lp = np.log(prop.p)[None, :]

    def obj(th):
        z = lp + Fall @ th             # (n, 100)
        zm = z.max(axis=1, keepdims=True)
        e = np.exp(z - zm)
        Zs = e.sum(axis=1, keepdims=True)
        logZ = (zm + np.log(Zs))[:, 0]
        P = e / Zs
        val = -(W @ (Fy @ th - logZ)) + 0.5 * kappa * th @ th
        grad = -(W @ (Fy - np.einsum("nk,nkd->nd", P, Fall))) + kappa * th
        return val, grad

    r = minimize(obj, theta0, jac=True, method="L-BFGS-B")
    return r.x


def fit(prob, target, sc, kind="pairwise", n_ce=2000, rho=0.1, max_iter=20, extra_iter=3,
        beta=0.7, kappa=1e-3, alpha=0.05, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    score, level, f = target_spec(prob, target, sc)
    prop = DoobProposal(prob.p, kind=kind, alpha=alpha)
    history, used, after = [], 0, 0
    for it in range(max_iter):
        Y, S, w = prop.sample(n_ce, rng)
        used += n_ce
        sc_final = score[S[:, -1]]
        gamma = min(level, np.quantile(sc_final, 1 - rho))
        # Discrete scores tie: if the elite set is too large, advance to the next attained level
        if gamma < level and (sc_final >= gamma).sum() > 2 * rho * n_ce:
            higher = sc_final[sc_final > gamma]
            if higher.size:
                gamma = min(level, higher.min())
        if gamma >= level:
            H = f[S[:, -1]]
            after += 1
        else:
            gamma = max(gamma, np.nextafter(0, 1)) if gamma <= 0 else gamma
            H = (sc_final >= gamma).astype(float)
        W = w * H
        new = np.array([_fit_year(prop, t, S[:, t], Y[:, t], W, kappa, prop.theta[t]) for t in range(YEARS)])
        prop.theta = beta * new + (1 - beta) * prop.theta
        history.append({"iter": it, "gamma": float(gamma), "level": float(level), "elite": int((W > 0).sum())})
        if verbose:
            print(history[-1])
        if after >= extra_iter:
            break
    return prop, {"history": history, "training_paths": used, "n_params": prop.n_params}
