"""The four brief baselines (MC, scrambled Sobol, stratified, IS) and the
zero-variance backward-induction estimator (our strongest classical baseline).

Each estimator: (prob, N, rng) -> {target: estimate}, using N paths.
"""
import numpy as np
from scipy.stats import qmc

from src.model.term import ADD, YEARS
from .common import inv_cdf, terminal_states


def _evaluate(prob, s, w=None):
    out = {}
    for t, f in prob.f.items():
        v = f[s] if w is None else w * f[s]
        out[t] = float(v.mean())
    return out


# --- 1. Plain Monte Carlo -------------------------------------------------
def mc(prob, N, rng):
    Y = inv_cdf(prob.cdf, rng.random((N, YEARS)))
    return _evaluate(prob, terminal_states(Y))


# --- 2. Scrambled Sobol: 8 independent scrambles, averaged ------------------
N_SCRAMBLES = 8


def sobol(prob, N, rng, return_ci=False):
    n = max(N // N_SCRAMBLES, 1)
    per = []
    for _ in range(N_SCRAMBLES):
        eng = qmc.Sobol(d=YEARS, scramble=True, seed=rng)
        U = eng.random(n)
        per.append(_evaluate(prob, terminal_states(inv_cdf(prob.cdf, U))))
    out = {t: float(np.mean([e[t] for e in per])) for t in prob.f}
    if return_ci:
        from scipy.stats import t as tdist
        q = tdist.ppf(0.975, N_SCRAMBLES - 1)
        ci = {t: q * np.std([e[t] for e in per], ddof=1) / np.sqrt(N_SCRAMBLES) for t in prob.f}
        return out, ci
    return out


# --- 3. Stratified by first year with any payout ----------------------------
def stratified(prob, N, rng):
    p0 = prob.p[0]
    pos = prob.p[1:] / (1 - p0)
    pos_cdf = np.cumsum(pos); pos_cdf[-1] = 1.0
    n = max(N // YEARS, 1)
    tot = {t: 0.0 for t in prob.f}
    for k in range(1, YEARS + 1):
        Y = np.zeros((n, YEARS), dtype=np.int64)  # years before k: no payout
        Y[:, k - 1] = 1 + inv_cdf(pos_cdf, rng.random(n))
        if k < YEARS:
            Y[:, k:] = inv_cdf(prob.cdf, rng.random((n, YEARS - k)))
        e = _evaluate(prob, terminal_states(Y))
        wk = p0 ** (k - 1) * (1 - p0)
        for t in tot:
            tot[t] += wk * e[t]
    return tot


# --- 4. Importance sampling: half true, half loss-tilted ---------------------
def is_proposal(prob):
    L = prob.f["term_expected_principal_loss_usd"]  # annual-state loss = same map
    tilt = prob.p * L / (prob.p @ L)
    return 0.5 * prob.p + 0.5 * tilt


def importance(prob, N, rng, q=None):
    q = is_proposal(prob) if q is None else q
    qcdf = np.cumsum(q); qcdf[-1] = 1.0
    Y = inv_cdf(qcdf, rng.random((N, YEARS)))
    w = np.prod(prob.p[Y] / q[Y], axis=1)
    return _evaluate(prob, terminal_states(Y), w)


# --- 5. Zero-variance backward induction (Doob h-transform) ------------------
def value_functions(p, f, years=YEARS):
    """V[t][s] = E[f(final) | cumulative state s after t years], t = 0..years."""
    V = [None] * (years + 1)
    V[years] = f.astype(float)
    for t in range(years - 1, -1, -1):
        V[t] = (V[t + 1][ADD] * p[None, :]).sum(axis=1)
    return V


def zero_variance_single(prob, target, N, rng):
    """Sample from the exact Doob-transformed chain; every path's weight*f == mu."""
    p, f = prob.p, prob.f[target]
    V = value_functions(p, f)
    s = np.zeros(N, dtype=np.int64)
    w = np.ones(N)
    for t in range(YEARS):
        Q = p[None, :] * V[t + 1][ADD[s]]          # (N, 100) unnormalised
        Z = Q.sum(axis=1, keepdims=True)
        Q = Q / Z
        u = rng.random((N, 1))
        y = np.minimum((Q.cumsum(axis=1) < u).sum(axis=1), 99)
        w *= p[y] / Q[np.arange(N), y]
        s = ADD[s, y]
    return float(np.mean(w * f[s]))


def zero_variance(prob, N, rng):
    return {t: zero_variance_single(prob, t, N, rng) for t in prob.f}
