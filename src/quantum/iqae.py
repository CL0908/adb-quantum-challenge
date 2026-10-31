"""Iterative quantum amplitude estimation (Grinko et al. 2021, Clopper-Pearson variant),
emulated exactly: a round with k Grover iterations returns '1' with probability
sin^2((2k+1)θ_a), a = sin^2 θ_a. This is the ideal noise-free behaviour of the circuit
Q^k A |0>, so oracle-call counts are exact for the algorithm; hardware noise is not modelled.

Also: the IS -> QAE cascade. With proposal q and bounded weights, A encodes
g(path) = w(path) f(path) / M ∈ [0,1], so a = μ / M.
"""
import itertools
import numpy as np
from scipy.stats import beta as beta_dist

from src.model.term import ADD, YEARS


def _find_next_k(k, th_l, th_u, up, r=2):
    K_i = 4 * k + 2
    K_max = int(np.floor(np.pi / (th_u - th_l)))
    K = K_max - (K_max - 2) % 4
    while K >= r * K_i:
        lo, hi = (K * th_l) % (2 * np.pi), (K * th_u) % (2 * np.pi)
        if lo <= np.pi and hi <= np.pi and lo <= hi:
            return (K - 2) // 4, True
        if lo >= np.pi and hi >= np.pi and lo <= hi:
            return (K - 2) // 4, False
        K -= 4
    return k, up


def _cp(n1, n, alpha):
    lo = 0.0 if n1 == 0 else beta_dist.ppf(alpha / 2, n1, n - n1 + 1)
    hi = 1.0 if n1 == n else beta_dist.ppf(1 - alpha / 2, n1 + 1, n - n1)
    return lo, hi


def iqae(a, eps, alpha=0.05, shots=100, rng=None, max_rounds=100000):
    """Estimate a to absolute half-width eps w.p. >= 1-alpha.
    Returns (estimate, A-operator calls, max k)."""
    rng = rng or np.random.default_rng()
    th = np.arcsin(np.sqrt(a))
    T = int(np.ceil(np.log2(np.pi / (8 * eps)))) + 1
    th_l, th_u, k, up = 0.0, np.pi / 2, 0, True
    n1_acc = n_acc = 0
    calls, kmax = 0, 0
    for _ in range(max_rounds):
        if np.sin(th_u) ** 2 - np.sin(th_l) ** 2 <= 2 * eps:
            break
        k_new, up = _find_next_k(k, th_l, th_u, up)
        if k_new != k:
            n1_acc = n_acc = 0
        k = k_new
        K = 4 * k + 2
        n1 = rng.binomial(shots, np.sin((2 * k + 1) * th) ** 2)
        n1_acc += n1; n_acc += shots
        calls += shots * (2 * k + 1)
        kmax = max(kmax, k)
        a_lo, a_hi = _cp(n1_acc, n_acc, alpha / T)
        if up:
            ph_lo, ph_hi = np.arccos(1 - 2 * a_lo), np.arccos(1 - 2 * a_hi)
        else:
            ph_lo, ph_hi = 2 * np.pi - np.arccos(1 - 2 * a_hi), 2 * np.pi - np.arccos(1 - 2 * a_lo)
        base_l = np.floor(K * th_l / (2 * np.pi)) * 2 * np.pi
        base_u = np.floor(K * th_u / (2 * np.pi)) * 2 * np.pi
        new_l, new_u = (base_l + ph_lo) / K, (base_u + ph_hi) / K
        th_l, th_u = max(th_l, new_l), min(th_u, new_u)
        if th_l > th_u:  # numerical guard
            th_l, th_u = min(th_l, th_u), max(th_l, th_u)
    est = 0.5 * (np.sin(th_l) ** 2 + np.sin(th_u) ** 2)
    return est, calls, kmax


# ---------------------------------------------------------------------------
# Amplitudes a = μ/M for different proposals (exact path enumeration, 10^6 paths)
# ---------------------------------------------------------------------------
def _all_paths():
    Y = np.array(list(itertools.product(range(100), repeat=YEARS)), dtype=np.int64)
    S = np.zeros((len(Y), YEARS + 1), dtype=np.int64)
    for t in range(YEARS):
        S[:, t + 1] = ADD[S[:, t], Y[:, t]]
    return Y, S


_PATHS = None


def paths():
    global _PATHS
    if _PATHS is None:
        _PATHS = _all_paths()
    return _PATHS


def amplitude_for_proposal(p, f, qfun):
    """qfun(t, s_batch) -> (n, 100) proposal rows. Returns dict with μ, M, a, and the
    exact per-path relative std of the classical IS estimator under the same proposal."""
    Y, S = paths()
    n = len(Y)
    logp = np.zeros(n); logq = np.zeros(n)
    for t in range(YEARS):
        Q = qfun(t, S[:, t])
        logq += np.log(Q[np.arange(n), Y[:, t]])
        logp += np.log(p[Y[:, t]])
    w = np.exp(logp - logq)
    q_path = np.exp(logq)
    g = w * f[S[:, -1]]
    mu = float(np.sum(q_path * g))
    M = float(g.max())
    m2 = float(np.sum(q_path * g ** 2))
    return {"mu": mu, "M": M, "a": mu / M, "rel_std_per_path": float(np.sqrt(max(m2 - mu ** 2, 0)) / mu)}
