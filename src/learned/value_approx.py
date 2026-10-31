"""Compact value-function approximations Ṽ_θ and the induced (approximate) Doob proposal.

Ṽ_{θ,t}(s) = exp(θ_t · φ(s)),   q_t(y | s) ∝ p(y) Ṽ_{θ,t}(s ⊕ y)   (mixed with p for safety)

Feature families (per cumulative state s = (a, b, r1, r2)):
  * "factorized": one-hot(a>=1..4), one-hot(b>=1..4), r1, r2         -> Ṽ = Π_i v_i(s_i)
                  (10 params/year, O(n) for n countries)
  * "pairwise":   factorized + cross-country terms a*b, a*r2, b*r1, r1*r2, 1[a=4,b=4]
                  (15 params/year, O(n + |edges|))
Using threshold indicators 1[a>=k] instead of one-hot keeps the family monotone-friendly.
"""
import numpy as np

from src.model.annual import STATES
from src.model.term import ADD, YEARS

A, B, R1, R2 = STATES.T.astype(float)


def features(kind):
    cols = [(A >= k) for k in range(1, 5)] + [(B >= k) for k in range(1, 5)] + [R1, R2]
    if kind == "pairwise":
        cols += [A * B / 16, A * R2 / 4, B * R1 / 4, R1 * R2, (A == 4) & (B == 4)]
    elif kind != "factorized":
        raise ValueError(kind)
    return np.stack(cols, axis=1).astype(float)  # (100, d)


class DoobProposal:
    """Time-inhomogeneous proposal q_t(y|s) from a log-linear Ṽ (one θ per year)."""

    def __init__(self, p, kind="pairwise", alpha=0.05, theta=None):
        self.p = p
        self.kind = kind
        self.Phi = features(kind)
        self.d = self.Phi.shape[1]
        self.alpha = alpha
        self.theta = np.zeros((YEARS, self.d)) if theta is None else np.array(theta, float)

    @property
    def n_params(self):
        return self.theta.size

    def logV(self, t):
        return self.Phi @ self.theta[t]  # over all 100 states

    def tilted(self, t, s):
        """Unmixed q_t(.|s) for a batch of current states s -> (N, 100)."""
        lv = self.logV(t)[ADD[s]]                     # (N, 100)
        lv -= lv.max(axis=1, keepdims=True)
        Q = self.p[None, :] * np.exp(lv)
        return Q / Q.sum(axis=1, keepdims=True)

    def q(self, t, s):
        return (1 - self.alpha) * self.tilted(t, s) + self.alpha * self.p[None, :]

    def sample(self, N, rng):
        """Returns paths Y (N, T), states S (N, T+1), likelihood ratio w (N,)."""
        S = np.zeros((N, YEARS + 1), dtype=np.int64)
        Y = np.zeros((N, YEARS), dtype=np.int64)
        w = np.ones(N)
        idx = np.arange(N)
        for t in range(YEARS):
            Q = self.q(t, S[:, t])
            u = rng.random((N, 1))
            y = np.minimum((Q.cumsum(axis=1) < u).sum(axis=1), 99)
            w *= self.p[y] / Q[idx, y]
            Y[:, t] = y
            S[:, t + 1] = ADD[S[:, t], y]
        return Y, S, w

    def estimate(self, f, N, rng):
        _, S, w = self.sample(N, rng)
        return float(np.mean(w * f[S[:, -1]]))
