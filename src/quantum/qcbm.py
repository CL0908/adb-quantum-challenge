"""Quantum circuit Born machine (QCBM) for the annual IS proposal, simulated exactly.

Encoding (8 qubits): |a2 a1 a0  b2 b1 b0  r1  r2>, a,b in 0..4 (codes 5..7 are invalid and
post-selected away; the leaked mass is reported). Ansatz: RY layer, then L x (CZ ring + RY layer).
RY/CZ keep amplitudes real, so a real statevector simulator is exact. Gradients use the exact
parameter-shift rule on the probability vector.

The QCBM is compared against classical *state-independent* (iid-per-year) proposal families on
the same task: fit the KL-optimal iid proposal q*(y) for a given target.
"""
import numpy as np
from scipy.optimize import minimize

from src.model.annual import STATES, encode
from src.model.term import ADD, YEARS, term_distribution
from src.baselines.estimators import value_functions

NQ = 8
DIM = 2 ** NQ


def _bit_index():
    """Map each of the 100 model states to its 8-bit basis index."""
    idx = np.empty(100, dtype=np.int64)
    for s, (a, b, r1, r2) in enumerate(STATES):
        idx[s] = (a << 5) | (b << 2) | (r1 << 1) | r2
    return idx


BIT = _bit_index()


def _ry(state, q, th):
    c, s = np.cos(th / 2), np.sin(th / 2)
    st = state.reshape(2 ** q, 2, -1)
    a0, a1 = st[:, 0, :].copy(), st[:, 1, :].copy()
    st[:, 0, :] = c * a0 - s * a1
    st[:, 1, :] = s * a0 + c * a1
    return state


_CZ_SIGN = None


def _cz_ring_sign():
    global _CZ_SIGN
    if _CZ_SIGN is None:
        x = np.arange(DIM)
        bits = (x[:, None] >> (NQ - 1 - np.arange(NQ))[None, :]) & 1
        par = np.zeros(DIM, dtype=int)
        for q in range(NQ):
            par += bits[:, q] & bits[:, (q + 1) % NQ]
        _CZ_SIGN = (-1.0) ** par
    return _CZ_SIGN


class QCBM:
    def __init__(self, layers=4, seed=0):
        self.L = layers
        self.n_params = NQ * (layers + 1)
        self.theta = np.random.default_rng(seed).normal(0, 0.1, self.n_params)

    def statevector(self, th=None):
        th = self.theta if th is None else th
        psi = np.zeros(DIM); psi[0] = 1.0
        sign = _cz_ring_sign()
        k = 0
        for layer in range(self.L + 1):
            if layer > 0:
                psi = psi * sign
            for q in range(NQ):
                psi = _ry(psi, q, th[k]); k += 1
        return psi

    def probs_full(self, th=None):
        return self.statevector(th) ** 2

    def model_probs(self, th=None):
        """Post-selected distribution on the 100 valid states, and leaked mass."""
        P = self.probs_full(th)
        v = P[BIT]
        return v / v.sum(), 1 - v.sum()

    def grad_full(self, th):
        """dP/dθ via exact parameter shift, shape (n_params, 256)."""
        G = np.empty((self.n_params, DIM))
        for k in range(self.n_params):
            e = np.zeros_like(th); e[k] = np.pi / 2
            G[k] = 0.5 * (self.probs_full(th + e) - self.probs_full(th - e))
        return G

    def fit_kl(self, target, restarts=3, seed=0, maxiter=500):
        """min_θ  -Σ_y target(y) log P_θ(y)  on the *unnormalised* valid-code probabilities.

        This equals KL(target || post-selected q) + (-log(1 - leak)), so it trades fit quality
        against post-selection success; it is what a hardware-deployable QCBM should optimise.
        Returns (KL of post-selected distribution, leak)."""
        best = None
        rng = np.random.default_rng(seed)
        for r in range(restarts):
            th0 = rng.normal(0, 0.1 if r == 0 else 1.0, self.n_params)

            def obj(th):
                v = self.probs_full(th)[BIT]
                ce = -float(target @ np.log(np.maximum(v, 1e-300)))
                G = self.grad_full(th)[:, BIT]
                return ce, -(G @ (target / np.maximum(v, 1e-300)))

            res = minimize(obj, th0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
            if best is None or res.fun < best.fun:
                best = res
        self.theta = best.x
        q, leak = self.model_probs()
        return kl(target, q), leak


# ---------------------------------------------------------------------------
# KL-optimal iid proposal ("teacher") and classical iid families
# ---------------------------------------------------------------------------
def optimal_iid_proposal(p, f):
    """q*(y) = (1/T) Σ_t P_{f-tilted}(Y_t = y): the minimiser of KL(p_f* || q^{⊗T})."""
    V = value_functions(p, f)
    mu = V[0][0]
    dist = np.zeros(100); dist[0] = 1.0
    q = np.zeros(100)
    for t in range(YEARS):
        # P*(Y_t = y) = Σ_s P(S_t = s) p(y) V_{t+1}(s⊕y) / μ
        q += (dist[:, None] * p[None, :] * V[t + 1][ADD]).sum(axis=0) / mu
        nxt = np.zeros(100)
        np.add.at(nxt, ADD, dist[:, None] * p[None, :])
        dist = nxt
    return q / YEARS


def factorized_fit(target):
    """Best product-of-marginals q(a)q(b)q(r1)q(r2) in KL: the product of target marginals."""
    T = target.reshape(5, 5, 2, 2)
    qa, qb, q1, q2 = T.sum((1, 2, 3)), T.sum((0, 2, 3)), T.sum((0, 1, 3)), T.sum((0, 1, 2))
    return np.einsum("a,b,r,s->abrs", qa, qb, q1, q2).ravel(), 4 + 4 + 1 + 1


def loglinear_fit(target, p):
    """q ∝ p · exp(θ·φ(y)) with the 'pairwise' feature set (15 params): the classical
    tilted-family analogue of the QCBM."""
    from src.learned.value_approx import features
    Phi = features("pairwise")
    lp = np.log(p)

    def obj(th):
        z = lp + Phi @ th; zm = z.max(); e = np.exp(z - zm); Z = e.sum(); q = e / Z
        val = -(target @ (z - zm - np.log(Z)))
        return val, -(Phi.T @ (target - q))

    th = minimize(obj, np.zeros(Phi.shape[1]), jac=True, method="L-BFGS-B").x
    z = lp + Phi @ th; q = np.exp(z - z.max())
    return q / q.sum(), Phi.shape[1]


def kl(a, b):
    m = a > 0
    return float(a[m] @ (np.log(a[m]) - np.log(np.maximum(b[m], 1e-300))))


def iid_is_variance(p, f, q):
    """Exact per-path second moment of the IS estimator under iid proposal q:
    E_q[(w f)^2] = Σ_paths p^2/q f^2 -> computed by the same capped convolution with p^2/q."""
    r = p ** 2 / q
    d = np.zeros(100); d[0] = 1.0
    for _ in range(YEARS):
        nxt = np.zeros(100); np.add.at(nxt, ADD, d[:, None] * r[None, :]); d = nxt
    m2 = float(d @ f ** 2)
    mu = float(term_distribution(p) @ f)
    return m2 - mu ** 2, mu
