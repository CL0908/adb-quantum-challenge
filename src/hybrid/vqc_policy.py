"""Hybrid quantum–classical RL for the Doob proposal.

Soft-RL view. The optimal IS proposal is the optimal policy of a KL-regularised control
problem (Todorov 2009, linearly solvable MDPs; Kappen 2005, path-integral control):
    π*(y | s) ∝ p(y) · V(s ⊕ y),   log V_t(s) = log Σ_y p(y) V_{t+1}(s ⊕ y)   (soft Bellman)
so log Ṽ is a *soft value function* and the proposal is its Boltzmann policy.

Quantum critic. log Ṽ_{θ,t}(s) = w_t · ⟨O⟩_{U(θ_c) E(s)}, where
  * E(s): per-country angle encoding (2 qubits per country: EQ level a -> RY(π a/4), rain r -> RY(π r)),
    re-uploaded every layer;
  * U(θ_c): L layers of trainable RY + CZ on *the dependence graph* (intra-country pair and the
    cross-country edges), so the circuit has O(n + |E|) parameters and depth O(L · maxdeg);
  * O: local ⟨Z_q⟩ and edge ⟨Z_q Z_q'⟩ observables (measured on hardware with shots).
The readout w_t is fitted by the convex CE step; θ_c by gradient steps on the same CE objective
(alternating), i.e. cross-entropy *policy search* with a variational quantum critic.

Caveat stated up front: for constant depth, local observables have constant light cones and are
classically simulable, so this is a scalable *architecture*, not a claimed advantage.
"""
import numpy as np
from scipy.optimize import minimize

from src.model.annual import STATES
from src.model.term import ADD, YEARS
from src.learned.value_approx import DoobProposal
from src.learned.cross_entropy import target_spec, _fit_year

NQ = 4                       # (A1, R1, A2, R2)
EDGES = [(0, 1), (2, 3), (0, 2), (1, 3)]   # intra-country + dependence edges


def _apply_ry(psi, q, th):
    """psi: (B, 2^NQ) real; th: scalar or (B,) angles on qubit q."""
    c, s = np.cos(np.asarray(th) / 2), np.sin(np.asarray(th) / 2)
    st = psi.reshape(psi.shape[0], 2 ** q, 2, -1)
    a0, a1 = st[:, :, 0, :].copy(), st[:, :, 1, :].copy()
    if np.ndim(c):
        c = c[:, None, None]; s = s[:, None, None]
    st[:, :, 0, :] = c * a0 - s * a1
    st[:, :, 1, :] = s * a0 + c * a1
    return psi


def _bits():
    x = np.arange(2 ** NQ)
    return (x[:, None] >> (NQ - 1 - np.arange(NQ))[None, :]) & 1


BITS = _bits()
CZ_SIGN = (-1.0) ** sum(BITS[:, i] & BITS[:, j] for i, j in EDGES)
ZQ = 1 - 2 * BITS                       # (16, 4) eigenvalues of Z_q
ZZ = np.stack([ZQ[:, i] * ZQ[:, j] for i, j in EDGES], 1)
OBS = np.concatenate([ZQ, ZZ], 1).astype(float)   # (16, 8)


def quantum_features(theta_c, layers):
    """Exact expectation values for all 100 cumulative states -> (100, 8)."""
    a, b, r1, r2 = STATES.T
    # per-layer angle scaled by 1/L: the total encoding rotation stays in [0, π] (monotone in a, r);
    # without this, re-uploading twice maps r -> RY(2π) = identity and a -> cos(πa/2) (non-monotone)
    enc = np.stack([np.pi * a / 4, np.pi * r1, np.pi * b / 4, np.pi * r2], 1) / layers  # (100, 4)
    psi = np.zeros((100, 2 ** NQ)); psi[:, 0] = 1.0
    k = 0
    for _ in range(layers):
        for q in range(NQ):                 # data re-uploading
            psi = _apply_ry(psi, q, enc[:, q])
        for q in range(NQ):
            psi = _apply_ry(psi, q, theta_c[k]); k += 1
        psi = psi * CZ_SIGN[None, :]
    return (psi ** 2) @ OBS


class VQCDoobProposal(DoobProposal):
    def __init__(self, p, layers=2, alpha=0.05, seed=0):
        self.layers = layers
        self.theta_c = np.random.default_rng(seed).normal(0, 0.5, NQ * layers)
        super().__init__(p, kind="factorized", alpha=alpha)
        self.set_circuit(self.theta_c)
        self.theta = np.zeros((YEARS, self.Phi.shape[1]))

    def set_circuit(self, theta_c):
        self.theta_c = np.array(theta_c, float)
        self.Phi = quantum_features(self.theta_c, self.layers)
        self.d = self.Phi.shape[1]

    @property
    def n_params(self):
        return self.theta.size + self.theta_c.size


def _ce_objective(prop, S, Y, W):
    """Total weighted CE (negative log-likelihood of elite paths) over years, given prop.Phi."""
    keep = W > 0
    s_all, y_all, Wk = S[keep], Y[keep], W[keep] / W[keep].sum()
    lp = np.log(prop.p)[None, :]
    tot = 0.0
    for t in range(YEARS):
        s, y = s_all[:, t], y_all[:, t]
        z = lp + prop.Phi[ADD[s]] @ prop.theta[t]
        zm = z.max(1, keepdims=True)
        logZ = (zm + np.log(np.exp(z - zm).sum(1, keepdims=True)))[:, 0]
        tot -= Wk @ (prop.Phi[ADD[s, y]] @ prop.theta[t] - logZ)
    return tot


def fit_vqc(prob, target, sc, layers=2, n_ce=2000, rho=0.1, max_iter=20, extra_iter=3,
            beta=0.7, kappa=1e-3, alpha=0.05, seed=0, circuit_steps=15):
    rng = np.random.default_rng(seed)
    score, level, f = target_spec(prob, target, sc)
    prop = VQCDoobProposal(prob.p, layers=layers, alpha=alpha, seed=seed)
    used, after = 0, 0
    for it in range(max_iter):
        Y, S, w = prop.sample(n_ce, rng); used += n_ce
        sf = score[S[:, -1]]
        gamma = min(level, np.quantile(sf, 1 - rho))
        if gamma < level and (sf >= gamma).sum() > 2 * rho * n_ce:
            higher = sf[sf > gamma]
            if higher.size:
                gamma = min(level, higher.min())
        if gamma >= level:
            H = f[S[:, -1]]; after += 1
        else:
            H = (sf >= max(gamma, np.nextafter(0, 1))).astype(float)
        W = w * H
        # (i) convex readout step (critic head)
        new = np.array([_fit_year(prop, t, S[:, t], Y[:, t], W, kappa, prop.theta[t]) for t in range(YEARS)])
        prop.theta = beta * new + (1 - beta) * prop.theta
        # (ii) variational circuit step on the same CE objective (finite-difference gradient)
        if (W > 0).any():
            def obj(tc):
                prop.set_circuit(tc)
                return _ce_objective(prop, S, Y, W)
            r = minimize(obj, prop.theta_c, method="L-BFGS-B", options={"maxiter": circuit_steps})
            prop.set_circuit(r.x)
        if after >= extra_iter:
            break
    return prop, {"training_paths": used, "n_params": prop.n_params}
