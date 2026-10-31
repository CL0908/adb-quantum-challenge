"""Information-theoretic policy refinement: minimise the Rényi-2 / χ² divergence.

Cross-entropy fits minimise KL(p* || q). The IS variance is instead governed by
    E_q[(w f)^2] / μ^2 = 1 + χ²(p* || q) = exp(D_2(p* || q))       (Rényi order 2),
so a proposal can have small KL and still a heavy-tailed weight distribution (this is how the
pairwise CE fits failed in §3.2). We refine any log-linear (or VQC) Doob policy by a
REINFORCE-type gradient on log E_q[(w f)^2]:
    ∇_θ E_q[(w f)^2] = − E_q[(w f)^2 ∇_θ log q_θ(path)].
This is variance-minimising adaptive IS (cf. Arouna 2004) written as a policy-gradient method:
reward = −(w f)^2, policy = q_θ. Normalised (self-normalised) weights and Adam keep it stable.
"""
import numpy as np

from src.model.term import ADD, YEARS


def _grad_logq(prop, S, Y):
    """∇_θ log q(path) for the mixture q = (1-α) q̃_θ + α p.  Returns (N, YEARS, d)."""
    N = S.shape[0]
    G = np.zeros((N, YEARS, prop.d))
    idx = np.arange(N)
    for t in range(YEARS):
        s, y = S[:, t], Y[:, t]
        Qt = prop.tilted(t, s)                          # (N, 100)
        Q = (1 - prop.alpha) * Qt + prop.alpha * prop.p[None, :]
        F = prop.Phi[ADD[s]]                            # (N, 100, d)
        mean_phi = np.einsum("nk,nkd->nd", Qt, F)
        frac = (1 - prop.alpha) * Qt[idx, y] / Q[idx, y]
        G[:, t, :] = frac[:, None] * (prop.Phi[ADD[s, y]] - mean_phi)
    return G


def renyi2_refine(prop, f, steps=40, batch=2000, lr=0.05, seed=0, clip=5.0):
    rng = np.random.default_rng(seed)
    m = np.zeros_like(prop.theta); v = np.zeros_like(prop.theta)
    hist = []
    best = (np.inf, prop.theta.copy())
    for k in range(1, steps + 1):
        Y, S, w = prop.sample(batch, rng)
        g2 = (w * f[S[:, -1]]) ** 2
        mu_hat = np.mean(w * f[S[:, -1]])
        m2 = np.mean(g2)
        rel_var = m2 / max(mu_hat, 1e-300) ** 2 - 1      # ≈ χ²(p*||q) estimate
        hist.append(float(rel_var))
        if rel_var < best[0]:
            best = (rel_var, prop.theta.copy())
        if m2 == 0:
            continue
        G = _grad_logq(prop, S, Y)                       # (N, T, d)
        grad = -np.einsum("n,ntd->td", g2 / m2, G) / batch   # ∇ log M2
        nrm = np.linalg.norm(grad)
        if nrm > clip:
            grad *= clip / nrm
        m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2
        prop.theta -= lr * (m / (1 - 0.9 ** k)) / (np.sqrt(v / (1 - 0.999 ** k)) + 1e-8)
    # keep the best iterate by the *estimated* χ² (no use of the exact answer)
    prop.theta = best[1]
    return prop, {"chi2_trace": hist, "refine_paths": steps * batch}


def exact_rel_std(prop, f, p_term_mu=None):
    """Exact per-path relative std of the IS estimator under a state-dependent policy, via the
    backward recursion U_t(s) = Σ_y p(y)^2 / q_t(y|s) · U_{t+1}(s⊕y), U_T = f^2  (second moment),
    and the same recursion with p for μ. No sampling noise; this is exp(D_2(p*||q)) - 1 in sqrt form."""
    U = f.astype(float) ** 2
    V = f.astype(float)
    for t in range(YEARS - 1, -1, -1):
        s_all = np.arange(100)
        Q = prop.q(t, s_all)                              # (100, 100)
        U = ((prop.p[None, :] ** 2 / Q) * U[ADD]).sum(1)
        V = (prop.p[None, :] * V[ADD]).sum(1)
    m2, mu = U[0], V[0]
    return float(np.sqrt(max(m2 - mu ** 2, 0)) / mu)
