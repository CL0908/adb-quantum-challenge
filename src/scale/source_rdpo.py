"""Scalable RDPO: a Rényi-Doob policy over *Poisson sources* instead of years.

The n-country term law is a product of independent sources (solo EQ / solo rain per country,
shared EQ / shared rain per edge), each with a finite capped-increment law of <= 25 values
(src/quantum/structured_loading.py). We sample sources sequentially, k = 1..K (K = 2n + 2|E|),
and learn a soft-RL (Doob) policy

    q_k(x | s) ∝ p_k(x) · Ṽ_θ(s ⊕ x),     log Ṽ_θ(s) = Σ_i θ_i · ψ(a_i, r_i) + θ_g · g(L(s))

with per-country features ψ = [1{a>=1..4}, r] (5 params / country) and global loss features
g(L) = [min(L,T)/T, 1{L>T}, (L-T)^+/(Lmax-T)].  Properties:
  * the normaliser sums over <= 25 values of ONE source (never over 10^n joint states);
  * a step changes the features of only the 1-2 countries it touches, so a path costs O(n + |E|);
  * O(n) parameters; the estimator stays unbiased (likelihood ratio, defensive mixture with p).
Training = RDPO: multilevel cross-entropy policy search (KL) followed by a Rényi-2 (χ²)
policy-gradient refinement (src/learned/renyi.py, same idea, source-level implementation).
"""
import numpy as np
from scipy.optimize import minimize

from src.tensor.synthetic import SyntheticModel
from src.quantum.structured_loading import sources as model_sources

NLOC = 5   # per-country features
NG = 3     # global loss features


def psi(a, r):
    return np.stack([a >= 1, a >= 2, a >= 3, a >= 4, r >= 1], -1).astype(float)


class SourceProblem:
    """Source-level description of an n-country model and one target."""

    def __init__(self, model: SyntheticModel, threshold, target="exceedance"):
        self.model, self.n, self.T = model, model.n, threshold
        unit = 2.5e6
        self.a_u = round(model.eq_limit / 4 / unit)   # loss units per quarter step (7)
        self.r_u = round(model.rain_limit / unit)     # loss units per rain trigger (4)
        self.Tu = threshold / unit
        self.Lmax = model.n * (4 * self.a_u + self.r_u)
        self.target = target
        self.src = []
        for s in model_sources(model):
            vals = np.arange(len(s.law))
            if s.kind == "eq1":
                da = [(s.sites[0], vals)]; dr = []
            elif s.kind == "eq2":
                da = [(s.sites[0], vals // 5), (s.sites[1], vals % 5)]; dr = []
            elif s.kind == "rain1":
                da = []; dr = [(s.sites[0], vals)]
            else:
                da = []; dr = [(s.sites[0], vals), (s.sites[1], vals)]
            self.src.append({"kind": s.kind, "p": s.law, "logp": np.log(np.maximum(s.law, 1e-300)),
                             "da": da, "dr": dr, "sites": s.sites})
        self.K = len(self.src)

    def loss_units(self, A, R):
        return (A * self.a_u + R * self.r_u).sum(-1)

    def g(self, L):
        T, Lm = self.Tu, self.Lmax
        return np.stack([np.minimum(L, T) / T, (L > T).astype(float), np.maximum(L - T, 0) / (Lm - T)], -1)

    def payoff(self, A, R):
        L = self.loss_units(A, R)
        if self.target == "exceedance":
            return (L > self.Tu).astype(float)
        if self.target == "stop_loss":
            return np.maximum(L - self.Tu, 0) * 2.5e6
        raise ValueError(self.target)

    def score(self, A, R):
        return self.loss_units(A, R).astype(float)

    def level(self):
        return self.Tu + 1e-9   # L > T


class SourcePolicy:
    def __init__(self, prob: SourceProblem, alpha=0.05):
        self.P = prob
        self.alpha = alpha
        self.th_loc = np.zeros((prob.n, NLOC))
        self.th_g = np.zeros(NG)

    @property
    def n_params(self):
        return self.th_loc.size + self.th_g.size

    def get(self):
        return np.concatenate([self.th_loc.ravel(), self.th_g])

    def set(self, v):
        self.th_loc = v[:self.th_loc.size].reshape(self.th_loc.shape); self.th_g = v[self.th_loc.size:]

    # --- one step: candidate next states for all values of source k --------------------------
    def _step_terms(self, k, A, R, L):
        """Return logits (N,V) of the *tilted* part, the local-feature deltas per touched site
        [(site, dpsi (N,V,5))], global delta dg (N,V,3), and new (A_i, R_i) per touched site."""
        s = self.P.src[k]
        N, V = A.shape[0], len(s["p"])
        dL = np.zeros((N, V))
        touched = {}
        for (i, inc) in s["da"]:
            a_new = np.minimum(A[:, i][:, None] + inc[None, :], 4)
            touched.setdefault(i, [A[:, i][:, None].repeat(V, 1), R[:, i][:, None].repeat(V, 1)])[0] = a_new
            dL += (a_new - A[:, i][:, None]) * self.P.a_u
        for (i, inc) in s["dr"]:
            r_new = np.minimum(R[:, i][:, None] + inc[None, :], 1)
            touched.setdefault(i, [A[:, i][:, None].repeat(V, 1), R[:, i][:, None].repeat(V, 1)])[1] = r_new
            dL += (r_new - R[:, i][:, None]) * self.P.r_u
        logit = np.zeros((N, V))
        dpsi = []
        for i, (a_new, r_new) in touched.items():
            d = psi(a_new, r_new) - psi(A[:, i], R[:, i])[:, None, :]
            dpsi.append((i, d))
            logit += d @ self.th_loc[i]
        dg = self.P.g(L[:, None] + dL) - self.P.g(L)[:, None, :]
        logit += dg @ self.th_g
        return logit, dpsi, dg, touched

    def _probs(self, k, logit):
        lp = self.P.src[k]["logp"][None, :] + logit
        lp -= lp.max(1, keepdims=True)
        qt = np.exp(lp); qt /= qt.sum(1, keepdims=True)
        q = (1 - self.alpha) * qt + self.alpha * self.P.src[k]["p"][None, :]
        return qt, q

    def sample(self, N, rng, record=False):
        P = self.P
        A = np.zeros((N, P.n), dtype=np.int64); R = np.zeros((N, P.n), dtype=np.int64)
        L = np.zeros(N); logw = np.zeros(N)
        acts = np.zeros((N, P.K), dtype=np.int64) if record else None
        idx = np.arange(N)
        for k in range(P.K):
            logit, _, _, touched = self._step_terms(k, A, R, L)
            _, q = self._probs(k, logit)
            v = np.minimum((q.cumsum(1) < rng.random((N, 1))).sum(1), q.shape[1] - 1)
            logw += P.src[k]["logp"][v] - np.log(q[idx, v])
            for i, (a_new, r_new) in touched.items():
                A[:, i] = a_new[idx, v]; R[:, i] = r_new[idx, v]
            L = P.loss_units(A, R).astype(float)
            if record:
                acts[:, k] = v
        return A, R, np.exp(logw), acts

    def estimate(self, N, rng):
        A, R, w, _ = self.sample(N, rng)
        return float(np.mean(w * self.P.payoff(A, R)))

    # --- replay recorded paths: weighted log-likelihood and gradients --------------------------
    def replay_grad(self, acts, coef, mode):
        """mode 'ce'  : returns (−Σ c_j log q̃(path_j), grad)  (unmixed tilted policy)
           mode 'logq': returns grad of Σ c_j log q(path_j) for the *mixture* (policy gradient)."""
        P = self.P
        N = acts.shape[0]
        A = np.zeros((N, P.n), dtype=np.int64); R = np.zeros((N, P.n), dtype=np.int64)
        L = np.zeros(N); idx = np.arange(N)
        gl = np.zeros_like(self.th_loc); gg = np.zeros_like(self.th_g); val = 0.0
        for k in range(P.K):
            v = acts[:, k]
            logit, dpsi, dg, touched = self._step_terms(k, A, R, L)
            qt, q = self._probs(k, logit)
            if mode == "ce":
                lp = P.src[k]["logp"][None, :] + logit
                m = lp.max(1, keepdims=True)
                logZ = (m + np.log(np.exp(lp - m).sum(1, keepdims=True)))[:, 0]
                val -= coef @ (lp[idx, v] - logZ)
                fac = coef
            else:
                fac = coef * (1 - self.alpha) * qt[idx, v] / q[idx, v]
            for i, d in dpsi:
                gl[i] += fac @ (d[idx, v] - np.einsum("nv,nvf->nf", qt, d))
            gg += fac @ (dg[idx, v] - np.einsum("nv,nvf->nf", qt, dg))
            for i, (a_new, r_new) in touched.items():
                A[:, i] = a_new[idx, v]; R[:, i] = r_new[idx, v]
            L = P.loss_units(A, R).astype(float)
        if mode == "ce":
            return val, -np.concatenate([gl.ravel(), gg])
        return np.concatenate([gl.ravel(), gg])


def fit_rdpo(prob: SourceProblem, n_ce=2000, rho=0.1, max_iter=25, extra_iter=2, beta=0.7, kappa=1e-3,
             renyi_steps=20, renyi_batch=2000, lr=0.05, seed=0, alpha=0.05, init_theta=None):
    rng = np.random.default_rng(seed)
    pol = SourcePolicy(prob, alpha)
    if init_theta is not None:
        pol.set(np.array(init_theta, float))
    level = prob.level()
    used, after, hist = 0, 0, []
    # ---- stage 1: multilevel CE policy search (KL) ----
    for it in range(max_iter):
        A, R, w, acts = pol.sample(n_ce, rng, record=True); used += n_ce
        sc = prob.score(A, R)
        gam = min(level, np.quantile(sc, 1 - rho))
        if gam < level and (sc >= gam).sum() > 2 * rho * n_ce:
            hi = sc[sc > gam]
            if hi.size:
                gam = min(level, hi.min())
        if gam >= level:
            H = prob.payoff(A, R); after += 1
        else:
            H = (sc >= max(gam, 1e-9)).astype(float)
        W = w * H
        hist.append({"iter": it, "gamma": float(gam), "elite": int((W > 0).sum())})
        keep = W > 0
        if keep.any():
            c = W[keep] / W[keep].sum()
            ak = acts[keep]

            def obj(v):
                pol.set(v)
                f, g = pol.replay_grad(ak, c, "ce")
                return f + 0.5 * kappa * v @ v, g + kappa * v

            v0 = pol.get()
            r = minimize(obj, v0, jac=True, method="L-BFGS-B", options={"maxiter": 100})
            pol.set(beta * r.x + (1 - beta) * v0)
        if after >= extra_iter:
            break
    ce_theta = pol.get().copy()
    # ---- stage 2: Rényi-2 (χ²) policy gradient ----
    m = np.zeros_like(ce_theta); s2 = np.zeros_like(ce_theta)
    best = (np.inf, ce_theta.copy()); trace = []
    for k in range(1, renyi_steps + 1):
        A, R, w, acts = pol.sample(renyi_batch, rng, record=True)
        gsq = (w * prob.payoff(A, R)) ** 2
        mu = np.mean(w * prob.payoff(A, R)); m2 = gsq.mean()
        chi2 = m2 / max(mu, 1e-300) ** 2 - 1 if mu > 0 else np.inf
        trace.append(float(chi2))
        if chi2 < best[0]:
            best = (chi2, pol.get().copy())
        if m2 == 0:
            continue
        grad = -pol.replay_grad(acts, gsq / m2 / renyi_batch, "logq")
        nrm = np.linalg.norm(grad)
        if nrm > 5:
            grad *= 5 / nrm
        m = 0.9 * m + 0.1 * grad; s2 = 0.999 * s2 + 0.001 * grad ** 2
        pol.set(pol.get() - lr * (m / (1 - 0.9 ** k)) / (np.sqrt(s2 / (1 - 0.999 ** k)) + 1e-8))
    pol.set(best[1])
    return pol, {"ce_paths": used, "renyi_paths": renyi_steps * renyi_batch, "n_params": pol.n_params,
                 "K_sources": prob.K, "ce_history": hist, "chi2_trace": trace, "ce_theta": ce_theta.tolist()}


def mc_estimate(prob: SourceProblem, N, rng):
    """Plain MC by sampling every source from its own law (same per-path cost)."""
    pol = SourcePolicy(prob, alpha=1.0)   # alpha=1 -> q = p exactly
    return pol.estimate(N, rng)


# ---------------------------------------------------------------------------
# Classical exponential twisting (state-independent, 1 parameter) and RDPO warm start
# ---------------------------------------------------------------------------
def twisted_estimate(prob: SourceProblem, eta, N, rng):
    """Sample every source from q_k ∝ p_k e^{η ℓ_k}; w = exp(−η U + Λ(η)) (U = uncapped loss)."""
    from src.quantum.twisted import source_losses
    n = prob.n
    A = np.zeros((N, n), dtype=np.int64); R = np.zeros((N, n), dtype=np.int64)
    logw = np.zeros(N)
    for (s, ell), src in zip(source_losses(prob.model), prob.src):
        q = s.law * np.exp(eta * ell); Z = q.sum(); q = q / Z
        v = np.minimum((np.cumsum(q)[None, :] < rng.random((N, 1))).sum(1), len(q) - 1)
        logw += -eta * ell[v] + np.log(Z)
        for (i, inc) in src["da"]:
            A[:, i] = np.minimum(A[:, i] + inc[v], 4)
        for (i, inc) in src["dr"]:
            R[:, i] = np.minimum(R[:, i] + inc[v], 1)
    return float(np.mean(np.exp(logw) * prob.payoff(A, R)))


def warm_start_theta(prob: SourceProblem):
    """Initialise RDPO from the optimal exponential twist: log Ṽ ≈ η* · min(L, T)."""
    from src.quantum.twisted import best_twist
    eta, _ = best_twist(prob.model, prob.Tu)
    th = np.zeros(prob.n * NLOC + NG); th[prob.n * NLOC] = eta * prob.Tu
    return th, eta
