"""Source-tilt: a learned product-form (state-independent) change of measure.

Every source k keeps its own tilted law over its finite increment values,

        q_k(x) ∝ p_k(x) · exp(θ_k · φ_k(x)),      φ_k(x) = [ℓ_k(x), 1{ℓ_k(x) > 0}]   (or [ℓ_k(x)] only),

with ℓ_k(x) the (uncapped) loss increment of the source in units of 2.5m. The sources are independent under q, so
q(x_1..x_K) = Π_k q_k(x_k): sampling is exactly as cheap as exponential twisting (one inverse-CDF draw per source, no
state), and the likelihood ratio is a table lookup per source. θ = 0 is the plain law p; θ_k = (η*, 0) for all k is
the optimal single exponential twist, which is where fitting starts. The family is exactly the measure that the
quantum structured loader prepares (src/quantum/structured_loading.py, twisted.py): product of source registers.

Fitting (cheap, fully vectorised, no per-step python state):
  * χ² (second-moment) fit by sample-average approximation. For a batch drawn from q_ref with weights w_ref,
        M2(θ) = E_{q_ref}[ f² w_ref w_θ ]  is log-convex in θ  (w_θ = p/q_θ = exp(-Σ_k θ_k·φ_k(x_k) + Σ_k log Z_k(θ_k))),
    so   min_θ  log Σ_j f_j² w_ref,j w_θ(x_j) + (κ/2)|θ − θ_0|²   is a convex problem solved by L-BFGS on the
    positive-payoff paths only.
  * cross-entropy fit (optional first step when the starting measure rarely reaches the target):
        min_θ  −Σ_j w_ref,j f_j log q_θ(x_j) + (κ/2)|θ − θ_0|²   (convex).
  * Each outer iteration draws a fresh batch from the current measure, fits, damps the move (β) and keeps the
    iterate with the smallest *estimated* relative variance; the initial twist is a candidate too (model selection).

Estimator: batches are drawn under measures fixed before the batch is sampled and combined with weights fixed in
advance (proportional to batch size), so the estimate is exactly unbiased; the standard error uses the batch-wise
sample variances (martingale argument), 95% interval is Student-t with Welch-Satterthwaite degrees of freedom.
All time (including fitting) is charged to the budget.
"""
import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as _tdist

from .problem import PortfolioProblem
from .train import _best_twist_sources


class TiltSampler:
    """Product-form sampler for a PortfolioProblem with parameter array theta of shape (K, m)."""

    def __init__(self, prob: PortfolioProblem, features="ell+ind"):
        self.P = P = prob
        self.features = features
        self.K = K = P.K
        self.Vk = [len(s["p"]) for s in P.src]
        self.Vmax = max(self.Vk)
        self.ell = []
        for s in P.src:
            e = np.zeros(len(s["p"]))
            for _, inc in s["da"]:
                e = e + P.a_u * inc
            for _, inc in s["dr"]:
                e = e + P.r_u * inc
            self.ell.append(e.astype(float))
        if features == "twist":
            self.Phi = [e[:, None] for e in self.ell]
        elif features == "full":                      # free per-source law (one-hot features)
            m = self.Vmax
            self.Phi = [np.eye(len(e), m) for e in self.ell]
        elif features == "poly":
            self.Phi = [np.stack([e, (e > 0).astype(float), (e / max(e.max(), 1)) ** 2 * 10], 1) for e in self.ell]
        else:
            self.Phi = [np.stack([e, (e > 0).astype(float)], 1) for e in self.ell]
        self.m = self.Phi[0].shape[1]
        self.logp = [s["logp"] for s in P.src]
        self.p = [s["p"] for s in P.src]
        # per-site increment tables (int16) for the capped state update
        self.inc_a = [[(i, np.asarray(inc, dtype=np.int16)) for i, inc in s["da"]] for s in P.src]
        self.inc_r = [[(i, np.asarray(inc, dtype=np.int16)) for i, inc in s["dr"]] for s in P.src]
        self.eta0 = None
        self.set_theta(np.zeros((K, self.m)))

    # ---- parameters -------------------------------------------------------------------------
    def twist_theta(self, eta):
        th = np.zeros((self.K, self.m))
        if self.features == "full":
            for k in range(self.K):
                th[k, :self.Vk[k]] = eta * self.ell[k]
        else:
            th[:, 0] = eta
        return th

    def set_theta(self, theta):
        """Compute the tilted laws and the log likelihood-ratio tables for theta (K, m)."""
        self.theta = np.array(theta, dtype=float).reshape(self.K, self.m)
        self.cdf, self.lr = [], np.zeros((self.K, self.Vmax))
        self.logZ = np.zeros(self.K)
        self.EPhi = np.zeros((self.K, self.m))
        self.q = []
        for k in range(self.K):
            lg = self.logp[k] + self.Phi[k] @ self.theta[k]
            mx = lg.max()
            e = np.exp(lg - mx); Z = e.sum(); q = e / Z
            self.q.append(q)
            c = np.cumsum(q); c[-1] = 1.0
            self.cdf.append(c)
            self.logZ[k] = mx + np.log(Z)              # log Σ p e^{Φθ} ... (logp already inside the exponent)
            self.lr[k, :self.Vk[k]] = self.logp[k] - np.log(np.maximum(q, 1e-300))
            self.EPhi[k] = q @ self.Phi[k]
        self.lr_flat = self.lr.ravel()

    # ---- sampling -----------------------------------------------------------------------------
    def sample(self, N, rng, record=False):
        """Draw N paths. Returns dict(f, w, acts (K,N) int8 or None)."""
        P, K = self.P, self.K
        A = np.zeros((P.n, N), dtype=np.int16); R = np.zeros((P.n, N), dtype=np.int16)
        logw = np.zeros(N)
        acts = np.empty((K, N), dtype=np.int8) if record else None
        U = rng.random((K, N))
        for k in range(K):
            v = np.searchsorted(self.cdf[k], U[k])
            np.minimum(v, self.Vk[k] - 1, out=v)
            logw += self.lr[k][v]
            for i, inc in self.inc_a[k]:
                A[i] += inc[v]
            for i, inc in self.inc_r[k]:
                R[i] += inc[v]
            if record:
                acts[k] = v
        np.minimum(A, 4, out=A); np.minimum(R, 1, out=R)
        Ls = P.a_u * A.astype(np.int64) + P.r_u * R
        i, j = P.pair
        f = P.payoff_from(Ls.sum(0), Ls[i], Ls[j])
        return {"f": f, "w": np.exp(logw), "acts": acts, "score": P.score_from(Ls.sum(0), Ls[i], Ls[j])}

    # ---- fitting objectives ------------------------------------------------------------------
    def _flat_index(self, acts):
        return (acts.astype(np.int32) + (np.arange(self.K, dtype=np.int32) * self.Vmax)[:, None]).T   # (n, K)

    def fit_chi2(self, acts, f, w_ref, theta0, kappa=1e-3, bound=None, maxiter=40, th_init=None):
        """argmin of the sample-average second moment E_ref[f² w_ref w_θ] (log-convex) + ridge towards theta0."""
        pos = f > 0
        if pos.sum() < 20:
            return self.theta.copy(), {"n_pos": int(pos.sum()), "ok": False}
        a = acts[:, pos]; f2w = np.log(f[pos] ** 2 * w_ref[pos])
        idx = self._flat_index(a)                                            # (n, K)
        K, m, Vmax = self.K, self.m, self.Vmax
        # Phi table (K, Vmax, m) padded
        PhiT = np.zeros((K, Vmax, m))
        for k in range(K):
            PhiT[k, :self.Vk[k]] = self.Phi[k]
        PhiFlat = PhiT.reshape(K * Vmax, m)
        # contribution table of the path-dependent part: Σ_k θ_k·Φ_k(v_k) = Σ_j gathered
        # use a (n, K, m) free formulation through bincount over flat (k,v) cells
        counts_cache = {}
        th0 = np.asarray(theta0, float).reshape(-1)
        n = idx.shape[0]
        flat_all = idx.ravel()

        def tables(th):
            T = th.reshape(K, m)
            S = np.einsum("kvm,km->kv", PhiT, T)                           # (K, Vmax) Σ θ φ
            lg = np.full((K, Vmax), -np.inf)
            for k in range(K):
                lg[k, :self.Vk[k]] = self.logp[k] + S[k, :self.Vk[k]]
            mx = lg.max(1, keepdims=True)
            logZ = (mx + np.log(np.exp(lg - mx).sum(1, keepdims=True)))[:, 0]
            return S, logZ, np.exp(lg - logZ[:, None])

        def obj(th):
            S, logZ, Q = tables(th)
            # log w_θ(x_j) = Σ_k (−S[k, v_jk] + logZ_k − 0)?  w = p/q = exp(−Σφθ + logZ)
            s = -S.ravel()[idx].sum(1) + logZ.sum()
            z = f2w + s
            mz = z.max(); e = np.exp(z - mz); tot = e.sum()
            val = mz + np.log(tot / n)
            om = e / tot                                                   # normalised path weights
            Wkv = np.bincount(flat_all, weights=np.repeat(om, K), minlength=K * Vmax).reshape(K, Vmax)
            # d/dθ_k = Σ_v Wkv[k,v] (−Φ_k(v)) + E_{q_k}Φ_k   (Σ_v Wkv[k,v] = 1)
            g = -np.einsum("kv,kvm->km", Wkv, PhiT) + np.einsum("kv,kvm->km", Q, PhiT)
            d = th - th0
            return val + 0.5 * kappa * d @ d, g.reshape(-1) + kappa * d

        x0 = (self.theta if th_init is None else th_init).reshape(-1).copy()
        bnds = None
        if bound is not None:
            lo, hi = th0 - bound.reshape(-1), th0 + bound.reshape(-1)
            bnds = list(zip(lo, hi))
        r = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=bnds, options={"maxiter": maxiter})
        return r.x.reshape(K, m), {"n_pos": int(pos.sum()), "ok": True, "nit": int(r.nit), "val": float(r.fun)}

    def fit_ce(self, acts, f, w_ref, theta0, kappa=1e-3, bound=None, maxiter=40, weights=None):
        """Weighted maximum likelihood θ = argmax Σ_j W_j log q_θ(x_j) − ridge, W = w_ref f (or `weights`)."""
        W = w_ref * f if weights is None else weights
        keep = W > 0
        if keep.sum() < 20:
            return self.theta.copy(), {"n_pos": int(keep.sum()), "ok": False}
        a = acts[:, keep]; c = W[keep] / W[keep].sum()
        idx = self._flat_index(a)
        K, m, Vmax = self.K, self.m, self.Vmax
        PhiT = np.zeros((K, Vmax, m))
        for k in range(K):
            PhiT[k, :self.Vk[k]] = self.Phi[k]
        Wkv = np.bincount(idx.ravel(), weights=np.repeat(c, K), minlength=K * Vmax).reshape(K, Vmax)
        target = np.einsum("kv,kvm->km", Wkv, PhiT)                         # weighted mean feature per source
        th0 = np.asarray(theta0, float).reshape(-1)

        def obj(th):
            T = th.reshape(K, m)
            S = np.einsum("kvm,km->kv", PhiT, T)
            lg = np.full((K, Vmax), -np.inf)
            for k in range(K):
                lg[k, :self.Vk[k]] = self.logp[k] + S[k, :self.Vk[k]]
            mx = lg.max(1, keepdims=True)
            logZ = (mx + np.log(np.exp(lg - mx).sum(1, keepdims=True)))[:, 0]
            Q = np.exp(lg - logZ[:, None])
            val = -((Wkv * S).sum() - logZ.sum())
            g = -(target - np.einsum("kv,kvm->km", Q, PhiT))
            d = th - th0
            return val + 0.5 * kappa * d @ d, g.reshape(-1) + kappa * d

        bnds = None
        if bound is not None:
            bnds = list(zip(th0 - bound.reshape(-1), th0 + bound.reshape(-1)))
        r = minimize(obj, self.theta.reshape(-1).copy(), jac=True, method="L-BFGS-B", bounds=bnds,
                     options={"maxiter": maxiter})
        return r.x.reshape(K, m), {"n_pos": int(keep.sum()), "ok": True, "nit": int(r.nit)}

    # ---- encoded amplitude (quantum loading of the same measure) ----------------------------------
    def amplitude_bound(self, mu):
        """For the measure q_θ loaded by source registers and the objective rotation g/C, a valid normaliser is
        C ≥ sup_x w(x) f(x)/f_max; the unconditional bound C_ub = Π_k max_v p_k(v)/q_k(v) is loose but always valid
        (for f ∈ {0,1}: a = μ / C_ub is a lower bound on the best encodable amplitude)."""
        logC = sum(float(self.lr[k, :self.Vk[k]][np.array(self.q[k]) > 1e-300].max()) for k in range(self.K))
        return {"log_C_ub": logC, "a_lower": float(mu * np.exp(-logC)) if mu is not None else None}


def _pack(g):
    """Compact batch record (n, mean, unbiased variance): the estimate only needs these."""
    n = len(g)
    return (n, float(g.mean()), float(g.var(ddof=1)) if n > 1 else 0.0)


def _stats(batches):
    """Combine packed batches with weights ∝ size: estimate, se, dof (Welch-Satterthwaite)."""
    n = np.array([b[0] for b in batches], float)
    N = n.sum()
    est = float(sum(b[0] * b[1] for b in batches) / N)
    contrib = np.array([b[0] * b[2] for b in batches]) / N ** 2
    var = contrib.sum()
    den = (contrib ** 2 / np.maximum(n - 1, 1)).sum()
    dof = var ** 2 / den if den > 0 else np.inf
    return est, float(np.sqrt(var)), float(dof)


def _ci(est, se, dof):
    zq = float(_tdist.ppf(0.975, dof)) if np.isfinite(dof) and dof > 0 else 1.96
    return (est - zq * se, est + zq * se)


def tilt_estimate(prob: PortfolioProblem, rng, seconds=None, paths=None, *, fit=True, features="ell+ind",
                  sampler=None, theta0=None, b_train=None, train_frac=0.35, max_iter=4, beta=0.8, kappa=1e-3,
                  bound=(1.0, 1.5), use_ce=False, recycle=True, b_final=None, margin=0.9, t_start=None):
    """Source-tilt estimate of E[f] under a wall-clock budget `seconds` (training and everything charged) or a path
    budget `paths`. fit=False gives plain exponential twisting (same sampler). Returns a dict with the estimate,
    95% interval, ρ̂² of the last measure, per-path seconds τ, and the fitted θ."""
    t0 = time.perf_counter() if t_start is None else t_start
    P = prob
    S = sampler or TiltSampler(P, features)
    thresholded = P.target in ("exceedance", "stop_loss")
    eta0 = _best_twist_sources(P) if (thresholded and theta0 is None) else 0.0
    th_init = S.twist_theta(eta0) if theta0 is None else np.asarray(theta0, float).reshape(S.K, S.m)
    S.set_theta(th_init)
    th_ref = th_init.copy()
    # trust region around the starting measure: |Δθ_ℓ| <= bound[0]·max(η0, 0.05) (+0.02), |Δθ_ind| <= bound[1]
    bnd = np.zeros((S.K, S.m)); bnd[:, 0] = bound[0] * max(abs(eta0), 0.05) + 0.02
    if S.m > 1:
        bnd[:, 1] = bound[1]
    batches, log = [], []
    t_batch = []   # (n, seconds) of every draw

    def left_s():
        return seconds - (time.perf_counter() - t0)

    def draw(n, record):
        ts = time.perf_counter()
        o = S.sample(n, rng, record)
        g = o["w"] * o["f"]
        t_batch.append((n, time.perf_counter() - ts))
        batches.append(_pack(g))
        return o, g

    def tau():
        n = sum(a for a, _ in t_batch); s = sum(b for _, b in t_batch)
        return s / max(n, 1)

    total_paths = paths
    if seconds is not None:       # first small batch to learn τ
        b0 = b_train or 2048
    else:
        b0 = b_train or int(max(512, paths // 16))
    # ---- training ---------------------------------------------------------------------------------
    best = {"rho2": np.inf, "theta": th_init.copy()}
    used = 0
    if fit and thresholded:
        for it in range(max_iter):
            o, g = draw(b0, True); used += b0
            mu = g.mean()
            rho2 = g.var() / mu ** 2 if mu > 0 else np.inf
            log.append({"it": it, "rho2_hat": float(rho2), "n_pos": int((o["f"] > 0).sum())})
            if rho2 < best["rho2"] * 0.97 or not np.isfinite(best["rho2"]):
                best = {"rho2": rho2, "theta": S.theta.copy()}
            elif it >= 1:
                break
            # stop training when time/paths share is exhausted
            if seconds is not None:
                spent = time.perf_counter() - t0
                if spent > train_frac * seconds:
                    break
            elif used >= train_frac * paths:
                break
            try:
                if use_ce and (o["f"] > 0).mean() < 0.02:
                    th_new, info = S.fit_ce(o["acts"], o["f"], o["w"], th_ref, kappa, bnd, maxiter=25)
                else:
                    th_new, info = S.fit_chi2(o["acts"], o["f"], o["w"], th_ref, kappa, bnd, maxiter=30)
            except FloatingPointError:
                break
            if not info.get("ok"):
                break
            S.set_theta(S.theta + beta * (th_new - S.theta))
        # final candidate: the best *estimated* iterate (or the last one if it is untested)
        S.set_theta(best["theta"])
    # ---- estimation -------------------------------------------------------------------------------------
    nb = lambda: sum(b[0] for b in batches)
    if seconds is not None:
        while True:
            rem = left_s() * margin
            if rem <= 0.0:
                break
            tt = max(tau(), 1e-9)
            b_fin = b_final or int(np.clip(0.01 * seconds / tt, 4096, max(4096, 4_000_000 // S.K)))
            n = int(min(b_fin, rem / tt))
            if n < 256:
                break
            draw(n, False)
    else:
        while nb() < total_paths:
            draw(min(b_final or 16384, total_paths - nb()), False)
    use = batches if recycle else batches[-max(1, len(batches) - len(log)):]
    est, se, dof = _stats(use)
    fin = batches[len(log):] if len(batches) > len(log) else batches[-1:]
    nf = sum(b[0] for b in fin); mf = sum(b[0] * b[1] for b in fin) / nf
    # pooled second moment of the final batches -> per-path relative variance rho^2
    var_pool = (sum((b[0] - 1) * b[2] + b[0] * (b[1] - mf) ** 2 for b in fin)) / max(nf - 1, 1)
    rho2 = float(var_pool / mf ** 2) if mf > 0 else float("inf")
    return {"estimate": est, "se": se, "ci": _ci(est, se, dof), "dof": dof, "paths_used": int(nb()),
            "seconds": time.perf_counter() - t0, "rho2": rho2, "tau": tau(), "theta": S.theta.copy(),
            "log": log, "eta0": eta0, "n_batches": len(batches)}
