"""Route C (scoped): classical vs quantum common-cause dimension across a cut.

For a bipartite joint distribution P(x, y) (x = one block of countries, y = the rest):
  * classical latent model  P = Σ_λ π(λ) P(x|λ) P(y|λ)  needs |Λ| >= rank_+(P)   (nonnegative rank)
  * quantum latent model    P(x,y) = Tr[ρ (E_x ⊗ F_y)]  — equivalently P = Tr(A_x B_y) with PSD
    A_x, B_y of size k — needs k >= rank_psd(P)                                     (PSD rank)
  Fiorini et al. (STOC 2012); Fawzi, Gouveia, Parrilo, Robinson, Thomas (Math. Prog. 2015).
  Always  sqrt(rank) <= rank_psd <= rank_+ <= min(|X|,|Y|)  (rank = ordinary rank, for
  real-symmetric PSD factors the lower bound is rank <= k(k+1)/2).

No-input, single-measurement setting => there is no feasibility separation (Bell-type); only a
*dimension* separation is possible. We compute numerical upper bounds on both ranks by
best-approximation at each k (NMF vs real PSD factorization), on the tilted matrix
D_g P D_g (diagonal positive scaling leaves both ranks unchanged and makes tail entries count).
These are heuristic (non-convex) upper bounds, not certificates.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.tensor.synthetic import SyntheticModel, topology, dense_term_distribution
from src.tensor.mps import site_gauges


def cut_matrix(model, left):
    P = dense_term_distribution(model)
    g = site_gauges(model)
    for i, gi in enumerate(g):  # tilt
        shape = [1] * model.n; shape[i] = -1
        P = P * gi.reshape(shape)
    n = model.n
    dl = 10 ** left
    return P.reshape(dl, -1)


def nmf_error(M, k, restarts=6, iters=4000, seed=0):
    """Best-of-restarts HALS NMF (Cichocki & Phan 2009) from NNDSVD + random inits.
    For k >= min(M.shape) the exact factorisation exists (rank_+ <= min dimension) -> 0."""
    if k >= min(M.shape):
        return 0.0
    rng = np.random.default_rng(seed)
    nrm = np.linalg.norm(M)
    best = np.inf
    inits = [_nndsvd(M, k)] + [(rng.random((M.shape[0], k)) * np.sqrt(M.mean() / k),
                                 rng.random((k, M.shape[1])) * np.sqrt(M.mean() / k)) for _ in range(restarts - 1)]
    for W, H in inits:
        W, H = W.copy(), H.copy()
        for _ in range(iters):
            WtM, WtW = W.T @ M, W.T @ W
            for j in range(k):
                H[j] = np.maximum(H[j] + (WtM[j] - WtW[j] @ H) / max(WtW[j, j], 1e-300), 0)
            MHt, HHt = M @ H.T, H @ H.T
            for j in range(k):
                W[:, j] = np.maximum(W[:, j] + (MHt[:, j] - W @ HHt[:, j]) / max(HHt[j, j], 1e-300), 0)
        best = min(best, np.linalg.norm(M - W @ H) / nrm)
    return float(best)


def _nndsvd(M, k):
    """NNDSVD initialisation (Boutsidis & Gallopoulos 2008)."""
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    W = np.zeros((M.shape[0], k)); H = np.zeros((k, M.shape[1]))
    for j in range(min(k, len(S))):
        u, v = U[:, j], Vt[j]
        up, un, vp, vn = np.maximum(u, 0), np.maximum(-u, 0), np.maximum(v, 0), np.maximum(-v, 0)
        a, b = (up, vp) if np.linalg.norm(up) * np.linalg.norm(vp) >= np.linalg.norm(un) * np.linalg.norm(vn) else (un, vn)
        na, nb = np.linalg.norm(a) + 1e-300, np.linalg.norm(b) + 1e-300
        sc = np.sqrt(S[j] * na * nb)
        W[:, j] = sc * a / na; H[j] = sc * b / nb
    return W + 1e-12, H + 1e-12


def psd_error(M, k, restarts=4, seed=0, maxiter=3000):
    """min Σ_xy (M_xy - <A_x, B_y>_F)^2 with A_x = U_x U_x^T, B_y = V_y V_y^T  (real k x k).
    <U U^T, V V^T> = ||U^T V||_F^2."""
    rng = np.random.default_rng(seed)
    X, Y = M.shape
    nrm = np.linalg.norm(M)
    best = np.inf

    def obj(z):
        U = z[:X * k * k].reshape(X, k, k); V = z[X * k * k:].reshape(Y, k, k)
        C = np.einsum("xia,yib->xyab", U, V)               # U_x^T V_y   (X, Y, k, k)
        R = (C ** 2).sum(axis=(2, 3)) - M
        gU = 4 * np.einsum("xy,yib,xyab->xia", R, V, C)
        gV = 4 * np.einsum("xy,xia,xyab->yib", R, U, C)
        return float((R ** 2).sum()), np.concatenate([gU.ravel(), gV.ravel()])

    for _ in range(restarts):
        s = (M.mean() / k) ** 0.25
        z0 = rng.normal(0, s, (X + Y) * k * k)
        r = minimize(obj, z0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
        best = min(best, np.sqrt(r.fun) / nrm)
    return float(best)


def numerical_rank(M, tol):
    s = np.linalg.svd(M, compute_uv=False)
    return int((s > tol * s[0]).sum()), s.tolist()


def run(out=Path("results/causal.json")):
    res = {"note": "heuristic upper bounds on rank_+ (NMF) and rank_psd (real PSD factorization); "
                   "relative Frobenius error on the tilted cut matrix", "cases": {}}
    cases = {
        "adb2_term_cut(KG|TJ)": (SyntheticModel(2, topology("chain", 2)), 1),
        "ring4_term_cut(01|23)": (SyntheticModel(4, topology("ring", 4)), 2),
        "complete4_term_cut(01|23)": (SyntheticModel(4, topology("complete", 4)), 2),
    }
    for name, (mdl, left) in cases.items():
        M = cut_matrix(mdl, left)
        r6, sv = numerical_rank(M, 1e-6)
        r12, _ = numerical_rank(M, 1e-12)
        ks = list(range(1, 11)) if M.shape[0] == 10 else [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 30]
        rows = []
        for k in ks:
            e_n = nmf_error(M, k)
            e_p = psd_error(M, k, restarts=2 if M.shape[0] > 10 else 4, maxiter=1500 if M.shape[0] > 10 else 3000) \
                if k <= 12 else None
            rows.append({"k": k, "nmf_rel_err": e_n, "psd_rel_err": e_p})
            print(f"{name:28s} k={k:2d} NMF={e_n:.2e} PSD={e_p if e_p is None else f'{e_p:.2e}'}",
                  file=sys.stderr, flush=True)
        # rank_+ <= k' implies rank_+ <= k for k >= k': report the monotone envelope of the heuristics
        for key in ("nmf_rel_err", "psd_rel_err"):
            run_min = np.inf
            for r in rows:
                if r[key] is not None:
                    run_min = min(run_min, r[key])
                    r[key + "_envelope"] = run_min
        for r in rows:
            r["params_per_outcome_nmf"] = r["k"]
            r["params_per_outcome_psd"] = r["k"] * (r["k"] + 1) // 2
        res["cases"][name] = {"shape": list(M.shape), "rank_tol1e-6": r6, "rank_tol1e-12": r12,
                              "singular_values_rel": (np.array(sv) / sv[0]).tolist()[:40], "curve": rows}
    out.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    run()
