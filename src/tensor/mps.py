"""MPS representation of the term distribution, built by applying the local source kernels
as MPOs and recompressing (exact up to an SVD truncation tolerance), plus exact contraction
of the five portfolio targets via a running-loss transfer.

This is the "strongest classical baseline at scale" of plan §4/§7-risk-1: when the dependence
graph has small cut-width, every target is computable exactly in poly(n) time.
"""
import numpy as np

from .synthetic import SyntheticModel, D


class MPS:
    def __init__(self, tensors, gauge=None):
        self.T = tensors  # list of (chi_l, d, chi_r)
        # tilted gauge: stored tensor represents p(x) * prod_i g_i(x_i); None = untilted
        self.gauge = gauge

    @classmethod
    def product(cls, n):
        t = np.zeros((1, D, 1)); t[0, 0, 0] = 1.0
        return cls([t.copy() for _ in range(n)])

    @property
    def bond_dims(self):
        return [t.shape[2] for t in self.T[:-1]]

    def apply_single(self, i, K):
        self.T[i] = np.einsum("lpr,pq->lqr", self.T[i], K)

    def apply_pair(self, i, j, K, tol):
        """Apply a two-site kernel on sites i<j (any distance) as an MPO, then compress."""
        assert i < j
        K4 = K.reshape(D, D, D, D)                       # [pi, pj, qi, qj]
        M = K4.transpose(0, 2, 1, 3).reshape(D * D, D * D)  # (pi qi) x (pj qj)
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        k = int((S > S[0] * 1e-14).sum())
        A = (U[:, :k] * S[:k]).reshape(D, D, k)          # [pi, qi, k]
        B = Vt[:k].reshape(k, D, D)                      # [k, pj, qj]
        t = self.T[i]
        self.T[i] = np.einsum("lpr,pqk->lqrk", t, A).reshape(t.shape[0], D, t.shape[2] * k)
        for s in range(i + 1, j):
            t = self.T[s]
            self.T[s] = np.einsum("lpr,km->lkprm", t, np.eye(k)).reshape(t.shape[0] * k, D, t.shape[2] * k)
        t = self.T[j]
        self.T[j] = np.einsum("lpr,kpq->lkqr", t, B).reshape(t.shape[0] * k, D, t.shape[2])
        self.compress(tol)

    def compress(self, tol):
        """Left-canonicalise with QR, then right-to-left truncated SVD (relative Frobenius tol per bond)."""
        n = len(self.T)
        for s in range(n - 1):
            l, d, r = self.T[s].shape
            Q, Rm = np.linalg.qr(self.T[s].reshape(l * d, r))
            self.T[s] = Q.reshape(l, d, -1)
            self.T[s + 1] = np.einsum("ab,bdr->adr", Rm, self.T[s + 1])
        for s in range(n - 1, 0, -1):
            l, d, r = self.T[s].shape
            U, S, Vt = np.linalg.svd(self.T[s].reshape(l, d * r), full_matrices=False)
            tot = (S ** 2).sum()
            keep = len(S)
            if tol > 0 and tot > 0:
                tail = np.cumsum((S ** 2)[::-1])[::-1]  # tail[k] = Σ_{i>=k} S_i^2
                keep = max(1, int((tail > (tol ** 2) * tot).sum()))
            self.T[s] = Vt[:keep].reshape(keep, d, r)
            self.T[s - 1] = np.einsum("ldr,rk->ldk", self.T[s - 1], U[:, :keep] * S[:keep])

    def to_dense(self):
        T = self.T if self.gauge is None else [t / g[None, :, None] for t, g in zip(self.T, self.gauge)]
        out = T[0]
        for t in T[1:]:
            out = np.tensordot(out, t, axes=([-1], [0]))
        return out.reshape(out.shape[1:-1])


def site_gauges(model: SyntheticModel):
    """g_i(x) = 1 / (marginal of site i under the *independent* model with the same total rates).
    Tilting by g puts rare tail entries on the same scale as the bulk, so SVD truncation
    (a Frobenius-norm criterion) controls *relative* error on tail targets. This is the
    tensor-network analogue of the Doob h-transform used in Phases 1-3."""
    ind = SyntheticModel(model.n, [], m=0.0, rain_mult=model.rain_mult, years=model.years,
                         eq_limit=model.eq_limit, rain_limit=model.rain_limit)
    return [1.0 / np.maximum(ind.single_kernel(i)[0], 1e-300) for i in range(model.n)]


def build_mps(model: SyntheticModel, tol=1e-12, tilt=True):
    psi = MPS.product(model.n)
    g = site_gauges(model) if tilt else [np.ones(D)] * model.n
    psi.gauge = g
    # p~ = p diag(g): kernels become diag(1/g) K diag(g); the product state |0> picks up g(0)
    for i in range(model.n):
        psi.T[i] = psi.T[i] * g[i][None, :, None]
        psi.apply_single(i, (model.single_kernel(i) / g[i][:, None]) * g[i][None, :])
    max_bonds = 1
    for (i, j) in sorted(model.edges):
        gij = np.kron(g[i], g[j])
        K = (model.pair_kernel(i, j) / gij[:, None]) * gij[None, :]
        psi.apply_pair(min(i, j), max(i, j), K, tol)
        max_bonds = max(max_bonds, max(psi.bond_dims) if psi.bond_dims else 1)
    return psi, max_bonds


def targets_from_mps(model: SyntheticModel, psi: MPS, threshold):
    """Exact (up to MPS truncation) portfolio targets via left-to-right transfer.

    Carries (bond, running loss in 2.5m units, #countries paying (capped at 2)).
    Returns E[L], Pr(>=2 countries pay), Pr(L > T), E[(L-T)+], Pr(all exhausted), total mass."""
    unit = 2.5e6
    lu = model.site_loss_units(unit)
    full = lu.max()
    n = model.n
    Lmax = full * n
    # state: v[chi, loss, npay(0,1,2)]
    v = np.zeros((1, Lmax + 1, 3)); v[0, 0, 0] = 1.0
    allx = np.ones(1)  # transfer for "all exhausted"
    for s, t in enumerate(psi.T):
        if psi.gauge is not None:
            t = t / psi.gauge[s][None, :, None]   # un-tilt on the fly
        chi_r = t.shape[2]
        nv = np.zeros((chi_r, Lmax + 1, 3))
        for x in range(D):
            A = t[:, x, :]                               # (chi_l, chi_r)
            contrib = np.einsum("lLc,lr->rLc", v, A)
            sh = lu[x]
            pay = 1 if sh > 0 else 0
            if pay:
                tmp = np.zeros_like(contrib)
                tmp[:, :, 1] += contrib[:, :, 0]
                tmp[:, :, 2] += contrib[:, :, 1] + contrib[:, :, 2]
                contrib = tmp
            nv[:, sh:, :] += contrib[:, :Lmax + 1 - sh, :]
        v = nv
        allx = allx @ t[:, int(np.argmax(lu)), :]
    dist = v[0]  # (Lmax+1, 3)
    pL = dist.sum(axis=1)
    Lval = np.arange(Lmax + 1) * unit
    mass = pL.sum()
    return {
        "expected_loss": float(pL @ Lval),
        "prob_at_least_two_pay": float(dist[:, 2].sum()),
        "exceedance_prob": float(pL[Lval > threshold].sum()),
        "stop_loss": float(pL @ np.maximum(Lval - threshold, 0)),
        "all_exhausted_prob": float(allx[0]),
        "total_mass": float(mass),
    }
