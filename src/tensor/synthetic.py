"""Synthetic n-country extension of the ADB model, on a dependence graph.

Each country i has a site of dimension 10: (a_i ∈ 0..4 EQ quarter steps, r_i ∈ {0,1} rain).
Independent Poisson sources:
  * EQ solo source per country:  rate Λ_i − Σ_{j~i} c_ij, sizes w_i
  * EQ shared source per edge:   rate c_ij = m · ½(0.075 Λ_i + 0.074 Λ_j), sizes drawn independently
  * rain solo per country:       rate μ_i − Σ_{j~i} c^r_ij
  * rain shared per edge:        rate c^r_ij = s_r · min(μ_i, μ_j) / max_degree   (s_r = 0.25)
For n = 2 and a single edge this is exactly the ADB benchmark.

Key structural fact (verified for n=2 in tests): capped addition is a commutative monoid action,
so the generators of different sources commute and the T-year term distribution is
    P_term = Π_sources exp(T · G_source) |0…0>,
i.e. a product of 1- and 2-site "gates" applied to a product state. Its tensor-network
complexity is therefore governed by how many shared-source edges cross each cut of the
site ordering (cut-width / pathwidth of the dependence graph).
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from src.model.poisson_fit import TAILS, P_TJ_GIVEN_KG, P_KG_GIVEN_TJ, compound_poisson_rates

D = 10  # site dimension: index = a*2 + r


def topology(kind, n):
    if kind == "chain":
        return [(i, i + 1) for i in range(n - 1)]
    if kind == "ring":
        return [(i, i + 1) for i in range(n - 1)] + ([(0, n - 1)] if n > 2 else [])
    if kind == "complete":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if kind == "star":
        return [(0, j) for j in range(1, n)]
    if kind == "none":
        return []
    raise ValueError(kind)


@dataclass
class SyntheticModel:
    n: int
    edges: list
    m: float = 1.0
    rain_mult: float = 1.0
    rain_shared: float = 0.25
    years: float = 3.0
    eq_limit: float = 70e6
    rain_limit: float = 10e6

    def __post_init__(self):
        names = ["KG", "TJ"]
        self.lam, self.Lam = [], []
        for i in range(self.n):
            l, L = compound_poisson_rates(TAILS[names[i % 2]])
            self.lam.append(l); self.Lam.append(L)
        self.w = [l / L for l, L in zip(self.lam, self.Lam)]
        mu = -np.log(1 - 0.02) * self.rain_mult
        self.mu = [mu] * self.n
        deg = np.zeros(self.n, int)
        for i, j in self.edges:
            deg[i] += 1; deg[j] += 1
        self.maxdeg = max(int(deg.max()) if self.n else 0, 1)
        self.c = {}
        self.cr = {}
        for i, j in self.edges:
            pij = P_TJ_GIVEN_KG if i % 2 == 0 else P_KG_GIVEN_TJ
            pji = P_KG_GIVEN_TJ if i % 2 == 0 else P_TJ_GIVEN_KG
            self.c[(i, j)] = self.m * 0.5 * (pij * self.Lam[i] + pji * self.Lam[j])
            self.cr[(i, j)] = self.rain_shared * min(self.mu[i], self.mu[j]) / self.maxdeg
        self.solo_eq = [self.Lam[i] - sum(v for e, v in self.c.items() if i in e) for i in range(self.n)]
        self.solo_rain = [self.mu[i] - sum(v for e, v in self.cr.items() if i in e) for i in range(self.n)]
        assert min(self.solo_eq) >= 0 and min(self.solo_rain) >= 0, "shared rates exceed totals"

    # --- local kernels (T-year transition matrices) ------------------------------
    def _site_jump(self, i):
        """Single-site EQ jump matrix J (10x10, rows sum to 1) for one event at site i."""
        J = np.zeros((D, D))
        for a in range(5):
            for r in range(2):
                for k in range(1, 5):
                    J[a * 2 + r, min(a + k, 4) * 2 + r] += self.w[i][k - 1]
        return J

    def _site_rain(self):
        J = np.zeros((D, D))
        for a in range(5):
            for r in range(2):
                J[a * 2 + r, a * 2 + 1] = 1.0
        return J

    def single_kernel(self, i):
        G = self.solo_eq[i] * (self._site_jump(i) - np.eye(D)) + self.solo_rain[i] * (self._site_rain() - np.eye(D))
        return expm(self.years * G)

    def pair_kernel(self, i, j):
        I = np.eye(D * D)
        Jeq = np.kron(self._site_jump(i), self._site_jump(j))
        Jr = np.kron(self._site_rain(), self._site_rain())
        G = self.c[(i, j)] * (Jeq - I) + self.cr[(i, j)] * (Jr - I)
        return expm(self.years * G)

    # --- loss maps per site -------------------------------------------------------
    def site_loss(self):
        a = np.repeat(np.arange(5), 2); r = np.tile(np.arange(2), 5)
        return self.eq_limit * a / 4 + self.rain_limit * r

    def site_loss_units(self, unit=2.5e6):
        L = self.site_loss() / unit
        assert np.allclose(L, np.round(L))
        return np.round(L).astype(int)


def dense_term_distribution(model: SyntheticModel):
    """Exact 10^n tensor by applying kernels (row-vector convention: p ← p K)."""
    n = model.n
    P = np.zeros((D,) * n); P[(0,) * n] = 1.0
    for i in range(n):
        P = np.moveaxis(np.tensordot(P, model.single_kernel(i), axes=([i], [0])), -1, i)
    for (i, j) in model.edges:
        K = model.pair_kernel(i, j).reshape(D, D, D, D)
        P = np.tensordot(P, K, axes=([i, j], [0, 1]))  # new axes at the end
        P = np.moveaxis(P, [-2, -1], [i, j])
    return P
