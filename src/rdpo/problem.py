"""Portfolio problems in source form (shared by RDPO, baselines and exact references).

A problem is an n-country portfolio (`SyntheticModel`, which reduces exactly to the ADB benchmark
for n = 2) plus one target functional f of the terminal state. The term law is a product of K
independent Poisson sources (solo EQ / solo rain per country, shared EQ / shared rain per
dependence edge); source k has a finite capped-increment law p_k on V_k <= 25 values.

Loss is measured in units of USD 2.5m: an EQ quarter step is a_u units, a rain trigger r_u
units, and country i's term loss is L_i = a_u·a_i + r_u·r_i (a_i ∈ {0..4}, r_i ∈ {0,1}).

Targets (f) and their multilevel *scores* (used only to schedule cross-entropy levels):
    expected_loss     f = L·unit                          (not rare: no level schedule)
    joint_attachment  f = 1{L_i > 0, L_j > 0}            score min(L_i, L_j)
    exceedance        f = 1{L > T}                        score L
    stop_loss         f = (L − T)^+ · unit                score L
    joint_exhaustion  f = 1{L_i = F, L_j = F}             score min(L_i, L_j)
(i, j) = `pair`, F = full limit in units.  Target-progress features h(·) depend only on the total
loss L and the pair losses (L_i, L_j), so updating them after a source step is O(1).
"""
import numpy as np

from src.tensor.synthetic import SyntheticModel, topology, dense_term_distribution
from src.quantum.structured_loading import sources as model_sources

UNIT = 2.5e6
TARGETS = ["expected_loss", "joint_attachment", "exceedance", "stop_loss", "joint_exhaustion"]
ADB_SCENARIOS = {   # brief Table 4, expressed as a 2-country model with one dependence edge
    "adb-base": {},
    "adb-independent": {"m": 0.0, "rain_shared": 0.0},
    "adb-common-shock": {"m": 2.0},
    "adb-rain-stress": {"rain_mult": 2.0},
    "adb-rain-cover": {"eq_limit": 60e6, "rain_limit": 20e6},
}


def adb_model(scenario="adb-base"):
    return SyntheticModel(2, topology("chain", 2), **ADB_SCENARIOS[scenario])


class PortfolioProblem:
    def __init__(self, model: SyntheticModel, target, threshold=None, pair=(0, 1)):
        assert target in TARGETS
        self.model, self.n, self.target, self.pair = model, model.n, target, pair
        self.edges = list(model.edges)
        self.a_u = round(model.eq_limit / 4 / UNIT)
        self.r_u = round(model.rain_limit / UNIT)
        assert np.isclose(self.a_u * UNIT * 4, model.eq_limit) and np.isclose(self.r_u * UNIT, model.rain_limit)
        self.F = 4 * self.a_u + self.r_u
        self.Lmax = self.n * self.F
        self.T = (threshold if threshold is not None else 40e6 * self.n) / UNIT
        self.src = []
        if hasattr(model, "custom_sources"):          # e.g. sources built from an event loss table (src/elt)
            self.src = [dict(c) for c in model.custom_sources]
            self.K = len(self.src)
            return
        for s in model_sources(model):
            v = np.arange(len(s.law))
            if s.kind == "eq1":
                da, dr = [(s.sites[0], v)], []
            elif s.kind == "eq2":
                da, dr = [(s.sites[0], v // 5), (s.sites[1], v % 5)], []
            elif s.kind == "rain1":
                da, dr = [], [(s.sites[0], v)]
            else:
                da, dr = [], [(s.sites[0], v), (s.sites[1], v)]
            self.src.append({"kind": s.kind, "sites": tuple(s.sites), "p": s.law,
                             "logp": np.log(np.maximum(s.law, 1e-300)), "da": da, "dr": dr})
        self.K = len(self.src)

    # --- losses, payoff, score, progress features -----------------------------------------
    def site_loss(self, a, r):
        return self.a_u * a + self.r_u * r

    def payoff_from(self, Ltot, Li, Lj):
        t = self.target
        if t == "expected_loss":
            return Ltot * UNIT
        if t == "joint_attachment":
            return ((Li > 0) & (Lj > 0)).astype(float)
        if t == "exceedance":
            return (Ltot > self.T).astype(float)
        if t == "stop_loss":
            return np.maximum(Ltot - self.T, 0) * UNIT
        return ((Li == self.F) & (Lj == self.F)).astype(float)

    def score_from(self, Ltot, Li, Lj):
        if self.target in ("joint_attachment", "joint_exhaustion"):
            return np.minimum(Li, Lj).astype(float)
        return Ltot.astype(float)

    def level(self):
        """Smallest score at which f > 0 (multilevel CE stops raising γ here)."""
        t = self.target
        if t == "expected_loss":
            return 0.0
        if t == "joint_attachment":
            return float(min(self.a_u, self.r_u))
        if t in ("exceedance", "stop_loss"):
            return float(np.floor(self.T) + 1)
        return float(self.F)

    NH = {"expected_loss": 1, "joint_attachment": 2, "exceedance": 3, "stop_loss": 3, "joint_exhaustion": 3}

    def h(self, Ltot, Li, Lj):
        """Target-progress features (shape (..., G))."""
        t = self.target
        if t == "expected_loss":
            return (Ltot / self.Lmax)[..., None]
        if t == "joint_attachment":
            return np.stack([((Li > 0).astype(float) + (Lj > 0)) / 2, ((Li > 0) & (Lj > 0)).astype(float)], -1)
        if t in ("exceedance", "stop_loss"):
            T = self.T
            return np.stack([np.minimum(Ltot, T) / T, (Ltot > T).astype(float),
                             np.maximum(Ltot - T, 0) / (self.Lmax - T)], -1)
        F = self.F
        return np.stack([np.minimum(Li, Lj) / F, (Li + Lj) / (2 * F), ((Li == F) & (Lj == F)).astype(float)], -1)

    # --- exact reference by dense enumeration (n <= 6) -------------------------------------
    def exact_dense(self):
        n = self.n
        if hasattr(self.model, "custom_sources"):
            P = self._dense_from_sources()
        else:
            P = dense_term_distribution(self.model).reshape(-1)
        idx = np.arange(10 ** n)
        digits = np.stack([(idx // 10 ** (n - 1 - i)) % 10 for i in range(n)], 1)
        a, r = digits // 2, digits % 2
        Ls = self.site_loss(a, r)
        i, j = self.pair
        return float(P @ self.payoff_from(Ls.sum(1), Ls[:, i], Ls[:, j]))

    # --- exact reduction for targets that depend on a country pair only -----------------------
    def exact_pair(self):
        """Exact E[f] for pair targets (joint attachment / exhaustion) in *any* portfolio size.

        f depends only on the terminal codes (s_i, s_j) ∈ {0..9}², and sources commute (Proposition 2), so
        the joint law of (s_i, s_j) is the product of the kernels of the sources touching i or j, acting on
        a 100-state space; all other sources integrate out. Cost O(K · 100 · 25)."""
        assert self.target in ("joint_attachment", "joint_exhaustion")
        i, j = self.pair
        dist = np.zeros((10, 10)); dist[0, 0] = 1.0
        a_of, r_of = np.arange(10) // 2, np.arange(10) % 2
        for src in self.src:
            if i not in src["sites"] and j not in src["sites"]:
                continue
            new = np.zeros((10, 10))
            for v, pv in enumerate(src["p"]):
                if pv == 0:
                    continue
                codes = []
                for site in (i, j):
                    da = sum(inc[v] for s, inc in src["da"] if s == site)
                    dr = sum(inc[v] for s, inc in src["dr"] if s == site)
                    codes.append(2 * np.minimum(a_of + da, 4) + np.minimum(r_of + dr, 1))
                np.add.at(new, (codes[0][:, None].repeat(10, 1), codes[1][None, :].repeat(10, 0)), pv * dist)
            dist = new
        Li = self.site_loss(a_of, r_of)
        f = self.payoff_from(np.zeros((10, 10)), Li[:, None].repeat(10, 1), Li[None, :].repeat(10, 0))
        return float((dist * f).sum())

    def _dense_from_sources(self):
        """Exact 10^n joint law by applying every source's term law (capped increments) to the start state."""
        n = self.n
        assert n <= 6, "dense enumeration is limited to 6 countries"
        P = np.zeros(10 ** n); P[0] = 1.0
        idx = np.arange(10 ** n)
        digits = np.stack([(idx // 10 ** (n - 1 - i)) % 10 for i in range(n)], 1)
        a0, r0 = digits // 2, digits % 2
        for src in self.src:
            new = np.zeros_like(P)
            for v, pv in enumerate(src["p"]):
                if pv == 0:
                    continue
                a, r = a0.copy(), r0.copy()
                for site, inc in src["da"]:
                    a[:, site] = np.minimum(a[:, site] + inc[v], 4)
                for site, inc in src["dr"]:
                    r[:, site] = np.minimum(r[:, site] + inc[v], 1)
                code = 2 * a + r
                target = (code * (10 ** np.arange(n - 1, -1, -1))).sum(1)
                np.add.at(new, target, pv * P)
            P = new
        return P
