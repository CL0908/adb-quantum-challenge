"""Structured ("Poisson-source") quantum loading of the n-country term distribution.

Concept (the scalable quantum route)
------------------------------------
P_term = Π_sources exp(T·G_source)|0…0⟩ (REPORT §1.3). Every source (solo EQ, solo rain,
shared EQ on an edge, shared rain on an edge) acts independently. Because quarter steps are
capped at 4 and rain is an OR, a source's whole effect over the term is a *capped increment*
with a small finite law:
    solo EQ  : Δa_i ∈ {0..4}          (5 outcomes, 3 qubits)
    shared EQ: (Δa_i, Δa_j) ∈ {0..4}² (25 outcomes, 6 qubits)
    rain     : Δr ∈ {0,1}             (1 qubit)
min(a + min(x,4), 4) = min(a + x, 4) makes this *exact*, with no truncation of Poisson counts.

Circuit A:
  1. prepare each source register in Σ_x √p_src(x) |x⟩       (independent, all in parallel)
  2. add increments into per-country *uncapped* sum registers (in-place adders are permutations,
     so no garbage; widths sized so sums never overflow)
  3. payoff oracle: cap (a = min(S,4), r = min(R,1)), compute L = Σ 7a_i + 4r_i (2.5m units),
     compare / evaluate f, rotate the objective qubit by arcsin √g.
Qubits and gates are O(n + |E|) for ANY dependence graph; depth is O((maxdeg+1)·w + n·w_L).
This is the analogue of Egger et al. (2021)'s latent-factor loading, with the shared Poisson
sources in the role of the latent factors.

This module: (i) builds the source laws, (ii) simulates A exactly with a sparse statevector
(the state after step 1 is a product state with finite support, and steps 2-3 are permutations,
so the simulation tracks every nonzero amplitude exactly), (iii) counts logical resources.
"""
import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from src.tensor.synthetic import SyntheticModel


# ---------------------------------------------------------------------------
# 1. source laws
# ---------------------------------------------------------------------------
@dataclass
class Source:
    kind: str          # "eq1" | "eq2" | "rain1" | "rain2"
    sites: tuple
    law: np.ndarray    # probabilities over the register's valid values
    qubits: int


def _eq_increment_law(rate, w, T):
    J = np.zeros((5, 5))
    for a in range(5):
        for k in range(1, 5):
            J[a, min(a + k, 4)] += w[k - 1]
    return expm(T * rate * (J - np.eye(5)))[0]


def _eq_pair_law(rate, wi, wj, T):
    J = np.zeros((25, 25))
    for a in range(5):
        for b in range(5):
            for k in range(1, 5):
                for l in range(1, 5):
                    J[a * 5 + b, min(a + k, 4) * 5 + min(b + l, 4)] += wi[k - 1] * wj[l - 1]
    return expm(T * rate * (J - np.eye(25)))[0]


def sources(model: SyntheticModel):
    T = model.years
    out = []
    for i in range(model.n):
        out.append(Source("eq1", (i,), _eq_increment_law(model.solo_eq[i], model.w[i], T), 3))
        p = 1 - math.exp(-T * model.solo_rain[i])
        out.append(Source("rain1", (i,), np.array([1 - p, p]), 1))
    for (i, j) in model.edges:
        out.append(Source("eq2", (i, j), _eq_pair_law(model.c[(i, j)], model.w[i], model.w[j], T), 6))
        p = 1 - math.exp(-T * model.cr[(i, j)])
        out.append(Source("rain2", (i, j), np.array([1 - p, p]), 1))
    return out


def degrees(model):
    d = np.zeros(model.n, int)
    for i, j in model.edges:
        d[i] += 1; d[j] += 1
    return d


def register_widths(model):
    d = degrees(model)
    wE = [math.ceil(math.log2(4 * (1 + di) + 1)) for di in d]   # max uncapped EQ sum 4(1+deg)
    wR = [math.ceil(math.log2((1 + di) + 1)) for di in d]       # max rain count 1+deg
    return wE, wR


# ---------------------------------------------------------------------------
# 2. exact sparse-statevector simulation of A
# ---------------------------------------------------------------------------
def simulate_A(model: SyntheticModel, payoff, max_support=5_000_000):
    """Run circuit A on the nonzero-amplitude support.

    payoff(a, r) -> g in [0,1] per basis state, a,r arrays of shape (K, n) (capped values).
    Returns dict: amplitude 'a' = P(objective=1), the joint law of capped (a, r) as a dense
    10^n tensor, support size and the register layout."""
    srcs = sources(model)
    support = int(np.prod([np.count_nonzero(s.law) for s in srcs]))
    if support > max_support:
        raise ValueError(f"support {support} too large for exact simulation")
    n = model.n
    wE, wR = register_widths(model)
    # step 1: product state over source registers -> enumerate support with amplitudes
    amp = np.ones(1)
    S = np.zeros((1, n), dtype=np.int64)   # uncapped EQ sum registers
    Rr = np.zeros((1, n), dtype=np.int64)  # rain count registers
    for s in srcs:
        vals = np.nonzero(s.law)[0]
        a_src = np.sqrt(s.law[vals])
        K = len(amp)
        amp = (amp[:, None] * a_src[None, :]).ravel()
        S = np.repeat(S, len(vals), axis=0); Rr = np.repeat(Rr, len(vals), axis=0)
        v = np.tile(vals, K)
        # step 2: in-place adders (permutations of the computational basis; never overflow)
        if s.kind == "eq1":
            S[:, s.sites[0]] += v
        elif s.kind == "eq2":
            S[:, s.sites[0]] += v // 5; S[:, s.sites[1]] += v % 5
        elif s.kind == "rain1":
            Rr[:, s.sites[0]] += v
        else:
            Rr[:, s.sites[0]] += v; Rr[:, s.sites[1]] += v
    for i in range(n):
        assert S[:, i].max() < 2 ** wE[i] and Rr[:, i].max() < 2 ** wR[i], "register overflow"
    # step 3: oracle (cap, payoff) and objective rotation RY(2 arcsin sqrt g)
    a_cap, r_cap = np.minimum(S, 4), np.minimum(Rr, 1)
    g = payoff(a_cap, r_cap)
    prob = amp ** 2
    good = float(prob @ g)
    idx = np.zeros(len(prob), dtype=np.int64)
    for i in range(n):
        idx = idx * 10 + a_cap[:, i] * 2 + r_cap[:, i]
    joint = np.bincount(idx, weights=prob, minlength=10 ** n).reshape((10,) * n)
    return {"a": good, "joint": joint, "support": support, "norm": float(prob.sum()),
            "widths": {"eq": wE, "rain": wR}}


def payoff_library(model, threshold):
    unit = 2.5e6
    a7 = round(model.eq_limit / 4 / unit)    # 7 units per quarter step
    r4 = round(model.rain_limit / unit)      # 4 units per rain trigger
    Lmax_units = model.n * (4 * a7 + r4)

    def L_units(a, r):
        return (a * a7 + r * r4).sum(axis=1)

    T = threshold / unit
    return {
        "exceedance": lambda a, r: (L_units(a, r) > T).astype(float),
        "stop_loss_scaled": lambda a, r: np.maximum(L_units(a, r) - T, 0) / (Lmax_units - T),
        "at_least_two_pay": lambda a, r: (((a > 0) | (r > 0)).sum(axis=1) >= 2).astype(float),
        "all_exhausted": lambda a, r: np.all((a == 4) & (r == 1), axis=1).astype(float),
        "expected_loss_scaled": lambda a, r: L_units(a, r) / Lmax_units,
    }, {"unit": unit, "Lmax_units": Lmax_units, "T_units": T}


# ---------------------------------------------------------------------------
# 3. logical resource counts for A (formulas; assumptions as in resources.py)
# ---------------------------------------------------------------------------
def a_operator_resources(model: SyntheticModel, target="exceedance"):
    n, E = model.n, len(model.edges)
    wE, wR = register_widths(model)
    wL = math.ceil(math.log2(n * 32 + 1))
    q_src = 4 * n + 7 * E                      # 3+1 per country, 6+1 per edge
    q_country = sum(wE) + sum(wR)
    q_oracle = 4 * n + 2 * wL + 2              # capped a,r copies, L register + workspace, flag
    qubits = q_src + q_country + q_oracle + 1  # + objective qubit
    # state preparation: Möttönen on 3-qubit (7 angles), 6-qubit (63 angles), 1-qubit (1 angle)
    rot = n * (7 + 1) + E * (63 + 1)
    cnot_prep = n * 6 + E * 62
    # adders (Gidney: w Toffolis for a w-bit in-place add; rain: increment ~ w Toffolis)
    tof_add = sum(wE[i] for i in range(n)) + sum(wR[i] for i in range(n))       # solo sources
    for (i, j) in model.edges:
        tof_add += wE[i] + wE[j] + wR[i] + wR[j]
    # oracle: capping comparators, L accumulation (2 adds/country), threshold compare / AND tree
    tof_orc = sum(wE) + sum(wR) + n * 3 + 2 * n * wL + wL + (n - 1)
    tof = 2 * (tof_add + tof_orc)              # compute + uncompute of arithmetic
    # depth: sources in parallel rounds by proper edge colouring (<= maxdeg+1 rounds, Vizing)
    maxdeg = int(degrees(model).max()) if n > 1 else 0
    depth_tof = (maxdeg + 2) * max(wE) + n * wL // max(1, int(math.log2(max(n, 2)))) + wL
    return {"n": n, "edges": E, "logical_qubits": qubits, "rotations": rot, "cnot_prep": cnot_prep,
            "toffoli": tof, "toffoli_depth_est": depth_tof}


IQAE_C = None  # fitted constant: A-calls ≈ C / (eps_rel * sqrt(a)), set by calibrate_iqae()


def calibrate_iqae(seed=0, reps=30):
    """Fit A-calls ≈ C/(ε_rel·√a) on the exact IQAE emulator over a ∈ [1e-6, 0.3]."""
    from .iqae import iqae
    global IQAE_C
    rng = np.random.default_rng(seed)
    pts = []
    for a in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3]:
        for er in [0.01, 0.1]:
            calls = np.mean([iqae(a, er * a, rng=rng)[1] for _ in range(reps)])
            pts.append((a, er, calls, calls * er * math.sqrt(a)))
    IQAE_C = float(np.median([p[3] for p in pts]))
    return IQAE_C, pts


def qae_total(model, mu, eps_rel=0.1, target="exceedance", delta_total=1e-3):
    from .resources import rotation_T, surface_code_footprint
    A = a_operator_resources(model, target)
    calls = IQAE_C / (eps_rel * math.sqrt(mu))
    tof = calls * (A["toffoli"] + A["logical_qubits"])
    rot = calls * A["rotations"]
    T = 4 * tof + rot * rotation_T(delta_total / max(rot, 1))
    sc = surface_code_footprint(A["logical_qubits"], T)
    return {**A, "mu": mu, "eps_rel": eps_rel, "a_calls": calls, "T_count": T, **{"sc_" + k: v for k, v in sc.items() if k != "assumptions"}}


def mc_total(model, mu, eps_rel=0.1):
    """Plain MC: samples for relative error eps (Bernoulli target) and ~ops per sample
    (one Poisson/categorical draw per source + O(n) payoff)."""
    N = (1 - mu) / (mu * eps_rel ** 2)
    ops = 2 * (2 * model.n + 2 * len(model.edges)) + 4 * model.n
    return {"samples": N, "ops_per_sample": ops, "total_ops": N * ops}
