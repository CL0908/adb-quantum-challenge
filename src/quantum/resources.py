"""Logical-level resource estimates for the IS -> IQAE pipeline on the ADB model.

All counts are *estimates from explicit component formulas*, not compiled circuits.
Assumptions (each is a parameter below so it can be challenged):
  * annual state register: 8 qubits (a:3, b:3, r1, r2) per year, 3 years -> 24 qubits;
  * cumulative state: 8 qubits, built by saturating adders (cap at 4 / 1);
  * log-weight accumulator: B_W-bit fixed point; payoff/weight -> amplitude via a
    piecewise-polynomial arcsin(sqrt(.)) evaluation (Häner et al. 2018 style);
  * QROM lookups via unary iteration: (L - 1) Toffolis for L entries (Babbush et al. 2018);
  * n-bit adder: n Toffolis (Gidney 2018, measurement-based uncompute);
  * arbitrary single-qubit rotation: T-count ≈ 3 log2(1/δ) (Ross & Selinger 2016);
  * exact state preparation of an 8-qubit real distribution: 2^8 - 1 uniformly-controlled
    RY angles -> 255 rotations + 254 CNOT (Möttönen et al. 2004) per year;
  * Toffoli -> 4 T (Jones 2013 with measurement-based uncompute);
  * Grover iterate Q = A S0 A^† Sχ costs 2 A + reflections (~ n_qubits Toffolis).
"""
import math

B_W = 16          # fixed-point bits for log-weight / amplitude argument
POLY_DEG = 3      # piecewise polynomial degree for arcsin(sqrt(exp(.))) evaluation
POLY_PIECES = 16


def rotation_T(delta):
    return 3 * math.log2(1 / delta)


def a_operator(loading="exact_iid", qcbm_layers=4, state_dependent=False, indicator_payoff=True):
    """Return dict of counts for one application of A."""
    n_year, T = 8, 3
    tof = 0
    rot = 0
    cnot = 0
    # 1) proposal loading
    if loading == "exact_iid":
        rot += T * (2 ** n_year - 1); cnot += T * (2 ** n_year - 2)
    elif loading == "qcbm":
        rot += T * n_year * (qcbm_layers + 1); cnot += T * n_year * qcbm_layers  # CZ ~ CNOT
    elif loading == "doob_state_dependent":
        # year 1 iid; years 2,3: rotations multiplexed on the 8-qubit cumulative state
        rot += (2 ** n_year - 1) + 2 * (2 ** 8) * (2 ** n_year - 1)
        cnot += (2 ** n_year - 2) + 2 * (2 ** 8) * (2 ** n_year)
    else:
        raise ValueError(loading)
    # 2) cumulative state: per year per country a 3-bit saturating add (~3 adder + 6 compare/set),
    #    rain OR (1 Toffoli). compute + uncompute.
    tof += 2 * T * (2 * (3 + 6) + 2)
    # 3) log-weight lookup: QROM over year register (or over (state, year) if state dependent)
    entries = 2 ** (n_year + (8 if loading == "doob_state_dependent" else 0))
    tof += 2 * T * ((entries - 1) + B_W)
    # 4) payoff: indicator -> 7-input AND (~7 Toffoli); general -> QROM over 256 final states
    tof += 2 * (7 if indicator_payoff else (2 ** 8 - 1) + B_W)
    # 5) amplitude function: piecewise polynomial, Horner: deg multiplications of B_W bits
    tof += 2 * (POLY_DEG * B_W ** 2 + POLY_PIECES * B_W)
    rot += B_W  # controlled rotations from the angle register onto the objective qubit
    qubits = 3 * n_year + 8 + B_W + B_W + 2 * B_W + 1  # path + state + logw + angle + workspace + obj
    if loading == "doob_state_dependent":
        qubits += 8  # extra intermediate cumulative-state copies
    return {"toffoli": tof, "rotations": rot, "cnot_loading": cnot, "logical_qubits": qubits}


def total_cost(a_calls, loading, delta_total=1e-3, **kw):
    """Total logical cost for a run of IQAE that makes `a_calls` applications of A."""
    A = a_operator(loading, **kw)
    refl = A["logical_qubits"]
    tof_total = a_calls * (A["toffoli"] + refl)
    rot_total = a_calls * A["rotations"]
    delta = delta_total / max(rot_total, 1)
    T_total = 4 * tof_total + rot_total * rotation_T(delta)
    return {
        "per_A": A,
        "a_calls": a_calls,
        "logical_qubits": A["logical_qubits"],
        "toffoli_total": tof_total,
        "rotations_total": rot_total,
        "T_count_total": T_total,
        "rotation_synthesis_precision": delta,
    }


def surface_code_footprint(logical_qubits, T_count, p_phys=1e-3, target_fail=1e-2,
                           cycle_time_s=1e-6, factories=4, factory_qubits=6000, t_per_factory_cycle=1 / 11):
    """Crude surface-code estimate (Fowler et al. 2012 style):
    p_L(d) ≈ 0.1 (100 p)^((d+1)/2) per logical qubit per round; spacetime volume ≈ Q · T-steps · d."""
    # T-states are the bottleneck: one per factory per ~11d cycles  ->  runtime in code cycles
    d = 3
    while True:
        cycles = T_count / (factories * t_per_factory_cycle) * d
        pL = 0.1 * (100 * p_phys) ** ((d + 1) / 2)
        if logical_qubits * cycles * pL < target_fail or d > 61:
            break
        d += 2
    phys = 2 * d ** 2 * logical_qubits + factories * factory_qubits
    return {"code_distance": d, "physical_qubits": int(phys), "runtime_s": cycles * cycle_time_s,
            "assumptions": {"p_phys": p_phys, "factories": factories, "cycle_time_s": cycle_time_s}}


def nisq_sampling_cost(n_paths, qcbm_layers=4, leak=0.0):
    """QCBM used purely as a proposal *sampler* (no QAE): one shot per year per path."""
    shots = math.ceil(3 * n_paths / max(1 - leak, 1e-12))
    return {"shots": shots, "two_qubit_gates_per_shot": 8 * qcbm_layers,
            "circuit_depth_per_shot": 2 * qcbm_layers + 1, "qubits": 8}
