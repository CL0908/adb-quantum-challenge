"""Q1: cheaper rotation synthesis for the Q-RDPO A operator (friend's src/ untouched).

The friend's advantage_map() prices every arbitrary rotation at 3·log2(1/δ) T (Ross–Selinger) with
δ = 1e-3 / (#rotations in the whole run). At ring-28 this is ~92% of the T-count. We re-price the same
circuits (same Toffoli and rotation counts, same surface-code model) under:
  rs        3·log2(1/δ)                    friend's model (reproduces quantum_T_count exactly)
  fallback  1.03·log2(1/δ) + 5.75          Kliuchnikov, Lauter, Minko, Paetznick, Petit, Quantum 7, 1208 (2023),
                                           Table 1, Clifford+T, Fallback [BRS15a], mean; a unitary, no randomisation
  mixed     0.53·log2(1/δ) + 4.86          same Table 1, Clifford+T, Mixed fallback, mean (max 0.57·log2 + 8.83).
                                           A random mixture of circuits: every gate instance in every shot must be
                                           re-sampled independently, otherwise the error is coherent and the 2x is lost
  qrom_<m>  <m>, but each Möttönen level with 2^l angles may instead load its angles by QROM
            (unary iteration, 2^l − 1 Toffolis, ×2 for unload) and apply one b-bit phase-gradient addition
            (b Toffolis, Gidney 2018), b = ceil(log2(2π/δ)); per level the cheaper option is taken.
All ε in that table are diamond distances (Ross–Selinger row: 3.02·log2 + 1.77), so the friend's 3·log2 and these
rows are on the same metric, and diamond-norm errors add sub-additively over any composition (incl. Q^k, A†):
the friend's union budget Σδ = 1e-3 stays valid. (0.56·log2 + 5.3 in the paper's abstract is Clifford+√T.)
Runtime is the friend's factory-throughput model; the reaction-time floor (T-depth × 10 µs) is not modelled.
Rotation inventory per A (from structured_loading.a_operator_resources + bitwise oracle):
  per country 7 (3-qubit EQ law: levels 1,2,4) + 1 (rain), per edge 63 (6-qubit law: 1..32) + 1, + w_U oracle.
"""
import math

from _src import load, save
from src.quantum.resources import surface_code_footprint

LEVELS_EQ, LEVELS_EDGE = [1, 2, 4], [1, 2, 4, 8, 16, 32]
MODELS = ["rs", "fallback", "qrom_fallback", "mixed", "qrom_mixed"]   # conservative -> optimistic


def t_rot(model, delta):
    lg = math.log2(1 / delta)
    return {"rs": 3 * lg, "fallback": 1.03 * lg + 5.75, "mixed": 0.53 * lg + 4.86}[model]


def inventory(row):
    n = row["n"]
    E = (row["rotations_A"] - 8 * n - row["w_U"]) / 64
    assert E == int(E), row
    E = int(E)
    # (count, angles-per-level list) groups; singletons (rain, oracle) are plain rotations
    return n, E, [(n, LEVELS_EQ), (E, LEVELS_EDGE)], n + E + row["w_U"]


def price(row, model, calls=None, inv=None):
    """calls overrides the friend's IQAE call count (used to combine with a better AE algorithm);
    inv = (n, E, groups, singles) overrides the rotation inventory (used for other loadings, q_loading.py)."""
    calls = row["quantum_calls"] if calls is None else calls
    qubits = row["logical_qubits"]
    tof_A = row["toffoli_A"] + qubits                    # friend: + reflection
    n, E, groups, singles = inventory(row) if inv is None else inv
    # first pass: rotation count for the δ budget (synthesised + phase-gradient both consume budget)
    extra_q = 0
    if model.startswith("qrom_"):
        base = model[5:]
        delta = 1e-3 / max(calls * row["rotations_A"], 1)   # start from the friend's budget, then re-split
        for _ in range(3):
            b = math.ceil(math.log2(2 * math.pi / delta))
            syn_A, pg_A, qrom_tof_A = singles, 0, 0
            for cnt, levels in groups:
                for L in levels:
                    if 4 * (2 * (L - 1) + b) < L * t_rot(base, delta):
                        qrom_tof_A += cnt * (2 * (L - 1) + b); pg_A += cnt   # one phase-gradient add per level
                    else:
                        syn_A += cnt * L
            delta = 1e-3 / max(calls * (syn_A + pg_A), 1)   # both kinds of error share the 1e-3 budget
        extra_q = 2 * b                                    # angle register + phase-gradient register
        # + b synthesised rotations once, to prepare the phase-gradient state
        T = 4 * calls * (tof_A + qrom_tof_A) + (calls * syn_A + b) * t_rot(base, delta)
    else:
        rot_total = calls * row["rotations_A"]
        delta = 1e-3 / max(rot_total, 1)
        T = 4 * calls * tof_A + rot_total * t_rot(model, delta)
    sc = surface_code_footprint(qubits + extra_q, T)
    return {"T_count": T, "runtime_s": sc["runtime_s"], "code_distance": sc["code_distance"],
            "physical_qubits": sc["physical_qubits"], "s_star": sc["runtime_s"] / row["classical_seconds"],
            "rotation_share": None if model.startswith("qrom_") else 1 - 4 * calls * tof_A / T}


def run():
    rows = load("quantum_advantage.json")["advantage"]
    out = []
    for r in rows:
        p = {m: price(r, m) for m in MODELS}
        out.append({"topology": r["topology"], "n": r["n"], "calls": r["quantum_calls"],
                    "rotations_A": r["rotations_A"], "toffoli_A": r["toffoli_A"],
                    "declared_T": r["quantum_T_count"], "declared_s_star": r["required_speedup"], **p,
                    **{f"T_reduction_{m}": p["rs"]["T_count"] / p[m]["T_count"] for m in MODELS[1:]},
                    **{f"s_star_reduction_{m}": p["rs"]["s_star"] / p[m]["s_star"] for m in MODELS[1:]}})
        print(f"{r['topology']:8s} n={r['n']:2d} rot share {p['rs']['rotation_share']:.0%}  "
              + "  ".join(f"{m} T/{out[-1][f'T_reduction_{m}']:.2f} s*/{out[-1][f's_star_reduction_{m}']:.2f}"
                          for m in MODELS[1:]))
    save("q1_rotations.json", {"source": "adb-repro/results/quantum_advantage.json", "rows": out})
    return out


def selftest():
    rows = load("quantum_advantage.json")["advantage"]
    for r in rows:
        p = price(r, "rs")
        assert abs(p["T_count"] / r["quantum_T_count"] - 1) < 1e-9, (r["n"], p["T_count"], r["quantum_T_count"])
        assert abs(p["s_star"] / r["required_speedup"] - 1) < 1e-9
        T = {m: price(r, m)["T_count"] for m in MODELS}
        assert T["qrom_mixed"] <= T["mixed"] * 1.0001 < T["fallback"] < T["rs"] and T["qrom_fallback"] <= T["fallback"] * 1.0001
    print("selftest ok: rs model reproduces all", len(rows), "declared T-counts and s*")


if __name__ == "__main__":
    import sys
    selftest()
    if "--selftest" not in sys.argv:
        run()
