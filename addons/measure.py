"""Q3: would a better *loaded measure* or a variance-based quantum mean estimator cut the quantum cost?

Today (friend's twisted.py): A loads the exponentially twisted source law q_η and encodes g = 1{L>T}·e^{−η(U−T)} ∈ [0,1];
bounded-amplitude AE needs ≈ c/(ε√a) calls, a = E_q[g].
Candidates, all compared on calls × (cost per A):
  (b0) same circuit, variance-based quantum mean estimation (Montanaro 2015; Kothari–O'Donnell 2023):
       calls ≈ c_var·σ_g/(ε·a) = c_var·√(ρ²_tw)/ε with ρ²_tw = E[g²]/a² − 1 (= classical twisted-IS relative variance).
       Gain over bounded AE with c_var = c (optimistic): √(E[g²]/a). No extra loading cost.
  (b1) load RDPO's learned policy instead and use variance-based QME (its weights are unbounded: α^−K bound):
       calls ∝ √ρ²_RDPO. Loading cost: the policy is state-dependent, so every source's Möttönen angles must be
       computed in the circuit. LOWER bound used here: (2^w − 1) arcsin√ evaluations per source, each at the
       friend's own price POLY_DEG·B_W² + POLY_PIECES·B_W Toffolis (resources.py), compute + uncompute.
Moments of g are estimated by sampling the twisted law (N paths per setting); a is checked against twisted.json.
"""
import math

import numpy as np

from _src import load, save
from src.quantum import twisted as TW
from src.quantum.resources import POLY_DEG, POLY_PIECES, B_W
from src.rdpo.problem import PortfolioProblem
from src.tensor.synthetic import SyntheticModel, topology

N_PATHS, CHUNK, SEED = 2_000_000, 200_000, 20261106
ARCSIN_TOF = POLY_DEG * B_W ** 2 + POLY_PIECES * B_W


def g_moments(model, T_units, rng, n_paths=N_PATHS):
    eta, _ = TW.best_twist(model, T_units)
    srcs = TW.source_losses(model)
    qs = []
    for s, ell in srcs:
        q = s.law * np.exp(eta * ell)
        qs.append(np.cumsum(q / q.sum()))
    P = PortfolioProblem(model, "exceedance")
    s1 = s2 = 0.0
    for start in range(0, n_paths, CHUNK):
        m = min(CHUNK, n_paths - start)
        S = np.zeros((m, model.n), np.int64); R = np.zeros((m, model.n), np.int64); U = np.zeros(m)
        for (s, ell), c in zip(srcs, qs):
            v = np.minimum(np.searchsorted(c, rng.random(m), side="right"), len(c) - 1)
            U += ell[v]
            i = s.sites[0]
            if s.kind == "eq1":
                S[:, i] += v
            elif s.kind == "eq2":
                S[:, i] += v // 5; S[:, s.sites[1]] += v % 5
            elif s.kind == "rain1":
                R[:, i] += v
            else:
                R[:, i] += v; R[:, s.sites[1]] += v
        L = (P.a_u * np.minimum(S, 4) + P.r_u * np.minimum(R, 1)).sum(1)
        g = (L > T_units) * np.exp(-eta * (U - T_units))
        s1 += g.sum(); s2 += (g * g).sum()
    a, m2 = s1 / n_paths, s2 / n_paths
    return a, m2


def loading_lower_bound(model):
    tof = 0
    for s, _ in TW.source_losses(model):
        w = math.ceil(math.log2(np.count_nonzero(s.law)))
        tof += 2 * (2 ** w - 1) * ARCSIN_TOF
    return tof


def run():
    adv = {(r["topology"], r["n"]): r for r in load("quantum_advantage.json")["advantage"]}
    import json
    from _src import HERE
    p = HERE.parent / "repro" / "i25" / "summary.json"          # our I25 supplementary run, if present
    i25 = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    rng = np.random.default_rng(SEED)
    out = []
    for (kind, n), r in sorted(adv.items(), key=lambda kv: (kv[0][0] != "ring", kv[0][1])):
        model = SyntheticModel(n, topology(kind, n))
        a, m2 = g_moments(model, 16 * n, rng)
        rho2_tw = m2 / a ** 2 - 1
        gain_b0 = math.sqrt(m2 / a)                                    # calls(b0)/calls(bounded), c_var = c
        rho2_rdpo = r["classical_rho2"]
        gain_b1_calls = math.sqrt(rho2_rdpo * r["a_twisted"])          # calls(b1)/calls(bounded), optimistic
        tofA_now = r["toffoli_A"] + r["logical_qubits"]
        tofA_b1 = tofA_now + loading_lower_bound(model)
        net_b1 = gain_b1_calls * tofA_b1 / tofA_now                    # ≥ this factor on Toffoli work
        row = {"topology": kind, "n": n, "a_twisted_declared": r["a_twisted"], "a_sampled": a,
               "a_rel_diff": a / r["a_twisted"] - 1, "E_g2_over_a": m2 / a, "rho2_twisted": rho2_tw,
               "inv_a_minus_1": 1 / r["a_twisted"] - 1, "rho2_rdpo": rho2_rdpo,
               "b0_calls_ratio": gain_b0, "b1_calls_ratio": gain_b1_calls,
               "toffoli_A_now": tofA_now, "toffoli_A_rdpo_loading_lb": tofA_b1, "b1_net_work_ratio_lb": net_b1}
        if i25 and kind == "ring" and n == 28:
            for c in i25["cells"]:
                if c["n"] == 28:
                    row[f"rho2_i25_{c['config']}"] = c["quantum"]["point"]["rho2"]
            row["b1_calls_ratio_i25_e2"] = math.sqrt(row.get("rho2_i25_v1", rho2_rdpo) * r["a_twisted"])
        out.append(row)
        print(f"{kind:8s} n={n:2d} a={a:.3g} (decl {r['a_twisted']:.3g}, {row['a_rel_diff']:+.1%})  "
              f"E[g²]/a={m2 / a:.3f}  ρ²_tw={rho2_tw:.3g} vs 1/a−1={1 / r['a_twisted'] - 1:.3g}  ρ²_RDPO={rho2_rdpo:.3g}  "
              f"b0 calls ×{gain_b0:.2f}  b1 calls ×{gain_b1_calls:.3f}, Tof/A ×{tofA_b1 / tofA_now:.0f} → net ≥×{net_b1:.1f}",
              flush=True)
    save("q3_measure.json", {"n_paths": N_PATHS, "seed": SEED, "arcsin_toffoli": ARCSIN_TOF, "rows": out})
    return out


def selftest():
    m = SyntheticModel(2, topology("ring", 2))
    exact = TW.simulate_twisted_A(m, 16 * 2 * 2.5e6)["a"]
    a, m2 = g_moments(m, 32, np.random.default_rng(1), 400_000)
    assert abs(a / exact - 1) < 0.02 and m2 <= a + 1e-12, (a, exact, m2)
    print(f"selftest ok: sampled a={a:.4g} vs exact simulate_twisted_A {exact:.4g}; E[g²] <= a")


if __name__ == "__main__":
    import sys
    selftest()
    if "--selftest" not in sys.argv:
        run()
