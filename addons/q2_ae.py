"""Q2: A-operator calls of better amplitude-estimation algorithms at the friend's own operating points.

For each advantage_map setting (a = a_twisted, ε_rel = 10%, α = 0.05) we emulate, with the same counting as the
friend's iqae (one shot at Grover power k = 2k+1 calls of A): the friend's IQAE (shots=100), ChebAE restricted to
odd degrees (Grover-power circuits only, so the friend's circuit family and resource model still apply), and
MIQAE with Chernoff intervals (the only variant with a proof). Coverage P(|â − a| <= ε) is reported; pL_required
is recomputed from kmax with the friend's formula, since deeper circuits need better logical qubits.
"""
import math

import numpy as np

import ae
from _src import load, save
from src.quantum.iqae import iqae

REPS, SEED = 200, 7
ALGOS = {"iqae": lambda a, e, g: iqae(a, e, 0.05, 100, g),
         "chebae_odd": lambda a, e, g: ae.chebae(a, e, 0.05, 100, g, odd_only=True),
         "miqae_chernoff": lambda a, e, g: ae.miqae(a, e, 0.05, 1, g, "chernoff")}


def run(reps=REPS):
    rows = load("quantum_advantage.json")["advantage"]
    out = []
    for r in rows:
        a, er = r["a_twisted"], r["eps_rel"]
        rec = {"topology": r["topology"], "n": r["n"], "a": a, "eps_rel": er, "declared_calls": r["quantum_calls"]}
        for name, f in ALGOS.items():
            g = np.random.default_rng(SEED)
            res = np.array([f(a, er * a, g) for _ in range(reps)], float)
            kmax = res[:, 2].max()
            rec[name] = {"calls_mean": float(res[:, 1].mean()), "c": float(res[:, 1].mean() * er * math.sqrt(a)),
                         "coverage": float(np.mean(abs(res[:, 0] - a) <= er * a)), "kmax_max": float(kmax),
                         "kmax_mean": float(res[:, 2].mean()), "k0_runs": int((res[:, 2] == 0).sum()),
                         "pL_required": math.log(2) / ((2 * kmax + 1) * r["cnot_Q"] / 2 + r["cnot_A"])}
        rec["calls_ratio_chebae_vs_iqae"] = rec["iqae"]["calls_mean"] / rec["chebae_odd"]["calls_mean"]
        out.append(rec)
        print(f"{r['topology']:8s} n={r['n']:2d} a={a:.3g}  " + "  ".join(
            f"{k}: c={v['c']:.1f} cover={v['coverage']:.3f} kmax={v['kmax_max']:.0f} k0={v['k0_runs']}"
            for k, v in rec.items() if isinstance(v, dict)) + f"  -> /{rec['calls_ratio_chebae_vs_iqae']:.2f}", flush=True)
    save("q2_ae.json", {"reps": reps, "seed": SEED, "alpha": 0.05, "rows": out})
    return out


if __name__ == "__main__":
    run()
