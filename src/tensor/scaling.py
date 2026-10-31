"""Phase 4 driver: bond-dimension and runtime scaling of the exact MPS method.

Exp A  correctness: MPS targets vs dense 10^n enumeration (n <= 6), and n=2 vs ADB Table 5.
Exp B  scaling in n for chain / ring / star / complete topologies at several tolerances.
Exp C  scaling with dependence strength m (common-shock multiplier).
Threshold for n countries: T_n = USD 40m * n (the ADB T = 80m for two countries).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from .synthetic import SyntheticModel, topology, dense_term_distribution, D
from .mps import build_mps, targets_from_mps, MPS


def dense_targets(model, P, threshold):
    psi = MPS([P.reshape(1, -1, 1)]) if model.n == 1 else None
    # direct evaluation on the dense tensor
    lu = model.site_loss()
    n = model.n
    grids = np.meshgrid(*[lu] * n, indexing="ij")
    L = sum(grids)
    pay = sum((g > 0).astype(int) for g in grids)
    full = lu.max()
    allx = np.all([np.isclose(g, full) for g in grids], axis=0)
    return {"expected_loss": float((P * L).sum()), "prob_at_least_two_pay": float(P[pay >= 2].sum()),
            "exceedance_prob": float(P[L > threshold].sum()),
            "stop_loss": float((P * np.maximum(L - threshold, 0)).sum()),
            "all_exhausted_prob": float(P[allx].sum())}


def exp_a():
    out = []
    for kind in ["chain", "ring", "complete"]:
        for n in range(2, 7 if kind != "complete" else 6):
            mdl = SyntheticModel(n, topology(kind, n))
            P = dense_term_distribution(mdl)
            T = 40e6 * n
            d = dense_targets(mdl, P, T)
            for tilt in [False, True]:
                for tol in [1e-12, 1e-9, 1e-6]:
                    psi, mb = build_mps(mdl, tol, tilt)
                    m = targets_from_mps(mdl, psi, T)
                    rel = {k: abs(m[k] - d[k]) / abs(d[k]) for k in d}
                    out.append({"topology": kind, "n": n, "tol": tol, "tilted_gauge": tilt, "max_bond": mb,
                                "max_rel_err": max(rel.values()), "rel_err": rel, "dense": d})
                    print(f"A {kind:8s} n={n} tilt={tilt:d} tol={tol:.0e} chi={mb:4d} "
                          f"maxrelerr={max(rel.values()):.2e}", file=sys.stderr, flush=True)
    return out


def exp_b(n_max={"chain": 30, "ring": 30, "star": 14, "complete": 7}):
    out = []
    for kind, nm in n_max.items():
        for tol in [1e-6, 1e-9]:
            for n in list(range(2, 11)) + list(range(12, nm + 1, 2 if nm <= 16 else 4)):
                if n > nm:
                    break
                try:
                    mdl = SyntheticModel(n, topology(kind, n))
                except AssertionError:
                    out.append({"topology": kind, "n": n, "tol": tol, "infeasible": True}); break
                t0 = time.perf_counter()
                psi, mb = build_mps(mdl, tol)
                tg = targets_from_mps(mdl, psi, 40e6 * n)
                dt = time.perf_counter() - t0
                out.append({"topology": kind, "n": n, "tol": tol, "max_bond": mb,
                            "final_max_bond": max(psi.bond_dims), "runtime_s": dt,
                            "n_edges": len(mdl.edges), "targets": tg})
                print(f"B {kind:8s} tol={tol:.0e} n={n:2d} chi={mb:4d} t={dt:7.2f}s "
                      f"Pall={tg['all_exhausted_prob']:.3e} mass={tg['total_mass']:.12f}", file=sys.stderr, flush=True)
                if dt > 60:
                    break
    return out


def exp_c(sizes={"ring": 8, "complete": 6}):
    out = []
    for kind, n in sizes.items():
        for m in [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]:
            try:
                mdl = SyntheticModel(n, topology(kind, n), m=m)
            except AssertionError:
                out.append({"topology": kind, "m": m, "infeasible": True}); continue
            psi, mb = build_mps(mdl, 1e-9)
            tg = targets_from_mps(mdl, psi, 40e6 * n)
            out.append({"topology": kind, "n": n, "m": m, "tol": 1e-9, "max_bond": mb, "targets": tg})
            print(f"C {kind:8s} n={n} m={m} chi={mb}", file=sys.stderr, flush=True)
    return out


def run(out=Path("results/tensor.json")):
    res = {"threshold_rule": "T_n = 40m * n"}
    for key, fn in [("A_correctness", exp_a), ("B_scaling_n", exp_b), ("C_scaling_m", exp_c)]:
        res[key] = fn()
        out.write_text(json.dumps(res, indent=2))  # save after each experiment
    return res


if __name__ == "__main__":
    run()
