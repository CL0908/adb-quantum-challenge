"""Phase 6a driver: structured (Poisson-source) loading -> exactness check + scaling comparison
of QAE vs plain MC vs exact tilted MPS as the portfolio grows."""
import json
import sys
from pathlib import Path

import numpy as np

from src.tensor.synthetic import SyntheticModel, topology, dense_term_distribution
from src.baselines.common import make_problem
from src.model.scenarios import TARGETS
from . import structured_loading as SL

OUT = Path("results/structured.json")


def verify():
    out = []
    ex = make_problem("adb-base").exact
    for kind, n in [("chain", 2), ("chain", 3)]:
        m = SyntheticModel(n, topology(kind, n))
        lib, info = SL.payoff_library(m, 40e6 * n)
        P = dense_term_distribution(m)
        row = {"topology": kind, "n": n}
        for k, f in lib.items():
            r = SL.simulate_A(m, f)
            row[k] = r["a"]
            row["support"] = r["support"]; row["norm_err"] = abs(r["norm"] - 1)
            row["max_abs_err_vs_dense"] = float(np.abs(r["joint"] - P).max())
        if n == 2:  # direct comparison with ADB Table-5 exact values
            row["adb_exact"] = {"exceedance": ex[TARGETS[2]], "at_least_two_pay": ex[TARGETS[1]],
                                "all_exhausted": ex[TARGETS[4]],
                                "stop_loss_scaled": ex[TARGETS[3]] / (info["Lmax_units"] - info["T_units"]) / info["unit"],
                                "expected_loss_scaled": ex[TARGETS[0]] / info["Lmax_units"] / info["unit"]}
        row["resources"] = SL.a_operator_resources(m)
        out.append(row)
        print("verify", kind, n, {k: v for k, v in row.items() if k not in ("resources", "adb_exact")}, file=sys.stderr)
    return out


def scaling():
    t = json.loads(Path("results/tensor.json").read_text())
    mps = {(r["topology"], r["n"]): r for r in t["B_scaling_n"] if r.get("tol") == 1e-9 and "max_bond" in r}
    mps6 = {(r["topology"], r["n"]): r for r in t["B_scaling_n"] if r.get("tol") == 1e-6 and "max_bond" in r}
    rows = []
    for kind in ["chain", "ring", "complete"]:
        for n in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 28, 50]:
            try:
                m = SyntheticModel(n, topology(kind, n))
            except AssertionError:
                rows.append({"topology": kind, "n": n, "infeasible": "shared rates exceed totals"}); continue
            A = SL.a_operator_resources(m)
            ref = mps.get((kind, n)) or mps6.get((kind, n))
            row = {"topology": kind, "n": n, "edges": len(m.edges), "A": A}
            if ref:
                mu = ref["targets"]["exceedance_prob"]
                row.update({"mu_exceedance": mu, "mps_chi": ref["max_bond"], "mps_tol": ref["tol"],
                            "mps_runtime_s": ref["runtime_s"]})
            for label, mu in ([("exceedance", row["mu_exceedance"])] if ref else []) + [("mu=1e-4", 1e-4), ("mu=1e-8", 1e-8)]:
                q = SL.qae_total(m, mu, eps_rel=0.1)
                c = SL.mc_total(m, mu, eps_rel=0.1)
                row[label] = {"qae_calls": q["a_calls"], "qae_T": q["T_count"], "qae_runtime_s": q["sc_runtime_s"],
                              "qae_phys_qubits": q["sc_physical_qubits"], "mc_samples": c["samples"], "mc_ops": c["total_ops"]}
            rows.append(row)
            print("scale", kind, n, "q", A["logical_qubits"], "tof", A["toffoli"], "chi", row.get("mps_chi"), file=sys.stderr)
    return rows


def run():
    C, pts = SL.calibrate_iqae()
    print("IQAE C =", C, file=sys.stderr)
    res = {"iqae_C": C, "iqae_calibration": pts, "verify": verify(), "scaling": scaling(),
           "assumptions": "eps_rel = 10% (target half-width), IQAE alpha=0.05; T-count & surface code as resources.py"}
    OUT.write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    run()
