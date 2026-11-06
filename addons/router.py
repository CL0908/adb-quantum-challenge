"""C2 (additive): find a cheap *exact reference* for (portfolio, target) among engines the friend's repo already has,
to attach NEXT TO RDPO's estimate (as the friend's own api/estimate.py already does for pair targets and n <= 5).
RDPO stays the main estimator; nothing on the main pipeline is replaced. Friend's src/ is used read-only.

Rules (first match wins; 'rdpo' means: no cheap exact reference, RDPO's estimate stands alone):
  1. pair target (joint_attachment / joint_exhaustion)  -> PortfolioProblem.exact_pair   any n, O(K·100·25)
  2. n <= 6                                             -> PortfolioProblem.exact_dense  10^n states
  3. MPS bond dimension <= CHI_MAX                      -> tensor.mps build_mps + targets_from_mps
  4. otherwise                                          -> rdpo.train.rdpo_estimate (sampling, has error)
"""
import time

import numpy as np

from _src import load, save
from src.rdpo.problem import PortfolioProblem
from src.tensor.mps import build_mps, targets_from_mps
from src.tensor.synthetic import SyntheticModel, topology

CHI_MAX = 512
MPS_KEY = {"exceedance": "exceedance_prob", "stop_loss": "stop_loss", "expected_loss": "expected_loss"}


def route(model, target, chi_max=CHI_MAX):
    """Returns (engine, value or None, explanation). Value is exact when engine != 'rdpo'."""
    P = PortfolioProblem(model, target)
    if target in ("joint_attachment", "joint_exhaustion"):
        return "exact_pair", P.exact_pair(), "pair target: only sources touching the pair matter (100-state law)"
    if model.n <= 6:
        return "exact_dense", P.exact_dense(), f"n={model.n} <= 6: dense enumeration of 10^{model.n} states"
    if target in MPS_KEY:
        psi, chi = build_mps(model)
        if chi <= chi_max:
            v = targets_from_mps(model, psi, P.T * 2.5e6)[MPS_KEY[target]]   # USD for stop_loss, as P's payoff
            return "mps", v, f"MPS bond dimension {chi} <= {chi_max}: exact contraction"
        return "rdpo", None, f"MPS bond dimension {chi} > {chi_max}: no cheap exact engine, sample with RDPO"
    return "rdpo", None, "no exact engine for this target"


def run():
    rows = load("rdpo_e2.json")["rows"]
    out = []
    for r in rows:
        model = SyntheticModel(r["n"], topology(r["topology"], r["n"]))
        t0 = time.perf_counter()
        eng, v, why = route(model, r["target"])
        dt = time.perf_counter() - t0
        rel_vs_ref = None if v is None else abs(v / r["mu"] - 1)
        out.append({"topology": r["topology"], "n": r["n"], "target": r["target"], "mu_ref": r["mu"],
                    "engine": eng, "value": v, "seconds": dt, "why": why, "rel_diff_vs_ref": rel_vs_ref,
                    "rdpo_rel_rmse": r["rdpo"]["rel_rmse"], "rdpo_seconds_per_fit": r["seconds_per_fit"],
                    "speedup_vs_rdpo": r["seconds_per_fit"] / dt})
        print(f"{r['topology']:8s} n={r['n']:2d} {r['target']:16s} {eng:11s} {dt:8.3f}s  "
              f"ref diff {rel_vs_ref if rel_vs_ref is None else f'{rel_vs_ref:.1e}'}  "
              f"RDPO {r['rdpo']['rel_rmse']:.2%} in {r['seconds_per_fit']:.1f}s", flush=True)
    save("c2_router.json", {"source": "adb-repro/results/rdpo_e2.json", "chi_max": CHI_MAX, "rows": out})
    return out


def selftest():
    m = SyntheticModel(4, topology("ring", 4))
    for tgt in ("exceedance", "joint_exhaustion", "stop_loss"):
        eng, v, _ = route(m, tgt)
        assert eng in ("exact_pair", "exact_dense") and v > 0
    # MPS branch must agree with dense enumeration where both apply (n = 6)
    m6 = SyntheticModel(6, topology("ring", 6))
    dense = PortfolioProblem(m6, "exceedance").exact_dense()
    psi, _ = build_mps(m6)
    mps = targets_from_mps(m6, psi, PortfolioProblem(m6, "exceedance").T * 2.5e6)["exceedance_prob"]
    assert abs(mps / dense - 1) < 1e-8, (mps, dense)
    print("selftest ok: routing branches, MPS == dense at ring-6 (rel diff %.1e)" % abs(mps / dense - 1))


if __name__ == "__main__":
    import sys
    selftest()
    if "--selftest" not in sys.argv:
        run()
