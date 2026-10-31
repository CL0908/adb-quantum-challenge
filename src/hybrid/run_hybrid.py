"""Phase 6b/6c driver.

6b  Hybrid quantum RL: VQC soft-value critic vs classical log-linear critics, CE policy search.
6c  Information-theoretic RL: Rényi-2 (χ²) policy-gradient refinement after CE; robustness over seeds.
6d  Cut-entropy diagnostics: classical mutual information and Born-state entanglement entropy
    across the middle cut vs the MPS bond dimension (what a quantum/TN representation must carry).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.baselines.common import make_problem
from src.model.scenarios import SCENARIOS
from src.learned.cross_entropy import fit
from src.learned.renyi import renyi2_refine
from src.hybrid.vqc_policy import fit_vqc

OUT = Path("results/hybrid.json")
SEEDS = [11, 22, 33]
N_EVAL, R_EVAL = 4096, 50
CASES = [(s, "term_both_bonds_exhausted_probability") for s in SCENARIOS] + \
        [("adb-base", "term_portfolio_loss_exceedance_probability"),
         ("adb-base", "term_portfolio_stop_loss_usd")]


def evaluate(prop, prob, t, seed):
    rng = np.random.default_rng(seed)
    x = np.array([prop.estimate(prob.f[t], N_EVAL, rng) for _ in range(R_EVAL)])
    mu = prob.exact[t]
    return float(np.sqrt(np.mean((x - mu) ** 2)) / mu)


def policies():
    return {
        "classical_factorized": lambda prob, t, sc, s: fit(prob, t, sc, kind="factorized", seed=s),
        "classical_pairwise": lambda prob, t, sc, s: fit(prob, t, sc, kind="pairwise", seed=s),
        "hybrid_vqc_L3": lambda prob, t, sc, s: fit_vqc(prob, t, sc, layers=3, seed=s),
    }


def exp_policies():
    rows = []
    for name, t in CASES:
        prob, sc = make_problem(name), SCENARIOS[name]
        for pol, fitter in policies().items():
            for s in SEEDS:
                t0 = time.perf_counter()
                prop, info = fitter(prob, t, sc, s)
                ce = evaluate(prop, prob, t, s + 1)
                prop, rinfo = renyi2_refine(prop, prob.f[t], seed=s + 2)
                rr = evaluate(prop, prob, t, s + 1)
                rows.append({"scenario": name, "target": t, "policy": pol, "seed": s, "n_params": info["n_params"],
                             "training_paths": info["training_paths"], "refine_paths": rinfo["refine_paths"],
                             "rel_rmse_ce": ce, "rel_rmse_ce_renyi2": rr, "chi2_trace": rinfo["chi2_trace"]})
                print(f"{name:17s} {t[5:30]:26s} {pol:21s} seed={s} CE={ce:.4f} +R2={rr:.4f} "
                      f"{time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True)
    return rows


def exp_entropy():
    from src.tensor.synthetic import SyntheticModel, topology, dense_term_distribution
    from src.tensor.mps import site_gauges
    t = json.loads(Path("results/tensor.json").read_text())
    chi = {(r["topology"], r["n"], r["tol"]): r["max_bond"] for r in t["A_correctness"] if r.get("tilted_gauge")}
    out = []
    for kind in ["chain", "ring", "complete"]:
        for n in [4, 5, 6] if kind != "complete" else [4, 5]:
            mdl = SyntheticModel(n, topology(kind, n))
            P = dense_term_distribution(mdl)
            left = n // 2
            M = P.reshape(10 ** left, -1)
            pa, pb = M.sum(1), M.sum(0)
            H = lambda q: float(-(q[q > 0] * np.log2(q[q > 0])).sum())
            mi = H(pa) + H(pb) - H(M.ravel())
            s = np.linalg.svd(np.sqrt(M), compute_uv=False); lam = s ** 2 / (s ** 2).sum()
            ent = H(lam)
            g = site_gauges(mdl)
            Pt = P.copy()
            for i, gi in enumerate(g):
                sh = [1] * n; sh[i] = -1; Pt = Pt * gi.reshape(sh)
            st = np.linalg.svd(Pt.reshape(10 ** left, -1), compute_uv=False); lt = st ** 2 / (st ** 2).sum()
            out.append({"topology": kind, "n": n, "cut": left, "mutual_info_bits": mi,
                        "born_entanglement_bits": ent, "tilted_spectrum_entropy_bits": H(lt),
                        "chi_tilted_tol1e-9": chi.get((kind, n, 1e-9)), "log2_chi": float(np.log2(chi.get((kind, n, 1e-9)) or 1))})
            print("entropy", out[-1], file=sys.stderr, flush=True)
    return out


def run():
    res = {"seeds": SEEDS, "n_eval": N_EVAL, "reps": R_EVAL}
    res["entropy"] = exp_entropy()
    OUT.write_text(json.dumps(res, indent=2))
    res["policies"] = exp_policies()
    OUT.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    run()
