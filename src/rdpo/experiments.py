"""Final RDPO experiments (all at a fixed TOTAL path budget, training included).

  E1  benchmark: 5 ADB scenarios x 5 targets x B in {4096, 16384, 65536}, vs the brief's baselines
  E2  scaling:   ring (2..28 countries) and complete (4..7) dependence; exceedance, stop-loss and the
                 joint exhaustion of countries (0, 1) inside the portfolio; vs MC and exponential twisting
  E3  ablation:  each RDPO component removed in turn
Usage:  python -m src.rdpo.experiments e1|e2|e3
"""
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.tensor.synthetic import SyntheticModel, topology
from .problem import PortfolioProblem, adb_model, ADB_SCENARIOS, TARGETS
from .train import rdpo_estimate, RDPOConfig

R = Path("results")


def summarise(est, mu, cis=None):
    e = np.asarray(est)
    out = {"rel_rmse": float(np.sqrt(np.mean((e - mu) ** 2)) / abs(mu)), "rel_bias": float((e.mean() - mu) / abs(mu)),
           "reps": len(e)}
    if cis is not None:
        out["ci_coverage"] = float(np.mean([lo <= mu <= hi for lo, hi in cis]))
    return out


# ------------------------------------------------------------------------------------------------
def e1(budgets=(4096, 16384, 65536), reps=40, seed=1000):
    base = json.loads((R / "baselines.json").read_text())["scenarios"]
    key = {"expected_loss": "term_expected_principal_loss_usd", "joint_attachment": "term_joint_attachment_probability",
           "exceedance": "term_portfolio_loss_exceedance_probability", "stop_loss": "term_portfolio_stop_loss_usd",
           "joint_exhaustion": "term_both_bonds_exhausted_probability"}
    prev = json.loads((R / "rdpo_e1.json").read_text())["rows"] if (R / "rdpo_e1.json").exists() else []
    done = {(r["scenario"], r["target"], r["budget"]) for r in prev}
    rows = list(prev)
    for si, sc in enumerate(ADB_SCENARIOS):
        for ti, t in enumerate(TARGETS):
            if all((sc, t, B) in done for B in budgets):
                continue
            P = PortfolioProblem(adb_model(sc), t, threshold=80e6)
            mu = P.exact_dense()
            for B in budgets:
                t0 = time.perf_counter()
                xs = [rdpo_estimate(P, B, np.random.default_rng(seed + 100000 * si + 1000 * ti + r)) for r in range(reps)]
                s = summarise([x["estimate"] for x in xs], mu, [x["ci"] for x in xs])
                bl = {m: base[sc]["methods"][m][key[t]][str(B)]["rel_rmse"] for m in ["mc", "sobol", "stratified", "importance"]}
                best = min(bl, key=bl.get)
                rows.append({"scenario": sc, "target": t, "budget": B, "mu": mu, "rdpo": s, "baselines": bl,
                             "best_baseline": best, "improvement_vs_best": bl[best] / s["rel_rmse"],
                             "n_params": xs[0]["n_params"], "mean_rel_se": float(np.mean([x["se"] for x in xs]) / mu),
                             "seconds_per_fit": (time.perf_counter() - t0) / reps})
                print(f"E1 {sc:17s} {t:17s} B={B:6d} RDPO={s['rel_rmse']:.4f} cov={s['ci_coverage']:.2f} "
                      f"best={best}:{bl[best]:.4f} x{bl[best] / s['rel_rmse']:.1f}", file=sys.stderr, flush=True)
            (R / "rdpo_e1.json").write_text(json.dumps({"reps": reps, "rows": rows}, indent=2))
    return rows


# ------------------------------------------------------------------------------------------------
def pair_exhaustion_exact(model, pair=(0, 1)):
    from src.tensor.mps import build_mps
    psi, _ = build_mps(model, tol=1e-12, tilt=True)
    v = np.ones(1)
    for s, t in enumerate(psi.T):
        t = t / psi.gauge[s][None, :, None]
        v = v @ (t[:, 9, :] if s in pair else t.sum(1))
    return float(v[0])


def twisted_est(P, B, rng):
    from src.scale.source_rdpo import SourceProblem, twisted_estimate
    from src.quantum.twisted import best_twist
    sp = SourceProblem(P.model, P.T * 2.5e6, "exceedance" if P.target == "exceedance" else "stop_loss")
    eta, _ = best_twist(P.model, P.T)
    return twisted_estimate(sp, eta, B, rng)


def e2(B=65536, seed=2000):
    tens = {(r["topology"], r["n"]): r for r in json.loads((R / "tensor.json").read_text())["B_scaling_n"]
            if "targets" in r and r["tol"] == (1e-6 if (r["topology"] == "complete" and r["n"] == 7) else 1e-9)}
    prev = json.loads((R / "rdpo_e2.json").read_text())["rows"] if (R / "rdpo_e2.json").exists() else []
    done = {(r["topology"], r["n"], r["target"]) for r in prev}
    rows = list(prev)
    cases = [("ring", n) for n in (2, 8, 16, 20, 28)] + [("complete", n) for n in (4, 6, 7)]
    for kind, n in cases:
        model = SyntheticModel(n, topology(kind, n))
        ref = tens[(kind, n)]["targets"]
        mus = {"exceedance": ref["exceedance_prob"], "stop_loss": ref["stop_loss"],
               "joint_exhaustion": pair_exhaustion_exact(model)}
        reps = 20 if n <= 4 else (10 if n <= 16 else 6)
        for t in ("exceedance", "stop_loss", "joint_exhaustion"):
            if (kind, n, t) in done:
                continue
            P = PortfolioProblem(model, t)
            mu = mus[t]
            t0 = time.perf_counter()
            xs = [rdpo_estimate(P, B, np.random.default_rng(seed + 1000 * n + 100 * len(rows) + r)) for r in range(reps)]
            s = summarise([x["estimate"] for x in xs], mu, [x["ci"] for x in xs])
            row = {"topology": kind, "n": n, "edges": len(model.edges), "target": t, "mu": mu, "budget": B,
                   "rdpo": s, "n_params": xs[0]["n_params"], "K_sources": P.K,
                   "seconds_per_fit": (time.perf_counter() - t0) / reps,
                   "mc_theory_rel_rmse": math.sqrt(max(1 - mu, 0) / (mu * B)) if t != "stop_loss" else None}
            if t in ("exceedance", "stop_loss"):
                rng = np.random.default_rng(seed + 7)
                row["twisted"] = summarise([twisted_est(P, B, rng) for _ in range(reps)], mu)
            rows.append(row)
            print(f"E2 {kind:8s} n={n:2d} {t:16s} mu={mu:.3e} RDPO={s['rel_rmse']:.4f} cov={s['ci_coverage']:.2f} "
                  f"twist={row.get('twisted', {}).get('rel_rmse', float('nan')):.4f} {row['seconds_per_fit']:.1f}s",
                  file=sys.stderr, flush=True)
            (R / "rdpo_e2.json").write_text(json.dumps({"rows": rows}, indent=2))
    return rows


# ------------------------------------------------------------------------------------------------
ABLATIONS = {
    "full RDPO": {},
    "no edge features": {"edge_features": False},
    "no target-relevance freezing": {"relevance": False},
    "no twist warm start": {"warm_start": False},
    "no cross-entropy stage": {"ce": False},
    "no Rényi-2 (χ²) stage": {"pg": False},
    "no sample recycling": {"recycle": False},
    "no RQMC final stage": {"qmc": False},
    "CE only (v1-style)": {"pg": False, "recycle": False, "qmc": False, "edge_features": False, "relevance": False},
}


def e3(B=16384, reps=40, seed=3000):
    cases = [("ADB base, joint exhaustion", PortfolioProblem(adb_model("adb-base"), "joint_exhaustion", threshold=80e6)),
             ("ADB base, Pr(L>80m)", PortfolioProblem(adb_model("adb-base"), "exceedance", threshold=80e6)),
             ("ring n=16, Pr(L>T)", PortfolioProblem(SyntheticModel(16, topology("ring", 16)), "exceedance")),
             ("ring n=8, joint exhaustion (0,1)", PortfolioProblem(SyntheticModel(8, topology("ring", 8)), "joint_exhaustion"))]
    tens = {(r["topology"], r["n"]): r for r in json.loads((R / "tensor.json").read_text())["B_scaling_n"]
            if "targets" in r and r["tol"] == 1e-9}
    prev = json.loads((R / "rdpo_e3.json").read_text())["rows"] if (R / "rdpo_e3.json").exists() else []
    done = {(r["case"], r["ablation"]) for r in prev}
    rows = list(prev)
    for name, P in cases:
        if P.n == 2:
            mu = P.exact_dense()
        elif P.target == "exceedance":
            mu = tens[("ring", P.n)]["targets"]["exceedance_prob"]
        else:
            mu = pair_exhaustion_exact(P.model)
        for ab, kw in ABLATIONS.items():
            if (name, ab) in done:
                continue
            cfg = replace(RDPOConfig(), **kw)
            xs = [rdpo_estimate(P, B, np.random.default_rng(seed + r), cfg) for r in range(reps)]
            s = summarise([x["estimate"] for x in xs], mu, [x["ci"] for x in xs])
            rows.append({"case": name, "ablation": ab, "mu": mu, "budget": B, **s})
            print(f"E3 {name:34s} {ab:24s} relRMSE={s['rel_rmse']:.4f} cov={s['ci_coverage']:.2f}", file=sys.stderr, flush=True)
        (R / "rdpo_e3.json").write_text(json.dumps({"rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    {"e1": e1, "e2": e2, "e3": e3}[sys.argv[1]]()
