"""Dense-dependence study: exact references on bounded-treewidth graphs denser than the ring, and RDPO
(Phase 2 engine, final v2 configuration) against exponential twisting at the same budget and seeds.

  exact  : junction/frontier solver (src/tensor/junction.py) for Pr(L > T_n) and E[(L - T_n)+], T_n = 40m * n
  run    : RDPO and twisting replications (resumable jobs in results/raw/dense_*.jsonl)
  report : results/dense_exact.json
Usage: python -m src.rdpo.experiments_dense exact|run|report
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.tensor.synthetic import SyntheticModel
from src.tensor.junction import dense_topology, junction_solve, treewidth_minfill
from .problem import PortfolioProblem
from .train import rdpo_estimate
from .experiments import summarise, twisted_est
from .experiments_v2 import run_jobs, final_config, RAW

R = Path("results")
B = 65536
TARGETS = ("exceedance", "stop_loss")
CASES = ([("band2", n) for n in (8, 12, 16, 20, 24, 28)] +
         [("band3", n) for n in (8, 12, 16, 20, 24, 28)] +
         [("band4", n) for n in (8, 12, 16, 20, 24, 28)] +
         [("ladder-xdiag", n) for n in (12, 20, 28)])
# subset on which the (slow) RDPO/twisting replications are run
RUN_CASES = [("band2", 28), ("band3", 20), ("band4", 12), ("band4", 20), ("ladder-xdiag", 28)]


def model_of(kind, n):
    return SyntheticModel(n, dense_topology(kind, n))


def exact_refs():
    """Exact references, cached in results/raw/dense_exact.jsonl (one record per case)."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "dense_exact.jsonl"
    recs = {}
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip():
                r = json.loads(l); recs[(r["graph"], r["n"])] = r
    with open(path, "a") as fh:
        for kind, n in CASES:
            if (kind, n) in recs:
                continue
            m = model_of(kind, n)
            P = PortfolioProblem(m, "exceedance")
            s = junction_solve(P)
            rec = {"graph": kind, "n": n, "edges": len(m.edges), "treewidth_ub": treewidth_minfill(n, m.edges),
                   "frontier_width": s["width"], "T_units": P.T, "exceedance": s["exceedance"],
                   "stop_loss": s["stop_loss"], "total_mass": s["total_mass"], "exact_seconds": s["seconds"]}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            recs[(kind, n)] = rec
            print(f"exact {kind} n={n} |E|={rec['edges']} w={rec['frontier_width']} mu={rec['exceedance']:.3e} "
                  f"{rec['exact_seconds']:.1f}s", file=sys.stderr, flush=True)
    return recs


def _job(args):
    jid, kind, n, target, variant, rep, seed = args
    P = PortfolioProblem(model_of(kind, n), target)
    t0 = time.perf_counter()
    if variant == "twist":
        est = float(twisted_est(P, B, np.random.default_rng(seed)))
        return {"id": jid, "key": [kind, n, target, variant], "est": est, "ci": None, "sec": time.perf_counter() - t0}
    x = rdpo_estimate(P, B, np.random.default_rng(seed), final_config(target))
    return {"id": jid, "key": [kind, n, target, variant], "est": x["estimate"], "ci": list(x["ci"]),
            "sec": time.perf_counter() - t0}


def run(reps=8, seed=11000, procs=3):
    exact_refs()
    jobs = [(f"{kind}-{n}-{t}-{v}-{r}", kind, n, t, v, r, seed + 1000 * n + 100 * TARGETS.index(t) + r)
            for r in range(reps) for (kind, n) in RUN_CASES for t in TARGETS for v in ("rdpo", "twist")]
    run_jobs("dense_runs", jobs, _job, procs)


def report():
    recs = exact_refs()
    res = {}
    for l in (RAW / "dense_runs.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l); res.setdefault(tuple(r["key"]), []).append(r)
    rows = []
    for (kind, n) in CASES:
        ex = recs[(kind, n)]
        for t in TARGETS:
            a, b = res.get((kind, n, t, "rdpo")), res.get((kind, n, t, "twist"))
            if not a or not b or len(a) < 4:      # cells with too few replications are left out
                continue
            mu = ex[t]
            s = summarise([x["est"] for x in a], mu, [x["ci"] for x in a])
            w = summarise([x["est"] for x in b], mu)
            rows.append({"graph": kind, "n": n, "k": int(kind[4:]) if kind.startswith("band") else None,
                         "edges": ex["edges"], "treewidth_ub": ex["treewidth_ub"], "frontier_width": ex["frontier_width"],
                         "target": t, "mu": mu, "budget": B, "reps": s["reps"],
                         "rdpo_rel_rmse": s["rel_rmse"], "rdpo_rel_bias": s["rel_bias"], "rdpo_coverage": s["ci_coverage"],
                         "twist_rel_rmse": w["rel_rmse"], "twist_over_rdpo": w["rel_rmse"] / s["rel_rmse"],
                         "rdpo_sec_per_estimate_median": float(np.median([x["sec"] for x in a])),
                         "twist_sec_per_estimate_median": float(np.median([x["sec"] for x in b])),
                         "exact_seconds": ex["exact_seconds"]})
    meta = {"budget": B, "threshold_rule": "T_n = 40m * n (units of 2.5m: 16 n)",
            "reference": "exact frontier variable elimination, src/tensor/junction.py; validated against dense "
                         "enumeration (n<=6) and the tilted MPS (ring/chain), see tests/test_junction.py",
            "rdpo": "Phase 2 engine, final v2 configuration (experiments_v2.final_config)",
            "baseline": "exponential twisting (experiments.twisted_est), same budget and seeds",
            "graphs": "bandk: country i linked to i+1..i+k (treewidth k); ladder-xdiag: 2 x n/2 ladder with both "
                      "diagonals per square (treewidth 3)",
            "notes": "RDPO/twisting replications (8 each, 7 for band4 n=20 after a worker was lost to memory pressure) "
                     "were run on a subset of the exact settings because one RDPO fit takes 45-110 s; exact times "
                     "for band4 n>=24 were measured under CPU contention; coverage is from t-intervals with few replications",
            "max_n_exact": max(r["n"] for r in recs.values()),
            "max_exact_seconds": max(r["exact_seconds"] for r in recs.values())}
    (R / "dense_exact.json").write_text(json.dumps({"meta": meta, "exact": list(recs.values()), "rows": rows}, indent=2))
    for r in rows:
        print(f"{r['graph']:13s} n={r['n']:2d} {r['target']:10s} mu={r['mu']:.2e} rdpo={r['rdpo_rel_rmse']:.4f} "
              f"cov={r['rdpo_coverage']:.2f} twist={r['twist_rel_rmse']:.4f} x{r['twist_over_rdpo']:.1f}")


if __name__ == "__main__":
    {"exact": exact_refs, "run": run, "report": report}[sys.argv[1]]()
