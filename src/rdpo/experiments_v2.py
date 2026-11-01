"""RDPO v2 experiments (roadmap WP1 + WP3), run against the v1 results at the same total budget.

  throughput  rollout paths/s, v1 engine vs table-driven engine (identical policy)
  scale       ring 16/28 and complete 6/7: exceedance, stop-loss, pair exhaustion; B = 65,536;
              variants v1 (original algorithm on the fast engine) and v2 (t-intervals + model selection);
              24 replications each (v1 study: 6–10)
  bench       all 25 ADB cells x 3 budgets with the final v2 configuration, 40 replications
  transfer    "sleep phase" warm start: a policy trained on one scenario is reused, unchanged in form, as the
              initial policy for a perturbed scenario; accuracy at small budgets with and without transfer
Usage: python -m src.rdpo.experiments_v2 throughput|scale|bench|transfer
"""
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.tensor.synthetic import SyntheticModel, topology
from .problem import PortfolioProblem, adb_model, ADB_SCENARIOS, TARGETS
from .policy import RDPOPolicy
from .fast import FastPolicy
from .train import rdpo_estimate, RDPOConfig, v2_config
from .experiments import summarise, pair_exhaustion_exact

R = Path("results")
VARIANTS = {
    "v1": RDPOConfig(engine="fast", ci="z"),     # original algorithm (fast engine; identical policy)
    "v2": v2_config(),
}
RAW = R / "raw"


def run_jobs(name, jobs, fn, procs=4):
    """Run jobs in a process pool, appending every finished job to results/raw/<name>.jsonl, so an interrupted
    run resumes where it stopped. Each job is a tuple whose first element is a unique id."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()}
    todo = [j for j in jobs if j[0] not in done]
    print(f"{name}: {len(done)} done, {len(todo)} to run", file=sys.stderr, flush=True)
    with Pool(procs) as pool, open(path, "a") as fh:
        for k, rec in enumerate(pool.imap_unordered(fn, todo, chunksize=1)):
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if (k + 1) % 20 == 0:
                print(f"{name}: {k + 1}/{len(todo)}", file=sys.stderr, flush=True)
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def final_config(target):
    """Final v2 configuration, chosen in development probes (see the report):
    stage-dependent progress parameters (4 stages) for portfolio-wide threshold targets; flexible (spline)
    soft values for the smooth expected-loss target; the v1 feature set for pair targets (which the product
    routes to the exact pair reduction, PortfolioProblem.exact_pair)."""
    if target == "expected_loss":
        return v2_config(value="spline")
    if target in ("exceedance", "stop_loss"):
        return v2_config(stages=4)
    return v2_config()


# ------------------------------------------------------------------------------------------------
def throughput(N=65536):
    rows = []
    for kind, n in [("ring", 2), ("ring", 8), ("ring", 16), ("ring", 28), ("complete", 4), ("complete", 7)]:
        P = PortfolioProblem(SyntheticModel(n, topology(kind, n)), "exceedance")
        row = {"topology": kind, "n": n, "K": P.K}
        for name, cls in (("v1", RDPOPolicy), ("fast", FastPolicy)):
            pol = cls(P); pol.theta = np.random.default_rng(0).normal(size=pol.d) * 0.1
            ts = []
            for rep in range(3):
                t0 = time.perf_counter(); pol.rollout(N, np.random.default_rng(rep)); ts.append(time.perf_counter() - t0)
            row[name] = N / min(ts)
        row["speedup"] = row["fast"] / row["v1"]
        rows.append(row)
        print(f"{kind} n={n}: v1 {row['v1']:,.0f}/s fast {row['fast']:,.0f}/s x{row['speedup']:.1f}", file=sys.stderr, flush=True)
    (R / "rdpo_v2_throughput.json").write_text(json.dumps({"N": N, "rows": rows}, indent=2))


# ------------------------------------------------------------------------------------------------
def _scale_refs():
    tens = {(r["topology"], r["n"]): r for r in json.loads((R / "tensor.json").read_text())["B_scaling_n"]
            if "targets" in r and r["tol"] == (1e-6 if (r["topology"] == "complete" and r["n"] == 7) else 1e-9)}
    out = {}
    for kind, n in [("ring", 16), ("ring", 28), ("complete", 6), ("complete", 7)]:
        ref = tens[(kind, n)]["targets"]
        out[(kind, n)] = {"exceedance": ref["exceedance_prob"], "stop_loss": ref["stop_loss"],
                          "joint_exhaustion": pair_exhaustion_exact(SyntheticModel(n, topology(kind, n)))}
    return out


def _job(args):
    jid, kind, n, target, variant, rep, B, seed = args
    P = PortfolioProblem(SyntheticModel(n, topology(kind, n)), target)
    cfg = VARIANTS[variant] if variant in VARIANTS else final_config(target)
    t0 = time.perf_counter()
    x = rdpo_estimate(P, B, np.random.default_rng(seed), cfg)
    return {"id": jid, "key": [kind, n, target, variant], "est": x["estimate"], "ci": list(x["ci"]),
            "sec": time.perf_counter() - t0, "selected": x.get("log", {}).get("selected")}


def scale(B=65536, reps=24, seed=7000, procs=4):
    refs = _scale_refs()
    jobs = [(f"{kind}-{n}-{t}-{v}-{r}", kind, n, t, v, r, B, seed + 1000 * n + 100 * TARGETS.index(t) + r)
            for r in range(reps) for (kind, n) in refs for t in ("exceedance", "stop_loss", "joint_exhaustion")
            for v in list(VARIANTS) + ["v2-final"] if not (v == "v2-final" and t == "joint_exhaustion")]
    res = {}
    for rec in run_jobs("scale", jobs, _job, procs):
        res.setdefault(tuple(rec["key"]), []).append((rec["est"], rec["ci"], rec["sec"], rec["selected"]))
    v1old = {(r["topology"], r["n"], r["target"]): r for r in json.loads((R / "rdpo_e2.json").read_text())["rows"]}
    rows = []
    exact_ms = {}
    for (kind, n) in refs:
        P = PortfolioProblem(SyntheticModel(n, topology(kind, n)), "joint_exhaustion")
        t0 = time.perf_counter(); val = P.exact_pair()
        exact_ms[f"{kind}-{n}"] = {"value": val, "ms": 1e3 * (time.perf_counter() - t0),
                                   "rel_err_vs_mps": val / refs[(kind, n)]["joint_exhaustion"] - 1}
    for (kind, n, t, v), xs in sorted(res.items()):
        mu = refs[(kind, n)][t]
        s = summarise([e for e, *_ in xs], mu, [c for _, c, *_ in xs])
        rows.append({"topology": kind, "n": n, "target": t, "variant": v, "mu": mu, "budget": B, **s,
                     "seconds_per_fit": float(np.mean([x[2] for x in xs])),
                     "selected_initial": float(np.mean([x[3] == "initial" for x in xs])),
                     "v1_study": {k: v1old[(kind, n, t)]["rdpo"][k] for k in ("rel_rmse", "ci_coverage", "reps")}
                     if (kind, n, t) in v1old else None,
                     "v1_study_seconds": v1old[(kind, n, t)]["seconds_per_fit"] if (kind, n, t) in v1old else None,
                     "twisted": v1old[(kind, n, t)].get("twisted", {}).get("rel_rmse") if (kind, n, t) in v1old else None})
        print(f"{kind:8s} n={n:2d} {t:16s} {v:7s} rel={s['rel_rmse']:.4f} cov={s['ci_coverage']:.2f}", file=sys.stderr)
    (R / "rdpo_v2_scale.json").write_text(json.dumps({"reps": reps, "rows": rows, "exact_pair": exact_ms}, indent=2))


# ------------------------------------------------------------------------------------------------
def _bench_job(args):
    jid, sc, t, B, seed = args
    P = PortfolioProblem(adb_model(sc), t, threshold=80e6)
    x = rdpo_estimate(P, B, np.random.default_rng(seed), final_config(t))
    return {"id": jid, "key": [sc, t, B], "est": x["estimate"], "ci": list(x["ci"])}


def bench(budgets=(4096, 16384, 65536), reps=40, seed=1000, procs=4):
    jobs = [(f"{sc}-{t}-{B}-{r}", sc, t, B, seed + 100000 * si + 1000 * ti + r) for si, sc in enumerate(ADB_SCENARIOS)
            for ti, t in enumerate(TARGETS) for B in budgets for r in range(reps)]
    res = {}
    for rec in run_jobs("bench", jobs, _bench_job, procs):
        res.setdefault(tuple(rec["key"]), []).append((rec["est"], rec["ci"]))
    v1 = {(r["scenario"], r["target"], r["budget"]): r for r in json.loads((R / "rdpo_e1.json").read_text())["rows"]}
    rows = []
    for (sc, t, B), xs in sorted(res.items()):
        mu = PortfolioProblem(adb_model(sc), t, threshold=80e6).exact_dense()
        s = summarise([e for e, _ in xs], mu, [c for _, c in xs])
        o = v1[(sc, t, B)]
        bl = o["baselines"]; best = min(bl, key=bl.get)
        rows.append({"scenario": sc, "target": t, "budget": B, "mu": mu, "v2": s, "v1": o["rdpo"],
                     "baselines": bl, "best_baseline": best, "improvement_vs_best": bl[best] / s["rel_rmse"],
                     "v1_improvement_vs_best": o["improvement_vs_best"]})
    (R / "rdpo_v2_bench.json").write_text(json.dumps({"reps": reps, "rows": rows}, indent=2))
    print("bench done", file=sys.stderr)


# ------------------------------------------------------------------------------------------------
def _transfer_job(args):
    jid, case, t, B, with_transfer, seed = args
    src_model, dst_model = TRANSFER_CASES[case]
    P_dst = PortfolioProblem(dst_model(), t, threshold=80e6 if dst_model().n == 2 else None)
    cfg = final_config(t)
    if with_transfer:
        # "overnight": train once on yesterday's model with a generous budget; reuse θ as today's start
        P_src = PortfolioProblem(src_model(), t, threshold=80e6 if src_model().n == 2 else None)
        th = rdpo_estimate(P_src, 65536, np.random.default_rng(seed + 999), cfg)["theta"]
        cfg = replace(cfg, theta0=th)
    x = rdpo_estimate(P_dst, B, np.random.default_rng(seed), cfg)
    return {"id": jid, "key": [case, t, B, with_transfer], "est": x["estimate"], "ci": list(x["ci"])}


TRANSFER_CASES = {
    "ADB base -> common shock": (lambda: adb_model("adb-base"), lambda: adb_model("adb-common-shock")),
    "ADB base -> rain stress": (lambda: adb_model("adb-base"), lambda: adb_model("adb-rain-stress")),
    "ring 16: rates +20% overnight": (lambda: SyntheticModel(16, topology("ring", 16)),
                                      lambda: SyntheticModel(16, topology("ring", 16), m=1.2)),
}


def transfer(budgets=(2048, 4096), reps=40, seed=9000, procs=4):
    targets = {"ADB base -> common shock": ("joint_exhaustion", "stop_loss"),
               "ADB base -> rain stress": ("joint_exhaustion", "stop_loss"),
               "ring 16: rates +20% overnight": ("exceedance",)}
    jobs = [(f"{c}-{t}-{B}-{wt}-{r}", c, t, B, wt, seed + r) for c in TRANSFER_CASES for t in targets[c]
            for B in budgets for wt in (False, True) for r in range(reps)]
    res = {}
    for rec in run_jobs("transfer", jobs, _transfer_job, procs):
        res.setdefault(tuple(rec["key"]), []).append((rec["est"], rec["ci"]))
    rows = []
    for (c, t, B, wt), xs in sorted(res.items()):
        dst = TRANSFER_CASES[c][1]()
        P = PortfolioProblem(dst, t, threshold=80e6 if dst.n == 2 else None)
        if dst.n == 2:
            mu = P.exact_dense()
        else:
            mu = _ring_target_exact(dst, P)
        s = summarise([e for e, _ in xs], mu, [ci for _, ci in xs])
        rows.append({"case": c, "target": t, "budget": B, "transfer": wt, "mu": mu, **s})
        print(f"{c:32s} {t:16s} B={B} transfer={wt} rel={s['rel_rmse']:.4f} cov={s['ci_coverage']:.2f}", file=sys.stderr)
    (R / "rdpo_v2_transfer.json").write_text(json.dumps({"reps": reps, "rows": rows}, indent=2))


def _ring_target_exact(model, P):
    """Exact Pr(L > T) for a ring model via the tilted MPS used for the scaling references."""
    from src.tensor.mps import build_mps, targets_from_mps
    psi, _ = build_mps(model, tol=1e-12, tilt=True)
    return float(targets_from_mps(model, psi, P.T * 2.5e6)["exceedance_prob"])


if __name__ == "__main__":
    {"throughput": throughput, "scale": scale, "bench": bench, "transfer": transfer}[sys.argv[1]]()
