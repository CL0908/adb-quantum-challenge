"""Cutset conditioning (src/tensor/cutset.py) against exponential twisting at equal wall-clock.

  python -m src.rdpo.experiments_cutset validate   # small dense checks (n <= 6): unbiasedness, enumeration bounds
  python -m src.rdpo.experiments_cutset run        # equal-time replications + references + |D| scaling + enumeration
  python -m src.rdpo.experiments_cutset report     # results/cutset.json
Resumable jobs in results/raw/cutset_*.jsonl; run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from src.rdpo.problem import PortfolioProblem
from src.rdpo.tilt import tilt_estimate
from src.rdpo.experiments_v2 import run_jobs
from src.tensor.cutset import CutsetSolver, split_edges
from src.tensor.junction import dense_topology, ladder_edges
from src.tensor.synthetic import SyntheticModel, topology

R = Path("results")
RAW = R / "raw"
TARGETS = ("exceedance", "stop_loss")

# case name -> (graph kind, n, keep rule, exact reference available)
CASES = {
    "band4-8/b2": ("band4", 8, ("band", 2), True),
    "band4-12/b2": ("band4", 12, ("band", 2), True),
    "band4-12/b1": ("band4", 12, ("band", 1), True),
    "band4-20/b2": ("band4", 20, ("band", 2), True),
    "ladder-xdiag-28/ladder": ("ladder-xdiag", 28, ("ladder",), True),
    "band6-12/b2": ("band6", 12, ("band", 2), False),
    "band6-20/b2": ("band6", 20, ("band", 2), False),
    "band8-12/b2": ("band8", 12, ("band", 2), False),
    "band8-14/b2": ("band8", 14, ("band", 2), False),
    "complete-8/chain": ("complete", 8, ("band", 1), False),
    "complete-10/chain": ("complete", 10, ("band", 1), False),
    "complete-12/chain": ("complete", 12, ("band", 1), False),
}
TIMED = [c for c in CASES if c not in ("band4-8/b2", "band4-12/b1", "band8-12/b2", "complete-10/chain")]   # 10 s budget (b1 and n=8 enter the |D| scaling study)
TIMED_60 = ["band4-20/b2"]   # additional 60 s budget
SCALING = [("band4-12", k) for k in (3, 2, 1, 0)] + [("band4-20", k) for k in (3, 2, 1)]
REPS = 8


def build(kind, n):
    return SyntheticModel(n, topology(kind, n) if kind == "complete" else dense_topology(kind, n))


def keep_fn(spec, n):
    if spec[0] == "band":
        return lambda i, j: j - i <= spec[1]
    ed = {tuple(sorted(e)) for e in ladder_edges(n, 0)}
    return lambda i, j: (i, j) in ed


def exact_ref(kind, n):
    d = json.loads((R / "dense_exact.json").read_text())["exact"]
    for r in d:
        if r["graph"] == kind and r["n"] == n:
            return {"exceedance": r["exceedance"], "stop_loss": r["stop_loss"]}
    return None


# ---------------------------------------------------------------------------------------------------------------
def make_solver(kind, n, spec):
    m = build(kind, n)
    P = PortfolioProblem(m, "exceedance")
    Rr, D = split_edges(m.edges, keep_fn(spec, n))
    return P, CutsetSolver(P, D), D


def _job(args):
    jid, typ = args[0], args[1]
    t00 = time.perf_counter()
    if typ == "cmp":
        _, _, case, budget, rep, method, seed = args
        kind, n, spec, _ex = CASES[case]
        rng = np.random.default_rng(seed)
        if method == "cutset":
            t0 = time.perf_counter()
            P, S, D = make_solver(kind, n, spec)
            o = S.sampled(rng, seconds=budget, t_start=t0)
            return {"id": jid, "typ": typ, "case": case, "budget": budget, "rep": rep, "method": method,
                    "n_samples": o["n_samples"], "seconds": o["seconds"], "n_contractions": o["n_contractions"],
                    "distinct_cached": o["distinct_cached"], "sec_per_contraction": o["sec_per_contraction"],
                    "ess_frac": o["ess_frac"], "KD": o["KD"], "width_R": o["width_R"], "D_edges": len(D),
                    "exceedance": o["exceedance"], "stop_loss": o["stop_loss"]}
        res = {}
        for t in TARGETS:
            P = PortfolioProblem(build(kind, n), t)
            x = tilt_estimate(P, rng, seconds=budget, fit=False)
            res[t] = {"estimate": x["estimate"], "se": x["se"], "ci": list(x["ci"]), "rel_var_per_sample": x["rho2"]}
        return {"id": jid, "typ": typ, "case": case, "budget": budget, "rep": rep, "method": method,
                "paths_used": x["paths_used"], "tau": x["tau"], "seconds": time.perf_counter() - t00, **res}
    if typ == "ref":      # large-budget twisting reference for graphs without an exact reference
        _, _, case, budget, seed = args
        kind, n, spec, _ex = CASES[case]
        res = {}
        for t in TARGETS:
            P = PortfolioProblem(build(kind, n), t)
            x = tilt_estimate(P, np.random.default_rng(seed), seconds=budget, fit=False)
            res[t] = {"estimate": x["estimate"], "se": x["se"], "ci": list(x["ci"]), "rel_var_per_sample": x["rho2"],
                      "paths": x["paths_used"]}
        return {"id": jid, "typ": typ, "case": case, "budget": budget, **res}
    if typ == "scal":      # |D| scaling and sensitivity of the sampling law, 10 s, cutset only
        _, _, name, keep, rep, eta_scale, delta, seed = args
        kind, n = name.split("-")[0], int(name.split("-")[1])
        t0 = time.perf_counter()
        P, S, D = make_solver(kind, n, ("band", keep))
        o = S.sampled(np.random.default_rng(seed), seconds=10.0, t_start=t0, eta_scale=eta_scale, delta=delta)
        return {"id": jid, "typ": typ, "name": name, "keep": keep, "rep": rep, "eta_scale": eta_scale, "delta": delta,
                "D_edges": len(D), "KD": o["KD"], "width_R": o["width_R"], "n_samples": o["n_samples"],
                "n_contractions": o["n_contractions"], "distinct_cached": o["distinct_cached"],
                "sec_per_contraction": o["sec_per_contraction"], "ess_frac": o["ess_frac"],
                "exceedance": o["exceedance"], "stop_loss": o["stop_loss"]}
    if typ == "enum":
        _, _, case, eta_mode, max_s = args
        kind, n, spec, _ex = CASES[case]
        P, S, D = make_solver(kind, n, spec)
        from src.rdpo.train import _best_twist_sources
        eta = 0.0 if eta_mode == "p" else _best_twist_sources(P)
        e = S.enumerate_best_first(max_configs=10 ** 7, max_seconds=max_s, eta_order=eta, check_every=500)
        return {"id": jid, "typ": typ, "case": case, "order": eta_mode, "eta_order": eta, "KD": S.KD, "D_edges": len(D),
                "width_R": S.width, **{k: v for k, v in e.items() if k != "trace"},
                "trace": e["trace"][::max(1, len(e["trace"]) // 40)] + e["trace"][-1:]}
    raise ValueError(typ)


# ---------------------------------------------------------------------------------------------------------------
def validate():
    """Dense references (n <= 6): unbiasedness of the sampler, containment by the enumeration bounds."""
    rows = []
    cases = [("complete", 4, ("band", 1)), ("complete", 5, ("band", 1)), ("complete", 6, ("band", 1)),
             ("band4", 6, ("band", 1)), ("complete", 5, ("band", 2))]
    for kind, n, spec in cases:
        m = build(kind, n)
        for t in TARGETS:
            P = PortfolioProblem(m, t)
            thr = P.T
            ex = P.exact_dense()
            Rr, D = split_edges(m.edges, keep_fn(spec, n))
            S = CutsetSolver(P, D)
            reps = []
            for r in range(10):
                o = S.sampled(np.random.default_rng(100 + r), n_samples=4000)
                reps.append(o[t])
            est = np.array([x["estimate"] for x in reps]); se = np.array([x["se"] for x in reps])
            cov = float(np.mean([x["ci"][0] <= ex <= x["ci"][1] for x in reps]))
            zmean = float((est.mean() - ex) / (est.std(ddof=1) / np.sqrt(len(est))))
            e = S.enumerate_best_first(max_configs=3000, check_every=1000)
            lo = e["exceedance_lower"] if t == "exceedance" else e["stop_loss_lower"]
            rem = e["remainder_bound_exceedance"] if t == "exceedance" else e["remainder_bound_stop_loss"]
            rows.append({"graph": kind, "n": n, "target": t, "cut_edges": len(D), "KD": S.KD, "width_R": S.width,
                         "dense_exact": ex, "sampled_mean_of_10": float(est.mean()),
                         "rel_err_of_mean": float(est.mean() / ex - 1), "z_of_mean": zmean, "ci_coverage_10": cov,
                         "mean_rel_var_per_sample": float(np.mean([x["rel_var_per_sample"] for x in reps])),
                         "enum_lower": lo, "enum_remainder_bound": rem,
                         "enum_contains_exact": bool(lo <= ex * (1 + 1e-12) and ex <= lo + rem), "enum_n": e["n_enumerated"],
                         "contraction_check": None})
            print("validate", rows[-1], file=sys.stderr, flush=True)
    # contraction check: empty cutset equals the exact solver (dense graph)
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "cutset_validate.json").write_text(json.dumps(rows, indent=1))
    return rows


def run(procs=2, seed=7000):
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ci, case in enumerate(TIMED):
        if not CASES[case][3]:
            jobs.append((f"ref-{case}", "ref", case, 30.0, seed + 31 * ci))
    for ci, case in enumerate(TIMED):
        for rep in range(REPS):
            for method in ("cutset", "twist"):
                jobs.append((f"cmp-{case}-10-{method}-{rep}", "cmp", case, 10.0, rep, method, seed + 1000 * ci + rep))
    for ci, case in enumerate(TIMED_60):
        for rep in range(REPS):
            for method in ("cutset", "twist"):
                jobs.append((f"cmp-{case}-60-{method}-{rep}", "cmp", case, 60.0, rep, method, seed + 50000 + 1000 * ci + rep))
    for name, keep in SCALING:
        for rep in range(REPS):
            jobs.append((f"scal-{name}-{keep}-1.0-0.1-{rep}", "scal", name, keep, rep, 1.0, 0.1, seed + 90000 + rep))
    for eta_scale, delta in ((0.5, 0.1), (1.0, 0.0), (1.0, 0.5)):
        for rep in range(REPS):
            jobs.append((f"scal-band4-12-2-{eta_scale}-{delta}-{rep}", "scal", "band4-12", 2, rep, eta_scale, delta,
                         seed + 95000 + rep))
    for case in ("band4-8/b2", "band4-12/b2", "band4-20/b2", "complete-8/chain"):
        for mode in ("p", "tilt"):
            jobs.append((f"enum-{case}-{mode}", "enum", case, mode, 40.0))
    jobs = sorted(jobs, key=lambda j: ({"ref": 0, "enum": 1, "scal": 2, "cmp": 3}[j[1]], j[1] == "cmp" and j[3] == 60.0))
    run_jobs("cutset_runs", jobs, _job, procs)


def _summ(vals, ref):
    v = np.array(vals, float)
    return {"mean": float(v.mean()), "rel_rmse": float(np.sqrt(np.mean((v / ref - 1) ** 2))) if ref else None,
            "rel_bias": float(v.mean() / ref - 1) if ref else None, "rel_sd": float(v.std(ddof=1) / v.mean())}


def report():
    recs = [json.loads(l) for l in (RAW / "cutset_runs.jsonl").read_text().splitlines() if l.strip()]
    by = {}
    for r in recs:
        by.setdefault(r["typ"], []).append(r)
    refs = {r["case"]: r for r in by.get("ref", [])}
    out = {"meta": {"reps": REPS, "budgets_s": [10, 60], "threshold": "T_n = 40m n", "twist": "tilt_estimate(fit=False), the "
           "fast product-form sampler of src/rdpo/tilt.py, one run per target", "cutset": "CutsetSolver.sampled, one run gives "
           "both targets, construction, zero-offset sweep and all contractions charged", "threads": "1 BLAS thread, 2 worker processes"},
           "equal_time": [], "scaling": [], "sensitivity": [], "enumeration": [], "references": list(refs.values())}
    rows = {}
    for r in by.get("cmp", []):
        rows.setdefault((r["case"], r["budget"]), {}).setdefault(r["method"], []).append(r)
    for (case, B), d in rows.items():
        kind, n, spec, hasex = CASES[case]
        ex = exact_ref(kind, n) if hasex else None
        row = {"case": case, "graph": kind, "n": n, "budget_s": B, "reps": {m: len(v) for m, v in d.items()}}
        c = d.get("cutset", [])
        if c:
            row["cutset_info"] = {"D_edges": c[0]["D_edges"], "KD": c[0]["KD"], "width_R": c[0]["width_R"],
                                  "median_samples": float(np.median([x["n_samples"] for x in c])),
                                  "median_contractions": float(np.median([x["n_contractions"] for x in c])),
                                  "median_sec_per_contraction": float(np.median([x["sec_per_contraction"] for x in c])),
                                  "median_ess_frac": float(np.median([x["ess_frac"] for x in c]))}
        for t in TARGETS:
            ref = ex[t] if ex else None
            e = {"exact": ref}
            if ref is None and case in refs:
                e["twist_reference"] = refs[case][t]["estimate"]; e["twist_reference_ci"] = refs[case][t]["ci"]
            for m in ("cutset", "twist"):
                xs = d.get(m, [])
                if not xs:
                    continue
                est = [x[t]["estimate"] for x in xs]
                e[m] = _summ(est, ref)
                e[m]["median_rel_var_per_sample"] = float(np.median([x[t]["rel_var_per_sample"] for x in xs]))
                if ref is not None:
                    e[m]["ci_coverage"] = float(np.mean([x[t]["ci"][0] <= ref <= x[t]["ci"][1] for x in xs]))
                if m == "twist":
                    e[m]["median_paths"] = float(np.median([x["paths_used"] for x in xs]))
            if "cutset" in e and "twist" in e:
                if ref:
                    e["rmse_ratio_cutset_over_twist"] = e["cutset"]["rel_rmse"] / e["twist"]["rel_rmse"]
                else:
                    e["rel_sd_ratio_cutset_over_twist"] = e["cutset"]["rel_sd"] / e["twist"]["rel_sd"]
                    r0 = e.get("twist_reference")
                    if r0:
                        e["cutset_mean_over_reference"] = e["cutset"]["mean"] / r0
                        e["twist_mean_over_reference"] = e["twist"]["mean"] / r0
            row[t] = e
        out["equal_time"].append(row)
    sc = {}
    for r in by.get("scal", []):
        sc.setdefault((r["name"], r["keep"], r["eta_scale"], r["delta"]), []).append(r)
    for (name, keep, es, de), xs in sorted(sc.items()):
        kind, n = name.split("-")[0], int(name.split("-")[1])
        ex = exact_ref(kind, n)
        row = {"graph": name, "keep_band": keep, "eta_scale": es, "delta": de, "reps": len(xs), "D_edges": xs[0]["D_edges"],
               "KD": xs[0]["KD"], "width_R": xs[0]["width_R"],
               "median_samples": float(np.median([x["n_samples"] for x in xs])),
               "median_distinct_cached": float(np.median([x["distinct_cached"] for x in xs])),
               "median_sec_per_contraction": float(np.median([x["sec_per_contraction"] for x in xs])),
               "median_ess_frac": float(np.median([x["ess_frac"] for x in xs]))}
        for t in TARGETS:
            row[t] = {**_summ([x[t]["estimate"] for x in xs], ex[t]),
                      "median_rel_var_per_sample": float(np.median([x[t]["rel_var_per_sample"] for x in xs]))}
        (out["scaling"] if (es, de) == (1.0, 0.1) else out["sensitivity"]).append(row)
    for r in by.get("enum", []):
        kind, n, spec, hasex = CASES[r["case"]]
        ex = exact_ref(kind, n) if hasex else None
        r = dict(r)
        r["exact"] = ex
        for t, lo, rem in (("exceedance", "exceedance_lower", "remainder_bound_exceedance"),
                           ("stop_loss", "stop_loss_lower", "remainder_bound_stop_loss")):
            if ex:
                r[f"{t}_rel_remainder_bound"] = r[rem] / ex[t]
                r[f"{t}_lower_over_exact"] = r[lo] / ex[t]
                r[f"{t}_contains_exact"] = bool(r[lo] <= ex[t] * (1 + 1e-9) and ex[t] <= r[lo] + r[rem])
        out["enumeration"].append(r)
    vf = RAW / "cutset_validate.json"
    out["validation_dense"] = json.loads(vf.read_text()) if vf.exists() else None
    (R / "cutset.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cmd = sys.argv[1]
    {"validate": validate, "run": run, "report": report}[cmd]()
