"""Equal wall-clock audit of the two headline result sets (the proposal compares methods at equal PATH budget).

  A  ADB two-country benchmark: 5 scenarios x 5 targets = 25 cells, exact values known.
     Methods: MC, scrambled Sobol, stratified, the brief's IS, SMC-on-twist, flow IS (src/baselines) and RDPO
     (Phase 2 engine, benchmark configuration, B = 16,384 paths, training included).
  B  Data-calibrated pools (CAREC 10 countries, Asia-Pacific 28 countries), the questions of results/wp2_*.json.
     Methods: plain MC, exponential twisting, RDPO (B = 65,536, as in the paper).

Protocol.  (1) Timing pass, ONE process, nothing of ours in parallel: for each cell the methods are timed back to back
(interleaved repeats, median), together with a fixed reference kernel so that drift of machine load is visible.
Per-path rates of the baselines come from a two-point linear fit t(N) = a + b N (small and large N), so fixed costs
are not charged per path. (2) Equal-time budget: every baseline gets N_eq = (t_RDPO - a) / b paths (capped at 2^24;
capped cells are flagged), and the budget is checked by timing each baseline at N_eq single-process. (3) Accuracy:
>= 24 (A) / 12 (B) replications, the same seeds for every method, relative RMSE against the exact value.
Plain MC for the pools is the lean sampler of src/baselines/stronger (the paper's "mc" runs inside the RDPO engine
and is ~16x slower per path; it is timed for reference but the equal-time MC uses the faster sampler).

Usage: python -m src.rdpo.experiments_equal_time timing_a | reps_a | timing_b | reps_b | summary | all
Raw results (resumable): results/raw/eqt_*.jsonl; summary: results/equal_time.json
"""
import hashlib
import json
import math
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from .problem import PortfolioProblem, adb_model, ADB_SCENARIOS, TARGETS
from .train import rdpo_estimate, RDPOConfig
from .experiments import summarise, twisted_est
from .experiments_v2 import final_config, run_jobs

R = Path("results")
RAW = R / "raw"
CAP = 2 ** 24
B_ADB, B_POOL = 16384, 65536
REPS_A, REPS_B = 24, 12
TIME_REPS_A, TIME_REPS_B = 11, 3
SEED_A = 424200
KEYS = {"expected_loss": "term_expected_principal_loss_usd", "joint_attachment": "term_joint_attachment_probability",
        "exceedance": "term_portfolio_loss_exceedance_probability", "stop_loss": "term_portfolio_stop_loss_usd",
        "joint_exhaustion": "term_both_bonds_exhausted_probability"}
BRIEF4 = ["mc", "sobol", "stratified", "importance"]
METHODS_A = BRIEF4 + ["smc", "flow"]
POOL_CACHE = Path("/tmp/claude-0/-home-user-adb-quantum-challenge/e302eff0-1cb7-5189-a4c3-b73bd85f3c16/scratchpad")

CELLS_A = [(sc, t) for sc in ADB_SCENARIOS for t in TARGETS]
CELLS_B = [("carec", "exceedance", 240), ("carec", "exceedance", 500), ("carec", "stop_loss", 240),
           ("carec", "stop_loss", 500), ("carec", "joint_exhaustion", None),
           ("asia_pacific", "exceedance", 450), ("asia_pacific", "exceedance", 850),
           ("asia_pacific", "stop_loss", 850), ("asia_pacific", "joint_exhaustion", None)]
SEED_B = {"carec": 5151, "asia_pacific": 6161}      # the seeds of the paper's WP2 runs


def cell_id_a(sc, t): return f"{sc}|{t}"
def cell_id_b(c, t, thr): return f"{c}|{t}|{thr}"


def read_jsonl(name):
    p = RAW / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def append_jsonl(name, rec):
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / f"{name}.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


_REF = np.random.default_rng(0).random(1_000_000)


def ref_kernel():
    """Fixed reference workload (ms): drift of this number across the pass shows machine contention."""
    t0 = time.perf_counter()
    for _ in range(3):
        np.sort(_REF.copy())
    return 1e3 * (time.perf_counter() - t0)


def med(x): return float(np.median(x))


def fit_linear(n1, t1, n2, t2):
    b = (t2 - t1) / (n2 - n1)
    a = t1 - b * n1
    return max(a, 0.0), max(b, 1e-12)


def n_eq_from(t_target, a, b, mult=1):
    n = int((t_target - a) / b)
    n = max(n // mult * mult, 1024)
    return min(n, CAP), n >= CAP


def code_hash():
    h = {}
    for f in ("fast.py", "train.py", "problem.py"):
        h[f] = hashlib.sha1((Path(__file__).parent / f).read_bytes()).hexdigest()[:10]
    return h


# ================================================================================================
# A: ADB benchmark
# ================================================================================================
_A = {}


def adb_objects(sc, t):
    from src.baselines.common import make_problem
    key = (sc, t)
    if key not in _A:
        if ("brief", sc) not in _A:
            _A[("brief", sc)] = make_problem(sc)
        _A[key] = (PortfolioProblem(adb_model(sc), t, threshold=80e6), _A[("brief", sc)])
    return _A[key]


def adb_fn(method, sc, t):
    from src.baselines import estimators as E, stronger as S
    P, prob = adb_objects(sc, t)
    k = KEYS[t]
    if method == "mc": return lambda N, rng: E.mc(prob, N, rng)[k]
    if method == "sobol": return lambda N, rng: E.sobol(prob, N, rng)[k]
    if method == "stratified": return lambda N, rng: E.stratified(prob, N, rng)[k]
    if method == "importance": return lambda N, rng: E.importance(prob, N, rng)[k]
    if method == "smc": return lambda N, rng: S.smc_estimate(P, N, rng)["estimate"]
    if method == "flow": return lambda N, rng: S.flow_estimate(P, N, rng)["estimate"]
    if method == "rdpo":
        cfg = final_config(t)
        return lambda N, rng: rdpo_estimate(P, N, rng, cfg)["estimate"]
    raise KeyError(method)


def timed(fn, N, seed):
    t0 = time.perf_counter(); fn(N, np.random.default_rng(seed)); return time.perf_counter() - t0


def timing_a():
    from src.model.annual import annual_distribution
    from src.model.term import term_distribution
    from src.model.scenarios import SCENARIOS
    from src.model.targets import exact_targets
    warnings.simplefilter("ignore")
    done = {r["id"] for r in read_jsonl("eqt_timing_a")}
    for ci, (sc, t) in enumerate(CELLS_A):
        cid = cell_id_a(sc, t)
        if cid in done:
            continue
        P, prob = adb_objects(sc, t)
        fns = {m: adb_fn(m, sc, t) for m in ["rdpo"] + METHODS_A}
        for m in fns:                                           # warm-up
            fns[m](2048, np.random.default_rng(1))
        ref0 = ref_kernel()
        ts = {m: [] for m in fns}
        for r in range(TIME_REPS_A):
            for m in fns:
                ts[m].append(timed(fns[m], B_ADB, r))
        t16 = {m: med(v) for m, v in ts.items()}
        big = {m: med([timed(fns[m], 2 ** 18, r) for r in range(3)]) for m in METHODS_A}
        fit = {m: fit_linear(B_ADB, t16[m], 2 ** 18, big[m]) for m in METHODS_A}
        t_r = t16["rdpo"]
        neq, capped = {}, {}
        for m in METHODS_A:
            neq[m], capped[m] = n_eq_from(t_r, *fit[m], mult=8)
        verify = {m: med([timed(fns[m], neq[m], 100 + r) for r in range(3)]) for m in METHODS_A}
        # exact references: dense enumeration of the source-form model, the brief model's own exact path, exact_pair
        ex_dense = med([(lambda t0: (P.exact_dense(), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(21)])

        def brief_exact():
            t0 = time.perf_counter(); p = annual_distribution(SCENARIOS[sc]); td = term_distribution(p)
            exact_targets(SCENARIOS[sc], td); return time.perf_counter() - t0
        ex_brief = med([brief_exact() for _ in range(21)])
        ex_pair = None
        if t in ("joint_attachment", "joint_exhaustion"):
            ex_pair = med([(lambda t0: (P.exact_pair(), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(21)])
        t0 = time.perf_counter(); PortfolioProblem(adb_model(sc), t, threshold=80e6); setup = time.perf_counter() - t0
        rec = {"id": cid, "scenario": sc, "target": t, "idx": ci, "budget": B_ADB, "t16384": t16, "t_raw": ts,
               "t_262144": big, "fit_a_b": {m: list(v) for m, v in fit.items()}, "n_eq": neq, "capped": capped,
               "t_at_n_eq": verify, "exact_dense_s": ex_dense, "exact_brief_model_s": ex_brief, "exact_pair_s": ex_pair,
               "rdpo_problem_setup_s": setup, "ref_kernel_ms": [ref0, ref_kernel()], "code": code_hash()}
        append_jsonl("eqt_timing_a", rec)
        print(f"timing_a {cid}: rdpo {1e3 * t_r:.0f} ms; N_eq " + " ".join(f"{m}={neq[m]}" for m in METHODS_A),
              file=sys.stderr, flush=True)


def _job_a(args):
    jid, ci, sc, t, method, mode, N = args
    warnings.simplefilter("ignore")
    fn = adb_fn(method, sc, t)
    base = SEED_A + 1000 * ci
    ests, secs = [], []
    for r in range(REPS_A):
        t0 = time.perf_counter(); ests.append(float(fn(N, np.random.default_rng(base + r)))); secs.append(time.perf_counter() - t0)
    return {"id": jid, "cell": cell_id_a(sc, t), "method": method, "mode": mode, "N": N, "est": ests,
            "sec_mean_contended": float(np.mean(secs))}


def reps_a(procs=2):
    tim = {r["id"]: r for r in read_jsonl("eqt_timing_a")}
    jobs = []
    for ci, (sc, t) in enumerate(CELLS_A):
        tr = tim[cell_id_a(sc, t)]
        jobs.append((f"{sc}|{t}|rdpo|paths", ci, sc, t, "rdpo", "paths", B_ADB))
        for m in METHODS_A:
            jobs.append((f"{sc}|{t}|{m}|paths", ci, sc, t, m, "paths", B_ADB))
            jobs.append((f"{sc}|{t}|{m}|time", ci, sc, t, m, "time", tr["n_eq"][m]))
    run_jobs("eqt_reps_a", jobs, _job_a, procs)


# ================================================================================================
# B: data-calibrated pools
# ================================================================================================
_B = {}


def pool_model(case):
    if case not in _B:
        f = POOL_CACHE / f"eqt_pool_{case}.pkl"
        if f.exists():
            _B[case] = pickle.loads(f.read_bytes())
        else:
            if case == "carec":
                from src.realdata import carec_pool as m
                c = m.calibrate(); obj = (c["model"], c["pair"])
            else:
                from src.realdata import asia_pacific_pool as m
                obj = m._model("asia_pacific")
            f.write_bytes(pickle.dumps(obj)); _B[case] = obj
    return _B[case]


def pool_refs():
    c = json.loads((R / "wp2_carec.json").read_text())["references"]
    a = json.loads((R / "wp2_asia_pacific.json").read_text())["cases"]["asia_pacific"]["refs"]
    return {"carec": c, "asia_pacific": a}


def pool_problem(case, t, thr):
    model, pair = pool_model(case)
    return PortfolioProblem(model, t, threshold=thr * 1e6 if thr else 400e6, pair=pair)


def chunked(fn, P, N, rng, chunk=2 ** 17):
    """Mean of N paths computed in chunks (memory); unbiased, equal in law to one batch of N."""
    tot, left = 0.0, N
    while left > 0:
        n = min(chunk, left)
        tot += n * fn(P, n, rng); left -= n
    return tot / N


def lean_mc(P, n, rng):
    from src.baselines import stronger as S
    return float(S._simulate(P, None, n, rng)["f"].mean())


def pool_fn(method, case, t, thr):
    P = pool_problem(case, t, thr)
    if method == "rdpo":
        cfg = final_config(t)
        return lambda N, rng: rdpo_estimate(P, N, rng, cfg)["estimate"]
    if method == "mc_paper":
        cfg = RDPOConfig(engine="fast", warm_start=False, ce=False, pg=False, qmc=False)
        return lambda N, rng: rdpo_estimate(P, N, rng, cfg)["estimate"]
    if method == "mc":
        return lambda N, rng: chunked(lean_mc, P, N, rng)
    if method == "twist":
        return lambda N, rng: chunked(lambda P_, n, r: float(twisted_est(P_, n, r)), P, N, rng, chunk=2 ** 18)
    raise KeyError(method)


def pool_methods(t):
    return ["mc", "twist"] if t in ("exceedance", "stop_loss") else ["mc"]


def timing_b():
    warnings.simplefilter("ignore")
    done = {r["id"] for r in read_jsonl("eqt_timing_b")}
    for case, t, thr in CELLS_B:
        cid = cell_id_b(case, t, thr)
        if cid in done:
            continue
        pool_model(case)
        fns = {m: pool_fn(m, case, t, thr) for m in ["rdpo", "mc_paper"] + pool_methods(t)}
        ref0 = ref_kernel()
        t_r = [timed(fns["rdpo"], B_POOL, r) for r in range(TIME_REPS_B)]
        t_mp = [timed(fns["mc_paper"], B_POOL, r) for r in range(2)]
        small, big = 2 ** 16, 2 ** 18
        ts = {m: med([timed(fns[m], small, r) for r in range(3)]) for m in pool_methods(t)}
        tb = {m: med([timed(fns[m], big, r) for r in range(3)]) for m in pool_methods(t)}
        fit = {m: fit_linear(small, ts[m], big, tb[m]) for m in pool_methods(t)}
        tr = med(t_r)
        neq, capped = {}, {}
        for m in pool_methods(t):
            neq[m], capped[m] = n_eq_from(tr, *fit[m])
        verify = {m: timed(fns[m], neq[m], 100) for m in pool_methods(t)}
        P = pool_problem(case, t, thr)
        ex_pair = None
        if t == "joint_exhaustion":
            ex_pair = med([(lambda t0: (P.exact_pair(), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(11)])
        rec = {"id": cid, "case": case, "target": t, "threshold_musd": thr, "budget": B_POOL, "t_rdpo": t_r, "t_rdpo_median": tr,
               "t_mc_paper_engine_65536": med(t_mp), "t_small": ts, "t_large": tb, "n_small": small, "n_large": big,
               "fit_a_b": {m: list(v) for m, v in fit.items()}, "n_eq": neq, "capped": capped, "t_at_n_eq": verify,
               "exact_pair_s": ex_pair, "ref_kernel_ms": [ref0, ref_kernel()], "code": code_hash()}
        append_jsonl("eqt_timing_b", rec)
        print(f"timing_b {cid}: rdpo {tr:.2f}s N_eq {neq}", file=sys.stderr, flush=True)


def _job_b(args):
    jid, case, t, thr, method, mode, N = args
    warnings.simplefilter("ignore")
    fn = pool_fn(method, case, t, thr)
    ests, secs = [], []
    for r in range(REPS_B):
        t0 = time.perf_counter(); ests.append(float(fn(N, np.random.default_rng(SEED_B[case] + r)))); secs.append(time.perf_counter() - t0)
    return {"id": jid, "cell": cell_id_b(case, t, thr), "method": method, "mode": mode, "N": N, "est": ests,
            "sec_mean_contended": float(np.mean(secs))}


def reps_b(procs=2):
    tim = {r["id"]: r for r in read_jsonl("eqt_timing_b")}
    for case in ("carec", "asia_pacific"):
        pool_model(case)                                   # calibrate once in the parent; workers fork from it
    jobs = []
    for case, t, thr in CELLS_B:
        tr = tim[cell_id_b(case, t, thr)]
        for m in pool_methods(t):
            jobs.append((f"{case}|{t}|{thr}|{m}|paths", case, t, thr, m, "paths", B_POOL))
        for m in pool_methods(t):
            jobs.append((f"{case}|{t}|{thr}|{m}|time", case, t, thr, m, "time", tr["n_eq"][m]))
        jobs.append((f"{case}|{t}|{thr}|rdpo|paths", case, t, thr, "rdpo", "paths", B_POOL))
    # longest first so the pool stays balanced
    jobs.sort(key=lambda j: -(0 if j[4] != "rdpo" else 1))
    run_jobs("eqt_reps_b", jobs, _job_b, procs)


# ================================================================================================
# Summary
# ================================================================================================
def rel_rmse(ests, mu):
    return summarise(ests, mu)["rel_rmse"]


def ratio(a, b):
    return None if b == 0 else a / b


def block_summary(rows, brief):
    """rows: per-cell dicts with rmse_paths / rmse_time (method -> rel rmse incl. rdpo)."""
    out = {}
    for tt in sorted({r["target"] for r in rows}):
        rs = [r for r in rows if r["target"] == tt]
        o = {"cells": len(rs)}
        for mode in ("paths", "time"):
            def best(r, group):
                v = {m: x for m, x in r[f"rmse_{mode}"].items() if m in group}
                return min(v.values()) if v else None
            allm = lambda r: [m for m in r[f"rmse_{mode}"] if m != "rdpo"]
            wins4 = [r["rmse_" + mode]["rdpo"] < best(r, brief) for r in rs]
            winsall = [r["rmse_" + mode]["rdpo"] < best(r, allm(r)) for r in rs]
            ratios = [ratio(best(r, allm(r)), r["rmse_" + mode]["rdpo"]) for r in rs]
            ratios4 = [ratio(best(r, brief), r["rmse_" + mode]["rdpo"]) for r in rs]
            winner = {}
            for r in rs:
                w = min(r[f"rmse_{mode}"], key=r[f"rmse_{mode}"].get); winner[w] = winner.get(w, 0) + 1
            o[mode] = {"rdpo_wins_vs_brief_methods": int(sum(wins4)), "rdpo_wins_vs_all_baselines": int(sum(winsall)),
                       "median_ratio_best_baseline_over_rdpo": med([x for x in ratios if x is not None]) if any(x is not None for x in ratios) else None,
                       "median_ratio_best_brief_over_rdpo": med([x for x in ratios4 if x is not None]) if any(x is not None for x in ratios4) else None,
                       "most_accurate_method_counts": winner}
        out[tt] = o
    return out


def summary():
    ta = {r["id"]: r for r in read_jsonl("eqt_timing_a")}
    ra = {}
    for r in read_jsonl("eqt_reps_a"):
        ra[(r["cell"], r["method"], r["mode"])] = r
    cells_a = []
    for sc, t in CELLS_A:
        cid = cell_id_a(sc, t)
        if cid not in ta or (cid, "rdpo", "paths") not in ra:
            continue
        tr = ta[cid]
        P, _ = adb_objects(sc, t)
        mu = P.exact_dense()
        row = {"scenario": sc, "target": t, "mu": mu, "reps": REPS_A, "seconds_rdpo_16384": tr["t16384"]["rdpo"],
               "seconds_baselines_16384": {m: tr["t16384"][m] for m in METHODS_A},
               "n_equal_time": tr["n_eq"], "capped_at_2^24": tr["capped"], "seconds_at_n_equal_time": tr["t_at_n_eq"],
               "exact_dense_enumeration_s": tr["exact_dense_s"], "exact_brief_model_s": tr["exact_brief_model_s"],
               "exact_pair_s": tr["exact_pair_s"]}
        row["rmse_paths"] = {"rdpo": rel_rmse(ra[(cid, "rdpo", "paths")]["est"], mu)}
        row["rmse_time"] = {"rdpo": row["rmse_paths"]["rdpo"]}
        for m in METHODS_A:
            if (cid, m, "paths") in ra: row["rmse_paths"][m] = rel_rmse(ra[(cid, m, "paths")]["est"], mu)
            if (cid, m, "time") in ra: row["rmse_time"][m] = rel_rmse(ra[(cid, m, "time")]["est"], mu)
        cells_a.append(row)
    tb = {r["id"]: r for r in read_jsonl("eqt_timing_b")}
    rb = {}
    for r in read_jsonl("eqt_reps_b"):
        rb[(r["cell"], r["method"], r["mode"])] = r
    refs = pool_refs()
    cells_b = []
    for case, t, thr in CELLS_B:
        cid = cell_id_b(case, t, thr)
        if cid not in tb or (cid, "rdpo", "paths") not in rb:
            continue
        tr = tb[cid]
        mu = refs[case][f"{t}|{thr}"]
        meths = pool_methods(t)
        row = {"case": case, "target": t, "threshold_musd": thr, "mu": mu, "reps": REPS_B,
               "seconds_rdpo_65536": tr["t_rdpo_median"], "seconds_mc_in_rdpo_engine_65536": tr["t_mc_paper_engine_65536"],
               "n_equal_time": tr["n_eq"], "capped_at_2^24": tr["capped"], "seconds_at_n_equal_time": tr["t_at_n_eq"],
               "exact_pair_s": tr["exact_pair_s"], "rmse_paths": {"rdpo": rel_rmse(rb[(cid, "rdpo", "paths")]["est"], mu)}}
        row["rmse_time"] = {"rdpo": row["rmse_paths"]["rdpo"]}
        for m in meths:
            if (cid, m, "paths") in rb: row["rmse_paths"][m] = rel_rmse(rb[(cid, m, "paths")]["est"], mu)
            if (cid, m, "time") in rb: row["rmse_time"][m] = rel_rmse(rb[(cid, m, "time")]["est"], mu)
        cells_b.append(row)
    cells_a = [r for r in cells_a if len(r["rmse_paths"]) > 1 and len(r["rmse_time"]) > 1]
    cells_b = [r for r in cells_b if len(r["rmse_paths"]) > 1 and len(r["rmse_time"]) > 1]
    for r in cells_a + cells_b:
        for mode in ("paths", "time"):
            best = min((v, m) for m, v in r[f"rmse_{mode}"].items() if m != "rdpo")
            r[f"best_baseline_{mode}"] = best[1]
            r[f"ratio_best_baseline_over_rdpo_{mode}"] = ratio(best[0], r[f"rmse_{mode}"]["rdpo"])
    # seconds per 1e4 paths
    def per1e4(timings, methods, key_small, n_small):
        out = {}
        for m in methods:
            sm = [r[key_small][m] / n_small * 1e4 for r in timings if m in r[key_small]]
            mg = [r["fit_a_b"][m][1] * 1e4 for r in timings if m in r["fit_a_b"]]
            if sm: out[m] = {"including_fixed_costs": med(sm), "marginal": med(mg)}
        return out
    sec_a = per1e4(list(ta.values()), METHODS_A, "t16384", B_ADB)
    sec_a["rdpo"] = {"including_training_16384": med([r["t16384"]["rdpo"] / B_ADB * 1e4 for r in ta.values()])}
    tbv = list(tb.values())
    sec_b = {m: {"marginal": med([r["fit_a_b"][m][1] * 1e4 for r in tbv if m in r["fit_a_b"]])} for m in ("mc", "twist")}
    sec_b["mc_in_rdpo_engine"] = {"including_fixed_costs": med([r["t_mc_paper_engine_65536"] / B_POOL * 1e4 for r in tbv])}
    sec_b["rdpo"] = {"including_training_65536": med([r["t_rdpo_median"] / B_POOL * 1e4 for r in tbv])}
    ex = {"exact_dense_enumeration_s_median": med([r["exact_dense_enumeration_s"] for r in cells_a]) if cells_a else None,
          "exact_brief_model_s_median": med([r["exact_brief_model_s"] for r in cells_a]) if cells_a else None,
          "exact_pair_s_median": med([r["exact_pair_s"] for r in cells_a if r["exact_pair_s"]]) if cells_a else None,
          "rdpo_seconds_16384_median": med([r["seconds_rdpo_16384"] for r in cells_a]) if cells_a else None}
    kern = [k for r in list(ta.values()) + tbv for k in r["ref_kernel_ms"]]
    out = {"protocol": {"A": {"budget": B_ADB, "reps": REPS_A, "timing_repeats": TIME_REPS_A, "seed_base": SEED_A,
                              "cap_paths": CAP, "methods": ["rdpo"] + METHODS_A},
                        "B": {"budget": B_POOL, "reps": REPS_B, "timing_repeats": TIME_REPS_B, "seeds": SEED_B, "cap_paths": CAP,
                              "methods": ["rdpo", "mc (lean sampler)", "twist"]},
                        "contention": {"reference_kernel_ms_min": min(kern), "reference_kernel_ms_max": max(kern),
                                       "note": "all timing passes single-process; methods timed back to back (interleaved repeats, medians); "
                                               "N_eq from a two-point linear fit and verified by timing each baseline at N_eq"},
                        "code_hashes": code_hash()},
           "A_cells": cells_a, "B_cells": cells_b,
           "exact_enumeration_A": ex,
           "summary": {"A_by_target": block_summary(cells_a, BRIEF4), "B_by_target": block_summary(cells_b, ["mc", "twist"]),
                       "seconds_per_1e4_paths_A": sec_a, "seconds_per_1e4_paths_B": sec_b}}
    for key, cells, brief in (("A_overall", cells_a, BRIEF4), ("B_overall", cells_b, ["mc", "twist"])):
        out["summary"][key] = {"cells": len(cells)}
        for mode in ("paths", "time"):
            if not cells: continue
            allm = lambda r: [m for m in r[f"rmse_{mode}"] if m != "rdpo"]
            out["summary"][key][mode] = {
                "rdpo_wins_vs_brief_methods": int(sum(r[f"rmse_{mode}"]["rdpo"] < min(r[f"rmse_{mode}"][m] for m in brief if m in r[f"rmse_{mode}"]) for r in cells)),
                "rdpo_wins_vs_all_baselines": int(sum(r[f"rmse_{mode}"]["rdpo"] < min(r[f"rmse_{mode}"][m] for m in allm(r)) for r in cells)),
                "median_ratio_best_baseline_over_rdpo": med([r[f"ratio_best_baseline_over_rdpo_{mode}"] for r in cells if r[f"ratio_best_baseline_over_rdpo_{mode}"] is not None])}
    (R / "equal_time.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"]["A_overall"]), json.dumps(out["summary"]["B_overall"]), file=sys.stderr)
    return out


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"timing_a": timing_a, "reps_a": reps_a, "timing_b": timing_b, "reps_b": reps_b, "summary": summary}
    for s in (steps if what == "all" else [what]):
        steps[s]()


if __name__ == "__main__":
    main()
