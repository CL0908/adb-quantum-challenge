"""I25 supplementary experiment: how well is rho^2 = B * rel. variance of RDPO pinned down for the ring-28 (and
ring-20) exceedance cells that feed the quantum comparison (src/quantum/advantage.py:78-80)?

Runs the authors' own rdpo_estimate (src/ untouched) for many fresh seeds, pools with the per-replication data that
already exists (raw/scale.jsonl of both runs) and the E2 summaries, and states the implied n = 28 / n = 20 quantum
numbers using the authors' own timing (tau_C, quantum seconds) so only rho^2 changes.

    cd adb-repro && PYTHONPATH=. python ../repro/i25_rho2.py [--procs 14]   # run (resumes) + analyse
    ... i25_rho2.py --analyse | --selftest
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "adb-repro"
OUT = HERE / "i25"
RAW = OUT / "raw.jsonl"
B, EPS = 65536, 0.1
OUTLIER = 0.05   # |relative error| above this counts as an outlier replication
# authors' E2 seed for ring-28 exceedance: seed 2000 + 1000 n + 100 len(rows) + r, the row sat at index 9
E2_AUTHOR_SEED0 = 2000 + 1000 * 28 + 100 * 9

# tag, topology, n, config, first seed, reps  (fresh seed bases do not overlap any seed used by the repo's drivers)
PLAN = [("A_author_seeds", "ring", 28, "e2", E2_AUTHOR_SEED0, 6),
        ("B_e2", "ring", 28, "e2", 600000, 120),
        ("C_v2final", "ring", 28, "v2-final", 700000, 120),
        ("D_e2", "ring", 20, "e2", 800000, 60),
        ("D_v2final", "ring", 20, "v2-final", 900000, 60)]


# ------------------------------------------------------------------------------------------------ run
def _fit(job):
    tag, kind, n, cfg_name, seed = job
    from src.rdpo.experiments_v2 import final_config
    from src.rdpo.problem import PortfolioProblem
    from src.rdpo.train import RDPOConfig, rdpo_estimate
    from src.tensor.synthetic import SyntheticModel, topology
    cfg = RDPOConfig() if cfg_name == "e2" else final_config("exceedance")   # e2 = exactly what E2 ran
    P = PortfolioProblem(SyntheticModel(n, topology(kind, n)), "exceedance")
    t0 = time.perf_counter()
    x = rdpo_estimate(P, B, np.random.default_rng(seed), cfg)
    return {"id": f"{tag}-{seed}", "tag": tag, "key": [kind, n, "exceedance", cfg_name], "seed": seed,
            "est": x["estimate"], "ci": list(x["ci"]), "sec": time.perf_counter() - t0}


def run(procs):
    from multiprocessing import Pool
    OUT.mkdir(exist_ok=True)
    done = {json.loads(l)["id"] for l in RAW.read_text().splitlines()} if RAW.exists() else set()
    jobs = [(tag, kind, n, cfg, s0 + r) for tag, kind, n, cfg, s0, reps in PLAN for r in range(reps)]
    jobs = sorted((j for j in jobs if f"{j[0]}-{j[4]}" not in done), key=lambda j: (-j[2], j[3] != "e2"))  # slow first
    print(f"i25: {len(done)} done, {len(jobs)} to run", flush=True)
    with Pool(procs) as pool, RAW.open("a") as fh:
        for i, rec in enumerate(pool.imap_unordered(_fit, jobs), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(jobs):
                print(f"i25: {i}/{len(jobs)}", flush=True)


# ------------------------------------------------------------------------------------------------ statistics
def rel_rmse(err):
    return float(np.sqrt(np.mean(np.square(err))))


def boot_ci(err, rng, nboot=20000):
    # ponytail: percentile bootstrap; with rare large outliers it can under-cover. The 6-rep simulation below is the
    # direct answer to "are 4.2% and 1.0% the same distribution", the CI is context.
    err = np.asarray(err)
    r = np.sqrt((err[rng.integers(0, len(err), (nboot, len(err)))] ** 2).mean(1))
    return [float(v) for v in np.percentile(r, [2.5, 97.5])]


def pooled(err, summaries):
    """rel RMSE over per-rep errors plus (reps, rel_rmse) summaries: sum of squares adds up."""
    n = len(err) + sum(k for k, _ in summaries)
    return math.sqrt((float(np.sum(np.square(err))) + sum(k * r * r for k, r in summaries)) / n), n


def s_star(rho2, Tq, t_path, eps=EPS):
    """required hardware speed-up, exactly as advantage_map: Tq / Tc, Tc = rho2 / eps^2 * t_path."""
    return Tq / (rho2 / eps ** 2 * t_path)


def six_rep(err, rng, k=6, nsim=200000):
    err = np.asarray(err)
    return np.sqrt((err[rng.integers(0, len(err), (nsim, k))] ** 2).mean(1))


# ------------------------------------------------------------------------------------------------ analyse
def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def analyse():
    rng = np.random.default_rng(25)
    e2 = {t: {(r["topology"], r["n"], r["target"]): r for r in _json(REPO / t / "rdpo_e2.json")["rows"]}
          for t in ("results_declared", "results")}
    qa = {(r["topology"], r["n"]): r for r in _json(REPO / "results_declared" / "quantum_advantage.json")["advantage"]}
    scale = {t: [json.loads(l) for l in (REPO / t / "raw" / "scale.jsonl").read_text().splitlines()]
             for t in ("results_declared", "results")}
    new = [json.loads(l) for l in RAW.read_text().splitlines()] if RAW.exists() else []
    out = {"budget": B, "eps_rel": EPS, "outlier_threshold": OUTLIER, "plan": PLAN, "cells": [], "author_seeds": None}

    for n in (28, 20):
        key = ("ring", n, "exceedance")
        mu = e2["results_declared"][key]["mu"]
        assert abs(e2["results"][key]["mu"] / mu - 1) < 1e-9
        q = qa[("ring", n)]
        t_path = q["classical_seconds"] / q["classical_paths"]
        for fam, scale_variant in (("e2", "v1"), ("v2-final", "v2-final")):
            mine = [r for r in new if r["key"] == ["ring", n, "exceedance", fam] and not r["tag"].startswith("A_")]
            if not mine:
                continue
            err_new = np.array([r["est"] for r in mine]) / mu - 1
            sources = [{"source": f"补充实验（{fam} 配置，新种子）", "reps": len(err_new), "rel_rmse": rel_rmse(err_new),
                        "per_rep": True}]
            err_all = [err_new]
            for t, who in (("results_declared", "作者"), ("results", "我们")):
                xs = [r["est"] for r in scale[t] if r["key"] == ["ring", n, "exceedance", scale_variant]]
                e = np.array(xs) / mu - 1
                err_all.append(e)
                sources.append({"source": f"v2 scale {scale_variant}（{who}）", "reps": len(e), "rel_rmse": rel_rmse(e),
                                "per_rep": True})
            summ = []
            if fam == "e2":
                for t, who in (("results_declared", "作者"), ("results", "我们")):
                    s = e2[t][key]["rdpo"]
                    summ.append((s["reps"], s["rel_rmse"]))
                    sources.append({"source": f"E2（{who}，仅汇总）", "reps": s["reps"], "rel_rmse": s["rel_rmse"],
                                    "per_rep": False})
            err_rep = np.concatenate(err_all)
            pool_all, n_all = pooled(err_rep, summ)
            sims = six_rep(err_rep, rng)
            ae = np.sort(np.abs(err_rep))[::-1]
            cell = {"n": n, "config": fam, "mu": mu, "sources": sources,
                    "new": {"reps": len(err_new), "rel_rmse": rel_rmse(err_new), "ci": boot_ci(err_new, rng),
                            "ci_coverage": float(np.mean([lo <= mu <= hi for lo, hi in (r["ci"] for r in mine)])),
                            "outlier_share": float(np.mean(np.abs(err_new) > OUTLIER)),
                            "max_abs_err": float(np.abs(err_new).max()),
                            "median_sec": float(np.median([r["sec"] for r in mine]))},
                    "pooled_per_rep": {"reps": len(err_rep), "rel_rmse": rel_rmse(err_rep), "ci": boot_ci(err_rep, rng),
                                       "outlier_share": float(np.mean(ae > OUTLIER)), "top_abs_err": ae[:5].tolist(),
                                       "rel_rmse_without_outliers": rel_rmse(err_rep[np.abs(err_rep) <= OUTLIER])},
                    "pooled_all": {"reps": n_all, "rel_rmse": pool_all},
                    "six_rep_sim": {"p_ge_author_e2": float(np.mean(sims >= e2["results_declared"][key]["rdpo"]["rel_rmse"])),
                                    "p_le_ours_e2": float(np.mean(sims <= e2["results"][key]["rdpo"]["rel_rmse"])),
                                    "author_e2": e2["results_declared"][key]["rdpo"]["rel_rmse"],
                                    "ours_e2": e2["results"][key]["rdpo"]["rel_rmse"],
                                    "q05_q50_q95": [float(v) for v in np.percentile(sims, [5, 50, 95])]},
                    "quantum": {"declared_rho2": q["classical_rho2"], "declared_s_star": q["required_speedup"],
                                "Tq_s": q["quantum_seconds"], "t_path_s": t_path,
                                "mc_ratio": (1 - q["mu"]) / q["mu"]}}
            rr = cell["pooled_per_rep"]
            for name, r in (("point", rr["rel_rmse"]), ("ci_lo", rr["ci"][0]), ("ci_hi", rr["ci"][1])):
                rho2 = r * r * B
                cell["quantum"][name] = {"rho2": rho2, "s_star": s_star(rho2, q["quantum_seconds"], t_path),
                                         "mc_overstatement": cell["quantum"]["mc_ratio"] / rho2,
                                         "classical_paths_10pct": rho2 / EPS ** 2}
            out["cells"].append(cell)

    a = [r for r in new if r["tag"].startswith("A_")]
    if a:
        mu = e2["results_declared"][("ring", 28, "exceedance")]["mu"]
        err = np.array([r["est"] for r in sorted(a, key=lambda r: r["seed"])]) / mu - 1
        out["author_seeds"] = {"seeds": sorted(r["seed"] for r in a), "rel_err": err.tolist(), "rel_rmse": rel_rmse(err),
                               "author_e2_rel_rmse": e2["results_declared"][("ring", 28, "exceedance")]["rdpo"]["rel_rmse"]}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def report(out):
    if out["author_seeds"]:
        s = out["author_seeds"]
        print(f"A author seeds {s['seeds'][0]}..: rel RMSE {s['rel_rmse']:.4f} (author E2 {s['author_e2_rel_rmse']:.4f}); "
              f"errs % {[round(100 * e, 2) for e in s['rel_err']]}")
    for c in out["cells"]:
        nw, pr, sx, qq = c["new"], c["pooled_per_rep"], c["six_rep_sim"], c["quantum"]
        print(f"ring-{c['n']} {c['config']:8s} new {nw['reps']} reps rel {nw['rel_rmse']:.4f} CI {nw['ci'][0]:.4f}-{nw['ci'][1]:.4f} "
              f"out>{OUTLIER:.0%} {nw['outlier_share']:.1%} max {nw['max_abs_err']:.3f} cov {nw['ci_coverage']:.2f} {nw['median_sec']:.0f}s | "
              f"pooled {pr['reps']} rel {pr['rel_rmse']:.4f} CI {pr['ci'][0]:.4f}-{pr['ci'][1]:.4f} (no-outl {pr['rel_rmse_without_outliers']:.4f}) "
              f"all {c['pooled_all']['reps']} {c['pooled_all']['rel_rmse']:.4f} | 6-rep P(>=auth)={sx['p_ge_author_e2']:.3f} "
              f"P(<=ours)={sx['p_le_ours_e2']:.3f} | rho2 {qq['point']['rho2']:.1f} [{qq['ci_lo']['rho2']:.1f},{qq['ci_hi']['rho2']:.1f}] "
              f"(decl {qq['declared_rho2']:.1f}) s* {qq['point']['s_star']:.3g} [{qq['ci_hi']['s_star']:.3g},{qq['ci_lo']['s_star']:.3g}] "
              f"(decl {qq['declared_s_star']:.3g})")


# ------------------------------------------------------------------------------------------------ self-check
def selftest():
    rng = np.random.default_rng(0)
    assert math.isclose(rel_rmse([0.03, -0.04]), math.sqrt(0.00125))
    e = rng.normal(0, 0.02, 50)
    r, k = pooled(e[:30], [(20, rel_rmse(e[30:]))])
    assert k == 50 and math.isclose(r, rel_rmse(e))
    lo, hi = boot_ci(e, rng)
    assert lo < rel_rmse(e) < hi
    rows = _json(REPO / "results_declared" / "rdpo_e2.json")["rows"]
    idx = next(i for i, x in enumerate(rows) if (x["topology"], x["n"], x["target"]) == ("ring", 28, "exceedance"))
    assert E2_AUTHOR_SEED0 == 2000 + 1000 * 28 + 100 * idx
    for q in _json(REPO / "results_declared" / "quantum_advantage.json")["advantage"]:   # formula = advantage_map
        t_path = q["classical_seconds"] / q["classical_paths"]
        assert math.isclose(s_star(q["classical_rho2"], q["quantum_seconds"], t_path), q["required_speedup"], rel_tol=1e-9)
    print("selftest ok")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
    else:
        if "--analyse" not in a:
            run(int(a[a.index("--procs") + 1]) if "--procs" in a else 14)
        report(analyse())
