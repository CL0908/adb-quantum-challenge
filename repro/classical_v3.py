"""Turn results/classical_v3_raw.jsonl (repro/classical_v3_bench.py) into results/quantum_advantage_v3.json.

The quantum side is src/quantum/advantage.py unchanged: advantage_map() is fed the new engine's measured RDPO rows (same
fields as results/rdpo_e2.json: rel_rmse over the replications, seconds_per_fit with training charged) and scaling_laws()
re-derives the crossover table from them. rho2 = rel_rmse^2 * B, t_path = seconds_per_fit / B, T_C = rho2 / eps^2 * t_path,
s* = T_Q / T_C, eps = 10% -- exactly the paper's formulas. The old engine run back-to-back on the same machine is processed
the same way, so the same-machine ratio is explicit. results/quantum_advantage.json (the earlier file) is kept.

    PYTHONPATH=. python repro/classical_v3.py
"""
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.quantum.advantage import advantage_map, scaling_laws
from src.rdpo.experiments import summarise

R = Path("results")
B = 65536


def rows_for(raw, eng):
    by = defaultdict(list)
    for r in raw:
        by[(r["topology"], r["n"])].append(r)
    out = []
    for (kind, n), rs in by.items():
        mu = rs[0]["mu"]
        s = summarise([r[eng]["est"] for r in rs], mu, [r[eng]["ci"] for r in rs])
        out.append({"topology": kind, "n": n, "target": "exceedance", "mu": mu, "budget": B, "rdpo": s,
                    "seconds_per_fit": float(np.mean([r[eng]["wall"] for r in rs])),
                    "cpu_seconds_per_fit": float(np.mean([r[eng]["cpu"] for r in rs]))})
    return out


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def main():
    raw = [json.loads(l) for l in (R / "classical_v3_raw.jsonl").read_text().splitlines()]
    e_old, e_new = rows_for(raw, "old"), rows_for(raw, "new")
    adv_old, adv_new = advantage_map(e2_rows=e_old), advantage_map(e2_rows=e_new)
    paper = {(r["topology"], r["n"]): r for r in json.loads((R / "quantum_advantage.json").read_text())["advantage"]}
    paper_e2 = {(r["topology"], r["n"]): r for r in json.loads((R / "rdpo_e2.json").read_text())["rows"] if r["target"] == "exceedance"}
    key = lambda r: (r["topology"], r["n"])
    old = {key(r): r for r in adv_old}
    rows = []
    for r in adv_new:
        k = key(r)
        o, p = old[k], paper[k]
        e_o = next(x for x in e_old if key(x) == k); e_n = next(x for x in e_new if key(x) == k)
        rows.append({**r, "reps": e_n["rdpo"]["reps"], "rel_rmse": e_n["rdpo"]["rel_rmse"], "ci_coverage": e_n["rdpo"]["ci_coverage"],
                     "seconds_per_fit": e_n["seconds_per_fit"], "t_path": e_n["seconds_per_fit"] / B,
                     "old_engine_same_machine": {"rel_rmse": e_o["rdpo"]["rel_rmse"], "ci_coverage": e_o["rdpo"]["ci_coverage"],
                                                 "seconds_per_fit": e_o["seconds_per_fit"], "classical_rho2": o["classical_rho2"],
                                                 "classical_seconds": o["classical_seconds"], "required_speedup": o["required_speedup"]},
                     "paper_old": {"rel_rmse": paper_e2[k]["rdpo"]["rel_rmse"], "seconds_per_fit": paper_e2[k]["seconds_per_fit"],
                                   "classical_rho2": p["classical_rho2"], "classical_seconds": p["classical_seconds"],
                                   "required_speedup": p["required_speedup"]},
                     "time_ratio_same_machine": e_o["seconds_per_fit"] / e_n["seconds_per_fit"],
                     "rho2_ratio_new_over_old_same_machine": r["classical_rho2"] / o["classical_rho2"],
                     "s_star_engine_speed_only": o["required_speedup"] * e_o["seconds_per_fit"] / e_n["seconds_per_fit"],   # old rho2, new time per path
                     "s_star_ratio_old_over_new": o["required_speedup"] / r["required_speedup"]})
    scaling = scaling_laws(adv=adv_new, out_file=False)
    tot = lambda e: sum(x["seconds_per_fit"] * x["rdpo"]["reps"] for x in e)
    lr = np.log([x["rho2_ratio_new_over_old_same_machine"] for x in rows])
    summ = {"speed_ratio_total_time": tot(e_old) / tot(e_new),
            "speed_ratio_range": [min(x["time_ratio_same_machine"] for x in rows), max(x["time_ratio_same_machine"] for x in rows)],
            "rho2_ratio_geomean": float(np.exp(lr.mean())), "rho2_ratio_max": float(np.exp(lr.max())),
            "s_star_ratio_range": [min(x["s_star_ratio_old_over_new"] for x in rows), max(x["s_star_ratio_old_over_new"] for x in rows)],
            "s_star_28_old_paper": paper[("ring", 28)]["required_speedup"], "s_star_28_old_same_machine": old[("ring", 28)]["required_speedup"],
            "s_star_28_speed_only": next(r for r in rows if key(r) == ("ring", 28))["s_star_engine_speed_only"],
            "note": "rho2 of ring-20 and ring-28 rests on 6 replications (as in the paper's E2) and is outlier-dominated (repro/i25); s_star_engine_speed_only keeps the old engine's rho2 and changes only the time per path",
            "s_star_28_new": next(r for r in rows if key(r) == ("ring", 28))["required_speedup"]}
    meta = {"machine": {"cpu": sh("bash", "-c", "lscpu | grep 'Model name' | sed 's/.*: *//'"), "logical_cpus": os.cpu_count(),
                        "platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__},
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "timing": "wall-clock perf_counter of one full rdpo_estimate (training charged), B = 65,536, one process, old and new alternating per replication with the same seed",
            "engines": {"old": "RDPOConfig() default, engine v1 (the engine of results/rdpo_e2.json and the paper)",
                        "new": "RDPOConfig(engine='fast') = FastPolicy (same policy) with addons/rdpo_engine.rollout for every draw without gradient (Xu Zi Yu, PR #3-#4)"},
            "repo_commit": sh("git", "rev-parse", "HEAD"), "git_blobs": {f: sh("git", "hash-object", f) for f in ("addons/rdpo_engine.py", "src/rdpo/fast.py", "src/rdpo/train.py", "src/quantum/advantage.py")},
            "seeds": "2000 + 1000*n + 300*case_index + replication (same seed for old and new); reps 20 (n<=4) / 10 (n<=16) / 6 (else)",
            "eps_rel": 0.1, "budget": B, "quantum_side": "src/quantum/advantage.py unchanged (resource model, IQAE, surface code)",
            "caveat": "classical times are this machine's; quantum seconds are the surface-code model's and do not depend on the machine"}
    out = {"meta": meta, "summary": summ, "advantage": rows, "scaling": scaling}
    (R / "quantum_advantage_v3.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(summ, indent=1))
    for r in rows:
        print(f"{r['topology']:8s} {r['n']:2d} time ratio {r['time_ratio_same_machine']:5.1f} | rho2 old {r['old_engine_same_machine']['classical_rho2']:.3g} new {r['classical_rho2']:.3g} paper {r['paper_old']['classical_rho2']:.3g} | "
              f"s* paper {r['paper_old']['required_speedup']:.3g} old-here {r['old_engine_same_machine']['required_speedup']:.3g} new {r['required_speedup']:.3g} | pL {r['pL_required']:.2e}")


if __name__ == "__main__":
    main()
