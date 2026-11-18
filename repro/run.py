"""Reproduction runner for the RDPO experiments (adb-repro/ is a separate clone; the original repo is untouched).

    python repro/run.py <phase> [--cpus 14] [--only name1,name2] [--dry] [--force] --i-understand-this-wipes-results

Runs every job of <phase> in dependency order, in parallel up to a CPU budget, inside adb-repro/ with the pinned
.venv. Each run appends one record to repro/runs.jsonl and writes its output to repro/logs/<name>.log.
A job runs only when its deps have a successful record in runs.jsonl (from this or an earlier phase).
"""
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "adb-repro"
PY = str(HERE.parent / ".venv" / "Scripts" / "python.exe")
RUNS, LOGS = HERE / "runs.jsonl", HERE / "logs"
ALL = 16  # cpu cost meaning "run alone" (timing claims must not share the machine)


def M(mod, *args):
    return [PY, "-m", mod, *args]


# name, phase, argv, deps, cpu cost (4 = driver with a 4-process pool), soft (failure does not block dependents)
JOBS = [
    ("exact", 1, M("src.model.exact"), [], 1, False),
    ("tensor", 1, M("src.tensor.scaling"), [], 1, False),
    ("eq_check", 1, M("src.realdata.eq_check"), [], 1, False),
    ("loss_backtest", 1, M("src.realdata.loss_backtest"), [], 1, False),
    ("rain_check", 1, M("src.realdata.rain_check"), ["loss_backtest"], 1, False),
    ("projected", 1, M("src.rdpo.projected"), [], 1, False),
    ("structured", 1, M("src.quantum.run_structured"), ["tensor"], 1, False),
    ("twisted", 1, M("src.quantum.run_twisted"), ["structured"], 1, False),
    ("pytest", 1, [PY, "-m", "pytest", "-q", "tests/"], ["loss_backtest", "rain_check"], 1, False),

    ("baselines", 2, M("src.baselines.run_baselines"), [], 1, False),
    ("e1", 2, M("src.rdpo.experiments", "e1"), ["baselines"], 1, False),
    ("v2_bench", 2, M("src.rdpo.experiments_v2", "bench"), ["e1"], 4, False),
    ("v2_transfer", 2, M("src.rdpo.experiments_v2", "transfer"), [], 4, False),
    ("stronger", 2, M("src.baselines.stronger"), [], 1, False),
    ("elt_demo", 2, M("src.elt.demo"), ["rain_check"], 4, False),
    ("param_uncertainty", 2, M("src.realdata.param_uncertainty"), ["loss_backtest", "rain_check"], 1, False),
    ("central_asia_pool", 2, M("src.realdata.central_asia_pool"), [], 4, False),
    ("carec_pool", 2, M("src.realdata.carec_pool"), [], 4, False),
    ("nisq", 2, M("src.quantum.run_nisq"), [], 1, False),
    ("v2_throughput", 2, M("src.rdpo.experiments_v2", "throughput"), [], ALL, False),
    ("timing", 2, M("src.rdpo.timing"), [], ALL, False),

    ("e2", 3, M("src.rdpo.experiments", "e2"), ["tensor"], 1, False),
    ("e3", 3, M("src.rdpo.experiments", "e3"), ["tensor"], 1, False),
    # ponytail: v2 scale reads rdpo_e2.json only after all its jobs, so it computes alongside e2 and is expected to
    # stop at that read; the second run resumes from raw/scale.jsonl (no recompute) and writes the table.
    ("v2_scale_jobs", 3, M("src.rdpo.experiments_v2", "scale"), ["tensor"], 4, True),
    ("v2_scale", 3, M("src.rdpo.experiments_v2", "scale"), ["v2_scale_jobs", "e2"], 4, False),
    ("asia_pacific_pool", 3, M("src.realdata.asia_pacific_pool"), ["rain_check"], 4, False),
    ("quantum_advantage", 3, M("src.quantum.advantage"), ["e2", "twisted"], 1, False),
    # no CLI entry point exists for scaling_laws() (defined after __main__), so it is called directly
    ("quantum_scaling", 3, [PY, "-c", "from src.quantum.advantage import scaling_laws; scaling_laws()"],
     ["quantum_advantage"], 1, False),
    # phase 5: our supplementary experiments (src/ untouched)
    ("i25", 5, [PY, str(HERE / "i25_rho2.py"), "--procs", "14"], [], ALL, False),
    # phase 6: computation add-ons in ../addons (import src/ read-only, write only addons/results/)
    *[(f"addon_{s}", 6, [PY, str(HERE.parent / "addons" / f"{s}.py")], deps, cpu, False)
      for s, deps, cpu in [("rotations", [], 1), ("q2_ae", [], 1), ("combine", ["addon_rotations", "addon_q2_ae"], 1)]],
    # phase 7: quality add-ons (methods borrowed from the QContagion CDS work; same read-only rules)
    *[(f"quality_{s}", 7, [PY, str(HERE.parent / "addons" / f"{s}.py")], [], 1, False)
      for s in ("eq_cluster", "rates_unc")],
    # phase 9: climate layers on the the main author's two-country model (exact, outside RDPO; src/ read-only)
    ("climate_cc", 9, [PY, str(HERE.parent / "addons" / "climate_cc.py")], [], 1, False),
    # phase 10 (NEX-GDDP ensemble, daily POT) and phase 12 M2 (rain tail dependence): tested and withdrawn, see archive/
    # phase 11: RDPO value experiments (E3 sensitivities, E2 nested uncertainty + transfer; E1 rain event set withdrawn)
    *[(s, 11, [PY, str(HERE.parent / "addons" / f"{s}.py")], [], ALL, False) for s in ("sensitivity", "nested_uq")],
    # phase 13 (exact tail, Rao–Blackwellised RDPO): tested and withdrawn, see archive/
    # phase 14: global sensitivity (Sobol) + value of information on the the main author's two-country bond (exact)
    ("voi", 14, [PY, str(HERE.parent / "addons" / "voi.py")], [], ALL, False),
    # phase 15 (quantum-stage loading variants): tested and withdrawn, see archive/
    # phase 16 (quantum nested expectation on a dense pool): tested and withdrawn, see archive/
    # phase 17: model-risk engine on the the main author's CAREC pool (posterior predictive, P(> main author), Sobol, value of information)
    ("model_risk", 17, [PY, str(HERE.parent / "addons" / "model_risk.py")], [], ALL, False),
    # phase 18 (earthquake data upgrade): tested and withdrawn, see archive/
    # phase 19 (ambiguity premium -> spread): tested and withdrawn, see archive/
    # phase 20: risk twin of the Asia-Pacific pool (yearly posterior update 2017-2025, answers by reweighting fixed paths;
    # ap_risk.py is its engine). Needs the ComCat catalogue from fetch_usgs.asia_pacific() in DATA.
    ("twin", 20, [PY, str(HERE.parent / "addons" / "twin.py")], [], ALL, False),
    # phase 21: expected-loss interval -> spread band with three published market curves (a calculation)
    ("spread_band", 21, [PY, str(HERE.parent / "addons" / "spread_band.py")], ["quality_rates_unc"], 1, False),
    # phase 22: event-response mode of the twin (Omori-type forecast after an event, out-of-sample test, the 3 Nov 2026
    # border earthquake). Needs carec_m45_1973_2025.csv from fetch_usgs.extended() in DATA.
    ("event_response", 22, [PY, str(HERE.parent / "addons" / "event_response.py")], [], 1, False),
    # phase 23: live mark of the two notes (remaining-term loss, indicative price, what each add-on is worth in points)
    ("live_mark", 23, [PY, str(HERE.parent / "addons" / "live_mark.py")], ["quality_rates_unc", "event_response"], 1, False),
    # phases 24-27: why RDPO ties twisting at equal CPU and what changes it (two-country model, exact references).
    # 24 exact ceiling of a state-dependent sampler; 25 measured variance and cost per path; 26 a faster rollout for the
    # same policy (same paths); 27 the six exchanges between RDPO and the twin
    ("ceiling", 24, [PY, str(HERE.parent / "addons" / "ceiling.py")], [], 1, False),
    ("rdpo_gap", 25, [PY, str(HERE.parent / "addons" / "rdpo_gap.py")], ["ceiling"], 1, False),
    ("rdpo_engine", 26, [PY, str(HERE.parent / "addons" / "rdpo_engine.py")], ["rdpo_gap"], 1, False),
    ("dreamwake", 27, [PY, str(HERE.parent / "addons" / "dreamwake.py")], ["rdpo_engine", "quality_rates_unc"], 1, False),
]
BY = {j[0]: j for j in JOBS}


def done_ok():
    if not RUNS.exists():
        return set()
    return {r["name"] for r in map(json.loads, RUNS.read_text().splitlines()) if r["exit"] == 0 or r.get("soft")}


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()


def run_one(job):
    name, phase, argv, deps, cpu, soft = job
    env = dict(os.environ, PYTHONPATH=".", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8")
    log = LOGS / f"{name}.log"
    t0, start = time.perf_counter(), datetime.now().isoformat(timespec="seconds")
    with open(log, "w", encoding="utf-8") as fh:
        code = subprocess.run(argv, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    rec = {"name": name, "phase": phase, "cmd": " ".join(argv[1:]), "start": start,
           "end": datetime.now().isoformat(timespec="seconds"), "seconds": round(time.perf_counter() - t0, 1),
           "exit": code, "soft": soft, "cpu_cost": cpu, "log": str(log.relative_to(HERE)), "commit": git("rev-parse", "HEAD"),
           "data_changed": git("status", "--porcelain", "--", "data", "src", "tests")}
    with open(RUNS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{rec['end']}] {name}: exit {code} in {rec['seconds']}s", flush=True)
    return rec


def env_record():
    out = subprocess.run([PY, "-c", "import sys,numpy,scipy,matplotlib,pytest;print(sys.version.split()[0],"
                          "numpy.__version__,scipy.__version__,matplotlib.__version__,pytest.__version__)"],
                         capture_output=True, text=True).stdout.split()
    return dict(zip(["python", "numpy", "scipy", "matplotlib", "pytest"], out),
                platform=platform.platform(), cpu=platform.processor(), cores=os.cpu_count(),
                commit=git("rev-parse", "HEAD"), recorded=datetime.now().isoformat(timespec="seconds"))


def main():
    args = sys.argv[1:]
    phase = int(args[0])
    cpus = int(args[args.index("--cpus") + 1]) if "--cpus" in args else 14
    if "--dry" not in args and "--i-understand-this-wipes-results" not in args:
        sys.exit("refusing to run: run.py re-runs the experiments in ../adb-repro and wipes its results/. "
                 "Pass --i-understand-this-wipes-results to proceed (or --dry to list jobs).")
    only = set(args[args.index("--only") + 1].split(",")) if "--only" in args else None
    todo = [j for j in JOBS if j[1] == phase and (only is None or j[0] in only)]
    ok = done_ok()
    if "--force" not in args:                    # --force: rerun jobs that already have a successful record
        todo = [j for j in todo if j[0] not in ok]
    if "--dry" in args:
        for j in todo:
            print(j[0], "deps:", j[3], "cpu:", j[4], "|", " ".join(j[2][1:]))
        return
    LOGS.mkdir(exist_ok=True)
    (HERE / f"env_phase{phase}.json").write_text(json.dumps(env_record(), indent=2))
    running, failed = {}, set()
    with ThreadPoolExecutor(max_workers=len(todo) or 1) as ex:
        while todo or running:
            used = sum(BY[n][4] for n in running.values())
            for j in list(todo):
                if any(d in failed for d in j[3]):
                    todo.remove(j); failed.add(j[0]); print(f"skip {j[0]}: a dependency failed", flush=True)
                elif all(d in ok for d in j[3]) and (not running or used + j[4] <= cpus) \
                        and not any(BY[n][4] >= ALL for n in running.values()):
                    running[ex.submit(run_one, j)] = j[0]; todo.remove(j); used += j[4]
            if not running:
                break
            fin, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in fin:
                rec = f.result(); running.pop(f)
                (ok if rec["exit"] == 0 or rec["soft"] else failed).add(rec["name"])
    for j in todo:
        print(f"not run {j[0]}: deps {j[3]} not satisfied", flush=True)


if __name__ == "__main__":
    main()
