"""Classical reference for the quantum crossover, new engine vs the engine the paper used, timed back-to-back on one core.

Same settings, budget and protocol as src/rdpo/experiments.py:e2 (B = 65,536, 20/10/6 replications, training charged inside the
timed fit). Old engine = RDPOConfig() default (v1 policy). New engine = the table-driven FastPolicy of the same policy with
addons/rdpo_engine.py's rollout for every draw that needs no gradient (Xu Zi Yu, PR #3-#4). The ring-16/20/28 sequences were later extended (arguments, see main) to stabilise rho2. Each replication runs old then new
with the same seed. Output: results/classical_v3_raw.jsonl (resumable). Run with OMP/MKL/OPENBLAS_NUM_THREADS=1.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:addons python repro/classical_v3_bench.py
"""
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "addons"))
import rdpo_engine as E
import src.rdpo.fast as F
from src.rdpo.problem import PortfolioProblem
from src.rdpo.train import RDPOConfig, rdpo_estimate
from src.tensor.synthetic import SyntheticModel, topology

R = Path("results")
RAW = R / "classical_v3_raw.jsonl"
B, SEED = 65536, 2000
CASES = [("ring", n) for n in (2, 8, 16, 20, 28)] + [("complete", n) for n in (4, 6, 7)]


class NewPolicy(F.FastPolicy):
    """Same policy and parameters as FastPolicy; draws without gradient go through the state-grouped rollout."""
    def rollout(self, N, rng, record=False, grad=None, U=None):
        if grad is not None:
            return super().rollout(N, rng, record=record, grad=grad, U=U)
        return E.rollout(self, N, rng, record=record, U=U)


def fit(P, engine, seed):
    cfg = RDPOConfig() if engine == "old" else replace(RDPOConfig(), engine="fast")
    orig = F.FastPolicy
    if engine == "new":
        F.FastPolicy = NewPolicy
    try:
        t0, c0 = time.perf_counter(), time.process_time()
        x = rdpo_estimate(P, B, np.random.default_rng(seed), cfg)
        return x, time.perf_counter() - t0, time.process_time() - c0
    finally:
        F.FastPolicy = orig


def main(only=None, extra=None):
    """extra: {(kind, n): total replications}; extends the sequence with the same seed formula (seed = SEED + 1000 n + 300 case + rep)."""
    tens = {(r["topology"], r["n"]): r for r in json.loads((R / "tensor.json").read_text())["B_scaling_n"]
            if "targets" in r and r["tol"] == (1e-6 if (r["topology"] == "complete" and r["n"] == 7) else 1e-9)}
    done = {json.loads(l)["id"] for l in RAW.read_text().splitlines()} if RAW.exists() else set()
    for ci, (kind, n) in enumerate(CASES):
        if only and (kind, n) not in only:
            continue
        model = SyntheticModel(n, topology(kind, n))
        mu = tens[(kind, n)]["targets"]["exceedance_prob"]
        P = PortfolioProblem(model, "exceedance")
        reps = (extra or {}).get((kind, n)) or (20 if n <= 4 else (10 if n <= 16 else 6))
        for r in range(reps):
            seed = SEED + 1000 * n + 100 * (3 * ci) + r
            rid = f"{kind}-{n}-{r}"
            if rid in done:
                continue
            rec = {"id": rid, "topology": kind, "n": n, "rep": r, "seed": seed, "mu": mu}
            for eng in ("old", "new") if r % 2 == 0 else ("new", "old"):       # alternate order: no systematic warm-up/drift bias
                x, wall, cpu = fit(P, eng, seed)
                rec[eng] = {"est": x["estimate"], "ci": list(x["ci"]), "wall": wall, "cpu": cpu}
            with RAW.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(rid, f"old {rec['old']['wall']:.1f}s new {rec['new']['wall']:.1f}s", flush=True)


if __name__ == "__main__":
    # optional: python repro/classical_v3_bench.py ring-28=36 ring-20=36 ring-16=20   (total replications; only these cases run)
    ex = {(a.split("=")[0].split("-")[0], int(a.split("=")[0].split("-")[1])): int(a.split("=")[1]) for a in sys.argv[1:]}
    main(only=set(ex) or None, extra=ex or None)
