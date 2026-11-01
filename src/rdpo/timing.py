"""Wall-clock of a full stress-test sweep: 5 brief scenarios x 5 targets of the ADB bond, one estimate each with
16,384 paths (Phase 2 engine), on one CPU core. Run: python -m src.rdpo.timing -> results/sweep_timing.json"""
import json
import time
from pathlib import Path

import numpy as np

from .problem import PortfolioProblem, adb_model, ADB_SCENARIOS, TARGETS
from .train import rdpo_estimate
from .experiments_v2 import final_config


def main(B=16384, seed=0):
    t0 = time.perf_counter(); n = 0
    for sc in ADB_SCENARIOS:
        for t in TARGETS:
            rdpo_estimate(PortfolioProblem(adb_model(sc), t, threshold=80e6), B, np.random.default_rng(seed + n), final_config(t))
            n += 1
    sec = time.perf_counter() - t0
    out = {"estimates": n, "paths_each": B, "seconds_total": sec, "seconds_per_estimate": sec / n}
    Path("results/sweep_timing.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
