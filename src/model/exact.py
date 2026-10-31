"""Phase 0 driver: exact reproduction of brief checkpoints + Table 5."""
import json
import time
from pathlib import Path

import numpy as np

from .annual import annual_distribution, eq_annual, rain_annual, eq_generator
from .poisson_fit import TAILS, compound_poisson_rates
from .scenarios import SCENARIOS, REFERENCE, TARGETS
from .term import term_distribution
from .targets import exact_targets


def run(out=Path("results/exact.json")):
    res = {"rates": {}, "checkpoints": {}, "scenarios": {}}
    for k, t in TAILS.items():
        lam, L = compound_poisson_rates(t)
        res["rates"][k] = {"lambda": lam.tolist(), "Lambda": L}
    base = SCENARIOS["adb-base"]
    _, c = eq_generator(base)
    R = rain_annual(base)
    res["checkpoints"] = {
        "shared_rate_c": c,
        "eq_no_payout": float(eq_annual(base)[0, 0]),
        "rain_neither": R[0, 0], "rain_one_only": R[1, 0], "rain_both": R[1, 1],
        "annual_no_payout": float(annual_distribution(base)[0]),
    }
    for name, sc in SCENARIOS.items():
        t0 = time.perf_counter()
        p = annual_distribution(sc)
        d = term_distribution(p)
        vals = exact_targets(sc, d)
        dt = time.perf_counter() - t0
        ref = dict(zip(TARGETS, REFERENCE[name]))
        res["scenarios"][name] = {
            "values": vals,
            "reference_rounded": ref,
            "runtime_s": dt,
            "annual_p0": float(p[0]),
        }
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["checkpoints"], indent=2))
    for n, s in r["scenarios"].items():
        print(n, {k[5:]: f"{v:.6g} (ref {s['reference_rounded'][k]:.6g})" for k, v in s["values"].items()})
