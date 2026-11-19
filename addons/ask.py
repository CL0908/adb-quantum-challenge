"""The three layers as three commands that print one JSON line each. The only door llm_gate.py lets a language model use.

  python ask.py small_world [--m 1.0 --rain_mult 1.0 --rain_shared 0.25] [--pool28]   coarse, exact in the uncapped world, no sampling
  python ask.py rdpo        [same] [--paths 200000]                                   unbiased estimate with a standard error
  python ask.py twin        [fix any of the three inputs] [--worlds 40]               spread over the worlds the twin cannot yet tell apart

Quantity: probability that the pool's total loss exceeds its threshold over the term. Pool: --countries N (2..28, default 2), the
the main author's synthetic ring of N countries with his default threshold of US$40m per country; or --pool28, the 28-country pool of
ap_risk.py, base world only. Unfixed inputs of `twin` are drawn uniformly from dreamwake.RANGE (assumed).
"""
import argparse
import json
import time

import numpy as np

import dreamwake as D
import rdpo_engine as E
import small_world as W
from src.rdpo.problem import PortfolioProblem
from src.tensor.synthetic import SyntheticModel, topology


def problem(a, world=None):
    if a.pool28:
        import ap_risk
        return PortfolioProblem(ap_risk.inputs()["model"], "exceedance", threshold=ap_risk.THRESH[0] * 1e6)
    world = world or {n: getattr(a, n) if getattr(a, n) is not None else D.BASE[n] for n in D.INPUTS}
    return PortfolioProblem(SyntheticModel(a.countries, topology("ring", a.countries), **world), "exceedance")


def estimate(P, n, s=1):
    pol, answer, _ = W.teacher(P)
    o = E.rollout(pol, n, np.random.default_rng(s))
    g = o["w"] * o["f"]
    return float(g.mean()), float(g.std() / np.sqrt(n)), answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", choices=("small_world", "rdpo", "twin"))
    for n in D.INPUTS:
        ap.add_argument(f"--{n}", type=float)
    ap.add_argument("--countries", type=int, default=2)
    ap.add_argument("--pool28", action="store_true")
    ap.add_argument("--paths", type=int, default=200_000)
    ap.add_argument("--worlds", type=int, default=40)
    a = ap.parse_args()
    for n in D.INPUTS:                                                         # trust boundary: the caller is a language model
        v = getattr(a, n)
        if v is not None and not D.RANGE[n][0] <= v <= D.RANGE[n][1]:
            ap.error(f"--{n} must be within {D.RANGE[n]}")
    if not 2 <= a.countries <= 28:
        ap.error("--countries must be 2..28")
    if not 1_000 <= a.paths <= 2_000_000 or not 5 <= a.worlds <= 200:
        ap.error("--paths must be 1000..2000000 and --worlds 5..200")
    if a.pool28 and (a.layer == "twin" or any(getattr(a, n) is not None for n in D.INPUTS)):
        ap.error("--pool28 is the base world only: no inputs, no twin")
    t0 = time.process_time()
    out = {"layer": a.layer, "quantity": "probability that total pool loss exceeds the threshold over the term",
           "model": "28-country pool (ap_risk)" if a.pool28 else f"ring of {a.countries} countries"}
    if a.layer == "small_world":
        out.update(probability=W.value_tables(problem(a))[1], kind="coarse: exact for the pool without per-country caps, so it can only be too high; no sampling, no error bar")
    elif a.layer == "rdpo":
        p, se, _ = estimate(problem(a), a.paths)
        out.update(probability=p, standard_error=se, interval_95=[p - 1.96 * se, p + 1.96 * se], paths=a.paths, kind="unbiased importance-sampling estimate")
    else:
        fixed = {n: getattr(a, n) for n in D.INPUTS if getattr(a, n) is not None}
        rng = np.random.default_rng(7)
        ps = np.array([estimate(problem(a, {**w, **fixed}), 20_000, s=i)[0] for i, w in enumerate(D.worlds(rng, a.worlds))])
        out.update(fixed_inputs=fixed, varied_inputs=[n for n in D.INPUTS if n not in fixed], worlds=a.worlds, mean=float(ps.mean()),
                   quantile_05=float(np.quantile(ps, 0.05)), median=float(np.median(ps)), quantile_95=float(np.quantile(ps, 0.95)),
                   kind="spread over worlds; unfixed inputs uniform over assumed ranges; each world rebuilt and estimated with 20000 paths")
    if not a.pool28 and a.layer != "twin":
        out["inputs"] = {n: getattr(a, n) if getattr(a, n) is not None else D.BASE[n] for n in D.INPUTS}
    out["cpu_seconds"] = round(time.process_time() - t0, 3)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
