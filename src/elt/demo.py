"""ELT/YLT demonstration on a public-data stochastic event set for the five Central Asian countries.

Event set (illustrative, built from public data; not a vendor model):
  * Earthquakes. Every catalogue main shock (USGS M >= 5.5, 1965–2016, declustered) within 300 km of the five
    countries is a source location. Each location generates magnitudes M = 7.0, 7.1, ..., 8.5 with Gutenberg–
    Richter rates (regional b), normalised so that each location carries its observed share of M >= 5.5 activity.
    An event pays in country i if the country lies within R(M) = 50 km · 10^(0.5 (M − 7)) of the epicentre
    (cat-in-a-circle style). Payout tiers follow the bond: quarter steps at M1 + ΔM_k (M1 = 7.0; ΔM_k from the
    brief's tier ratios).
  * Rain. A 2% (1-in-50) parametric rain trigger per country; shared rain events for neighbouring pairs with the
    shared share measured in station data (src/realdata/rain_check.py).
YLT: 10,000 simulated years drawn from the ELT (Poisson occurrences), as a modelling agent would deliver.
Tests: exact answers (dense enumeration) vs RDPO on the converted sources, plain Monte Carlo, and the
vendor-style estimate taken directly from a 10,000-year YLT (3,333 three-year terms).
Run: python -m src.elt.demo -> data/elt/*.csv, results/elt_demo.json
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from src.model.poisson_fit import TAILS
from src.realdata import eq_check
from src.realdata.central_asia_pool import _load, ISO
from src.rdpo.problem import PortfolioProblem
from src.rdpo.train import rdpo_estimate, RDPOConfig
from src.rdpo.experiments import summarise
from src.rdpo.experiments_v2 import final_config, run_jobs
from src.rdpo.projected import ProjectedDoob
from .convert import read_elt, read_ylt, ELTModel

OUT = Path("data/elt")
ELT_CSV, YLT_CSV = OUT / "central_asia_stochastic_elt.csv", OUT / "central_asia_ylt_10k.csv"
NEIGH = [("KG", "TJ"), ("KG", "UZ"), ("KG", "KZ"), ("TJ", "UZ"), ("UZ", "TM"), ("UZ", "KZ"), ("KZ", "TM")]
M1, MAGS = 7.0, np.round(np.arange(7.0, 8.51, 0.1), 1)


def build_elt():
    cat, geo = eq_check.DATA / "usgs_m55_central_asia_wide_1965_2016.csv", eq_check.DATA / "central_asia5_boundaries.geojson"
    ev, polys = _load(cat, geo)
    ev = eq_check.decluster([e for e in ev if eq_check.YEARS[0] <= e[0].year <= eq_check.YEARS[1]])
    years = eq_check.YEARS[1] - eq_check.YEARS[0] + 1
    dist = np.stack([[eq_check.dist_to_country(la, lo, polys[c]) for _, la, lo, _ in ev] for c in ISO], 1)
    near = dist.min(1) <= 300
    mag = np.array([e[3] for e in ev])
    b = math.log10(math.e) / (mag[near].mean() - (5.5 - 0.05))
    tier = np.mean([np.log10(TAILS[c][0] / TAILS[c]) / b for c in ("KG", "TJ")], axis=0)
    rows, eid = [], 0
    for j in np.flatnonzero(near):
        for M in MAGS:
            rate = (10 ** (-b * (M - 5.5)) - 10 ** (-b * (M + 0.1 - 5.5))) / years
            R = 50.0 * 10 ** (0.5 * (M - 7.0))
            steps = int(np.sum(M + 1e-9 >= M1 + tier))
            hit = [i for i in range(len(ISO)) if dist[j, i] <= R]
            if steps == 0 or not hit:
                continue
            eid += 1
            for i in hit:
                rows.append({"event_id": f"EQ{eid}", "rate": rate, "country": ISO[i], "eq_steps": steps, "rain": 0})
    share = json.loads(Path("results/realdata_rain.json").read_text())["pooled"]["0.2"]["share"]
    mu = -math.log(1 - 0.02)
    deg = {c: sum(c in p for p in NEIGH) for c in ISO}
    for a, c in NEIGH:
        eid += 1
        r = share * mu / max(deg[a], deg[c])
        for x in (a, c):
            rows.append({"event_id": f"RN{eid}", "rate": r, "country": x, "eq_steps": 0, "rain": 1})
    for c in ISO:
        eid += 1
        solo = mu - sum(share * mu / max(deg[a], deg[x]) for a, x in NEIGH if c in (a, x))
        rows.append({"event_id": f"RN{eid}", "rate": solo, "country": c, "eq_steps": 0, "rain": 1})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(ELT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["event_id", "rate", "country", "eq_steps", "rain"]); w.writeheader(); w.writerows(rows)
    return {"b": b, "events": len({r["event_id"] for r in rows}), "rows": len(rows), "locations": int(near.sum())}


def simulate_ylt(n_years=10000, seed=0, path=YLT_CSV):
    rng = np.random.default_rng(seed)
    ev = {}
    for row in csv.DictReader(open(ELT_CSV)):
        ev.setdefault(row["event_id"], {"rate": float(row["rate"]), "rows": []})["rows"].append(row)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["year", "event_id", "country", "eq_steps", "rain"])
        for e, d in ev.items():
            k = rng.poisson(d["rate"], n_years)
            for y in np.flatnonzero(k):
                for _ in range(int(k[y])):
                    for row in d["rows"]:
                        w.writerow([int(y), e, row["country"], row["eq_steps"], row["rain"]])
    return n_years


def ylt_estimate(path, n_years, target, threshold_musd, pair=(0, 1)):
    """Vendor-style estimate from a YLT: consecutive 3-year terms, payouts capped per country and term."""
    a = np.zeros((n_years // 3, len(ISO)), int); r = np.zeros_like(a)
    for row in csv.DictReader(open(path)):
        t = int(row["year"]) // 3
        if t >= a.shape[0]:
            continue
        i = ISO.index(row["country"])
        a[t, i] = min(a[t, i] + int(row["eq_steps"]), 4); r[t, i] = min(r[t, i] + int(row["rain"]), 1)
    L = a * 17.5 + r * 10.0
    tot = L.sum(1)
    if target == "exceedance":
        v = tot > threshold_musd
    elif target == "stop_loss":
        v = np.maximum(tot - threshold_musd, 0) * 1e6
    elif target == "expected_loss":
        v = tot * 1e6
    else:
        v = (L[:, pair[0]] == 80) & (L[:, pair[1]] == 80)
    return float(np.mean(v))


TESTS = [("exceedance", 160), ("exceedance", 320), ("stop_loss", 240), ("expected_loss", None), ("joint_exhaustion", None)]
_M = {}


def _model():
    if "m" not in _M:
        _M["m"] = ELTModel(len(ISO), read_elt(ELT_CSV, ISO))
    return _M["m"]


def _job(args):
    jid, t, thr, method, seed = args
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    if method == "ylt":
        path = OUT / ("_ylt_" + "".join(ch if ch.isalnum() else "_" for ch in jid) + ".csv")
        simulate_ylt(10000, seed, path)
        est = ylt_estimate(path, 10000, t, thr); path.unlink()
        return {"id": jid, "key": [t, thr, method], "est": est, "ci": None, "sec": time.perf_counter() - t0}
    P = PortfolioProblem(_model(), t, threshold=thr * 1e6 if thr else 200e6)
    if method == "rdpo":
        x = rdpo_estimate(P, 16384, rng, final_config(t))
    else:
        x = rdpo_estimate(P, 16384, rng, RDPOConfig(engine="fast", warm_start=False, ce=False, pg=False, qmc=False))
    return {"id": jid, "key": [t, thr, method], "est": x["estimate"], "ci": list(x["ci"]), "sec": time.perf_counter() - t0}


def run(reps=24, seed=8080, procs=4):
    info = build_elt()
    t0 = time.perf_counter(); m = _model(); conv = time.perf_counter() - t0
    n_years = simulate_ylt()
    m_ylt = ELTModel(len(ISO), read_ylt(YLT_CSV, ISO, n_years))
    exact, exact_ylt = {}, {}
    for t, thr in TESTS:
        exact[f"{t}|{thr}"] = PortfolioProblem(m, t, threshold=thr * 1e6 if thr else 200e6).exact_dense()
        exact_ylt[f"{t}|{thr}"] = PortfolioProblem(m_ylt, t, threshold=thr * 1e6 if thr else 200e6).exact_dense()
    pd_ = ProjectedDoob(PortfolioProblem(m, "joint_exhaustion"))
    print("exact", exact, file=sys.stderr, flush=True)
    jobs = [(f"{t}-{thr}-{meth}-{r}", t, thr, meth, seed + r) for r in range(reps) for t, thr in TESTS for meth in ("rdpo", "mc", "ylt")]
    res = {}
    for rec in run_jobs("elt_demo", jobs, _job, procs):
        res.setdefault(tuple(rec["key"]), []).append(rec)
    rows = []
    for (t, thr, meth), xs in sorted(res.items(), key=lambda kv: [str(x) for x in kv[0]]):
        mu = exact[f"{t}|{thr}"]
        s = summarise([x["est"] for x in xs], mu, None if meth == "ylt" else [x["ci"] for x in xs])
        rows.append({"target": t, "threshold_musd": thr, "method": meth, "mu": mu, **s, "seconds": float(np.mean([x["sec"] for x in xs]))})
        print(f"{t:16s} {str(thr):5s} {meth:5s} mu={mu:.3e} rel={s['rel_rmse']:.4f} cov={s.get('ci_coverage', float('nan')):.2f}", file=sys.stderr)
    out = {"event_set": info, "conversion_seconds": conv, "footprints": {"-".join(ISO[i] for i in k): v for k, v in m.footprints.items()},
           "n_sources": len(m.custom_sources), "max_footprint": max(len(k) for k in m.footprints),
           "exact": exact, "exact_from_ylt": exact_ylt, "projected_doob_pair": pd_.mu, "reps": reps, "rows": rows}
    Path("results/elt_demo.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    run()
