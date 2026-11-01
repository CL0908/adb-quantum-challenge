"""A data-calibrated five-country Central Asian pool, and RDPO tested on it against exact answers.

Calibration (all from public data; see src/realdata/eq_check.py for the catalogue and boundaries):
  * Earthquake main shocks (USGS significant earthquakes, M >= 5.5, 1965–2016, Gardner–Knopoff declustered)
    within 50 km of each country -> yearly rate of M >= 5.5 events per country.
  * One Gutenberg–Richter slope b for the region (Aki maximum likelihood).
  * Payout tiers. Every country holds the bond's design: US$70m earthquake cover paid in quarter steps plus
    US$10m rain cover. The 25% tier is triggered at the magnitude M1 at which the Kyrgyz and Tajik data
    reproduce the brief's 25%-payout frequencies; the 50/75/100% tiers sit at the magnitude offsets implied by
    the brief's tier ratios under Gutenberg–Richter. Country i's annual tier exceedances are then
    t_k(i) = 1 − exp(−λ_i(5.5) · 10^(−b (M1 + ΔM_k − 5.5))).
  * Shared earthquakes. For each pair of countries, the rate of main shocks within 50 km of both, scaled to
    trigger level with the same law; an edge of the dependence graph wherever such events occurred.
  * Rain: the brief's 2% yearly trigger per country and 25% shared component on each edge (hydro loss
    data do not constrain it; see src/realdata/loss_backtest.py).
Test: exact answers by dense enumeration (10^5 states) vs RDPO (Phase 2 engine), plain Monte Carlo and
exponential twisting, all at the same budget, 40 replications.
Run: python -m src.realdata.central_asia_pool -> results/realdata_pool.json
"""
import json
import math
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src.tensor.synthetic import SyntheticModel
from src.model.poisson_fit import TAILS, compound_poisson_rates
from src.rdpo.problem import PortfolioProblem
from src.rdpo.train import rdpo_estimate, RDPOConfig
from src.rdpo.experiments import summarise, twisted_est
from src.rdpo.experiments_v2 import final_config
from . import eq_check

ISO = ["KG", "TJ", "UZ", "KZ", "TM"]
NAMES = {"KG": "Kyrgyz Republic", "TJ": "Tajikistan", "UZ": "Uzbekistan", "KZ": "Kazakhstan", "TM": "Turkmenistan"}
BOX = (34.0, 56.0, 45.0, 89.0)
R_KM = 50.0


@dataclass
class DataModel(SyntheticModel):
    """SyntheticModel with country-specific tails and data-based shared rates (same source structure)."""
    tails: list = field(default_factory=list)
    shared_eq: dict = field(default_factory=dict)
    rain_p: list = field(default_factory=list)       # annual rain-trigger probability per country (default 2%)

    def __post_init__(self):
        self.lam, self.Lam = [], []
        for t in self.tails:
            l, L = compound_poisson_rates(np.asarray(t))
            self.lam.append(l); self.Lam.append(L)
        self.w = [l / L for l, L in zip(self.lam, self.Lam)]
        rp = self.rain_p if self.rain_p else [0.02] * self.n
        self.mu = [-np.log(1 - p) * self.rain_mult for p in rp]
        deg = np.zeros(self.n, int)
        for i, j in self.edges:
            deg[i] += 1; deg[j] += 1
        self.maxdeg = max(int(deg.max()) if self.n else 0, 1)
        self.c = {e: self.m * self.shared_eq[e] for e in self.edges}
        self.cr = {(i, j): self.rain_shared * min(self.mu[i], self.mu[j]) / self.maxdeg for i, j in self.edges}
        self.solo_eq = [self.Lam[i] - sum(v for e, v in self.c.items() if i in e) for i in range(self.n)]
        self.solo_rain = [self.mu[i] - sum(v for e, v in self.cr.items() if i in e) for i in range(self.n)]
        assert min(self.solo_eq) >= 0 and min(self.solo_rain) >= 0


def calibrate():
    import json as _j
    import urllib.request
    data = eq_check.DATA
    cat = data / "usgs_m55_central_asia_wide_1965_2016.csv"
    geo = data / "central_asia5_boundaries.geojson"
    if not cat.exists():
        raw = urllib.request.urlopen(eq_check.EQ_URL, timeout=120).read().decode().splitlines()
        keep = [raw[0]] + [l for l in raw[1:] if BOX[0] <= float(l.split(",")[1]) <= BOX[1] and BOX[2] <= float(l.split(",")[2]) <= BOX[3]]
        cat.write_text("# source: " + eq_check.EQ_URL + "\n" + "\n".join(keep) + "\n")
    if not geo.exists():
        g = _j.loads(urllib.request.urlopen(eq_check.GEO_URL, timeout=300).read())
        feats = [f for f in g["features"] if f["properties"]["ISO3166-1-Alpha-2"] in ISO]
        geo.write_text(_j.dumps({"type": "FeatureCollection", "source": eq_check.GEO_URL, "features": feats}))
    # reuse the loaders with the wider files
    eq_check_files = (eq_check.DATA / "usgs_m55_central_asia_1965_2016.csv", eq_check.DATA / "kg_tj_boundaries.geojson")
    ev, polys = _load(cat, geo)
    ev = eq_check.decluster([e for e in ev if eq_check.YEARS[0] <= e[0].year <= eq_check.YEARS[1]])
    years = eq_check.YEARS[1] - eq_check.YEARS[0] + 1
    dist = {c: np.array([eq_check.dist_to_country(la, lo, polys[c]) for _, la, lo, _ in ev]) for c in ISO}
    mag = np.array([e[3] for e in ev])
    near = {c: dist[c] <= R_KM for c in ISO}
    anyc = np.any([near[c] for c in ISO], axis=0)
    mm = mag[anyc]
    b = math.log10(math.e) / (mm.mean() - (5.5 - 0.05))
    rate55 = {c: near[c].sum() / years for c in ISO}
    # M1: magnitude at which KG and TJ reproduce the brief's 25%-payout frequency (average of the two)
    m1 = np.mean([5.5 + math.log10(rate55[c] / -math.log(1 - TAILS[c][0])) / b for c in ("KG", "TJ")])
    tier = np.mean([np.log10(TAILS[c][0] / TAILS[c]) / b for c in ("KG", "TJ")], axis=0)    # ΔM_k (k=1..4), ΔM_1 = 0
    tails = {c: 1 - np.exp(-rate55[c] * 10 ** (-b * (m1 + tier - 5.5))) for c in ISO}
    scale = 10 ** (-b * (m1 - 5.5))
    # each main shock reaching two or more countries is one shared event of the two nearest countries
    # (the model has pairwise shared sources only, so an event is never counted twice)
    D = np.stack([dist[c] for c in ISO], 1)
    pairs = {}
    for e in range(len(ev)):
        hit = np.flatnonzero(D[e] <= R_KM)
        if len(hit) >= 2:
            a, bb = sorted(hit[np.argsort(D[e, hit])[:2]])
            pairs[(int(a), int(bb))] = pairs.get((int(a), int(bb)), 0) + 1
    edges, shared, nboth = [], {}, {}
    for (a, bb), k in sorted(pairs.items()):
        nboth[f"{ISO[a]}-{ISO[bb]}"] = k
        edges.append((a, bb)); shared[(a, bb)] = k / years * scale
    return {"b": b, "m1": float(m1), "tier_dm": tier.tolist(), "rate55": rate55, "tails": {c: tails[c].tolist() for c in ISO},
            "edges": edges, "shared": {f"{ISO[i]}-{ISO[j]}": v for (i, j), v in shared.items()}, "n_both": nboth,
            "events": int(len(ev)), "model": DataModel(5, edges, tails=[tails[c] for c in ISO], shared_eq=shared)}


def _load(cat, geo):
    from datetime import datetime
    rows = [l.split(",") for l in cat.read_text().splitlines() if l and not l.startswith(("#", "Date"))]
    ev = []
    for d, la, lo, m in rows:
        try:
            t = datetime.strptime(d, "%m/%d/%Y")
        except ValueError:
            t = datetime.fromisoformat(d[:19].replace("Z", ""))
        ev.append((t, float(la), float(lo), float(m)))
    ev.sort()
    polys = {}
    for f in json.loads(geo.read_text())["features"]:
        g = f["geometry"]
        rings = [p[0] for p in g["coordinates"]] if g["type"] == "MultiPolygon" else [g["coordinates"][0]]
        polys[f["properties"]["ISO3166-1-Alpha-2"]] = [np.array(r) for r in rings]
    return ev, polys


TESTS = [("exceedance", 200e6), ("exceedance", 320e6), ("stop_loss", 200e6), ("joint_exhaustion", None),
         ("joint_attachment", None), ("expected_loss", None)]
METHODS = ("rdpo", "mc", "twist")
_CAL = None


def _problem(t, thr):
    global _CAL
    if _CAL is None:
        _CAL = calibrate()
    return PortfolioProblem(_CAL["model"], t, threshold=thr if thr else 200e6)


def _job(args):
    jid, t, thr, method, B, seed = args
    P = _problem(t, thr)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if method == "rdpo":
        x = rdpo_estimate(P, B, rng, final_config(t)); est, ci = x["estimate"], list(x["ci"])
    elif method == "mc":
        x = rdpo_estimate(P, B, rng, RDPOConfig(engine="fast", warm_start=False, ce=False, pg=False, qmc=False))
        est, ci = x["estimate"], list(x["ci"])
    else:
        est, ci = float(twisted_est(P, B, rng)), None
    return {"id": jid, "key": [t, thr, method, B], "est": est, "ci": ci, "sec": time.perf_counter() - t0}


def run(budgets=(16384,), reps=40, seed=4242, procs=4):
    from src.rdpo.experiments_v2 import run_jobs
    cal = calibrate()
    exact = {}
    for t, thr in TESTS:
        P = PortfolioProblem(cal["model"], t, threshold=thr if thr else 200e6)
        exact[f"{t}|{thr}"] = P.exact_dense()
    jobs = [(f"{t}-{thr}-{m}-{B}-{r}", t, thr, m, B, seed + r) for r in range(reps) for t, thr in TESTS for m in METHODS
            for B in budgets if not (m == "twist" and t not in ("exceedance", "stop_loss"))]
    res = {}
    for rec in run_jobs("ca_pool", jobs, _job, procs):
        res.setdefault(tuple(rec["key"]), []).append(rec)
    rows = []
    for (t, thr, m, B), xs in sorted(res.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), kv[0][2])):
        mu = exact[f"{t}|{thr}"]
        s = summarise([x["est"] for x in xs], mu, None if m == "twist" else [x["ci"] for x in xs])
        rows.append({"target": t, "threshold_musd": thr / 1e6 if thr else None, "method": m, "budget": B, "mu": mu, **s,
                     "seconds": float(np.mean([x["sec"] for x in xs]))})
        print(f"{t:17s} {str(thr):8s} {m:6s} mu={mu:.3e} rel={s['rel_rmse']:.4f} cov={s.get('ci_coverage', float('nan')):.2f}",
              file=sys.stderr)
    cal_out = {k: v for k, v in cal.items() if k != "model"}
    cal_out["edges"] = [[ISO[i], ISO[j]] for i, j in cal["edges"]]
    cal_out["brief_tails"] = {c: TAILS[c].tolist() for c in ("KG", "TJ")}
    Path("results/realdata_pool.json").write_text(json.dumps({"calibration": cal_out, "exact": exact, "reps": reps,
                                                              "rows": rows}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        c = calibrate()
        print(json.dumps({k: v for k, v in c.items() if k != "model"}, indent=1, default=str))
    else:
        run()
