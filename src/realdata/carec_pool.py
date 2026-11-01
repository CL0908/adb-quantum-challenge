"""WP2: a data-calibrated CAREC regional pool (10 countries), exact tensor-network references, and RDPO tested
against them, including the roadmap's WP2 success criterion (relative RMSE <= 10% at mu <= 1e-8 within 1e5 paths).

Countries: the CAREC members Afghanistan, Azerbaijan, Georgia, Kazakhstan, Kyrgyz Republic, Mongolia, Pakistan,
Tajikistan, Turkmenistan and Uzbekistan (China is excluded: only some of its provinces take part in CAREC, and a
single country-wide cover for China is not comparable).

Calibration (public data, cached in data/real/):
  * Earthquakes, exactly as in central_asia_pool.py: catalogue main shocks within 50 km of each country,
    one regional Gutenberg–Richter slope, the bond's tier design (25% tier at the magnitude where the Kyrgyz and
    Tajik data reproduce the brief), shared events from shocks reaching two countries (two nearest countries).
  * Rain (heterogeneous): each country's yearly frequency f_i of major hydro-meteorological losses in EM-DAT
    1990–2020 (floods, landslides, mass movements, storms with >= 10 deaths, >= 10,000 affected or >= US$10m
    damage). The annual rain-trigger probability is 2% x f_i / f_ref with f_ref the Kyrgyz–Tajik average, so the
    brief's 2% is kept on average for the benchmark pair, and clipped to [0.5%, 8%]. Shared rain on each
    earthquake edge as in the brief (25% of the smaller rate / max degree).
Exact references: tilted matrix-product state (tol 1e-12) with sites ordered west to east; the pair targets by
the exact pair reduction. Run: python -m src.realdata.carec_pool -> results/wp2_carec.json
"""
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.tensor.mps import MPS, targets_from_mps, D
from src.rdpo.problem import PortfolioProblem
from src.rdpo.train import rdpo_estimate, RDPOConfig
from src.rdpo.experiments import summarise, twisted_est
from src.rdpo.experiments_v2 import final_config, run_jobs
from src.model.poisson_fit import TAILS
from . import eq_check
from .central_asia_pool import DataModel, _load

ISO = ["AF", "AZ", "GE", "KZ", "KG", "MN", "PK", "TJ", "TM", "UZ"]
NAMES = {"AF": "Afghanistan", "AZ": "Azerbaijan", "GE": "Georgia", "KZ": "Kazakhstan", "KG": "Kyrgyz Republic",
         "MN": "Mongolia", "PK": "Pakistan", "TJ": "Tajikistan", "TM": "Turkmenistan", "UZ": "Uzbekistan"}
OWID_NAMES = {"AF": "Afghanistan", "AZ": "Azerbaijan", "GE": "Georgia", "KZ": "Kazakhstan", "KG": "Kyrgyzstan",
              "MN": "Mongolia", "PK": "Pakistan", "TJ": "Tajikistan", "TM": "Turkmenistan", "UZ": "Uzbekistan"}
BOX = (22.0, 56.0, 38.0, 122.0)
R_KM = 50.0
DATA = eq_check.DATA


def _cache():
    import urllib.request
    cat, geo, hyd = DATA / "usgs_m55_carec_1965_2016.csv", DATA / "carec_boundaries.geojson", DATA / "owid_emdat_carec.csv"
    if not cat.exists():
        raw = urllib.request.urlopen(eq_check.EQ_URL, timeout=120).read().decode().splitlines()
        keep = [raw[0]] + [l for l in raw[1:] if BOX[0] <= float(l.split(",")[1]) <= BOX[1] and BOX[2] <= float(l.split(",")[2]) <= BOX[3]]
        cat.write_text("# source: " + eq_check.EQ_URL + "\n" + "\n".join(keep) + "\n")
    if not geo.exists():
        g = json.loads(urllib.request.urlopen(eq_check.GEO_URL, timeout=300).read())
        feats = [f for f in g["features"] if f["properties"]["ISO3166-1-Alpha-2"] in ISO]
        geo.write_text(json.dumps({"type": "FeatureCollection", "source": eq_check.GEO_URL, "features": feats}))
    if not hyd.exists():
        tmp = Path("/tmp/owid_ds"); rel = "datasets/Natural disasters (EMDAT)/Natural disasters (EMDAT).csv"
        if not tmp.exists():
            subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--depth", "1",
                            "https://github.com/owid/owid-datasets", str(tmp)], check=True)
        subprocess.run(["git", "-C", str(tmp), "checkout", "-q", "HEAD", "--", rel], check=True)
        rows = list(csv.reader(open(tmp / rel)))
        keep = [rows[0]] + [r for r in rows[1:] if r[0] in OWID_NAMES.values()]
        import io
        buf = io.StringIO(); csv.writer(buf).writerows(keep)
        hyd.write_text("# source: https://github.com/owid/owid-datasets " + rel + "\n" + buf.getvalue())
    return cat, geo, hyd


def hydro_frequency(hyd, years=(1990, 2020)):
    rows = list(csv.DictReader([l for l in hyd.read_text().splitlines() if not l.startswith("#")]))
    f = {}
    for c in ISO:
        ys = set()
        for r in rows:
            if r["Entity"] != OWID_NAMES[c] or not years[0] <= int(r["Year"]) <= years[1]:
                continue
            g = lambda k: float(r.get(k) or 0)
            if any(g(f"deaths_{h}") >= 10 or g(f"total_affected_{h}") >= 1e4 or g(f"total_damages_{h}") >= 1e4
                   for h in ("flood", "landslide", "mass_movement", "storm")):
                ys.add(int(r["Year"]))
        f[c] = len(ys) / (years[1] - years[0] + 1)
    return f


def calibrate():
    cat, geo, hyd = _cache()
    ev, polys = _load(cat, geo)
    ev = eq_check.decluster([e for e in ev if eq_check.YEARS[0] <= e[0].year <= eq_check.YEARS[1]])
    years = eq_check.YEARS[1] - eq_check.YEARS[0] + 1
    # west-to-east site order keeps the tensor-network cut-width small
    lon = {c: float(np.mean(np.concatenate([r[:, 0] for r in polys[c]]))) for c in ISO}
    order = sorted(ISO, key=lambda c: lon[c])
    dist = {c: np.array([eq_check.dist_to_country(la, lo, polys[c]) for _, la, lo, _ in ev]) for c in order}
    mag = np.array([e[3] for e in ev])
    near = {c: dist[c] <= R_KM for c in order}
    anyc = np.any([near[c] for c in order], axis=0)
    mm = mag[anyc]
    b = math.log10(math.e) / (mm.mean() - (5.5 - 0.05))
    rate55 = {c: float(near[c].sum() / years) for c in order}
    m1 = float(np.mean([5.5 + math.log10(rate55[c] / -math.log(1 - TAILS[c][0])) / b for c in ("KG", "TJ")]))
    tier = np.mean([np.log10(TAILS[c][0] / TAILS[c]) / b for c in ("KG", "TJ")], axis=0)
    tails = {c: 1 - np.exp(-rate55[c] * 10 ** (-b * (m1 + tier - 5.5))) for c in order}
    scale = 10 ** (-b * (m1 - 5.5))
    Dm = np.stack([dist[c] for c in order], 1)
    pairs = {}
    for e in range(len(ev)):
        hit = np.flatnonzero(Dm[e] <= R_KM)
        if len(hit) >= 2:
            a, bb = sorted(int(x) for x in hit[np.argsort(Dm[e, hit])[:2]])
            pairs[(a, bb)] = pairs.get((a, bb), 0) + 1
    edges = sorted(pairs)
    shared = {e: pairs[e] / years * scale for e in edges}
    fh = hydro_frequency(hyd)
    fref = 0.5 * (fh["KG"] + fh["TJ"])
    rain_p = {c: float(np.clip(0.02 * fh[c] / fref, 0.005, 0.08)) for c in order}
    model = DataModel(len(order), edges, tails=[tails[c] for c in order], shared_eq=shared, rain_p=[rain_p[c] for c in order])
    return {"order": order, "b": b, "m1": m1, "rate55": rate55, "tails": {c: tails[c].tolist() for c in order},
            "edges": [[order[i], order[j]] for i, j in edges], "n_both": {f"{order[i]}-{order[j]}": k for (i, j), k in pairs.items()},
            "shared_rate": {f"{order[i]}-{order[j]}": v for (i, j), v in shared.items()}, "hydro_freq": fh, "rain_p": rain_p,
            "events": len(ev), "model": model, "pair": (order.index("KG"), order.index("TJ"))}


def build_mps_data(model, tol=1e-12):
    """Tilted MPS as in src/tensor/mps.build_mps, with gauges from this model's own independent marginals."""
    ind = DataModel(model.n, [], tails=model.tails, shared_eq={}, rain_p=model.rain_p, years=model.years)
    g = [1.0 / np.maximum(ind.single_kernel(i)[0], 1e-300) for i in range(model.n)]
    psi = MPS.product(model.n); psi.gauge = g
    for i in range(model.n):
        psi.T[i] = psi.T[i] * g[i][None, :, None]
        psi.apply_single(i, (model.single_kernel(i) / g[i][:, None]) * g[i][None, :])
    for (i, j) in sorted(model.edges):
        gij = np.kron(g[i], g[j])
        psi.apply_pair(i, j, (model.pair_kernel(i, j) / gij[:, None]) * gij[None, :], tol)
    return psi


_CAL = None


def _cal():
    global _CAL
    if _CAL is None:
        _CAL = calibrate()
    return _CAL


def _job(args):
    jid, t, thr, method, B, seed = args
    cal = _cal()
    P = PortfolioProblem(cal["model"], t, threshold=thr * 1e6 if thr else 400e6, pair=cal["pair"])
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    if method == "rdpo":
        x = rdpo_estimate(P, B, rng, final_config(t)); est, ci = x["estimate"], list(x["ci"])
    elif method == "mc":
        x = rdpo_estimate(P, B, rng, RDPOConfig(engine="fast", warm_start=False, ce=False, pg=False, qmc=False))
        est, ci = x["estimate"], list(x["ci"])
    else:
        est, ci = float(twisted_est(P, B, rng)), None
    return {"id": jid, "key": [t, thr, method, B], "est": est, "ci": ci, "sec": time.perf_counter() - t0}


def references(cal):
    m = cal["model"]
    t0 = time.perf_counter()
    psi = build_mps_data(m, 1e-12)
    sec = time.perf_counter() - t0
    psi10 = build_mps_data(m, 1e-10)
    # thresholds: a moderate one and the rarest round US$m level with mu in [1e-10, 1e-8]
    grid = {}
    for T in range(100, 801, 20):
        grid[T] = targets_from_mps(m, psi, T * 1e6)["exceedance_prob"]
    moderate = min(grid, key=lambda T: abs(math.log10(max(grid[T], 1e-300)) + 3))
    rare = [T for T in grid if 1e-10 <= grid[T] <= 1e-8]
    rare = min(rare) if rare else max(grid)
    refs = {}
    for T in (moderate, rare):
        a = targets_from_mps(m, psi, T * 1e6); a10 = targets_from_mps(m, psi10, T * 1e6)
        refs[f"exceedance|{T}"] = a["exceedance_prob"]; refs[f"stop_loss|{T}"] = a["stop_loss"]
        refs[f"check|{T}"] = abs(a10["exceedance_prob"] / a["exceedance_prob"] - 1)
    refs["expected_loss|None"] = targets_from_mps(m, psi, 0)["expected_loss"]
    for t in ("joint_exhaustion", "joint_attachment"):
        refs[f"{t}|None"] = PortfolioProblem(m, t, pair=cal["pair"]).exact_pair()
    return refs, moderate, rare, {"mps_seconds": sec, "max_bond": int(max(psi.bond_dims)), "grid": grid}


def run(reps=24, seed=5151, procs=4):
    cal = _cal()
    refs, Tm, Tr, info = references(cal)
    print("refs", {k: v for k, v in refs.items()}, "T", Tm, Tr, "bond", info["max_bond"], file=sys.stderr, flush=True)
    tests = [("exceedance", Tm, 65536), ("exceedance", Tr, 65536), ("stop_loss", Tm, 65536), ("stop_loss", Tr, 65536),
             ("exceedance", Tr, 16384), ("joint_exhaustion", None, 65536), ("expected_loss", None, 65536)]
    jobs = [(f"{t}-{thr}-{m}-{B}-{r}", t, thr, m, B, seed + r) for r in range(reps) for t, thr, B in tests
            for m in ("rdpo", "mc", "twist") if not (m == "twist" and t not in ("exceedance", "stop_loss"))]
    res = {}
    for rec in run_jobs("wp2_carec", jobs, _job, procs):
        res.setdefault(tuple(rec["key"]), []).append(rec)
    rows = []
    for (t, thr, m, B), xs in sorted(res.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][3], kv[0][2])):
        mu = refs[f"{t}|{thr}"]
        if t in ("expected_loss", "stop_loss"):
            mu_cmp = mu                      # MPS money targets in USD, estimator in USD
        else:
            mu_cmp = mu
        s = summarise([x["est"] for x in xs], mu_cmp, None if m == "twist" else [x["ci"] for x in xs])
        rows.append({"target": t, "threshold_musd": thr, "method": m, "budget": B, "mu": mu_cmp, **s,
                     "seconds": float(np.mean([x["sec"] for x in xs]))})
        print(f"{t:17s} {str(thr):5s} B={B:6d} {m:6s} mu={mu_cmp:.3e} rel={s['rel_rmse']:.4f} cov={s.get('ci_coverage', float('nan')):.2f}",
              file=sys.stderr)
    out = {k: v for k, v in cal.items() if k not in ("model",)}
    Path("results/wp2_carec.json").write_text(json.dumps({"calibration": out, "references": refs, "T_moderate": Tm,
                                                          "T_rare": Tr, "mps": {k: v for k, v in info.items() if k != "grid"},
                                                          "exceedance_curve": info["grid"], "reps": reps, "rows": rows},
                                                         indent=2, default=str))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        c = calibrate()
        print(json.dumps({k: v for k, v in c.items() if k != "model"}, indent=1, default=str))
        t0 = time.time(); r = references(c); print(r[0], r[1], r[2], r[3]["max_bond"], r[3]["mps_seconds"], time.time() - t0)
    else:
        run()
