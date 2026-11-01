"""WP2 at scale: a data-calibrated Asia–Pacific pool of 31 ADB developing member countries, and a 50-country
synthetic ring, with RDPO tested against exact tensor-network references.

Design (differs from the CAREC pool on purpose): every country buys the same *return-period* cover, as parametric
sovereign covers are designed in practice. Its 25% earthquake tier attaches at the country's own 1-in-25.6-year
event (the brief's average 3.9% annual trigger), with the brief's tier ratios above it. Country i's trigger
magnitude M1_i therefore follows from its own main-shock rate (USGS catalogue, 50 km, Gutenberg–Richter with one
regional slope). A shared earthquake source joins two countries whose territory one main shock reached; its
rate is the observed rate scaled to the larger of the two trigger magnitudes, so it triggers both. A shock reaching
three or more countries counts for its two nearest. Rain: the brief's 2% (1-in-50) parametric trigger in every
country, with the shared share measured in station data (src/realdata/rain_check.py, pooled 1-in-5-year level),
on every earthquake edge.
Exact references: tilted MPS with sites ordered by reverse Cuthill–McKee (small bandwidth); pair targets exact.
Run: python -m src.realdata.asia_pacific_pool -> results/wp2_asia_pacific.json
"""
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from src.tensor.synthetic import SyntheticModel, topology
from src.tensor.mps import targets_from_mps, build_mps
from src.rdpo.problem import PortfolioProblem
from src.rdpo.train import rdpo_estimate, RDPOConfig
from src.rdpo.experiments import summarise, twisted_est
from src.rdpo.experiments_v2 import final_config, run_jobs
from src.model.poisson_fit import TAILS
from . import eq_check
from .central_asia_pool import DataModel, _load
from .carec_pool import build_mps_data

ISO = ["AF", "AM", "AZ", "BD", "BT", "KH", "FJ", "GE", "IN", "ID", "KZ", "KG", "LA", "MY", "MN", "MM", "NP", "PK",
       "PG", "PH", "SB", "LK", "TJ", "TH", "TL", "TM", "UZ", "VU", "VN", "TO", "WS"]
DATA = eq_check.DATA
R_KM = 50.0
T1 = float(np.mean([TAILS["KG"][0], TAILS["TJ"][0]]))


def _cache():
    cat, geo = DATA / "usgs_m55_asia_pacific_1965_2016.csv", DATA / "asia_pacific_boundaries.geojson"
    if not cat.exists():
        raw = urllib.request.urlopen(eq_check.EQ_URL, timeout=120).read().decode().splitlines()
        def inbox(l):
            la, lo = float(l.split(",")[1]), float(l.split(",")[2])
            return -30 <= la <= 56 and (lo >= 38 or lo <= -165)
        cat.write_text("# source: " + eq_check.EQ_URL + "\n" + "\n".join([raw[0]] + [l for l in raw[1:] if inbox(l)]) + "\n")
    if not geo.exists():
        g = json.loads(urllib.request.urlopen(eq_check.GEO_URL, timeout=300).read())
        feats = [f for f in g["features"] if f["properties"]["ISO3166-1-Alpha-2"] in ISO]
        geo.write_text(json.dumps({"type": "FeatureCollection", "source": eq_check.GEO_URL, "features": feats}))
    return cat, geo


def calibrate(rain_share=None):
    cat, geo = _cache()
    ev, polys = _load(cat, geo)
    iso = [c for c in ISO if c in polys]
    ev = eq_check.decluster([e for e in ev if eq_check.YEARS[0] <= e[0].year <= eq_check.YEARS[1]])
    years = eq_check.YEARS[1] - eq_check.YEARS[0] + 1
    dist = np.stack([[eq_check.dist_to_country(la, lo, polys[c]) for _, la, lo, _ in ev] for c in iso], 1)
    mag = np.array([e[3] for e in ev])
    near = dist <= R_KM
    keep = near.sum(0) > 0
    iso = [c for c, k in zip(iso, keep) if k]                 # countries with at least one main shock
    dist, near = dist[:, keep], near[:, keep]
    mm = mag[near.any(1)]
    b = math.log10(math.e) / (mm.mean() - (5.5 - 0.05))
    rate55 = near.sum(0) / years
    lam_t = -math.log(1 - T1)
    m1 = 5.5 + np.log10(rate55 / lam_t) / b                   # country trigger magnitudes (1-in-25.6 years)
    tier = np.mean([np.log10(TAILS[c][0] / TAILS[c]) / b for c in ("KG", "TJ")], axis=0)
    tails = [1 - np.exp(-rate55[i] * 10 ** (-b * (m1[i] + tier - 5.5))) for i in range(len(iso))]
    pairs = {}
    for e in range(len(ev)):
        hit = np.flatnonzero(near[e])
        if len(hit) >= 2:
            a, c = sorted(int(x) for x in hit[np.argsort(dist[e, hit])[:2]])
            pairs[(a, c)] = pairs.get((a, c), 0) + 1
    # site order with small bandwidth (reverse Cuthill–McKee on the shared-event graph)
    n = len(iso)
    A = np.zeros((n, n))
    for (a, c) in pairs:
        A[a, c] = A[c, a] = 1
    perm = list(reverse_cuthill_mckee(csr_matrix(A), symmetric_mode=True))
    pos = {old: new for new, old in enumerate(perm)}
    iso = [iso[i] for i in perm]; tails = [tails[i] for i in perm]; m1o = [float(m1[i]) for i in perm]
    rate55o = [float(rate55[i]) for i in perm]
    shared, edges = {}, []
    lam = [ -math.log(1 - t[0]) for t in tails]
    for (a, c), k in pairs.items():
        i, j = sorted((pos[a], pos[c]))
        rate = k / years * 10 ** (-b * (max(m1o[i], m1o[j]) - 5.5))
        shared[(i, j)] = rate; edges.append((i, j))
    edges.sort()
    # feasibility: shared rates may not exceed a country's total rate
    for i in range(n):
        tot = sum(v for e, v in shared.items() if i in e)
        if tot > 0.95 * lam[i]:
            f = 0.95 * lam[i] / tot
            for e in shared:
                if i in e:
                    shared[e] *= f
    if rain_share is None:
        rain_share = json.loads(Path("results/realdata_rain.json").read_text())["pooled"]["0.2"]["share"]
    model = DataModel(n, edges, tails=tails, shared_eq=shared, rain_p=[0.02] * n, rain_shared=rain_share)
    deg = np.zeros(n, int)
    for i, j in edges:
        deg[i] += 1; deg[j] += 1
    bw = max((j - i for i, j in edges), default=0)
    return {"order": iso, "b": b, "m1": dict(zip(iso, m1o)), "rate55": dict(zip(iso, rate55o)),
            "edges": [[iso[i], iso[j]] for i, j in edges], "n_both": {f"{iso[pos[a]]}-{iso[pos[c]]}": k for (a, c), k in pairs.items()},
            "rain_share": rain_share, "events": len(ev), "bandwidth": int(bw), "max_degree": int(deg.max()),
            "model": model, "pair": (iso.index("KG"), iso.index("TJ"))}


_CAL = {}


def _model(case):
    if case not in _CAL:
        if case == "asia_pacific":
            c = calibrate(); _CAL[case] = (c["model"], c["pair"])
        else:
            _CAL[case] = (SyntheticModel(50, topology("ring", 50)), (0, 1))
    return _CAL[case]


def _job(args):
    jid, case, t, thr, method, B, seed = args
    model, pair = _model(case)
    P = PortfolioProblem(model, t, threshold=thr * 1e6 if thr else None, pair=pair)
    rng = np.random.default_rng(seed); t0 = time.perf_counter()
    if method == "rdpo":
        x = rdpo_estimate(P, B, rng, final_config(t)); est, ci = x["estimate"], list(x["ci"])
    elif method == "mc":
        x = rdpo_estimate(P, B, rng, RDPOConfig(engine="fast", warm_start=False, ce=False, pg=False, qmc=False))
        est, ci = x["estimate"], list(x["ci"])
    else:
        est, ci = float(twisted_est(P, B, rng)), None
    return {"id": jid, "key": [case, t, thr, method, B], "est": est, "ci": ci, "sec": time.perf_counter() - t0}


def references(model, pair, tol=1e-10, data_gauge=True, grid=range(200, 4001, 50)):
    t0 = time.perf_counter()
    psi = build_mps_data(model, tol) if data_gauge else build_mps(model, tol=tol, tilt=True)[0]
    sec = time.perf_counter() - t0
    curve = {T: targets_from_mps(model, psi, T * 1e6)["exceedance_prob"] for T in grid}
    rare = [T for T, v in curve.items() if 1e-10 <= v <= 1e-8]
    mod = [T for T, v in curve.items() if 1e-4 <= v <= 1e-2]
    Tr = min(rare) if rare else None
    Tm = min(mod, key=lambda T: abs(math.log10(curve[T]) + 3)) if mod else None
    refs = {}
    for T in (Tm, Tr):
        if T:
            a = targets_from_mps(model, psi, T * 1e6)
            refs[f"exceedance|{T}"] = a["exceedance_prob"]; refs[f"stop_loss|{T}"] = a["stop_loss"]
    refs["joint_exhaustion|None"] = PortfolioProblem(model, "joint_exhaustion", pair=pair).exact_pair()
    # precision check: a 100x tighter truncation must agree on the rare target
    psi2 = build_mps_data(model, tol / 100) if data_gauge else build_mps(model, tol=tol / 100, tilt=True)[0]
    check = abs(targets_from_mps(model, psi2, Tr * 1e6)["exceedance_prob"] / refs[f"exceedance|{Tr}"] - 1) if Tr else None
    return refs, Tm, Tr, {"mps_seconds": sec, "max_bond": int(max(psi.bond_dims)), "curve": curve, "check": check}


def run(reps=24, seed=6161, procs=4):
    out = {"reps": reps, "cases": {}}
    jobs = []
    for case in ("ring50", "asia_pacific"):
        model, pair = _model(case)
        t0 = time.time()
        refs, Tm, Tr, info = references(model, pair, data_gauge=(case == "asia_pacific"))
        print(case, "refs", refs, Tm, Tr, "bond", info["max_bond"], f"{time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        out["cases"][case] = {"refs": refs, "T_moderate": Tm, "T_rare": Tr, "max_bond": info["max_bond"],
                              "mps_seconds": info["mps_seconds"], "n": model.n, "edges": len(model.edges),
                              "tol_check": info["check"]}
        for r in range(reps):
            for t, thr in (("exceedance", Tm), ("exceedance", Tr), ("stop_loss", Tr)):
                if thr is None:
                    continue
                for m in ("rdpo", "mc", "twist"):
                    jobs.append((f"{case}-{t}-{thr}-{m}-{r}", case, t, thr, m, 65536, seed + r))
            jobs.append((f"{case}-pairsampled-{r}", case, "joint_exhaustion", None, "rdpo", 65536, seed + r))
    if "asia_pacific" in out["cases"]:
        c = calibrate()
        out["calibration"] = {k: v for k, v in c.items() if k != "model"}
    res = {}
    for rec in run_jobs("wp2_asia_pacific", jobs, _job, procs):
        res.setdefault(tuple(rec["key"]), []).append(rec)
    rows = []
    for (case, t, thr, m, B), xs in sorted(res.items(), key=lambda kv: [str(x) for x in kv[0]]):
        mu = out["cases"][case]["refs"][f"{t}|{thr}"]
        s = summarise([x["est"] for x in xs], mu, None if m == "twist" else [x["ci"] for x in xs])
        rows.append({"case": case, "target": t, "threshold_musd": thr, "method": m, "budget": B, "mu": mu, **s,
                     "seconds": float(np.mean([x["sec"] for x in xs]))})
        print(f"{case:12s} {t:16s} {str(thr):5s} {m:6s} mu={mu:.3e} rel={s['rel_rmse']:.4f} cov={s.get('ci_coverage', float('nan')):.2f}",
              file=sys.stderr)
    out["rows"] = rows
    Path("results/wp2_asia_pacific.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        c = calibrate()
        print(json.dumps({k: v for k, v in c.items() if k not in ("model", "n_both")}, indent=0, default=str)[:3000])
    else:
        run()


def pool_map(path="figures/paper/pool_map.png"):
    """Map of the data-calibrated Asia–Pacific pool: countries, CAREC members, shared-event links."""
    import matplotlib.pyplot as plt
    from src.viz import style as S
    S.apply()
    d = json.loads(Path("results/wp2_asia_pacific.json").read_text())
    cal = d["calibration"]
    geo = json.loads((DATA / "asia_pacific_boundaries.geojson").read_text())
    carec = {"AF", "AZ", "GE", "KZ", "KG", "MN", "PK", "TJ", "TM", "UZ"}
    cen = {}
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    for f in geo["features"]:
        c = f["properties"]["ISO3166-1-Alpha-2"]
        g = f["geometry"]
        rings = [p[0] for p in g["coordinates"]] if g["type"] == "MultiPolygon" else [g["coordinates"][0]]
        rings = [np.array(r) for r in rings]
        rings = [np.where(r[:, :1] < 0, r + [[360, 0]], r) for r in rings]   # Pacific: shift west longitudes
        inpool = c in cal["order"]
        for r in rings:
            ax.fill(r[:, 0], r[:, 1], color=(S.BLUE if c in carec else S.TEAL) if inpool else S.RULE,
                    alpha=0.35 if inpool else 0.6, lw=0)
            ax.plot(r[:, 0], r[:, 1], color=S.INK2, lw=0.4)
        big = max(rings, key=len)
        cen[c] = big.mean(0)
    for a, b in cal["edges"]:
        k = cal["n_both"].get(f"{a}-{b}", cal["n_both"].get(f"{b}-{a}", 1))
        (x1, y1), (x2, y2) = cen[a], cen[b]
        ax.plot([x1, x2], [y1, y2], color=S.TERRA, lw=0.8 + 0.12 * k, alpha=0.85, solid_capstyle="round")
    for c in cal["order"]:
        ax.plot(*cen[c], "o", ms=4, color=S.NAVY)
        ax.text(cen[c][0] + 0.8, cen[c][1] + 0.8, c, fontsize=7, color=S.NAVY)
    ax.set_xlim(38, 192); ax.set_ylim(-24, 56); ax.set_aspect(1.15)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.plot([], [], "s", color=S.BLUE, alpha=0.5, label="CAREC members in the pool")
    ax.plot([], [], "s", color=S.TEAL, alpha=0.5, label="other ADB developing members in the pool")
    ax.plot([], [], "-", color=S.TERRA, lw=2, label="shared-earthquake link found in the catalogue")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    S.save(fig, Path(path))
