"""Real-world check of the earthquake assumptions behind the ADB model.

Data (public, fetched from GitHub mirrors reachable from this environment; cached in data/real/):
  * Earthquake catalogue: USGS significant earthquakes, M >= 5.5, 1965–2016, as distributed in
    plotly/datasets (earthquakes-23k.csv; the "Significant Earthquakes, 1965–2016" extract of the USGS NEIC catalogue).
  * Country boundaries: datasets/geo-countries (Natural Earth derived) for Kyrgyzstan and Tajikistan.

Questions tested against the brief's model:
  1. Are yearly event counts consistent with a Poisson process (after removing aftershocks)?
  2. What earthquake size corresponds to the brief's 25%-payout trigger frequency (Gutenberg–Richter)?
  3. How often does one earthquake plausibly hit *both* countries? The brief assumes 7.5% / 7.4% of one country's
     triggering events also trigger the other ("shared events"); we measure the share of large quakes within
     damaging distance of both countries.
  4. Are the two countries' event years dependent beyond shared events?
  5. Implied range of the joint-exhaustion probability when the shared-event intensity is set from data.
Run: python -m src.realdata.eq_check  ->  results/realdata_eq.json, figures/paper/realdata_eq.png
"""
import json
import math
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath
from scipy import stats

ROOT = Path(".")
DATA = ROOT / "data/real"
EQ_URL = "https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv"
GEO_URL = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
BOX = (34.0, 46.0, 63.0, 84.0)            # lat min/max, lon min/max: Central Asia window
YEARS = (1965, 2016)
BRIEF = {"t1": {"KG": 0.03883, "TJ": 0.03925}, "shared": {"KG": 0.075, "TJ": 0.074}}


def _fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    eq, geo = DATA / "usgs_m55_central_asia_1965_2016.csv", DATA / "kg_tj_boundaries.geojson"
    if not eq.exists():
        raw = urllib.request.urlopen(EQ_URL, timeout=120).read().decode().splitlines()
        keep = [raw[0]]
        for line in raw[1:]:
            d, la, lo, m = line.split(",")
            if BOX[0] <= float(la) <= BOX[1] and BOX[2] <= float(lo) <= BOX[3]:
                keep.append(line)
        eq.write_text("# source: " + EQ_URL + "\n" + "\n".join(keep) + "\n")
    if not geo.exists():
        g = json.loads(urllib.request.urlopen(GEO_URL, timeout=300).read())
        feats = [f for f in g["features"] if f["properties"]["ISO3166-1-Alpha-2"] in ("KG", "TJ")]
        geo.write_text(json.dumps({"type": "FeatureCollection", "source": GEO_URL, "features": feats}))
    return eq, geo


def load():
    eq, geo = _fetch()
    rows = [l.split(",") for l in eq.read_text().splitlines() if l and not l.startswith(("#", "Date"))]
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


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def dist_to_country(lat, lon, rings):
    """0 inside the country, otherwise the distance (km) to the nearest boundary vertex."""
    for r in rings:
        if MPath(r).contains_point((lon, lat)):
            return 0.0
    return float(min(haversine(lat, lon, r[:, 1], r[:, 0]).min() for r in rings))


def decluster(ev):
    """Gardner–Knopoff (1974) space–time windows: remove events inside the window of a larger event."""
    order = sorted(range(len(ev)), key=lambda i: -ev[i][3])
    removed = set()
    for i in order:
        if i in removed:
            continue
        t, la, lo, m = ev[i]
        L = 10 ** (0.1238 * m + 0.983)
        T = 10 ** (0.032 * m + 2.7389) if m >= 6.5 else 10 ** (0.5409 * m - 0.547)
        for j in range(len(ev)):
            if j != i and j not in removed and ev[j][3] <= m:
                if abs((ev[j][0] - t).days) <= T and haversine(la, lo, ev[j][1], ev[j][2]) <= L:
                    removed.add(j)
    return [e for k, e in enumerate(ev) if k not in removed]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


def analyse(radii=(25, 50, 100), mags=(5.5, 6.0, 6.5)):
    ev_all, polys = load()
    ev_all = [e for e in ev_all if YEARS[0] <= e[0].year <= YEARS[1]]
    ev = decluster(ev_all)
    nyears = YEARS[1] - YEARS[0] + 1
    dist = {c: np.array([dist_to_country(la, lo, polys[c]) for _, la, lo, _ in ev]) for c in ("KG", "TJ")}
    mag = np.array([e[3] for e in ev]); year = np.array([e[0].year for e in ev])
    out = {"data": {"catalogue": EQ_URL, "boundaries": GEO_URL, "years": YEARS, "events_in_window": len(ev_all),
                    "events_declustered": len(ev)}, "radii_km": {}}
    for R in radii:
        hit = {c: dist[c] <= R for c in dist}
        res = {}
        for M in mags:
            sel = mag >= M
            r = {}
            for c, o in (("KG", "TJ"), ("TJ", "KG")):
                own = hit[c] & sel
                both = own & hit[o]
                counts = np.bincount(year[own] - YEARS[0], minlength=nyears)
                mean, var = counts.mean(), counts.var(ddof=1)
                disp = var / mean if mean > 0 else float("nan")
                p_disp = float(stats.chi2.sf((nyears - 1) * disp, nyears - 1)) if mean > 0 else float("nan")
                r[c] = {"events": int(own.sum()), "rate_per_year": float(own.sum() / nyears),
                        "dispersion_index": float(disp), "poisson_p_value": p_disp,
                        "also_hit_other": int(both.sum()), "shared_share": float(both.sum() / max(own.sum(), 1)),
                        "shared_share_ci95": wilson(int(both.sum()), int(own.sum()))}
            # year-level dependence: 2x2 table of (KG year has an event, TJ year has an event)
            yk = np.zeros(nyears, bool); yt = np.zeros(nyears, bool)
            yk[np.unique(year[hit["KG"] & sel] - YEARS[0])] = True
            yt[np.unique(year[hit["TJ"] & sel] - YEARS[0])] = True
            tab = [[int((yk & yt).sum()), int((yk & ~yt).sum())], [int((~yk & yt).sum()), int((~yk & ~yt).sum())]]
            odds, p_f = stats.fisher_exact(tab)
            r["years"] = {"table_both_KGonly_TJonly_none": tab, "fisher_p": float(p_f), "odds_ratio": float(odds),
                          "p_both_observed": tab[0][0] / nyears, "p_both_if_independent": float(yk.mean() * yt.mean())}
            res[str(M)] = r
        out["radii_km"][str(R)] = res
    # Gutenberg–Richter b-value (Aki ML estimate, both countries, 50 km, completeness 5.5, 0.1 binning)
    near = (dist["KG"] <= 50) | (dist["TJ"] <= 50)
    mm = mag[near & (mag >= 5.5)]
    b = math.log10(math.e) / (mm.mean() - (5.5 - 0.05))
    b_se = b / math.sqrt(len(mm))
    gr = {"b": b, "b_se": b_se, "n": int(len(mm))}
    for c in ("KG", "TJ"):
        lam55 = out["radii_km"]["50"]["5.5"][c]["rate_per_year"]
        lam_t = -math.log(1 - BRIEF["t1"][c])
        gr[c] = {"rate_M55": lam55, "trigger_rate_brief": lam_t,
                 "implied_trigger_magnitude": 5.5 + math.log10(lam55 / lam_t) / b if lam55 > 0 else None}
    out["gutenberg_richter"] = gr
    out["implied_scenarios"] = implied_scenarios(out)
    return out, ev, dist, polys


def implied_scenarios(out, R="50", M="6.0"):
    """Set the shared-event intensity m from the observed shared share (m = observed share / brief share) and
    recompute the exact joint targets of the two-country model, at the point estimate and the 95% bounds."""
    from src.tensor.synthetic import SyntheticModel, topology
    from src.rdpo.problem import PortfolioProblem
    r = out["radii_km"][R][M]
    share = (r["KG"]["also_hit_other"] + r["TJ"]["also_hit_other"]) / max(r["KG"]["events"] + r["TJ"]["events"], 1)
    lo, hi = wilson(r["KG"]["also_hit_other"] + r["TJ"]["also_hit_other"], r["KG"]["events"] + r["TJ"]["events"])
    brief = 0.5 * (BRIEF["shared"]["KG"] + BRIEF["shared"]["TJ"])
    rows = {}
    for lab, s in (("observed", share), ("ci_low", lo), ("ci_high", hi), ("brief", brief)):
        m = s / brief
        try:
            model = SyntheticModel(2, topology("chain", 2), m=m)
        except AssertionError:
            rows[lab] = {"shared_share": s, "m": m, "error": "shared rate exceeds total rate"}
            continue
        vals = {t: PortfolioProblem(model, t, threshold=80e6).exact_dense()
                for t in ("joint_attachment", "joint_exhaustion", "exceedance")}
        rows[lab] = {"shared_share": s, "m": m, **vals}
    return {"radius_km": float(R), "min_magnitude": float(M), "rows": rows}


def figure(out, ev, dist, polys, path="figures/paper/realdata_eq.png"):
    import matplotlib.pyplot as plt
    from src.viz import style as S
    S.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw={"width_ratios": [1.1, 1.2], "wspace": 0.28})
    ax = axes[0]
    for c, col in (("KG", S.BLUE), ("TJ", S.TEAL)):
        for r in polys[c]:
            ax.plot(r[:, 0], r[:, 1], color=col, lw=1.1)
    mag = np.array([e[3] for e in ev]); la = np.array([e[1] for e in ev]); lo = np.array([e[2] for e in ev])
    both = (dist["KG"] <= 50) & (dist["TJ"] <= 50)
    one = ((dist["KG"] <= 50) | (dist["TJ"] <= 50)) & ~both
    far = ~(both | one)
    ax.scatter(lo[far], la[far], s=(mag[far] - 5) * 14, color=S.MUTED, alpha=0.35, lw=0, label="other events")
    ax.scatter(lo[one], la[one], s=(mag[one] - 5) * 14, color=S.BLUE, alpha=0.8, lw=0, label="within 50 km of one country")
    ax.scatter(lo[both], la[both], s=(mag[both] - 5) * 14, color=S.TERRA, alpha=0.95, lw=0, label="within 50 km of both")
    ax.set_xlim(66, 81); ax.set_ylim(35.5, 44.5); ax.set_aspect(1 / math.cos(math.radians(40)))
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("(a) M ≥ 5.5 main shocks, 1965–2016", fontsize=9)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax = axes[1]
    xs, labels = [], []
    k = 0
    for R in ("25", "50", "100"):
        for M in ("5.5", "6.0", "6.5"):
            for c, col in (("KG", S.BLUE), ("TJ", S.TEAL)):
                r = out["radii_km"][R][M][c]
                lo_, hi_ = r["shared_share_ci95"]
                ax.errorbar(k, r["shared_share"], yerr=[[r["shared_share"] - lo_], [hi_ - r["shared_share"]]], fmt="o",
                            color=col, ms=4.5, capsize=2)
                k += 1
            labels.append(f"{R} km\nM≥{M}"); xs.append(k - 1.5)
            k += 0.3
        k += 0.6
    ax.axhline(0.075, color=S.TERRA, lw=1.4, ls="--"); ax.text(0, 0.083, "brief: 7.5% / 7.4%", fontsize=7.5, color=S.TERRA)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylabel("share of a country's events that also\nreach the other country (95% CI)")
    ax.set_title("(b) Shared-event share: data vs brief", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    S.save(fig, Path(path))


if __name__ == "__main__":
    out, ev, dist, polys = analyse()
    Path("results/realdata_eq.json").write_text(json.dumps(out, indent=2, default=str))
    figure(out, ev, dist, polys)
    print(json.dumps(out["gutenberg_richter"], indent=1))
    for R, d in out["radii_km"].items():
        for M, r in d.items():
            print(R, M, {c: (r[c]["events"], round(r[c]["shared_share"], 3), round(r[c]["dispersion_index"], 2),
                             round(r[c]["poisson_p_value"], 3)) for c in ("KG", "TJ")}, r["years"]["fisher_p"],
                  r["years"]["p_both_observed"], round(r["years"]["p_both_if_independent"], 3))
    print(json.dumps(out["implied_scenarios"], indent=1))
