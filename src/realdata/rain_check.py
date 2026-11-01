"""Meteorological calibration of the rain cover from station data (NOAA GHCN-Daily, AWS open-data mirror).

The bond's rain cover is parametric: it pays when an extreme-precipitation index exceeds a level with an annual
exceedance probability of 2% (a 1-in-50-year design). The brief assumes that 25% of the smaller country's rain
triggers are shared events that hit both countries. We measure that share in daily station data.

Method:
  * Stations: per country, the 6 GHCN-Daily stations with the longest precipitation records (>= 30 years with
    >= 300 valid days, 1951–2023); quality-flagged values and the code values in [99, 100) mm are dropped.
  * Country index: each station's daily precipitation divided by its median annual maximum (removes climate
    differences between stations); the country index of a day is the maximum over its stations reporting that day.
  * Country events at annual exceedance probability q: days on which the country index exceeds the (1−q)-quantile
    of the country's annual maxima, merged into events when less than 3 days apart.
  * Shared share at level q: for a pair of countries, an event of the rarer country is shared when the other
    country has an event within ±1 day; share = shared events / events of the rarer country (the brief's definition:
    shared rate = 25% of the smaller rate). Years where both countries report >= 300 days only; Wilson 95% intervals.
  * Levels q = 50%, 20%, 10% (and 5% where enough events exist); the trend in q indicates the tail dependence
    towards the 2% design level.
Run: python -m src.realdata.rain_check -> results/realdata_rain.json, data/real/ghcn_country_index.csv.gz
"""
import csv
import gzip
import io
import json
import math
import subprocess
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

DATA = Path("data/real")
S3 = "https://noaa-ghcn-pds.s3.amazonaws.com"
FIPS = {"KG": "KG", "TI": "TJ", "UZ": "UZ", "KZ": "KZ", "TX": "TM", "PK": "PK", "AJ": "AZ", "GG": "GE", "MG": "MN"}
YEARS = (1951, 2023)
NST = 6
NEIGHBOURS = [("KG", "TJ"), ("KG", "UZ"), ("KG", "KZ"), ("TJ", "UZ"), ("UZ", "TM"), ("UZ", "KZ"), ("KZ", "TM"),
              ("AZ", "GE"), ("TJ", "PK"), ("KZ", "MN")]
LEVELS = (0.5, 0.2, 0.1, 0.05)


def _get(url, timeout=180):
    return urllib.request.urlopen(url, timeout=timeout).read()


def select_stations():
    inv = DATA / "ghcnd-inventory-prcp.txt"
    if not inv.exists():
        lines = _get(f"{S3}/ghcnd-inventory.txt", 600).decode().splitlines()
        inv.write_text("\n".join(l for l in lines if l[31:35] == "PRCP" and l[:2] in FIPS) + "\n")
    st = defaultdict(list)
    for l in inv.read_text().splitlines():
        sid, y0, y1 = l[:11], int(l[36:40]), int(l[41:45])
        span = min(y1, YEARS[1]) - max(y0, YEARS[0])
        if span >= 30:
            st[FIPS[sid[:2]]].append((span, sid))
    return {c: [s for _, s in sorted(v, reverse=True)[:NST]] for c, v in st.items()}


def station_prcp(sid):
    """Daily PRCP (mm) for one station, cached as a compact gzip CSV."""
    cache = DATA / "ghcn" / f"{sid}.csv.gz"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw = _get(f"{S3}/csv/by_station/{sid}.csv").decode().splitlines()
        out = io.StringIO()
        for l in raw:
            p = l.split(",")
            if len(p) > 5 and p[2] == "PRCP" and p[5] == "" and p[3] != "":
                y = int(p[1][:4])
                if YEARS[0] <= y <= YEARS[1]:
                    out.write(f"{p[1]},{int(p[3]) / 10:.1f}\n")
        cache.write_bytes(gzip.compress(out.getvalue().encode()))
    rows = gzip.decompress(cache.read_bytes()).decode().splitlines()
    out = {}
    for r in rows:
        k, v = r.split(",")
        v = float(v)
        # 99.1 / 99.3 / 99.6 / 99.8 mm occur hundreds of times in former-Soviet station records: source codes
        # (trace / missing markers), not rainfall. Values in [99, 100) mm are dropped.
        if not 99.0 <= v < 100.0:
            out[k] = v
    return out


def country_index(stations):
    """Daily country index (max over stations of precip / station median annual max) and valid years."""
    series = {}
    for sid in stations:
        d = station_prcp(sid)
        byyear = defaultdict(list)
        for k, v in d.items():
            byyear[int(k[:4])].append(v)
        good = {y for y, v in byyear.items() if len(v) >= 300}
        if len(good) < 20:
            continue
        med = float(np.median([max(byyear[y]) for y in good]))
        if med <= 0:
            continue
        series[sid] = ({k: v / med for k, v in d.items() if int(k[:4]) in good}, good)
    idx, cover = {}, defaultdict(int)
    for sid, (d, good) in series.items():
        for k, v in d.items():
            idx[k] = max(idx.get(k, 0.0), v)
    days_per_year = defaultdict(int)
    for k in idx:
        days_per_year[int(k[:4])] += 1
    years = sorted(y for y, n in days_per_year.items() if n >= 300)
    return idx, years, list(series)


def events(idx, years, q):
    yset = set(years)
    amax = {}
    for k, v in idx.items():
        y = int(k[:4])
        if y in yset:
            amax[y] = max(amax.get(y, 0.0), v)
    thr = float(np.quantile(list(amax.values()), 1 - q))
    days = sorted(date.fromisoformat(f"{k[:4]}-{k[4:6]}-{k[6:]}") for k, v in idx.items() if v > thr and int(k[:4]) in yset)
    ev = []
    for d in days:
        if not ev or (d - ev[-1][-1]).days >= 3:
            ev.append([d])
        else:
            ev[-1].append(d)
    return [(e[0], e[-1]) for e in ev], thr


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(c - h, 0.0), min(c + h, 1.0)]


def shared(eva, evb, years):
    ys = set(years)
    a = [e for e in eva if e[0].year in ys]; b = [e for e in evb if e[0].year in ys]
    rare, other = (a, b) if len(a) <= len(b) else (b, a)
    hit = 0
    for s0, s1 in rare:
        if any((o0 - s1).days <= 1 and (s0 - o1).days <= 1 for o0, o1 in other):
            hit += 1
    return {"events_A": len(a), "events_B": len(b), "years": len(ys), "shared": hit, "share": hit / max(len(rare), 1),
            "share_ci95": wilson(hit, len(rare))}


def run():
    st = select_stations()
    ci = {}
    for c, sids in st.items():
        idx, years, used = country_index(sids)
        ci[c] = {"idx": idx, "years": years, "stations": used}
    out = {"source": f"{S3} (NOAA GHCN-Daily)", "stations": {c: v["stations"] for c, v in ci.items()},
           "years_with_data": {c: [v["years"][0], v["years"][-1], len(v["years"])] if v["years"] else None for c, v in ci.items()},
           "pairs": {}}
    for a, b in NEIGHBOURS:
        if a not in ci or b not in ci:
            continue
        common = sorted(set(ci[a]["years"]) & set(ci[b]["years"]))
        res = {}
        for q in LEVELS:
            ea, _ = events(ci[a]["idx"], ci[a]["years"], q)
            eb, _ = events(ci[b]["idx"], ci[b]["years"], q)
            res[str(q)] = shared(ea, eb, common)
        out["pairs"][f"{a}-{b}"] = res
    # pooled share across neighbouring pairs (pairs with >= 20 common years)
    pooled = {}
    for q in LEVELS:
        k = n = 0
        for pr, r in out["pairs"].items():
            x = r[str(q)]
            if x["years"] >= 20:
                k += x["shared"]; n += min(x["events_A"], x["events_B"])
        pooled[str(q)] = {"shared": k, "events": n, "share": k / max(n, 1), "share_ci95": wilson(k, n)}
    out["pooled"] = pooled
    # implied two-country bond: earthquake share from the loss-data backtest, rain share from these data (q = 20%)
    try:
        from src.tensor.synthetic import SyntheticModel, topology
        from src.rdpo.problem import PortfolioProblem
        eq = json.loads(Path("results/realdata_losses.json").read_text())["tests"]["eq_noaa_1900"]
        brief_eq = 0.5 * (0.075 + 0.074)
        rows = {}
        for lab, m, sr in (("brief", 1.0, 0.25), ("rain from station data only", 1.0, pooled["0.2"]["share"]),
                           ("earthquake from loss data only", eq["share_of_A"] / brief_eq, 0.25),
                           ("both from data", eq["share_of_A"] / brief_eq, pooled["0.2"]["share"])):
            mdl = SyntheticModel(2, topology("chain", 2), m=m, rain_shared=sr)
            rows[lab] = {"m": m, "rain_shared": sr, **{t: PortfolioProblem(mdl, t, threshold=80e6).exact_dense()
                                                       for t in ("joint_attachment", "exceedance", "joint_exhaustion")}}
        out["implied"] = rows
    except FileNotFoundError:
        pass
    # compact daily country index for reproducibility
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(["country", "yyyymmdd", "index"])
    for c, v in ci.items():
        for k in sorted(v["idx"]):
            w.writerow([c, k, f"{v['idx'][k]:.3f}"])
    (DATA / "ghcn_country_index.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode()))
    return out


if __name__ == "__main__":
    o = run()
    Path("results/realdata_rain.json").write_text(json.dumps(o, indent=2, default=str))
    print(json.dumps(o["years_with_data"]))
    for p, r in o["pairs"].items():
        print(p, {q: (x["years"], x["events_A"], x["events_B"], x["shared"], round(x["share"], 2)) for q, x in r.items()})
    print("pooled", o["pooled"])
    for k, v in o.get("implied", {}).items():
        print(k, {a: (round(b, 6) if isinstance(b, float) else b) for a, b in v.items()})
