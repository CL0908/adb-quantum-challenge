"""Precompute the data behind the public site's interactive stage (frontend/public/data/).

    PYTHONPATH=. python -m src.viz.web_precompute

Writes
  grid.json      exact two-country answers on a 61 x 31 grid of (earthquake shared share, rain shared share).
                 Every grid point is one exact 10x10 term law from the repository's model
                 (src.finance.analysis.term_law -> SyntheticModel + dense_term_distribution, the same functions that
                 src/realdata/param_uncertainty.py and src/finance/analysis.py use) and the metrics are
                 src.finance.analysis.metrics. No sampling anywhere. The brief's point and the data point are stored
                 exactly (not read off the grid) so the page can show them to full precision.
  mc_sample.json one real plain Monte Carlo sample at the brief's base case (16,384 three-year terms, the
                 repository's sampler src.baselines.stronger._simulate) and how often 200 such runs return zero.
  map.json       Kyrgyz Republic and Tajikistan boundaries (simplified), earthquake main shocks of the cached USGS
                 catalogue (M >= 5.5, 1965-2016, declustered), the shared border zone (within 50 km of both
                 countries, the radius used by src/realdata/eq_check.py), and the 10-country CAREC pool map.
  dem.png        Real terrain: AWS Open Data "Terrain Tiles" (Terrarium encoding, zoom 7), resampled to a regular
                 lon/lat grid over the map area and stored as a 16-bit height in metres (R = high byte, G = low byte,
                 offset 500 m, B unused) plus dem.json (extent, size, min/max, source, licence). If the tiles cannot be
                 fetched nothing is written and the site falls back to the flat SVG map; terrain is never invented.
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "frontend" / "public" / "data"

EQ_MAX, RAIN_MAX, STEP = 0.60, 0.30, 0.01       # shared-share axes: 61 x 31 points
ZONE_KM = 50


def _round(x, n):
    return [round(float(v), n) for v in x]


# ------------------------------------------------------------------ exact grid
def grid():
    from src.finance.analysis import term_law, metrics, data_setting, BRIEF_M, BRIEF_RAIN_SHARED
    from src.realdata.param_uncertainty import BRIEF_EQ

    def point(s_eq, rain, m=None):
        P = term_law(s_eq / BRIEF_EQ if m is None else m, rain)
        r = metrics(P)
        return {"joint_exhaustion": r["joint_exhaustion"], "exceed": r["prob_exceed"],
                "delivered": r["delivered_fraction"], "joint_attachment": r["joint_attachment"],
                "expected_loss": r["expected_loss"], "shortfall": r["shortfall"]}

    eq_axis = np.round(np.arange(0, EQ_MAX + 1e-9, STEP), 4)
    rain_axis = np.round(np.arange(0, RAIN_MAX + 1e-9, STEP), 4)
    keys = ["joint_exhaustion", "exceed", "delivered"]
    cube = {k: np.zeros((len(eq_axis), len(rain_axis))) for k in keys}
    for i, s in enumerate(eq_axis):
        for j, r in enumerate(rain_axis):
            p = point(float(s), float(r))
            for k in keys:
                cube[k][i, j] = p[k]
    m_d, rain_d, src_d = data_setting()
    brief = point(BRIEF_EQ, BRIEF_RAIN_SHARED, m=BRIEF_M)
    data = point(m_d * BRIEF_EQ, rain_d, m=m_d)
    return {
        "method": "Exact dense enumeration of the two-country model (10x10 term law), one law per grid point; "
                  "src.finance.analysis.term_law / metrics. No sampling.",
        "eq_axis": eq_axis.tolist(), "rain_axis": rain_axis.tolist(),
        "units": {"joint_exhaustion": "probability both bonds exhausted in one 3-year term",
                  "exceed": "probability the pool's total loss exceeds US$80m in one term",
                  "delivered": "share of expected claims paid by one shared US$80m limit"},
        "values": {k: [[float(f"{v:.9g}") for v in row] for row in cube[k]] for k in keys},
        "brief": {"eq": BRIEF_EQ, "rain": BRIEF_RAIN_SHARED, "m": BRIEF_M, **brief},
        "data": {"eq": m_d * BRIEF_EQ, "rain": rain_d, "m": m_d, "source": src_d, **data},
    }


# ------------------------------------------------------------------ Monte Carlo
def mc_sample(n=16384, seed=0, repeats=400):
    from src.rdpo.problem import PortfolioProblem, adb_model, UNIT
    from src.baselines.stronger import _simulate
    P = PortfolioProblem(adb_model("adb-base"), "joint_exhaustion", threshold=80e6)
    out = _simulate(P, None, n, np.random.default_rng(seed))
    li, lj = out["Li"] * UNIT / 1e6, out["Lj"] * UNIT / 1e6
    cells = {}
    for a, b in zip(li.tolist(), lj.tolist()):
        cells[(a, b)] = cells.get((a, b), 0) + 1
    exact = P.exact_dense()
    zero = sum(int((_simulate(P, None, n, np.random.default_rng(1000 + k))["f"] > 0).sum()) == 0 for k in range(repeats))
    exceed = PortfolioProblem(adb_model("adb-base"), "exceedance", threshold=80e6)
    from src.finance.analysis import term_law, LOSS
    law = term_law()                                   # exact 10x10 joint law at the brief's base case
    return {"n": n, "seed": seed, "exact_joint_exhaustion": exact,
            "levels_musd": [float(x) / 1e6 for x in LOSS], "law": [[float(f"{v:.12g}") for v in row] for row in law], "exact_exceed": exceed.exact_dense(),
            "hits_joint_exhaustion": int(out["f"].sum()),
            "hits_exceed": int(((li + lj) > 80 + 1e-9).sum()),
            "cells": [[a, b, c] for (a, b), c in sorted(cells.items())],
            "zero_runs": zero, "repeats": repeats, "zero_share": zero / repeats,
            "expected_zero_share": float(np.exp(-n * exact))}


# ------------------------------------------------------------------ maps
def _dp(pts, eps):
    """Douglas-Peucker on an (n, 2) array."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        d = q - p
        seg = np.hypot(*d) or 1e-12
        dist = np.abs(d[0] * (pts[a + 1:b, 1] - p[1]) - d[1] * (pts[a + 1:b, 0] - p[0])) / seg
        k = int(dist.argmax())
        if dist[k] > eps:
            k += a + 1
            keep[k] = True
            stack += [(a, k), (k, b)]
    return pts[keep]


def _rings(feature, eps, min_area=0.0):
    g = feature["geometry"]
    polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    out = []
    for poly in polys:
        ring = np.array(poly[0])
        area = 0.5 * abs(np.dot(ring[:, 0], np.roll(ring[:, 1], 1)) - np.dot(ring[:, 1], np.roll(ring[:, 0], 1)))
        if area < min_area:
            continue
        # closed ring: split at the vertex farthest from the start so no segment is degenerate
        k = int(np.hypot(*(ring - ring[0]).T).argmax())
        s = np.vstack([_dp(ring[:k + 1], eps)[:-1], _dp(ring[k:], eps)])
        if len(s) >= 4:
            out.append([[round(float(x), 2), round(float(y), 2)] for x, y in s])
    return out


def _dist_field(rings, lon, lat):
    """Distance in km to a country: 0 inside, else to the nearest boundary vertex (as src/realdata/eq_check.py)."""
    from matplotlib.path import Path as MPath
    from src.realdata.eq_check import haversine
    pts = np.column_stack([lon.ravel(), lat.ravel()])
    d = np.full(len(pts), np.inf)
    inside = np.zeros(len(pts), bool)
    for r in rings:
        inside |= MPath(r).contains_points(pts)
    verts = np.vstack(rings)
    for k in range(0, len(pts), 400):
        blk = pts[k:k + 400]
        dd = haversine(blk[:, 1][:, None], blk[:, 0][:, None], verts[None, :, 1], verts[None, :, 0])
        d[k:k + 400] = dd.min(1)
    d[inside] = 0.0
    return d.reshape(lon.shape)


def maps():
    from src.realdata import eq_check
    ev, polys = eq_check.load()
    ev = [e for e in ev if eq_check.YEARS[0] <= e[0].year <= eq_check.YEARS[1]]
    main = eq_check.decluster(ev)
    geo = json.loads((ROOT / "data/real/kg_tj_boundaries.geojson").read_text())
    countries = {f["properties"]["ISO3166-1-Alpha-2"]: _rings(f, 0.02, 0.01) for f in geo["features"]}
    # shared border zone: within ZONE_KM of both countries
    lon, lat = np.meshgrid(np.arange(66.0, 82.01, 0.1), np.arange(35.0, 44.01, 0.1))
    dk, dt = _dist_field(polys["KG"], lon, lat), _dist_field(polys["TJ"], lon, lat)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cs = plt.contourf(lon, lat, np.maximum(dk, dt), levels=[-1, ZONE_KM])
    zone = [[[round(float(x), 2), round(float(y), 2)] for x, y in seg] for seg in cs.allsegs[0] if len(seg) > 3]
    plt.close("all")
    quakes = []
    for t, la, lo, m in main:
        quakes.append([round(la, 2), round(lo, 2), round(m, 1), t.year])
    # which main shocks reach both countries (within the zone radius): the shared events
    d_kg = np.array([eq_check.dist_to_country(la, lo, polys["KG"]) for _, la, lo, _ in main])
    d_tj = np.array([eq_check.dist_to_country(la, lo, polys["TJ"]) for _, la, lo, _ in main])
    for q, a, b in zip(quakes, d_kg, d_tj):
        q.append(int(a <= ZONE_KM and b <= ZONE_KM))

    # CAREC pool (10 countries)
    carec = json.loads((ROOT / "data/real/carec_boundaries.geojson").read_text())
    res = json.loads((ROOT / "results/wp2_carec.json").read_text())["calibration"]
    cc = {}
    for f in carec["features"]:
        iso = f["properties"]["ISO3166-1-Alpha-2"]
        rings = _rings(f, 0.06, 0.2)
        allp = np.vstack([np.array(r) for r in rings])
        big = max(rings, key=len)
        cc[iso] = {"name": f["properties"]["name"], "rings": rings,
                   "c": [round(float(np.mean(np.array(big)[:, 0])), 2), round(float(np.mean(np.array(big)[:, 1])), 2)],
                   "p25": res["tails"][iso][0] if iso in res["tails"] else None}
    return {"bbox": [66.9, 36.3, 80.7, 43.6], "countries": countries, "zone_km": ZONE_KM, "zone": zone,
            "quakes": quakes,
            "carec": {"countries": cc, "edges": res["edges"], "order": res["order"]},
            "catalogue": "USGS significant earthquakes M>=5.5, 1965-2016, Gardner-Knopoff declustered main shocks"}


# ------------------------------------------------------------------ terrain
DEM_BOX = (66.4, 35.9, 81.2, 44.0)           # lon0, lat0, lon1, lat1
DEM_Z = 7
DEM_STEP = 0.02                              # degrees per output pixel (about 2 km)
DEM_OFFSET = 500
DEM_URLS = ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
            "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"]


def _tile(z, x, y):
    import io
    import urllib.request
    from PIL import Image
    err = None
    for u in DEM_URLS:
        try:
            with urllib.request.urlopen(u.format(z=z, x=x, y=y), timeout=30) as r:
                im = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"), dtype=np.float64)
            return im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768
        except Exception as e:  # noqa: BLE001
            err = e
    raise RuntimeError(f"terrain tile {z}/{x}/{y} unavailable: {err}")


def dem():
    from PIL import Image
    lon0, lat0, lon1, lat1 = DEM_BOX
    n = 2 ** DEM_Z

    def tx(lon):
        return (lon + 180) / 360 * n

    def ty(lat):
        r = np.radians(lat)
        return (1 - np.log(np.tan(r) + 1 / np.cos(r)) / np.pi) / 2 * n

    xa, xb = int(np.floor(tx(lon0))), int(np.floor(tx(lon1)))
    ya, yb = int(np.floor(ty(lat1))), int(np.floor(ty(lat0)))
    mosaic = np.zeros(((yb - ya + 1) * 256, (xb - xa + 1) * 256))
    for yy in range(ya, yb + 1):
        for xx in range(xa, xb + 1):
            mosaic[(yy - ya) * 256:(yy - ya + 1) * 256, (xx - xa) * 256:(xx - xa + 1) * 256] = _tile(DEM_Z, xx, yy)
    lons = np.arange(lon0, lon1 + 1e-9, DEM_STEP)
    lats = np.arange(lat1, lat0 - 1e-9, -DEM_STEP)
    fx = (tx(lons) - xa) * 256 - 0.5
    fy = (ty(lats) - ya) * 256 - 0.5
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    ux, uy = (fx - x0)[None, :], (fy - y0)[:, None]
    x1, y1 = np.minimum(x0 + 1, mosaic.shape[1] - 1), np.minimum(y0 + 1, mosaic.shape[0] - 1)
    g = (mosaic[np.ix_(y0, x0)] * (1 - ux) * (1 - uy) + mosaic[np.ix_(y0, x1)] * ux * (1 - uy)
         + mosaic[np.ix_(y1, x0)] * (1 - ux) * uy + mosaic[np.ix_(y1, x1)] * ux * uy)
    g = np.clip(np.round(g + DEM_OFFSET), 0, 65535).astype(np.uint16)
    rgb = np.dstack([(g >> 8).astype(np.uint8), (g & 255).astype(np.uint8), np.zeros_like(g, dtype=np.uint8)])
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(OUT / "dem.png", optimize=True)
    print(f"{(OUT / 'dem.png').relative_to(ROOT)}: {(OUT / 'dem.png').stat().st_size / 1024:.0f} kB, {g.shape[1]} x {g.shape[0]}")
    return {"extent": [lon0, lat0, lon1, lat1], "width": int(g.shape[1]), "height": int(g.shape[0]), "step_deg": DEM_STEP,
            "offset_m": DEM_OFFSET, "min_m": float(g.min()) - DEM_OFFSET, "max_m": float(g.max()) - DEM_OFFSET,
            "source": "AWS Open Data Terrain Tiles (Terrarium), zoom 7, resampled bilinearly",
            "licence": "Mapzen Terrain Tiles on AWS Open Data: composite of SRTM, ETOPO1, GMTED2010 and other public DEMs; "
                       "see https://github.com/tilezen/joerd/blob/master/docs/attribution.md"}


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, separators=(",", ":")))
    print(f"{p.relative_to(ROOT)}: {p.stat().st_size / 1024:.0f} kB")


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    parts = sys.argv[1:] or ["grid", "mc", "map", "dem"]
    t0 = time.time()
    if "grid" in parts:
        g = grid()
        write("grid.json", g)
        print(f"  grid {len(g['eq_axis'])} x {len(g['rain_axis'])} in {time.time() - t0:.1f}s")
    if "mc" in parts:
        m = mc_sample()
        write("mc_sample.json", m)
        print(f"  MC: {m['hits_joint_exhaustion']} joint-exhaustion hits of {m['n']}, zero in {m['zero_runs']}/{m['repeats']} runs")
    if "map" in parts:
        write("map.json", maps())
    if "dem" in parts:
        try:
            write("dem.json", dem())
        except Exception as e:  # noqa: BLE001  (keep any earlier dem.png/dem.json; the site has a 2D fallback)
            print(f"  terrain not updated: {e}")


if __name__ == "__main__":
    main()
