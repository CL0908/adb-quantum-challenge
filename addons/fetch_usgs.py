"""Download USGS ComCat earthquakes (with depth) and ShakeMap grids into the field-organised data library:
  DATA/seismology/catalogs/usgs_comcat/carec_m55_1965_2016.csv      (same window and threshold as the friend's
                                                                     catalogue, which has no depth column)
  DATA/seismology/shakemap/<event id>_grid.xml.gz                    (M >= 6, 1973–2016, events within 300 km of a
                                                                     CAREC country: validation set of the archived eq_footprint)
Source: USGS FDSN event web service and ComCat product pages (public, no login). Resumable; manifest.jsonl next to
the data records url, bytes and sha256 of every file.

    python fetch_usgs.py
"""
import gzip
import hashlib
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _src import DATA, REPO

FDSN = "https://earthquake.usgs.gov/fdsnws/event/1/query"
BOX = "&minlatitude=22&maxlatitude=56&minlongitude=38&maxlongitude=122"   # the friend's CAREC window
CAT = DATA / "seismology" / "catalogs" / "usgs_comcat"
SM = DATA / "seismology" / "shakemap"
NEAR_KM = 300.0


def get(url, tries=4, timeout=120):
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def log(path, url, b):
    with open(path.parent / "manifest.jsonl", "a", encoding="utf-8") as m:
        m.write(json.dumps({"file": str(path.relative_to(DATA)), "url": url, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}) + "\n")


EXTENDED = {                                                                # eq_data.py (earthquake data upgrade)
    "carec_m55_1965_2025.csv": [("1965-01-01", "2025-12-31T23:59:59", 5.5)],
    "carec_m45_1973_2025.csv": [("1973-01-01", "1999-12-31T23:59:59", 4.5), ("2000-01-01", "2025-12-31T23:59:59", 4.5)],
    "carec_m60_1900_1964.csv": [("1900-01-01", "1964-12-31T23:59:59", 6.0)],   # ~92% ISC-GEM origins in ComCat
}


def extended():
    """Longer / smaller-magnitude ComCat windows; split queries stay under the 20,000-event limit."""
    CAT.mkdir(parents=True, exist_ok=True)
    for name, parts in EXTENDED.items():
        out = CAT / name
        if out.exists():
            continue
        text, urls = "", []
        for k, (t0, t1, m) in enumerate(parts):
            url = f"{FDSN}?format=csv&orderby=time-asc&starttime={t0}&endtime={t1}&minmagnitude={m}{BOX}"
            body = get(url, timeout=300).decode()
            text += body if k == 0 else body.split("\n", 1)[1]
            urls.append(url)
        b = text.encode()
        out.write_bytes(b); log(out, " | ".join(urls), b)
        print(name, text.count("\n") - 1, "events", flush=True)


def catalogue():
    CAT.mkdir(parents=True, exist_ok=True)
    out = CAT / "carec_m55_1965_2016.csv"
    if not out.exists():
        url = f"{FDSN}?format=csv&orderby=time-asc&starttime=1965-01-01&endtime=2016-12-31T23:59:59&minmagnitude=5.5{BOX}"
        b = get(url)
        out.write_bytes(b); log(out, url, b)
    return out


def carec_rings():
    g = json.loads((REPO / "data" / "real" / "carec_boundaries.geojson").read_text())
    rings = []
    for f in g["features"]:
        geo = f["geometry"]
        rings += [np.array(p[0]) for p in geo["coordinates"]] if geo["type"] == "MultiPolygon" else [np.array(geo["coordinates"][0])]
    return np.concatenate(rings)                                   # boundary vertices (lon, lat)


def near_carec(lat, lon, pts):
    la1, lo1, la2, lo2 = map(np.radians, (lat, lon, pts[:, 1], pts[:, 0]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h)).min() <= NEAR_KM


def shakemap(eid):
    dst = SM / f"{eid}_grid.xml.gz"
    if dst.exists():
        return eid, "cached"
    d = json.loads(get(f"{FDSN}?eventid={eid}&format=geojson"))
    prods = d["properties"]["products"].get("shakemap", [])
    for p in prods:
        c = p.get("contents", {})
        key = next((k for k in ("download/grid.xml", "download/grid.xml.zip") if k in c), None)
        if key:
            url = c[key]["url"]
            b = get(url, timeout=300)
            if key.endswith(".zip"):
                import io, zipfile
                b = zipfile.ZipFile(io.BytesIO(b)).read("grid.xml")
            gz = gzip.compress(b)
            dst.with_suffix(".part").write_bytes(gz); dst.with_suffix(".part").replace(dst)
            log(dst, url, gz)
            return eid, "ok"
    return eid, "no grid"


def main():
    catalogue()
    SM.mkdir(parents=True, exist_ok=True)
    url = f"{FDSN}?format=geojson&starttime=1973-01-01&endtime=2016-12-31T23:59:59&minmagnitude=6.0&producttype=shakemap{BOX}"
    feats = json.loads(get(url))["features"]
    pts = carec_rings()
    ids = [f["id"] for f in feats if near_carec(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0], pts)]
    (SM / "events.json").write_text(json.dumps({"query": url, "near_km": NEAR_KM, "ids": ids}, indent=1))
    print(f"{len(feats)} M>=6 events with ShakeMap; {len(ids)} within {NEAR_KM:.0f} km of a CAREC country", flush=True)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda e: _safe(e), ids))
    for s in ("ok", "cached", "no grid", "failed"):
        print(s, sum(r[1] == s for r in res))


def _safe(eid):
    try:
        return shakemap(eid)
    except Exception as e:
        print("FAIL", eid, re.sub(r"\s+", " ", str(e))[:120], flush=True)
        return eid, "failed"


if __name__ == "__main__":
    main()
