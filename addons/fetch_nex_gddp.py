"""Download NASA NEX-GDDP-CMIP6 daily precipitation, subset to the box around the friend's KG / TJ GHCN stations,
into the field-organised data library: DATA/climate/projections/nex-gddp-cmip6/pr_day/<model>/<scenario>/.

Source: NASA NCCS THREDDS NetCDF Subset Service (public, no login). One request per model-scenario-year; files
are NetCDF-3 (readable with scipy.io.netcdf_file). Resumable: existing files are skipped; .part files are retried.
A manifest (manifest.jsonl next to the data) records url, bytes and sha256 of every file.

    python fetch_nex_gddp.py [--workers 6] [--models A,B]
"""
import hashlib
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from _src import DATA

CAT = "https://ds.nccs.nasa.gov/thredds/catalog/AMES/NEX/GDDP-CMIP6"
NCSS = "https://ds.nccs.nasa.gov/thredds/ncss/grid/AMES/NEX/GDDP-CMIP6"
BOX = dict(north=43.25, south=37.0, west=68.5, east=78.5)       # friend's 12 stations: 37.2–42.8 N, 68.8–78.2 E
PERIODS = {"historical": (1951, 2014), "ssp245": (2021, 2050)}   # climate_ens uses 2021–2050
ROOT = DATA / "climate" / "projections" / "nex-gddp-cmip6" / "pr_day"


def get(url, tries=4, timeout=300):
    for k in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=timeout).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def titles(path):
    return re.findall(r'xlink:title="([^"]+)"', get(f"{CAT}/{path}/catalog.xml", timeout=60).decode())


def files(model, scen, var):
    names = re.findall(r'name="(pr_day_[^"]+\.nc)"', get(f"{CAT}/{model}/{scen}/{var}/pr/catalog.xml", timeout=60).decode())
    best = {}
    for n in names:                                   # several versions per year: keep the highest (v2.0 > v1.1 > none)
        m = re.search(r"_(\d{4})(?:_v(\d+)\.(\d+))?\.nc$", n)
        y, ver = int(m.group(1)), (int(m.group(2) or 0), int(m.group(3) or 0))
        if y not in best or ver > best[y][0]:
            best[y] = (ver, n)
    return {y: n for y, (_, n) in best.items()}


def jobs(models):
    """Job list for all models, listed in parallel and cached in jobs.json (delete it to re-list)."""
    cache = ROOT.parent / "jobs.json"
    if cache.exists():
        return [tuple(j) for j in json.loads(cache.read_text()) if j[0] in models and PERIODS[j[1]][0] <= j[3] <= PERIODS[j[1]][1]]
    with ThreadPoolExecutor(8) as ex:
        out = [j for js in ex.map(model_jobs, models) for j in js]
    cache.write_text(json.dumps(out))
    return out


def model_jobs(model):
    try:
        hv, sv = titles(f"{model}/historical"), titles(f"{model}/ssp245")
    except Exception as e:                            # some models lack one of the two scenarios
        print(f"skip {model}: {e}", flush=True)
        return []
    var = next((v for v in hv if v in sv), None)
    if var is None:
        print(f"skip {model}: no common variant", flush=True)
        return []
    out = []
    for scen, (y0, y1) in PERIODS.items():
        fs = files(model, scen, var)
        out += [(model, scen, var, y, fs[y]) for y in range(y0, y1 + 1) if y in fs]
    return out


def fetch(model, scen, var, y, name):
    d = ROOT / model / scen
    d.mkdir(parents=True, exist_ok=True)
    dst = d / name.replace(".nc", "_kgtj.nc")
    if dst.exists() and dst.stat().st_size > 0:
        return None
    q = "&".join(f"{k}={v}" for k, v in BOX.items())
    url = (f"{NCSS}/{model}/{scen}/{var}/pr/{name}?var=pr&{q}&horizStride=1"
           f"&time_start={y}-01-01T00:00:00Z&time_end={y}-12-31T23:59:59Z&accept=netcdf")
    try:
        b = get(url, tries=1)
    except urllib.error.HTTPError as e:                # 360-day calendars (HadGEM3, KACE, UKESM) have no 31 Dec
        if e.code != 400:
            b = get(url)
        else:
            url = url.replace(f"time_end={y}-12-31", f"time_end={y}-12-30")
            b = get(url)
    if b[:3] != b"CDF":
        raise ValueError(f"not NetCDF: {url}")
    part = dst.with_suffix(".part")
    part.write_bytes(b)
    part.replace(dst)
    return {"file": str(dst.relative_to(DATA)), "url": url, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest(),
            "model": model, "scenario": scen, "variant": var, "year": y}


def main():
    a = sys.argv[1:]
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 6
    models = a[a.index("--models") + 1].split(",") if "--models" in a else titles("")
    models = [m for m in models if m not in ("GFDL-CM4_gr2",)]   # second grid of GFDL-CM4, same model
    todo = jobs(models)
    print(f"{len(todo)} files for {len(models)} models -> {ROOT}", flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    done = bad = 0
    with ThreadPoolExecutor(workers) as ex, open(ROOT.parent / "manifest.jsonl", "a", encoding="utf-8") as man:
        futs = {ex.submit(fetch, *j): j for j in todo}
        for f in as_completed(futs):
            try:
                rec = f.result()
                if rec:
                    man.write(json.dumps(rec) + "\n"); man.flush()
                done += 1
            except Exception as e:
                bad += 1
                print(f"FAIL {futs[f][:4]}: {e}", flush=True)
            if (done + bad) % 200 == 0:
                print(f"{done + bad}/{len(todo)} ({bad} failed)", flush=True)
    print(f"finished: {done} ok, {bad} failed", flush=True)


if __name__ == "__main__":
    main()
