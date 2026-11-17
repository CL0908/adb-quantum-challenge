"""Put every dataset this project downloaded into the shared data library DATA_ROOT (default E:/data), organised by
field (not by project), and (re)write DATA_ROOT/catalog.csv + DATA_ROOT/files.sha256.

The friend's repository keeps its own copies under data/real (his code reads them by relative path and run.py
checks that data/ is unchanged), so those are copied from the clean original clone, not moved. Datasets that only
we downloaded are moved. Re-running is safe: existing identical files are skipped, the catalog is rebuilt.

    python repro/data_catalog.py
"""
import csv
import hashlib
import os
import shutil
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_ROOT", "E:/data"))
ORIG = HERE.parent / "adb-quantum-challenge" / "data" / "real"      # the friend's committed data, never modified
FRIEND = "adb-quantum-challenge: src/realdata (friend)"

# (field path, dataset, source, used by, files: (source path or glob, copy|move))
DATASETS = [
    ("meteorology/precipitation/ghcn-daily/stations", "NOAA GHCN-Daily station precipitation (KG, TJ, UZ, KZ, TM, PK, AZ, GE, MN)",
     "https://noaa-ghcn-pds.s3.amazonaws.com (NOAA GHCN-Daily, AWS open data)", FRIEND + "; addons rates_unc, climate_cc (archived: rain_tail)",
     [(ORIG / "ghcn", "*.csv.gz", "copy")]),
    ("meteorology/precipitation/ghcn-daily", "GHCN-Daily precipitation station inventory",
     "https://noaa-ghcn-pds.s3.amazonaws.com/ghcnd-inventory.txt (PRCP rows)", FRIEND, [(ORIG, "ghcnd-inventory-prcp.txt", "copy")]),
    ("meteorology/precipitation/derived", "Daily country precipitation index (max over 6 stations of precip / station median annual max)",
     "derived by the friend's src/realdata/rain_check.py from GHCN-Daily", FRIEND, [(ORIG, "ghcn_country_index.csv.gz", "copy")]),
    ("seismology/catalogs/usgs_m55_1965_2016", "USGS M>=5.5 earthquakes 1965–2016, regional cuts",
     "https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv (USGS significant earthquakes)",
     FRIEND + "; addons rates_unc, eq_cluster", [(ORIG, "usgs_m55_*.csv", "copy")]),
    ("seismology/catalogs/usgs_comcat", "USGS ComCat earthquakes with depth, CAREC window (22–56 N, 38–122 E): M>=5.5 1965–2016 and 1965–2025, M>=4.5 1973–2025, M>=6 1900–1964 (mostly ISC-GEM origins)",
     "https://earthquake.usgs.gov/fdsnws/event/1/query (fetched by addons/fetch_usgs.py; url + sha256 in manifest.jsonl)",
     "addons (archived: eq_footprint, eq_data)", []),
    ("seismology/shakemap", "USGS ShakeMap grids (MMI, PGA, PGV) of M>=6 events 1973–2016 within 300 km of a CAREC country, gzip",
     "USGS ComCat shakemap products (fetched by addons/fetch_usgs.py; url + sha256 in manifest.jsonl)",
     "addons (archived: eq_footprint validation)", []),
    ("disasters/impacts/emdat", "EM-DAT natural disasters by country-year (deaths, affected, damages per hazard)",
     "https://github.com/owid/owid-datasets (Natural disasters (EMDAT))", FRIEND, [(ORIG, "owid_emdat_*.csv", "copy")]),
    ("disasters/impacts/ngdc-noaa-earthquakes", "NGDC/NOAA significant earthquake events and deaths by country-year",
     "https://github.com/owid/owid-datasets (Earthquake deaths (NGDC-NOAA); Significant earthquake events (NGDC-NASA))",
     FRIEND, [(ORIG, "owid_noaa_*.csv", "copy")]),
    ("geography/boundaries/geo-countries", "Country boundaries (KG/TJ, Central Asia 5, CAREC, Asia-Pacific cuts)",
     "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson", FRIEND,
     [(ORIG, "*_boundaries.geojson", "copy")]),
    ("climate/indices/enso/noaa_cpc_oni", "Oceanic Niño Index (3-month running SST anomaly, Niño 3.4), 1950–",
     "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", "addons climate_regime (archived)",
     [(HERE.parent / "addons" / "data", "oni.ascii.txt", "move")]),
    ("climate/projections/nex-gddp-cmip6", "NASA NEX-GDDP-CMIP6 daily precipitation (bias-corrected, 0.25°), 33 models, "
     "historical 1951–2014 + SSP2-4.5 2021–2050, box 37–43.25 N, 68.5–78.5 E", "https://ds.nccs.nasa.gov/thredds (NCSS subset; "
     "fetched by addons/fetch_nex_gddp.py, per-file url + sha256 in manifest.jsonl)", "addons (archived: climate_ens, rain_elt, rain_share_gcm)", []),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def place():
    for field, _, _, _, specs in DATASETS:
        dst = DATA / field
        for src_dir, pat, how in specs:
            for src in sorted(src_dir.glob(pat)):
                dst.mkdir(parents=True, exist_ok=True)
                t = dst / src.name
                if t.exists() and sha(t) == sha(src):
                    if how == "move":
                        src.unlink()
                    continue
                (shutil.move if how == "move" else shutil.copy2)(str(src), str(t))
            if how == "move" and src_dir.exists() and not any(src_dir.iterdir()):
                src_dir.rmdir()


def catalog():
    rows, sums = [], []
    for field, name, source, used, _ in DATASETS:
        d = DATA / field
        fs = [p for p in d.rglob("*") if p.is_file() and p.suffix != ".part"] if d.exists() else []
        if field.endswith("ghcn-daily"):                       # stations/ is its own entry
            fs = [p for p in fs if "stations" not in p.parts]
        rows.append({"field": field.split("/")[0], "path": field, "dataset": name, "source": source, "files": len(fs),
                     "size_mb": round(sum(p.stat().st_size for p in fs) / 2 ** 20, 2), "used_by": used,
                     "catalogued": date.today().isoformat()})
        if not field.startswith("climate/projections"):         # per-file hashes there live in manifest.jsonl
            sums += [f"{sha(p)}  {p.relative_to(DATA).as_posix()}" for p in sorted(fs)]
    with open(DATA / "catalog.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (DATA / "files.sha256").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    place()
    for r in catalog():
        print(f"{r['path']:<55} {r['files']:>5} files {r['size_mb']:>9.2f} MB")
