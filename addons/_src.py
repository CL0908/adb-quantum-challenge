"""Locate the friend's repo (read-only), import its src/, and keep our outputs in addons/results/."""
import contextlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# inside the friend's repo (addons/ next to src/) use it; otherwise the sibling reproduction clone ../adb-repro
_DEFAULT = HERE.parent if (HERE.parent / "src").is_dir() else HERE.parent / "adb-repro"
REPO = Path(os.environ.get("ADB_REPO", _DEFAULT)).resolve()
# ADDON_OUT: write results somewhere else (scorecard.py reruns without touching the committed results)
OUT = Path(os.environ.get("ADDON_OUT", HERE / "results"))
# shared data library, organised by field (not by project); see DATA/README.md
# DATA_ROOT if set; else E:/data when that drive is there; else a local cache next to the add-ons (public data is
# downloaded into it on first use, see fetch_usgs.py)
DATA = Path(os.environ["DATA_ROOT"]) if os.environ.get("DATA_ROOT") else (Path("E:/data") if Path("E:/data").is_dir() else HERE / "data_cache")
sys.path.insert(0, str(REPO))


@contextlib.contextmanager
def in_repo():
    """Friend functions that read Path('results') relatively must run with cwd = REPO (read-only)."""
    old = os.getcwd()
    os.chdir(REPO)
    try:
        yield
    finally:
        os.chdir(old)


def seed(default):
    """A module's seed; ADDON_SEED_OFFSET shifts every seed at once (also in worker processes) for seed-stability runs."""
    return default + int(os.environ.get("ADDON_SEED_OFFSET", "0"))


def load(name):
    return json.loads((REPO / "results" / name).read_text())


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=float))
    return p
