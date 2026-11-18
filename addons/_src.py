"""Locate the the main author's repo (read-only), import its src/, and keep our outputs in addons/results/."""
import contextlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# inside the the main author's repo (addons/ next to src/) use it; otherwise the sibling reproduction clone ../adb-repro
_DEFAULT = HERE.parent if (HERE.parent / "src").is_dir() else HERE.parent / "adb-repro"
REPO = Path(os.environ.get("ADB_REPO", _DEFAULT)).resolve()
OUT = HERE / "results"
# shared data library, organised by field (not by project); see DATA/README.md
# set ADB_DATA_DIR (legacy name DATA_ROOT also accepted); default is a repo-relative addons/data (may not exist)
DATA = Path(os.environ.get("ADB_DATA_DIR") or os.environ.get("DATA_ROOT") or HERE / "data")
sys.path.insert(0, str(REPO))


@contextlib.contextmanager
def in_repo():
    """Main-author functions that read Path('results') relatively must run with cwd = REPO (read-only)."""
    old = os.getcwd()
    os.chdir(REPO)
    try:
        yield
    finally:
        os.chdir(old)


def load(name):
    return json.loads((REPO / "results" / name).read_text())


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=float))
    return p
