"""Export everything the public demo site needs. The site is a React app in frontend/; `npm run build` there writes
the static site to web/ (committed, so Vercel serves static files with no build step). This script fills
frontend/public/ with the inputs; src/viz/web_precompute.py writes the interactive stage's data beside it.

    PYTHONPATH=. python -m src.viz.export_web_data [--fallback]

Writes frontend/public/data/values.json (every number shown on the page: build_paper._collect_values() plus the slide-only
values of build_slides.slide_values()), frontend/public/data/fallback.json (the "try it" panel's precomputed answers, produced
by the same function api/estimate.py runs, used when the live function is unreachable), and copies the three PDFs into frontend/public/. No result number is typed by hand anywhere on the page.
"""
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "frontend" / "public"
PDFS = ["docs/RDPO_Proposal_Short.pdf", "docs/RDPO_Proposal.pdf", "docs/RDPO_Presentation.pdf"]
SCENARIOS = ["adb-base", "adb-independent", "adb-common-shock", "adb-rain-stress", "adb-rain-cover"]
TARGETS = ["expected_loss", "joint_attachment", "exceedance", "stop_loss", "joint_exhaustion"]
BUDGETS = [2048, 4096, 8192, 16384, 32768, 65536]


def copy(rel, sub):
    dst = WEB / sub / Path(rel).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / rel, dst)


def values():
    sys.path.insert(0, str(ROOT))
    from src.viz.build_paper import _collect_values
    from src.viz.build_slides import slide_values
    v = _collect_values()
    v.update(slide_values(v))
    return v


def fallback():
    """Precomputed answers for 5 scenarios x 5 targets x 6 budgets (seed 0), from api/estimate.py's own run()."""
    sys.path.insert(0, str(ROOT / "api"))
    import estimate
    out = {}
    for s in SCENARIOS:
        for t in TARGETS:
            for b in BUDGETS:
                r = estimate.run({"scenario": s, "target": t, "budget": b, "threshold_musd": 80, "seed": 0})
                keep = ("estimate", "ci", "exact", "mc", "money", "seconds", "paths_used")
                out[f"{s}|{t}|{b}"] = {k: r[k] for k in keep}
    return out


FRONTEND_KEY = re.compile(r"""\bk=["']([A-Z][A-Z0-9_]+)["']|\bv\.([A-Z][A-Z0-9_]+)\b|\bv\[["']([A-Z][A-Z0-9_]+)["']\]""")


def check_frontend_keys(v):
    """Fail (before anything is written) if the site's source reads a value key that this export did not produce.
    Same rule as frontend/scripts/check-values.mjs, which guards `npm run build`."""
    used = {}
    for f in (ROOT / "frontend" / "src").rglob("*.ts*"):
        for m in FRONTEND_KEY.finditer(f.read_text()):
            used.setdefault(m.group(1) or m.group(2) or m.group(3), set()).add(f.name)
    missing = {k: sorted(fs) for k, fs in used.items() if not str(v.get(k, "")).strip()}
    if missing:
        lines = "\n".join(f"  {k}  ({', '.join(fs)})" for k, fs in sorted(missing.items()))
        raise SystemExit(f"export_web_data: {len(missing)} value key(s) read by frontend/src are missing from the export "
                         f"(a values group probably failed, see 'placeholder group failed' above):\n{lines}")


def main():
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    v = values()
    check_frontend_keys(v)
    from src.viz import export_story
    export_story.write(WEB)
    # No merging with the previous file: every value on the page is recomputed from results/*.json, so a stale number cannot survive.
    old = WEB / "data" / "values.json"
    if old.exists():
        prev = json.loads(old.read_text())
        gone = sorted(set(prev) - set(v))
        if gone:
            print(f"dropped {len(gone)} previous values no longer produced, e.g. {gone[:3]}")
        changed = [k for k in v if k in prev and prev[k] != v[k]]
        print(f"{len(changed)} values changed since the previous export")
    (WEB / "data" / "values.json").write_text(json.dumps(v, ensure_ascii=False, indent=1, sort_keys=True))
    t0 = time.time()
    fb = WEB / "data" / "fallback.json"
    if "--fallback" in sys.argv or not fb.exists():     # 150 exact/MC/RDPO runs, about 90 s; the function is unchanged
        fb.write_text(json.dumps(fallback(), separators=(",", ":")))
    for f in PDFS:
        copy(f, ".")
    print(f"frontend/public/data/values.json: {len(v)} values; fallback in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
