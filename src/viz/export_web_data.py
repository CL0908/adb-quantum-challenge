"""Export everything the public demo site needs. The site is a React app in frontend/; `npm run build` there writes
the static site to web/ (committed, so Vercel serves static files with no build step). This script fills
frontend/public/ with the inputs; src/viz/web_precompute.py writes the interactive stage's data beside it.

    PYTHONPATH=. python -m src.viz.export_web_data [--fallback]

Writes frontend/public/data/values.json (every number shown on the page: build_paper._collect_values() plus the slide-only
values of build_slides.slide_values()), frontend/public/data/fallback.json (the "try it" panel's precomputed answers, produced
by the same function api/estimate.py runs, used when the live function is unreachable), and copies the three PDFs into frontend/public/. No result number is typed by hand anywhere on the page.
"""
import json
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


def main():
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    v = values()
    # A value group can fail while its result files are being regenerated; keep the last good value instead of
    # dropping the number from the page (and say so).
    old = WEB / "data" / "values.json"
    if old.exists():
        prev = json.loads(old.read_text())
        kept = sorted(set(prev) - set(v))
        if kept:
            print(f"kept {len(kept)} previous values that could not be recomputed, e.g. {kept[:3]}")
        v = {**prev, **v}
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
