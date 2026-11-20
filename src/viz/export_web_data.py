"""Export everything the public demo site (web/) needs, so Vercel serves static files with no build step.

    PYTHONPATH=. python -m src.viz.export_web_data

Writes web/data/values.json (every number shown on the page: build_paper._collect_values() plus the slide-only
values of build_slides.slide_values()), web/data/fallback.json (the "try it" panel's precomputed answers, produced
by the same function api/estimate.py runs, used when the live function is unreachable), and copies the figures,
illustrations and PDFs into web/. No result number is typed by hand anywhere on the page.
"""
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
FIGURES = ["figures/slides/exact.svg", "figures/slides/heat.svg", "figures/slides/phase.svg"]
ART = ["docs/art/cat_sitting.svg", "docs/art/cat_peek.svg", "docs/art/cat_team.svg", "docs/art/cat_stopwatch.svg",
       "docs/art/cat_chart.svg", "docs/art/cat_globe.svg", "docs/art/cat_laptop.svg", "docs/art/cat_umbrella.svg"]
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
    (WEB / "data" / "values.json").write_text(json.dumps(v, ensure_ascii=False, indent=1, sort_keys=True))
    t0 = time.time()
    (WEB / "data" / "fallback.json").write_text(json.dumps(fallback(), separators=(",", ":")))
    for f in FIGURES:
        copy(f, "assets")
    for f in ART:
        copy(f, "assets")
    for f in PDFS:
        copy(f, ".")
    print(f"web/data/values.json: {len(v)} values; fallback in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
