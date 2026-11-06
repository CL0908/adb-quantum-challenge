"""Every number the proposal prints, computed twice by the paper's own code (src/viz/build_paper.py):
once from results_declared/ (what the authors' files say) and once from results/ (our re-run).

    python repro/paper_values.py        # writes repro/paper_values.json
"""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "adb-repro"


def collect():
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import src.viz.build_paper as bp
    out = {}
    for tag in ("results_declared", "results"):
        bp.R = Path(tag)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            vals = {}
            for fn in _groups(bp):
                try:
                    vals.update({k: {"value": v, "group": fn.__name__} for k, v in fn().items()})
                except Exception as ex:
                    vals[f"<{fn.__name__}>"] = {"value": f"FAILED: {type(ex).__name__}: {ex}", "group": fn.__name__}
        out[tag] = vals
    return out


def _groups(bp):
    # same order as build_paper._collect_values
    names = ["e1_values", "e2_values", "e3_values", "q_values", "mk_values", "v2_values", "rd_values", "loss_values",
             "pool_values", "wp2_values", "proj_values", "rain_values", "ap_values", "headline_values", "elt_values",
             "qs_values", "stronger_values", "pu_values", "qs_inline_values"]
    return [getattr(bp, n) for n in names]


if __name__ == "__main__":
    v = collect()
    (HERE / "paper_values.json").write_text(json.dumps(v, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    d, o = v["results_declared"], v["results"]
    same = sum(1 for k in d if k in o and d[k]["value"] == o[k]["value"])
    print(f"placeholders: declared {len(d)}, ours {len(o)}, identical strings {same}")
    for k in sorted(d):
        if k not in o or d[k]["value"] != o[k]["value"]:
            print(f"  {k} [{d[k]['group']}]: {str(d[k]['value'])[:70]!r} -> {str(o.get(k, {}).get('value'))[:70]!r}")
