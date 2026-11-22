"""Write the site's add-on values from the committed result files, so no number on the page is typed by hand.

Reads branch-xzy/addons/results (cap_modules.json, dense_pools.json, q_gap_map.json), counts the collected tests of
tests/ and addons/test_addons.py with pytest, and merges the keys into frontend/public/data/values.json:
  SW_N, SW_PATH, SW_TIME       pools where the small world with saturation modules was tested, and how many it won
                               against exponential twisting at the same paths and at the same CPU time
  Q_LOADER_<n>, Q_TWIST_<n>    required speed-up at 1% error with the cap-aware loader and with twisting, rings 28/40/56
  N_TESTS                      collected tests in the whole repository (was tests/ only)
It also writes frontend/public/data/nest.json, the numbers behind "three models, one inside the other" on the home page:
  cap_modules.json ring 28 (relative variance, answer over truth, max |z| against exact, equal-CPU gain) and
  twinstar.json why.fragility (tail probabilities with the inputs fixed and with them uncertain).
Run from the repository root after the add-on results change: python addons/export_web_values.py
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "addons" / "results"
VALUES = REPO / "frontend" / "public" / "data" / "values.json"
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x):
    e = math.floor(math.log10(x))
    m = round(x / 10 ** e, 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return f"{m:.1f}×10{str(e).translate(SUP)}"


def small_world():
    cells = []
    cm = json.loads((RES / "cap_modules.json").read_text(encoding="utf-8"))
    dp = json.loads((RES / "dense_pools.json").read_text(encoding="utf-8"))
    for src in (cm, dp):
        for k, v in src.items():
            if k != "checks":
                cells.append((v["modules_1"]["rho2"] < v["rho2_twist"], v["modules_1"]["equal_cpu_gain"] > 1))
    return {"SW_N": str(len(cells)), "SW_PATH": str(sum(p for p, _ in cells)), "SW_TIME": str(sum(t for _, t in cells))}


def loader():
    p = json.loads((RES / "q_gap_map.json").read_text(encoding="utf-8"))["pools"]
    out = {}
    for n in (28, 40, 56):
        r = p[f"ring {n}"]
        out[f"Q_LOADER_{n}"], out[f"Q_TWIST_{n}"] = sci(r["loader_tree"]["s_star"]), sci(r["twisting"]["s_star"])
    return out


def n_tests():
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests", "addons/test_addons.py"], cwd=REPO,
                         capture_output=True, text=True, timeout=300, env={**__import__("os").environ, "PYTHONUTF8": "1"}).stdout
    m = re.search(r"(\d+) tests? collected", out)
    if not m:
        raise SystemExit("pytest collection failed:\n" + out[-2000:])
    return {"N_TESTS": m.group(1)}


def nest():
    r = json.loads((RES / "cap_modules.json").read_text(encoding="utf-8"))["ring 28"]
    f = json.loads((RES / "twinstar.json").read_text(encoding="utf-8"))["why"]["fragility"]
    out = {
        "pool": "ring 28",
        "rho2": {"twisting": r["rho2_twist"], "guide": r["uncapped"]["rho2"], "guide_tables": r["modules_1"]["rho2"]},
        "equal_cpu_gain": r["modules_1"]["equal_cpu_gain"],
        "answer_over_truth": {"guide": r["uncapped"]["answer_over_truth"], "guide_tables": r["modules_1"]["answer_over_truth"]},
        "rdpo_z_max": r["uncapped"]["z_max"],
        "twin": {k: {"fixed": f[k]["p_point"], "uncertain": f[k]["p_with_uncertainty"], "rel": f[k]["premium_relative"]} for k in ("exceedance", "joint_exhaustion")},
        "source": "branch-xzy/addons/results/cap_modules.json, twinstar.json",
    }
    (VALUES.parent / "nest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main():
    nest()
    v = json.loads(VALUES.read_text(encoding="utf-8"))
    new = {**small_world(), **loader(), **n_tests()}
    v.update(new)
    VALUES.write_text(json.dumps(v, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(new, ensure_ascii=False))


if __name__ == "__main__":
    assert sci(81527) == "8.2×10⁴" and sci(1009587) == "1.0×10⁶" and sci(99999) == "1.0×10⁵"
    main()
