"""Deep comparison of declared vs reproduced result files.

    python repro/compare.py exact.json tensor.json ...      # adb-repro/results_declared/<f> vs adb-repro/results/<f>
    python repro/compare.py                                 # self-check

Every numeric leaf is compared. Keys that look like timings are reported separately (machine-dependent).
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "adb-repro"
DECL, OURS = ROOT / "results_declared", ROOT / "results"
PREC = 1e-12  # relative difference treated as floating-point noise (last-bit BLAS / summation-order effects)
TIMING = re.compile(r"(^|_)(sec|secs|seconds|ms|time|times|runtime|wall|elapsed|throughput|per_s)($|_)", re.I)


def last_name(path):
    names = [s for s in re.split(r"[.\[\]]+", path) if s and not s.isdigit()]
    return names[-1] if names else ""


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, x


def rel(a, b):
    if a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
        return 0.0
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def compare(a, b):
    """-> dict(n, equal, max_rel, diffs=[(path, declared, ours, rel)], timing=[...], missing=[...], extra=[...])"""
    A, B = dict(leaves(a)), dict(leaves(b))
    out = {"n": 0, "equal": 0, "max_rel": 0.0, "diffs": [], "timing": [], "missing": sorted(set(A) - set(B)),
           "extra": sorted(set(B) - set(A))}
    for p in A.keys() & B.keys():
        x, y = A[p], B[p]
        num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y))
        if num and TIMING.search(last_name(p)):
            out["timing"].append((p, x, y, rel(x, y)))
            continue
        out["n"] += 1
        r = rel(x, y) if num else (0.0 if x == y else math.inf)
        if r == 0.0:
            out["equal"] += 1
        else:
            out["diffs"].append((p, x, y, r))
            out["max_rel"] = max(out["max_rel"], r)
    out["diffs"].sort(key=lambda d: -d[3])
    return out


def report(name):
    d, o = DECL / name, OURS / name
    if not o.exists():
        return print(f"{name}: NOT PRODUCED")
    c = compare(json.loads(d.read_text()), json.loads(o.read_text()))
    same_shape = not c["missing"] and not c["extra"]
    status = ("IDENTICAL" if not c["diffs"] else "MACHINE-PRECISION" if c["max_rel"] <= PREC else "DIFFERS") \
        if same_shape else "SHAPE-DIFFERS"
    print(f"{name}: {status}  values {c['equal']}/{c['n']} equal, max rel diff {c['max_rel']:.3g}, "
          f"timing fields {len(c['timing'])}, missing {len(c['missing'])}, extra {len(c['extra'])}")
    for p, x, y, r in c["diffs"][:8]:
        print(f"    {p}: declared {x!r} vs ours {y!r} (rel {r:.3g})")
    for p in c["missing"][:5]:
        print(f"    missing in ours: {p}")
    for p in c["extra"][:5]:
        print(f"    extra in ours: {p}")
    return c


def _selftest():
    c = compare({"a": 1.0, "b": [1, 2], "seconds": 3.0, "s": "x"}, {"a": 1.0 + 1e-9, "b": [1, 2], "seconds": 9.0, "s": "y"})
    assert c["n"] == 4 and c["equal"] == 2 and len(c["timing"]) == 1
    assert c["diffs"][0][3] == math.inf and abs(c["diffs"][1][3] - 1e-9) < 1e-12
    assert TIMING.search(last_name("rows[2].mps_seconds")) and not TIMING.search(last_name("sector.secondary"))
    c = compare({"r": [{"v": 1}]}, {"r": [{"v": 1}, {"v": 2}]})
    assert c["extra"] == ["r[1].v"] and c["equal"] == 1
    print("compare.py self-check passed")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _selftest()
    for f in sys.argv[1:]:
        report(f)
