"""Self-check of the run.py scheduler with dummy jobs (no repo code runs).  python repro/test_run.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run  # noqa: E402

P = sys.executable
ok = [P, "-c", "import time; time.sleep(0.3)"]
bad = [P, "-c", "import sys; sys.exit(3)"]
run.JOBS = [
    ("a", 9, ok, [], 1, False),
    ("b", 9, ok, ["a"], 1, False),          # waits for a
    ("x", 9, bad, [], 1, False),            # fails
    ("y", 9, ok, ["x"], 1, False),          # must be skipped
    ("s", 9, bad, [], 1, True),             # soft failure
    ("t", 9, ok, ["s"], 1, False),          # still runs
    ("solo", 9, ok, [], run.ALL, False),    # must run alone
]
run.BY = {j[0]: j for j in run.JOBS}
tmp = Path(tempfile.mkdtemp())
run.RUNS, run.LOGS, run.HERE = tmp / "runs.jsonl", tmp / "logs", tmp
sys.argv = ["run.py", "9"]
run.main()

recs = {r["name"]: r for r in map(json.loads, run.RUNS.read_text().splitlines())}
assert set(recs) == {"a", "b", "x", "s", "t", "solo"}, recs.keys()          # y skipped
assert recs["b"]["start"] >= recs["a"]["end"]
assert recs["x"]["exit"] == 3 and recs["t"]["exit"] == 0
solo = recs["solo"]
assert all(r["end"] <= solo["start"] or r["start"] >= solo["end"] for n, r in recs.items() if n != "solo")
print("run.py scheduler self-check passed")
