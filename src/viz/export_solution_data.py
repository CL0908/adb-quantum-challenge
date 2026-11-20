"""Numbers behind the /solution/ page (frontend/public/data/solution.json), all read from the committed results files.

Run:  PYTHONPATH=. python -m src.viz.export_solution_data
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = ROOT / "frontend" / "public" / "data" / "solution.json"
SCENARIOS = ["adb-base", "adb-independent", "adb-common-shock", "adb-rain-stress", "adb-rain-cover"]


def build() -> dict:
    from src.rdpo.problem import adb_model

    m = adb_model("adb-base")
    c = float(m.c[(0, 1)])
    cr = float(m.cr[(0, 1)])
    sources = [
        {"id": "kg_eq", "label": "Earthquake, Kyrgyz Republic only", "kind": "eq", "scope": "KG", "rate": float(m.solo_eq[0])},
        {"id": "tj_eq", "label": "Earthquake, Tajikistan only", "kind": "eq", "scope": "TJ", "rate": float(m.solo_eq[1])},
        {"id": "sh_eq", "label": "Earthquake, both countries", "kind": "eq", "scope": "both", "rate": c},
        {"id": "kg_rain", "label": "Extreme rain, Kyrgyz Republic only", "kind": "rain", "scope": "KG", "rate": float(m.solo_rain[0])},
        {"id": "tj_rain", "label": "Extreme rain, Tajikistan only", "kind": "rain", "scope": "TJ", "rate": float(m.solo_rain[1])},
        {"id": "sh_rain", "label": "Extreme rain, both countries", "kind": "rain", "scope": "both", "rate": cr},
    ]
    law = json.loads((R.parent / "frontend" / "public" / "data" / "mc_sample.json").read_text())["law"]
    n = len(law)
    margins = {
        "KG": {"attach": 1 - sum(law[0]), "exhaust": sum(law[n - 1])},
        "TJ": {"attach": 1 - sum(r[0] for r in law), "exhaust": sum(r[n - 1] for r in law)},
    }
    eq_t = json.loads((R / "equal_time.json").read_text())
    car = json.loads((R / "wp2_carec.json").read_text())
    ap = json.loads((R / "wp2_asia_pacific.json").read_text())
    pools = [
        {"id": "pair", "label": "2 countries", "sub": "Kyrgyz Republic, Tajikistan", "n": 2, "edges": 1, "bond": None,
         "seconds": eq_t["exact_enumeration_A"]["exact_dense_enumeration_s_median"], "method": "dense enumeration"},
        {"id": "carec", "label": "10 CAREC", "sub": "Central Asia and neighbours", "n": len(car["calibration"]["order"]),
         "edges": len(car["calibration"]["edges"]), "bond": car["mps"]["max_bond"], "seconds": car["mps"]["mps_seconds"],
         "method": "tensor network"},
        {"id": "ap", "label": "28 Asia-Pacific", "sub": "Asia-Pacific pool", "n": ap["cases"]["asia_pacific"]["n"],
         "edges": ap["cases"]["asia_pacific"]["edges"], "bond": ap["cases"]["asia_pacific"]["max_bond"],
         "seconds": ap["cases"]["asia_pacific"]["mps_seconds"], "method": "tensor network"},
    ]
    e1 = {}
    for r in json.loads((R / "rdpo_e1.json").read_text())["rows"]:
        if r["budget"] == 16384:
            e1.setdefault(r["scenario"], {})[r["target"]] = {
                "rdpo": r["rdpo"]["rel_rmse"], "mc": r["baselines"]["mc"], "best": min(r["baselines"].values()),
                "coverage": r["rdpo"]["ci_coverage"], "seconds": r["seconds_per_fit"]}
    return {"budget": 16384, "scenarios": SCENARIOS, "sources": sources, "term_years": float(m.years), "margins": margins,
            "pools": pools, "e1": e1, "carec_edges": car["calibration"]["edges"], "carec_order": car["calibration"]["order"]}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1))
    print("wrote", OUT)
