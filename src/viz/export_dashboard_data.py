"""Numbers behind the Banker view (frontend/public/data/dashboard.json, page /dashboard/).

Every figure is read from the committed results files; nothing is typed here except human-readable labels.
Run: PYTHONPATH=. python -m src.viz.export_dashboard_data   (also called by export_web_data.main()).
"""
import json
import subprocess
from pathlib import Path

from src.model.scenarios import SCENARIOS, Scenario

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = ROOT / "frontend" / "public" / "data" / "dashboard.json"
E1_BUDGET = 16384
TARGETS = ["expected_loss", "joint_attachment", "exceedance", "stop_loss", "joint_exhaustion"]
VALUE_KEY = dict(zip(TARGETS, [
    "term_expected_principal_loss_usd", "term_joint_attachment_probability",
    "term_portfolio_loss_exceedance_probability", "term_portfolio_stop_loss_usd", "term_both_bonds_exhausted_probability"]))
SCENARIO_LABEL = {
    "adb-base": "Base", "adb-independent": "Independent", "adb-common-shock": "Common shock",
    "adb-rain-stress": "Rain stress", "adb-rain-cover": "Rain cover",
}
# result file -> (script that writes it, what the dashboard uses it for)
RUNS = {
    "exact.json": ("src/model/exact.py", "Brief Table 5: exact values of the five scenarios"),
    "finance.json": ("src/finance/analysis.py", "Pooling analysis: exact grid, shared-limit payout"),
    "rdpo_e1.json": ("src/rdpo/experiments.py", "RDPO error and coverage per scenario and target"),
    "realdata_rain.json": ("src/realdata/rain_check.py", "Station-data rain shared share"),
    "wp2_carec.json": ("src/realdata/carec_pool.py", "CAREC 10-country pool: exact references and RDPO"),
    "wp2_asia_pacific.json": ("src/realdata/asia_pacific_pool.py", "Asia-Pacific 28-country pool: exact references and RDPO"),
}
POOL_RUNS = {
    "kgtj": ["exact.json", "finance.json", "rdpo_e1.json", "realdata_rain.json"],
    "carec": ["wp2_carec.json", "realdata_rain.json"],
    "ap": ["wp2_asia_pacific.json", "realdata_rain.json"],
}


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def load(name: str) -> dict:
    return json.loads((R / name).read_text())


def scenario_desc(s: Scenario) -> str:
    d, bits = Scenario("x"), []
    if s.m != d.m:
        bits.append(f"shared-quake multiplier m = {s.m:g}")
    if s.rain_mult != d.rain_mult:
        bits.append(f"rain rate x{s.rain_mult:g}")
    if s.rain_shared_share != d.rain_shared_share:
        bits.append(f"rain shared share {s.rain_shared_share:.0%}")
    if (s.eq_limit, s.rain_limit) != (d.eq_limit, d.rain_limit):
        bits.append(f"limits US${s.eq_limit / 1e6:g}m quake + US${s.rain_limit / 1e6:g}m rain")
    return "; ".join(bits) if bits else "brief calibration, no change"


def rows_for(rows: list, **kw) -> list:
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def kgtj() -> dict:
    ex, e1 = load("exact.json")["scenarios"], load("rdpo_e1.json")
    out = {}
    for name, sc in SCENARIOS.items():
        v = ex[name]["values"]
        el, sl = v[VALUE_KEY["expected_loss"]], v[VALUE_KEY["stop_loss"]]
        targets = {}
        for t in TARGETS:
            r = rows_for(e1["rows"], scenario=name, target=t, budget=E1_BUDGET)[0]
            targets[t] = {
                "value": v[VALUE_KEY[t]], "brief_rounded": ex[name]["reference_rounded"][VALUE_KEY[t]],
                "rdpo_rel_rmse": r["rdpo"]["rel_rmse"], "rdpo_coverage": r["rdpo"]["ci_coverage"],
                "reps": r["rdpo"]["reps"], "budget": r["budget"], "best_baseline": r["best_baseline"],
                "best_baseline_rel_rmse": r["baselines"][r["best_baseline"]], "mc_rel_rmse": r["baselines"]["mc"],
            }
        out[name] = {"label": SCENARIO_LABEL[name], "desc": scenario_desc(sc), "targets": targets,
                     "delivered": (el - sl) / el, "threshold_musd": sc.threshold / 1e6}
    return out


def wp2(refs: dict, rows: list, thr_mod: int, thr_rare: int, case: str | None, exact_label: str) -> dict:
    def pick(target, thr):
        kw = {"target": target, "threshold_musd": thr}
        if case:
            kw["case"] = case
        by = {m: rows_for(rows, method=m, budget=65536, **kw) for m in ("mc", "twist", "rdpo")}
        return {m: (x[0] if x else None) for m, x in by.items()}

    def ref(target, thr):
        key = f"{target}|{thr}" if thr else f"{target}|None"
        return refs.get(key)

    spec = [("expected_loss", None), ("joint_attachment", None), ("exceedance", thr_mod), ("exceedance", thr_rare),
            ("stop_loss", thr_mod), ("stop_loss", thr_rare), ("joint_exhaustion", None)]
    targets = []
    for t, thr in spec:
        r = pick(t, thr)
        rd = r["rdpo"]
        targets.append({
            "target": t, "threshold_musd": thr, "value": ref(t, thr),
            "rdpo_rel_rmse": rd["rel_rmse"] if rd else None, "rdpo_coverage": rd.get("ci_coverage") if rd else None,
            "reps": rd["reps"] if rd else None, "budget": rd["budget"] if rd else None,
            "mc_rel_rmse": r["mc"]["rel_rmse"] if r["mc"] else None,
            "twist_rel_rmse": r["twist"]["rel_rmse"] if r["twist"] else None,
        })
    targets = [t for t in targets if t["value"] is not None]
    el = ref("expected_loss", None)
    sl = ref("stop_loss", thr_mod)
    delivered = (el - sl) / el if el and sl is not None else None
    return {"delivered": delivered, "targets": targets, "exact_label": exact_label, "T_moderate": thr_mod, "T_rare": thr_rare}


def provenance(names: list) -> list:
    runs = []
    for n in names:
        script, use = RUNS[n]
        assert (ROOT / script).exists(), script
        h, d = git("log", "-1", "--format=%h|%cs", "--", f"results/{n}").split("|")
        meta = {}
        data = load(n)
        for k in ("seed", "reps"):
            v = (data.get("meta") or {}).get(k, data.get(k))
            if v is not None:
                meta[k] = v
        runs.append({"file": f"results/{n}", "script": script, "use": use, "commit": h, "date": d, **meta})
    return runs


def main() -> dict:
    carec, ap = load("wp2_carec.json"), load("wp2_asia_pacific.json")
    apc = ap["cases"]["asia_pacific"]
    ccurve = carec["exceedance_curve"]
    pools = {
        "kgtj": {"label": "Kyrgyz Republic + Tajikistan bond", "short": "KG + TJ", "n_countries": 2, "scenarios": kgtj(),
                 "exact_label": "exact, dense enumeration", "budget": E1_BUDGET},
        "carec": {"label": "CAREC 10-country pool", "short": "CAREC 10", "n_countries": 10,
                  **wp2(carec["references"], carec["rows"], carec["T_moderate"], carec["T_rare"], None,
                        "exact, tensor network (bond %d)" % carec["mps"]["max_bond"]),
                  "curve": {"x": [int(k) for k in ccurve], "y": list(ccurve.values())}, "reps": carec["reps"]},
        "ap": {"label": "Asia–Pacific 28-country pool", "short": "Asia–Pacific 28", "n_countries": 28,
               **wp2(apc["refs"], ap["rows"], apc["T_moderate"], apc["T_rare"], "asia_pacific",
                     "exact, tensor network (bond %d)" % apc["max_bond"]), "reps": ap["reps"]},
    }
    for k, p in pools.items():
        p["runs"] = provenance(POOL_RUNS[k])
    used = [r for p in pools.values() for r in p["runs"]]
    latest = max(used, key=lambda r: r["date"])
    out = {"asof": {"date": latest["date"], "commit": latest["commit"], "file": latest["file"]}, "pools": pools,
           "scenario_order": list(SCENARIOS)}
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    o = main()
    print("wrote", OUT, "asof", o["asof"])
