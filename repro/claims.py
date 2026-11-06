"""Claim -> numbers, evaluated the same way on the authors' results (results_declared/) and on ours (results/).

Each claim maps to (a) paper placeholders computed by build_paper.py (see paper_values.py) and/or
(b) a small extractor reading the result files directly, for claims the paper code does not print.
`primary(R)` gives one headline number per claim when there is one (used for the relative-difference formula).
"""
import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "adb-repro"
INVENTORY = Path(__file__).resolve().parent / "claims_inventory.md"


def J(R, name):
    return json.loads((REPO / R / name).read_text())


def parse_inventory():
    rows = []
    for line in INVENTORY.read_text(encoding="utf-8").splitlines():
        if not re.match(r"\| C\d+ \|", line):
            continue
        c = [x.strip() for x in line.strip().strip("|").split(" | ")]
        c += [""] * (11 - len(c))
        rows.append(dict(zip(["id", "claim", "declared", "where", "source", "producer", "seeds", "runtime", "deps",
                              "audit_match", "tier"], c[:11])))
    return rows


# ---------------------------------------------------------------- extractors (R = "results_declared" | "results")
def e1_stats(R, B=16384):
    rows = [r for r in J(R, "rdpo_e1.json")["rows"] if r["budget"] == B]
    wins = sum(r["improvement_vs_best"] > 1 for r in rows)
    ex = sorted(r["improvement_vs_best"] for r in rows if r["target"] == "joint_exhaustion")
    cov = [r["rdpo"]["ci_coverage"] for r in rows]
    return {"wins": wins, "cells": len(rows), "exh_min": ex[0], "exh_med": ex[len(ex) // 2], "exh_max": ex[-1],
            "cov_mean": sum(cov) / len(cov), "cov_min": min(cov)}


def e1_text(R):
    s = e1_stats(R)
    return (f"{s['wins']}/{s['cells']} cells won; exhaustion x{s['exh_min']:.1f}-x{s['exh_max']:.0f} (median x{s['exh_med']:.0f}); "
            f"coverage mean {s['cov_mean']:.1%}, min {s['cov_min']:.0%}")


def e1_per_target(R, B=16384):
    rows = [r for r in J(R, "rdpo_e1.json")["rows"] if r["budget"] == B]
    out = []
    for t in ["expected_loss", "joint_attachment", "exceedance", "stop_loss", "joint_exhaustion"]:
        v = sorted(r["improvement_vs_best"] for r in rows if r["target"] == t)
        out.append(f"{t} x{v[len(v) // 2]:.2f}")
    return "; ".join(out)


def e2_twist(R):
    rows = [r for r in J(R, "rdpo_e2.json")["rows"] if r.get("twisted")]
    wins = sum(r["rdpo"]["rel_rmse"] < r["twisted"]["rel_rmse"] for r in rows)
    return f"RDPO beats twisting in {wins}/{len(rows)} E2 settings"


def e2_pair(R):
    rows = [r for r in J(R, "rdpo_e2.json")["rows"] if r["target"] == "joint_exhaustion"]
    if not rows:
        return "no pair rows"
    err = [r["rdpo"]["rel_rmse"] for r in rows]
    cov = [r["rdpo"]["ci_coverage"] for r in rows]
    return f"pair exhaustion rel RMSE {min(err):.2%}-{max(err):.2%}, coverage {min(cov):.0%}-{max(cov):.0%}"


def e3_summary(R):
    rows = J(R, "rdpo_e3.json")["rows"]
    full = {r["case"]: r["rel_rmse"] for r in rows if r["ablation"] == "full RDPO"}
    worst = {}
    for r in rows:
        if r["ablation"] != "full RDPO" and full.get(r["case"]):
            worst[r["ablation"]] = max(worst.get(r["ablation"], 0), r["rel_rmse"] / full[r["case"]])
    return "worst error ratio vs full RDPO: " + "; ".join(f"{k} x{v:.1f}" for k, v in sorted(worst.items(), key=lambda kv: -kv[1]))


def mitigation(R):
    s = J(R, "quantum_advantage.json")["mitigation_summary"]
    worse = [v["mitigation_worse_than_raw"] for v in s.values()]
    p99 = [v["guard2"]["p99"] for v in s.values()]
    fb = [v["fallback2_rate"] for v in s.values()]
    return (f"mitigation worse than raw {min(worse):.0%}-{max(worse):.0%}; guard p99 <= {max(p99):.2f} "
            f"at fallback {min(fb):.0%}-{max(fb):.0%}")


def tensor_bonds(R):
    d = J(R, "tensor.json")["B_scaling_n"]
    out = []
    for kind in ("chain", "star", "ring", "complete"):
        rs = [r for r in d if r["topology"] == kind and r.get("tol") == 1e-9 and "max_bond" in r]
        bonds = sorted({r["max_bond"] for r in rs}) if kind != "complete" else [r["max_bond"] for r in rs]
        out.append(f"{kind} {bonds if kind == 'complete' else bonds[-1]}")
    return "; ".join(out)


def tensor_untilted(R):
    d = J(R, "tensor.json")["A_correctness"]
    m = max(r["max_rel_err"] for r in d if not r.get("tilted_gauge") and r.get("tol") == 1e-12)
    return f"untilted tol 1e-12 max rel err {m:.2e}"


def tensor_ring28(R):
    r = next(r for r in J(R, "tensor.json")["B_scaling_n"] if r["topology"] == "ring" and r["n"] == 28 and r.get("tol") == 1e-9)
    return f"ring-28 tilted MPS {r['runtime_s']:.1f} s"


def twisted28(R):
    r = next(r for r in J(R, "twisted.json")["rows"] if r["topology"] == "ring" and r["n"] == 28)
    return f"AE calls {r['qae_calls_plain']:.2g} -> {r['qae_calls_twisted']:.2g}; MC paths {r['mc_paths']:.2g}"


def oracle28(R):
    r = next(r for r in J(R, "quantum_advantage.json")["advantage"] if r["topology"] == "ring" and r["n"] == 28)
    return f"w_U {r['w_U']} controlled rotations (U_max {r['U_max']})"


def mc_overstate(R):
    a = next(r for r in J(R, "quantum_advantage.json")["advantage"] if r["topology"] == "ring" and r["n"] == 28)
    return f"(1-mu)/mu / rho2 = {((1 - a['mu']) / a['mu']) / a['classical_rho2']:.2g}"


def speedup_range(R):
    v = [r["required_speedup"] for r in J(R, "quantum_advantage.json")["advantage"]]
    return f"required speed-up {min(v):.2g}-{max(v):.2g}"


def nisq_qcbm(R):
    d = J(R, "nisq.json")["qcbm_bias"]
    ideal = [r["rel_bias_ideal_weights"] for r in d]
    cal = [r["rel_bias_calibrated_rms"] for r in d if r["calib_shots"] >= 1e5]
    return f"noise bias {min(ideal):.0%}-{max(ideal):.0%}; 1e5-shot calibrated rms <= {max(cal):.2%}"


def structured_verify(R):
    d = J(R, "structured.json")["verify"]
    d = d if isinstance(d, list) else [d]
    return f"max abs err vs dense {max(r['max_abs_err_vs_dense'] for r in d):.1e}"


def exact_runtime(R):
    s = J(R, "exact.json")["scenarios"]
    return f"{1000 * max(v['runtime_s'] for v in s.values()):.2f} ms per scenario (max)"


def sweep(R):
    return f"{J(R, 'sweep_timing.json')['seconds_total']:.2f} s for 25 estimates"


def runs_total(R):
    if R != "results":
        return "n/a (hard-coded in README)"
    recs = [json.loads(l) for l in (REPO.parent / "repro" / "runs.jsonl").read_text().splitlines()]
    cpu = sum(r["seconds"] * min(r["cpu_cost"], 4) for r in recs) / 3600
    return f"our re-run: {sum(r['seconds'] for r in recs) / 3600:.2f} job-hours, ~{cpu:.1f} CPU-hours"


def pytest_result(R):
    if R != "results":
        return "59 passing (declared)"
    log = (REPO.parent / "repro" / "logs" / "pytest.log").read_text(encoding="utf-8", errors="replace")
    return re.findall(r"\d+ passed[^\n]*", log)[-1]


# ---------------------------------------------------------------- claim -> (placeholders, extractor, primary)
def pct(s):
    m = re.search(r"([\d.]+)%", s or "")
    return float(m.group(1)) / 100 if m else None


def num(s):
    s = (s or "").translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-"))
    m = re.search(r"([\d.]+)(?:×10(-?\d+))?", s.replace(",", ""))
    return float(m.group(1)) * (10 ** int(m.group(2)) if m.group(2) else 1) if m else None


MAP = {
    "C01": (["PU_EXC_MED", "PU_EXC_X", "PU_EXC_LO", "PU_EXC_HI", "PU_EXC_BRIEF"], None, ("PU_EXC_MED", pct)),
    "C02": (["RAIN_NST"], None, ("RAIN_NST", num)),
    "C03": (["HL_EQ_SHARE", "HL_RAIN_SHARE", "LOSS_SHARE"], None, ("HL_EQ_SHARE", pct)),
    "C04": (["HL_AP_MU", "HL_AP_ERR", "HL_AP_SEC"], None, ("HL_AP_ERR", pct)),
    "C05": (["HL_SWEEP"], sweep, None),
    "C06": (["V2_TRANSFER_GAIN"], None, None),
    "C07": (["RAIN_EXC_FACTOR", "HL_EXC_DATA", "HL_EXC_BRIEF"], None, ("RAIN_EXC_FACTOR", num)),
    "C08": (["Q_SSTAR_28"], None, ("Q_SSTAR_28", num)),
    "C09": ([], e1_text, None),
    "C10": (["E1_EXH_RANGE", "E1_EXH_MED"], e1_text, None),
    "C11": (["E1_COVERAGE"], e1_text, ("E1_COVERAGE", pct)),
    "C12": ([], e1_per_target, None),
    "C13": (["E1_SEC_16K"], None, ("E1_SEC_16K", num)),
    "C14": (["E2_RANGE", "E2_RING28_EXC"], None, ("E2_RING28_EXC", pct)),
    "C15": ([], e2_twist, None),
    "C16": ([], e2_pair, None),
    "C17": (["E2_SEC_28"], None, ("E2_SEC_28", num)),
    "C18": ([], e3_summary, None),
    "C19": ([], tensor_bonds, None),
    "C20": ([], tensor_untilted, None),
    "C21": (["V2_SPEEDUP"], None, None),
    "C22": ([], None, None),
    "C23": (["V2_GAIN_MAX", "V2_TW_WINS"], None, None),
    "C24": (["V2_TIME"], None, None),
    "C25": (["V2_COV"], None, None),
    "C26": (["V2_EXACT_MS", "V2_EXACT_ERR"], None, None),
    "C27": ([], None, None),
    "C28": (["V2_BENCH_COV"], None, None),
    "C29": (["PROJ_SD", "PROJ_MS", "PROJ_COND_MEAN", "PROJ_COND_P95", "PROJ_COND_400", "PROJ_TRIPLE"], None,
            ("PROJ_COND_MEAN", num)),
    "C30": ([], None, None),
    "C31": (["STR_RANGE"], None, None),
    "C32": (["RD_N", "RD_DISP", "RD_DISP_P", "RD_B", "RD_SHARE"], None, ("RD_N", num)),
    "C33": (["RD_ATT", "RD_EXH_FACTOR"], None, None),
    "C34": (["LOSS_SHARE", "LOSS_P"], None, ("LOSS_SHARE", pct)),
    "C35": (["LOSS_EXH", "LOSS_EXH_FACTOR"], None, ("LOSS_EXH", num)),
    "C36": (["RAIN_SHARE"], None, ("RAIN_SHARE", pct)),
    "C37": (["POOL_GAIN", "POOL_EL", "POOL_PAIR", "POOL_PAIR_COV"], None, ("POOL_PAIR", pct)),
    "C38": (["CAREC_MU_RARE", "CAREC_CRIT", "CAREC_CRIT16", "CAREC_CRIT_OK", "CAREC_BOND", "CAREC_MPS_S"], None,
            ("CAREC_CRIT", pct)),
    "C39": (["AP_CRIT", "AP_CRIT_OK", "AP_BOND", "R50_BOND", "AP_MPS_S"], None, None),
    "C40": (["ELT_EVENTS", "ELT_SOURCES", "ELT_MAXFP", "ELT_CONV", "ELT_RARE_RDPO", "ELT_RARE_YLT"], None,
            ("ELT_RARE_RDPO", pct)),
    "C41": (["PU_ATT_MED", "PU_ATT_X", "PU_ATT_LO", "PU_ATT_HI", "PU_EXH_MED", "PU_EXH_X", "PU_EXH_LO", "PU_EXH_HI"],
            None, ("PU_ATT_MED", pct)),
    "C42": ([], twisted28, None),
    "C43": ([], oracle28, None),
    "C44": (["QS_FIT_ERR"], None, None),
    "C45": (["QS_BREAKEVEN_1", "QS_RHO2"], None, None),
    "C46": (["Q_SSTAR_28"], speedup_range, None),
    "C47": ([], mc_overstate, None),
    "C48": ([], mitigation, None),
    "C49": ([], nisq_qcbm, None),
    "C50": ([], structured_verify, None),
    "C51": ([], exact_runtime, None),
    "C52": ([], tensor_ring28, None),
    "C53": ([], runs_total, None),
    "C54": ([], None, None),
    "C55": ([], pytest_result, None),
}


# ---------------------------------------------------------------- verbatim source sentences (never edited)
# P = proposal.md with the authors' own declared values substituted (exactly what their PDF shows),
# R = README.md, S = docs/paper/short.filled.md (both from the untouched original repo).
DOCS = {"R": (HERE_ORIG := REPO.parent / "adb-quantum-challenge") / "README.md",
        "P": REPO.parent / "repro" / "proposal.declared.md",
        "S": HERE_ORIG / "docs" / "paper" / "short.filled.md"}
QUOTES = {
    "C01": [("S", "3.76%")], "C02": [("S", "48 GHCN-Daily")], "C03": [("P", "about 48% of events")],
    "C04": [("P", "Plain Monte Carlo sees no such event")], "C05": [("P", "**5 seconds** on one core")],
    "C06": [("P", "×1.8–×83.8")], "C07": [("R", "Rain dependence from station data")],
    "C08": [("R", "Falsifiable wall-clock criterion")], "C09": [("R", "in **20/25** cells")],
    "C10": [("R", "**×6.3 to ×171**")], "C11": [("S", "Interval coverage is 94% mean")],
    "C12": [("P", "20 of the 25 scenario–target cells")], "C13": [("P", "RDPO estimate, benchmark, B = 16,384")],
    "C14": [("R", "Scaling, ring of 2 → 28 countries")], "C15": [("P", "As countries are added to the ring")],
    "C16": [("P", "As countries are added to the ring")], "C17": [("P", "RDPO estimate, 28 countries, B = 65,536")],
    "C18": [("R", "| Ablation |")], "C19": [("P", "For chain, star and ring dependence the bond dimension is constant")],
    "C20": [("P", "Plain truncation incurs up to 2% error")], "C21": [("R", "table-driven, identical policy (tested)")],
    "C22": [("P", "each evaluation is 50–500× cheaper")], "C23": [("R", "more accurate than Phase 1 in **13/16**")],
    "C24": [("S", "2.3–11.9× faster per estimate")], "C25": [("R", "Coverage of 95% intervals (portfolio-wide targets)")],
    "C26": [("R", "Pair targets (joint attachment / exhaustion)")], "C27": [("R", "keeps 20/25")],
    "C28": [("P", "coverage is 82%–100% per cell")], "C29": [("P", "returns the exact value (relative spread")],
    "C30": [("P", "Spline soft values *hurt*")],
    "C31": [("R", "**Stronger rare-event baselines**"), ("P", "RDPO beats every one of them")],
    "C32": [("S", "Poisson (dispersion 0.66–1.24")],
    "C33": [("P", "With the data-based shared intensity, the joint-exhaustion probability")],
    "C34": [("R", "Loss records (NOAA 1900–2020")], "C35": [("R", "| Implied bond risk |")],
    "C36": [("R", "Rain dependence from station data")], "C37": [("R", "Data-calibrated five-country Central Asian pool")],
    "C38": [("R", "**WP2: data-calibrated 10-country CAREC pool**")], "C39": [("R", "**WP2 at scale**")],
    "C40": [("R", "**ELT/YLT converter**")], "C41": [("P", "| Joint attachment | 4.41%"), ("P", "| Joint exhaustion | 3.2×10⁻⁵")], "C42": [("R", "| Loading |")],
    "C43": [("P", "**Bitwise oracle.**")], "C44": [("S", "Compiled resources fit logical qubits")],
    "C45": [("R", "| Algorithm and scalability |")], "C46": [("P", "break-even needs a hardware speed-up between")],
    "C47": [("S", "overstates the")], "C48": [("R", "Error-mitigation trust boundary")],
    "C49": [("P", "reduces the bias to at most 0.6%")], "C50": [("P", "**Exact loading.**")],
    "C51": [("P", "Exact benchmark model (5 scenarios)")], "C52": [("P", "Exact tilted tensor network, 28 countries (ring)")],
    "C53": [("R", "≈ 7 CPU-hours")], "C54": [("P", "Projected logical and physical resources per amplitude")],
    "C55": [("S", "**59 passing tests**")],
}
_START = re.compile(r"\s*([*-] |#|\||\d+\.\s|:::|!\[)")


def _lines(doc, cache={}):
    if doc not in cache:
        cache[doc] = DOCS[doc].read_text(encoding="utf-8").splitlines()
    return cache[doc]


def quote(doc, needle):
    """Whole paragraph / list item / table row containing `needle`, verbatim, and where it is."""
    L = _lines(doc)
    i = next((k for k, l in enumerate(L) if needle in l), None)
    if i is None:
        return f"[{doc}] not found: {needle}", ""
    if L[i].lstrip().startswith("|"):
        a = b = i
    else:
        a = i
        while a > 0 and not _START.match(L[a]) and L[a - 1].strip() and not L[a - 1].lstrip().startswith("|"):
            a -= 1
        b = i
        while b + 1 < len(L) and L[b + 1].strip() and not _START.match(L[b + 1]):
            b += 1
    text = " ".join(x.strip() for x in L[a:b + 1])
    if doc == "P":
        head = next((L[k] for k in range(i, -1, -1) if L[k].startswith("#")), "")
        where = f"proposal（PDF）{head.lstrip('# ').split(' {')[0]}"
    else:
        where = f"{'README.md' if doc == 'R' else 'short.md（6 页版）'}:{a + 1}"
    return text[:3000], where


def quotes_for(cid):
    qs = [quote(d, n) for d, n in QUOTES.get(cid, [])]
    return "\n\n".join(q for q, _ in qs), "；".join(w for _, w in qs)


def evaluate(paper):
    """paper = output of paper_values.collect(). Returns one dict per claim with stored and reproduced text/number."""
    out = []
    for row in parse_inventory():
        ph, fn, prim = MAP[row["id"]]
        res = {}
        for tag in ("results_declared", "results"):
            parts = [f"{k}={paper[tag].get(k, {}).get('value')}" for k in ph]
            if fn:
                try:
                    parts.append(fn(tag))
                except Exception as ex:
                    parts.append(f"{fn.__name__} failed: {type(ex).__name__}")
            p = None
            if prim:
                p = prim[1](str(paper[tag].get(prim[0], {}).get("value")))
            res[tag] = ("; ".join(parts) or "—", p)
        q, qwhere = quotes_for(row["id"])
        out.append({**row, "quote": q, "quote_where": qwhere,
                    "stored_text": res["results_declared"][0], "ours_text": res["results"][0],
                    "stored_num": res["results_declared"][1], "ours_num": res["results"][1]})
    return out
