"""Build the formal report: figures -> numbers from results/*.json -> Markdown -> HTML -> PDF (+ cover).

Every number substituted into docs/paper/proposal.md comes from the result files written by the
experiment scripts; nothing is typed by hand.
Usage: python -m src.viz.build_paper [--cv path/to/cv.pdf]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import art, art2, art3, paper_figs
from ..business import market

ROOT = Path(".")
PAPER = ROOT / "docs/paper"
R = ROOT / "results"
TGT = paper_figs.TGT
SCN = paper_figs.SCN
TLAB = {"expected_loss": "Expected loss", "joint_attachment": "Joint attachment", "exceedance": "Pr(L > 80m)",
        "stop_loss": "Stop-loss", "joint_exhaustion": "Joint exhaustion"}
BLN = {"mc": "Monte Carlo", "sobol": "Sobol", "stratified": "Stratified", "importance": "Brief IS"}


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def fold(x):
    return f"{x:.1f}" if x < 10 else f"{x:.0f}"


def e1_values(B=16384):
    d = json.loads((R / "rdpo_e1.json").read_text())["rows"]
    rows = {(r["scenario"], r["target"], r["budget"]): r for r in d}
    v = {}
    exh = [rows[(s, "joint_exhaustion", B)] for s in SCN if (s, "joint_exhaustion", B) in rows]
    imp = [r["improvement_vs_best"] for r in exh]
    v["E1_EXH_MED"] = pct(float(np.median([r["rdpo"]["rel_rmse"] for r in exh])), 1)
    v["E1_EXH_RANGE"] = f"{fold(min(imp))}–{fold(max(imp))}-fold"
    v["E1_EXH_RANGE_SHORT"] = f"×{fold(min(imp))}–{fold(max(imp))}"
    v["E1_COVERAGE"] = pct(float(np.mean([r["rdpo"]["ci_coverage"] for r in d])), 0)
    v["E1_SEC_16K"] = f"{np.mean([r['seconds_per_fit'] for r in d if r['budget'] == B]):.1f}"
    # table: targets x scenarios at B
    hdr = "| Target | " + " | ".join(paper_figs.SCN_LABEL[s] for s in SCN) + " |"
    lines = [f"**Table [tab:e1].** RDPO relative RMSE at B = {B:,} paths (training included) and, in parentheses, the improvement factor over the best of the four reference methods in the same cell. Bold: RDPO more accurate.", "", hdr, "|---" * 6 + "|"]
    wins = 0
    for t in TGT:
        cells = []
        for s in SCN:
            r = rows.get((s, t, B))
            if not r:
                cells.append("—"); continue
            f = r["improvement_vs_best"]
            cell = f"{pct(r['rdpo']['rel_rmse'])} (×{fold(f)})"
            if f > 1:
                wins += 1; cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {TLAB[t]} | " + " | ".join(cells) + " |")
    v["TABLE_E1"] = "\n".join(lines)
    # text
    med_imp = {t: float(np.median([rows[(s, t, B)]["improvement_vs_best"] for s in SCN if (s, t, B) in rows])) for t in TGT}
    best_el = {}
    for s in SCN:
        r = rows.get((s, "expected_loss", B))
        if r:
            best_el[r["best_baseline"]] = best_el.get(r["best_baseline"], 0) + 1
    small = [rows[(s, "joint_exhaustion", 4096)]["improvement_vs_best"] for s in SCN if (s, "joint_exhaustion", 4096) in rows]
    big = [rows[(s, "joint_exhaustion", 65536)]["improvement_vs_best"] for s in SCN if (s, "joint_exhaustion", 65536) in rows]
    covs = [r["rdpo"]["ci_coverage"] for r in d]
    v["E1_TEXT"] = (
        f"At a total budget of {B:,} paths, RDPO is more accurate than the best reference method in {wins} of the 25 "
        f"scenario–target cells. The gain grows with the rarity of the target. The median improvement factor is "
        f"{fold(med_imp['joint_exhaustion'])} for joint exhaustion, {fold(med_imp['stop_loss'])} for the stop-loss, "
        f"{fold(med_imp['exceedance'])} for pool exceedance and {fold(med_imp['joint_attachment'])} for joint attachment. "
        f"For the expected loss, a smooth and non-rare target, scrambled Sobol sampling remains the most accurate method "
        f"(median factor {fold(med_imp['expected_loss'])}), and we recommend it for that quantity. "
        f"Because the training cost is charged to the budget, the advantage on joint exhaustion depends on the budget: "
        f"the median factor is {fold(float(np.median(small)))} at 4,096 paths, where the cross-entropy stage consumes most "
        f"of the budget, and {fold(float(np.median(big)))} at 65,536 paths (Figure [fig:e1budget]). Across all 75 budget–scenario–target settings the "
        f"nominal 95% intervals cover the exact value in {pct(float(np.mean(covs)), 0)} of replications "
        f"(range {pct(min(covs), 0)}–{pct(max(covs), 0)}).")
    return v


def mcfmt(x):
    return pct(x, 0) if x < 10 else f"{x:,.0f}×"


def sci(x):
    m, e = f"{x:.1e}".split("e")
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m}×10{str(int(e)).translate(sup)}"


def e2_values():
    d = json.loads((R / "rdpo_e2.json").read_text())["rows"]
    get = {(r["topology"], r["n"], r["target"]): r for r in d}
    v = {}
    ring_exc = sorted([r for r in d if r["topology"] == "ring" and r["target"] == "exceedance"], key=lambda r: r["n"])
    errs = [r["rdpo"]["rel_rmse"] for r in ring_exc]
    v["E2_RANGE"] = f"{pct(min(errs), 1)}–{pct(max(errs), 1)}"
    r28 = get.get(("ring", 28, "exceedance"))
    v["E2_RING28_EXC"] = pct(r28["rdpo"]["rel_rmse"], 1) if r28 else "—"
    v["E2_SEC_28"] = f"{r28['seconds_per_fit']:.0f}" if r28 else "—"
    hdr = ("| Graph | n | μ exceedance | RDPO | Exp. twisting | MC (theory) | RDPO stop-loss | Exp. twisting stop-loss | μ pair exhaustion | RDPO | MC (theory) |")
    lines = ["**Table [tab:e2].** Scaling results at a total budget of 65,536 paths (relative RMSE; training included). MC entries are theoretical relative errors at the same budget.", "", hdr, "|---" * 11 + "|"]
    keys = sorted({(r["topology"], r["n"]) for r in d}, key=lambda k: (k[0] != "ring", k[1]))
    for kind, n in keys:
        e, s, j = get.get((kind, n, "exceedance")), get.get((kind, n, "stop_loss")), get.get((kind, n, "joint_exhaustion"))
        f = lambda r, k="rdpo": pct(r[k]["rel_rmse"]) if r and k in r else "—"
        lines.append(f"| {kind} | {n} | {sci(e['mu'])} | **{f(e)}** | {f(e, 'twisted')} | {mcfmt(e['mc_theory_rel_rmse']) if e else '—'} | "
                     f"**{f(s)}** | {f(s, 'twisted')} | {sci(j['mu'])} | **{f(j)}** | {mcfmt(j['mc_theory_rel_rmse']) if j else '—'} |"
                     if e and s and j else f"| {kind} | {n} | — | — | — | — | — | — | — | — | — |")
    v["TABLE_E2"] = "\n".join(lines)
    tw = [r for r in d if "twisted" in r]
    ratios = [r["twisted"]["rel_rmse"] / r["rdpo"]["rel_rmse"] for r in tw]
    wins = sum(x > 1 for x in ratios)
    losses = [f"{r['target'].replace('_', '-')} on the {r['topology']} with n = {r['n']} "
              f"({pct(r['rdpo']['rel_rmse'], 1)} vs {pct(r['twisted']['rel_rmse'], 1)})" for r in tw
              if r["twisted"]["rel_rmse"] < r["rdpo"]["rel_rmse"]]
    jbad = [r for r in d if r["target"] == "joint_exhaustion" and r["rdpo"]["ci_coverage"] < 0.85]
    j28 = get.get(("ring", 28, "joint_exhaustion"))
    covs = [r["rdpo"]["ci_coverage"] for r in d]
    v["E2_TEXT"] = (
        f"As countries are added to the ring, the exceedance probability falls from {ring_exc[0]['mu']:.1e} to "
        f"{ring_exc[-1]['mu']:.1e}, while RDPO's relative error stays within {v['E2_RANGE']}. At the same budget, plain "
        f"Monte Carlo would have a relative error of about {ring_exc[-1]['mc_theory_rel_rmse']:,.0f} (i.e. {ring_exc[-1]['mc_theory_rel_rmse']:,.0f} times the value itself) at 28 countries; it "
        f"would need on the order of 10¹⁵ paths for 10% accuracy. Against classical exponential twisting, the strongest "
        f"simple rare-event method for threshold targets, RDPO is more accurate in {wins} of {len(ratios)} settings "
        f"(median factor {fold(float(np.median(ratios)))})."
        + (f" The exception is " + "; ".join(losses) + ": at the largest size, the one-parameter twist is already close "
           f"to optimal for a smooth loss threshold, and the 5n + 3|E| + 3 parameters of RDPO are not fully trained within the budget." if losses else "")
        + f" For the joint exhaustion of a single country pair inside the portfolio, a target that depends on two "
          f"specific countries among many, RDPO reaches "
        + ", ".join(f"{pct(r['rdpo']['rel_rmse'], 1)} (n = {r['n']})" for r in sorted([r for r in d if r['target'] == 'joint_exhaustion' and r['topology'] == 'ring'], key=lambda r: r['n']))
        + f" on the ring. Interval coverage across the scaling study averages {pct(float(np.mean(covs)), 0)}"
        + (f", but it falls to {pct(min(r['rdpo']['ci_coverage'] for r in jbad), 0)} for pair exhaustion in larger portfolios, "
           f"with only 6–10 replications; §6.5 re-examines coverage with 24" if jbad else "")
        + ". Replications are fewer at the largest sizes (6–20).")
    return v


def e3_values():
    d = json.loads((R / "rdpo_e3.json").read_text())["rows"]
    cases = list(dict.fromkeys(r["case"] for r in d))
    abl = list(dict.fromkeys(r["ablation"] for r in d))
    get = {(r["case"], r["ablation"]): r for r in d}
    hdr = "| Configuration | " + " | ".join(cases) + " |"
    lines = ["**Table [tab:e3].** Ablation: relative RMSE at B = 16,384 paths (40 replications). Ratios to full RDPO in parentheses.", "", hdr, "|---" * (len(cases) + 1) + "|"]
    for a in abl:
        cells = []
        for c in cases:
            r, f0 = get.get((c, a)), get.get((c, "full RDPO"))
            if not r:
                cells.append("—"); continue
            ratio = r["rel_rmse"] / f0["rel_rmse"]
            cells.append(pct(r["rel_rmse"]) + ("" if a == "full RDPO" else f" (×{ratio:.1f})"))
        lines.append(f"| {a} | " + " | ".join(cells) + " |")
    worst = {}
    for c in cases:
        f0 = get[(c, "full RDPO")]["rel_rmse"]
        for a in abl:
            if a != "full RDPO" and (c, a) in get:
                worst.setdefault(a, []).append(get[(c, a)]["rel_rmse"] / f0)
    def med(a):
        return float(np.median(worst[a])) if a in worst else float("nan")
    def ratios(a):
        return {c: get[(c, a)]["rel_rmse"] / get[(c, "full RDPO")]["rel_rmse"] for c in cases if (c, a) in get}
    def describe(a):
        r = ratios(a)
        hi = max(r, key=r.get); lo = min(r, key=r.get)
        return r, hi, lo
    essential, others = [], []
    for a in abl:
        if a in ("full RDPO", "CE only (v1-style)"):
            continue
        r, hi, lo = describe(a)
        name = a.replace("no ", "")
        if r[hi] >= 2.5:
            essential.append(f"{name} (removing it degrades {hi} by ×{fold(r[hi])})")
        else:
            hurt = sum(x > 1.15 for x in r.values()); helped = sum(x < 1 / 1.15 for x in r.values())
            others.append(f"{name[0].upper() + name[1:]}: removing it was worse in {hurt}, better in {helped} and within ±15% in "
                          f"{len(r) - hurt - helped} of {len(r)} cases (ratios ×{r[lo]:.1f} to ×{r[hi]:.1f})")
    v1 = ratios("CE only (v1-style)")
    better = [f"×{x:.1f} on {c}" for c, x in v1.items() if x > 1]
    worse = [f"×{1 / x:.1f} on {c}" for c, x in v1.items() if x <= 1]
    txt = ("\n".join(lines) + "\n\n" +
           "The ablation separates indispensable components from refinements. **Indispensable:** "
           + "; ".join(essential) + ". The cross-entropy stage is what reaches the rarest targets within the budget; "
           "the twist warm start and target-relevance freezing are what make the method work at portfolio scale, "
           "even though both are irrelevant for two countries. **Refinements** have smaller effects of either sign at "
           "this budget. " + "; ".join(others) + ". With 40 replications the relative RMSE itself carries a sampling "
           "error of roughly 10–20% (more in heavy-tailed cases), so ratios within this range are not significant. "
           "In particular, sample recycling gave no measurable benefit at these budgets; it is retained because it is "
           "unbiased and costless in compute, and whether to pool training batches will be decided by pilot effective "
           "sample size in Phase 2 (WP3). The Rényi-2 stage is likewise neutral here; its documented role is the repair "
           "of heavy-tailed fits (§4.4). Compared with the first prototype (cross-entropy only, factorised features, "
           "no relevance freezing), the full method is more accurate by " + ", ".join(better)
           + ((", and less accurate by " + ", ".join(worse) + ", where a 16-country model with more parameters competes "
               "with estimation for a small budget") if worse else "") + ".")
    return {"E3_TEXT": txt}


def q_values():
    d = json.loads((R / "quantum_advantage.json").read_text())
    adv = sorted(d["advantage"], key=lambda r: (r["topology"] != "ring", r["n"]))
    v = {}
    r28 = next(r for r in adv if r["topology"] == "ring" and r["n"] == 28)
    v["Q_SSTAR_28"] = sci(r28["required_speedup"]).replace("1.0×", "")
    v["Q_PL_28"] = sci(r28["pL_required"])
    def dur(s):
        return f"{s * 1e3:.1f} ms" if s < 1 else (f"{s:.1f} s" if s < 600 else (f"{s / 3600:.1f} h" if s < 2 * 86400 else f"{s / 86400:.0f} d"))
    lines = ["**Table [tab:qadv].** Falsifiable wall-clock comparison at 10% relative error on Pr(L > Tₙ). Classical: measured "
             "single-core RDPO time (training included). Quantum: projected IQAE runtime on the stated surface-code baseline. "
             "s*: hardware speed-up required for break-even. p_L: logical two-qubit error rate required by the deepest circuit.", "",
             "| Graph | n | μ | twisted amplitude a | RDPO paths needed | RDPO time | AE calls | max Grover depth | quantum time | s* | p_L required |",
             "|---" * 11 + "|"]
    for r in adv:
        lines.append(f"| {r['topology']} | {r['n']} | {sci(r['mu'])} | {r['a_twisted']:.3f} | {r['classical_paths']:,.0f} | "
                     f"{dur(r['classical_seconds'])} | {r['quantum_calls']:,.0f} | {r['quantum_kmax']:.0f} | {dur(r['quantum_seconds'])} | "
                     f"**{sci(r['required_speedup'])}** | {sci(r['pL_required'])} |")
    v["Q_ADV_TABLE"] = "\n".join(lines)
    ss = [r["required_speedup"] for r in adv]
    ring = [r for r in adv if r["topology"] == "ring"]
    v["Q_ADV_TEXT"] = "\n".join([
        f"* **Result.** In every setting the quantum stage is slower: break-even needs a hardware speed-up between "
        f"{sci(min(ss))} and {sci(max(ss))}. The reason is that RDPO needs only "
        f"{min(r['classical_paths'] for r in adv):,.0f}–{max(r['classical_paths'] for r in adv):,.0f} paths for 10% error, so the "
        f"quadratic query advantage has almost nothing to act on.",
        f"* **Trend.** The gap closes with portfolio size: on the ring, s* falls from {sci(ring[0]['required_speedup'])} (n = 2) to "
        f"{sci(ring[-1]['required_speedup'])} (n = 28), because RDPO's per-path variance grows "
        f"(ρ² from {ring[0]['classical_rho2']:.2f} to {ring[-1]['classical_rho2']:.0f}) while the twisted amplitude decays only slowly.",
        f"* **Hardware requirement.** The deepest circuit at n = 28 has Grover depth {ring[-1]['quantum_kmax']:.0f}, "
        f"{ring[-1]['logical_qubits']} logical qubits and about {ring[-1]['physical_qubits'] / 1e6:.1f} million physical qubits, and needs a "
        f"logical error rate of {sci(ring[-1]['pL_required'])}.",
        "* **Falsification statements.** The quantum stage is *not* useful for a portfolio class if (i) s* stays above the "
        "projected speed-up of available fault-tolerant hardware, or (ii) RDPO's per-path relative variance ρ² stays below "
        "about 10³ as portfolios grow. Conversely, a target class on which RDPO's ρ² grows quickly (dense dependence, "
        "many-country compound triggers) is where the stage should be re-tested first (WP4).",
        "* **What the comparison already includes.** The classical side is our *best* method, timed with training; the quantum "
        "side includes state preparation, the weight oracle and all IQAE rounds. Comparing against plain Monte Carlo instead would "
        "show a misleading quantum advantage of several orders of magnitude."])
    ms = d["mitigation_summary"]
    lines = ["**Table [tab:mit].** Error mitigation under control-offset miscalibration (250 random instances per level; relative error "
             "of the estimated amplitude). Guard 2 combines the goodness-of-fit and pilot-disagreement guards with fallback to RDPO.", "",
             "| offset δ_max | mitigation worse than none | p99 error: none | mitigated | guard 2 | max error: none | mitigated | guard 2 | fallback rate |",
             "|---" * 9 + "|"]
    for k in sorted(ms, key=float):
        m = ms[k]
        lines.append(f"| {float(k):g} | {pct(m['mitigation_worse_than_raw'], 0)} | {m['raw']['p99']:.2f} | {m['mit']['p99']:.2f} | "
                     f"**{m['guard2']['p99']:.2f}** | {m['raw']['max']:.2f} | {m['mit']['max']:.2f} | **{m['guard2']['max']:.2f}** | "
                     f"{pct(m['fallback2_rate'], 0)} |")
    v["MIT_TABLE"] = "\n".join(lines)
    ks = sorted(ms, key=float)
    worse = [ms[k]["mitigation_worse_than_raw"] for k in ks]
    v["MIT_TEXT"] = "\n".join([
        f"* **Mitigation is not monotone.** Noise-aware fitting removes the worst raw errors, but it is worse than no mitigation "
        f"in {pct(worse[0], 0)} of instances without miscalibration, rising to {pct(worse[-1], 0)} at δ_max = {float(ks[-1]):g}. "
        "This reproduces the qualitative finding of [@errmit2026] on our own workload.",
        f"* **Goodness of fit alone is not enough.** The likelihood-ratio guard triggers in only "
        f"{pct(min(ms[k]['fallback_rate'] for k in ks), 0)}–{pct(max(ms[k]['fallback_rate'] for k in ks), 0)} of instances: a coherent "
        "offset is largely absorbed by a shifted amplitude, so the fit still looks good.",
        f"* **An independent classical check bounds the tail.** Adding the pilot-disagreement test caps the 99th-percentile error "
        f"at {min(ms[k]['guard2']['p99'] for k in ks):.2f}–{max(ms[k]['guard2']['p99'] for k in ks):.2f} and the maximum at "
        f"{max(ms[k]['guard2']['max'] for k in ks):.2f} across all offset levels, at a fallback rate of "
        f"{pct(min(ms[k]['fallback2_rate'] for k in ks), 0)}–{pct(max(ms[k]['fallback2_rate'] for k in ks), 0)}. The median error "
        "is unchanged or better.",
        "* **Design rule.** The quantum estimate is accepted only when it is consistent with an unbiased classical pilot. This is "
        "where RDPO and the quantum stage meet: the classical RL estimator is the calibration anchor, not a competitor to be hidden."])
    lines = ["**Table [tab:qres].** Projected logical and physical resources per amplitude-estimation run (10% relative error, "
             "twisted loading with the bitwise weight oracle). A: one state preparation; Q: one Grover iterate.", "",
             "| Graph | n | logical qubits | Toffoli (A) | rotations (A) | CNOT (Q) | lookup Toffolis avoided | code distance | physical qubits | runtime |",
             "|---" * 10 + "|"]
    for r in adv:
        lines.append(f"| {r['topology']} | {r['n']} | {r['logical_qubits']} | {r['toffoli_A']:,} | {r['rotations_A']:,} | "
                     f"{r['cnot_Q']:,} | {r['lookup_toffoli_avoided']:,} | {r['code_distance']} | {r['physical_qubits']:,} | {dur(r['quantum_seconds'])} |")
    v["Q_RES_TABLE"] = "\n".join(lines)
    return v


def v2_values():
    tp = json.loads((R / "rdpo_v2_throughput.json").read_text())["rows"]
    sc = json.loads((R / "rdpo_v2_scale.json").read_text())
    rows = sc["rows"]; reps = sc["reps"]
    bench = json.loads((R / "rdpo_v2_bench.json").read_text())["rows"]
    tr = json.loads((R / "rdpo_v2_transfer.json").read_text())["rows"]
    v = {}
    sp = [r["speedup"] for r in tp if r["n"] >= 4]
    v["V2_SPEEDUP"] = f"{min(sp):.1f}–{max(sp):.1f}×"
    lines = ["**Table [tab:v2tp].** Rollout throughput (paths per second, one core) of the v1 engine and the table-driven v2 "
             "engine. Both sample from the identical policy (tested).", "",
             "| Graph | n | sources K | v1 | v2 | speed-up |", "|---|---|---|---|---|---|"]
    for r in tp:
        lines.append(f"| {r['topology']} | {r['n']} | {r['K']} | {r['v1']:,.0f} | {r['fast']:,.0f} | **{r['speedup']:.1f}×** |")
    v["V2_TP_TABLE"] = "\n".join(lines)
    get = {(r["topology"], r["n"], r["target"], r["variant"]): r for r in rows}
    keys = sorted({(r["topology"], r["n"], r["target"]) for r in rows}, key=lambda k: (k[0] != "ring", k[1], k[2]))
    tl = {"exceedance": "Pr(L > T)", "stop_loss": "stop-loss", "joint_exhaustion": "pair exhaustion"}
    lines = [f"**Table [tab:v2scale].** Scaling cases at B = 65,536 with {reps} replications. *v1 study*: the original results "
             "(6–10 replications). *v1 rerun* and *v2*: same seeds, {reps} replications. Relative RMSE, with empirical 95%-interval "
             "coverage in parentheses; time per estimate on one core.".replace("{reps}", str(reps)), "",
             "| Graph | n | target | v1 study | v1 rerun | v2 | exp. twisting | time v1 → v2 |", "|---|---|---|---|---|---|---|---|"]
    gains, covs_v2, covs_v1 = [], [], []
    for k in keys:
        a = get.get(k + ("v1",)); b = get.get(k + ("v2-final",)) or get.get(k + ("v2",))
        old = a["v1_study"]
        lines.append(f"| {k[0]} | {k[1]} | {tl[k[2]]} | {pct(old['rel_rmse'])} ({pct(old['ci_coverage'], 0)}) | "
                     f"{pct(a['rel_rmse'])} ({pct(a['ci_coverage'], 0)}) | **{pct(b['rel_rmse'])}** ({pct(b['ci_coverage'], 0)}) | "
                     f"{pct(a['twisted']) if a['twisted'] else '—'} | {a['v1_study_seconds']:.0f} s → {b['seconds_per_fit']:.0f} s |")
        covs_v1.append(a["ci_coverage"]); covs_v2.append(b["ci_coverage"])
        if k[2] != "joint_exhaustion":
            gains.append(a["rel_rmse"] / b["rel_rmse"])
    v["V2_SCALE_TABLE"] = "\n".join(lines)
    tw_rows = [k for k in keys if get[k + ("v1",)]["twisted"]]
    tw_wins = sum(get[k + ("v1",)]["twisted"] > (get.get(k + ("v2-final",)) or get[k + ("v2",)])["rel_rmse"] for k in tw_rows)
    times = [(get[k + ("v1",)]["v1_study_seconds"] / (get.get(k + ("v2-final",)) or get[k + ("v2",)])["seconds_per_fit"]) for k in keys]
    better = [g for g in gains if g > 1]; worse = [g for g in gains if g <= 1]
    v["V2_GAIN"] = (f"is more accurate than v1 in {len(better)} of {len(gains)} settings (error lower by up to ×{max(gains):.1f})"
                    + (f" and slightly less accurate in {len(worse)} (×{min(worse):.2f}–×{max(worse):.2f})" if worse else ""))
    v["V2_GAIN_MAX"] = f"{max(gains):.1f}×"
    v["V2_TIME"] = f"{min(times):.1f}–{max(times):.1f}×"
    ep = sc["exact_pair"]
    v["V2_EXACT_MS"] = f"{max(e['ms'] for e in ep.values()):.0f} ms"
    v["V2_EXACT_ERR"] = sci(max(abs(e["rel_err_vs_mps"]) for e in ep.values()))
    v["V2_TW_WINS"] = f"{tw_wins} of {len(tw_rows)}"
    v["V2_COV"] = f"{pct(min(covs_v1), 0)}–{pct(max(covs_v1), 0)}"
    # benchmark
    def summ(B):
        bb = [r for r in bench if r["budget"] == B]
        wins = sum(r["improvement_vs_best"] > 1 for r in bb); w1 = sum(r["v1_improvement_vs_best"] > 1 for r in bb)
        better = sum(r["v2"]["rel_rmse"] < r["v1"]["rel_rmse"] for r in bb)
        exh = [r for r in bb if r["target"] == "joint_exhaustion"]
        el = [r for r in bb if r["target"] == "expected_loss"]
        npair = [r for r in bb if r["target"] not in ("joint_attachment", "joint_exhaustion")]
        return wins, w1, better, float(np.median([r["improvement_vs_best"] for r in el])), \
            float(np.median([r["v1_improvement_vs_best"] for r in el])), float(np.mean([r["v2"]["ci_coverage"] for r in bb])), \
            float(np.median([r["v1"]["rel_rmse"] / r["v2"]["rel_rmse"] for r in npair]))
    lines = ["**Table [tab:v2bench].** Benchmark (25 scenario–target cells, 40 replications) with the final v2 configuration.", "",
             "| Budget | cells better than best reference: v1 → v2 | cells where v2 beats v1 | median v1/v2 error ratio (non-pair cells) | expected loss, median factor vs Sobol: v1 → v2 | mean coverage v2 |",
             "|---|---|---|---|---|---|"]
    for B in (4096, 16384, 65536):
        w, w1, bt, m2, m1, cv, ratio = summ(B)
        lines.append(f"| {B:,} | {w1} → **{w}** | {bt} of 25 | {ratio:.2f} | ×{m1:.2f} → **×{m2:.2f}** | {pct(cv, 0)} |")
    v["V2_BENCH_TABLE"] = "\n".join(lines)
    cv = [r["v2"]["ci_coverage"] for r in bench]
    v["V2_BENCH_COV"] = f"{pct(min(cv), 0)}–{pct(max(cv), 0)}"
    # transfer
    lines = ["**Table [tab:v2transfer].** Overnight transfer (sleep phase): today's estimate starts from a policy trained on yesterday's "
             "model instead of from scratch. Relative RMSE, 40 replications.", "",
             "| Change overnight | target | budget | from scratch | with transfer | gain |", "|---|---|---|---|---|---|"]
    tg = {}
    for r in tr:
        tg.setdefault((r["case"], r["target"], r["budget"]), {})[r["transfer"]] = r
    tgains = []
    for (c, t, B), d in sorted(tg.items()):
        if False in d and True in d:
            g = d[False]["rel_rmse"] / d[True]["rel_rmse"]; tgains.append(g)
            lines.append(f"| {c} | {tl.get(t, t)} | {B:,} | {pct(d[False]['rel_rmse'])} | **{pct(d[True]['rel_rmse'])}** | ×{g:.1f} |")
    v["V2_TRANSFER_TABLE"] = "\n".join(lines)
    v["V2_TRANSFER_GAIN"] = f"×{min(tgains):.1f}–×{max(tgains):.1f}" if tgains else "—"
    return v


def mk_values():
    m = json.loads((R / "market_model.json").read_text())
    return {"MK_SAM": f"{m['sam_m']:.0f}", "MK_SOM": f"{m['som_m']:.1f}", "MK_CUST": str(m["customers_y5"]),
            "MK_INST": str(sum(r["institutions"] for r in m["segments"])), "MK_SHARE": pct(m["som_share_of_sam"], 0)}


def number_labels(md):
    """Replace [fig:key] / [tab:key] by numbers in order of first appearance."""
    for kind in ("fig", "tab"):
        order = []
        for k in re.findall(r"\[" + kind + r":(\w+)\]", md):
            if k not in order:
                order.append(k)
        for i, k in enumerate(order, 1):
            md = md.replace(f"[{kind}:{k}]", str(i))
    left = re.findall(r"\[(?:fig|tab):\w+\]", md)
    if left:
        print("UNRESOLVED LABELS:", left, file=sys.stderr)
    return md


def toc(md):
    heads = [(n, re.sub(r"\s*\{[^}]*\}\s*$", "", t)) for n, t in re.findall(r"^# (\d+) (.+)$", md, flags=re.M)]
    items = "\n\n".join(f"**{n}**  {t}" for n, t in heads)
    return f"::: {{.toc}}\n{items}\n\n**·**  References\n:::"


def fill():
    md = (PAPER / "proposal.md").read_text()
    vals = {}
    for fn in (e1_values, e2_values, e3_values, q_values, mk_values, v2_values):
        try:
            vals.update(fn())
        except Exception as ex:
            print("placeholder group failed:", fn.__name__, type(ex).__name__, ex, file=sys.stderr)
    md = md.replace("# 1 Introduction {.newpage}", toc(md) + "\n\n# 1 Introduction", 1)
    for k, val in vals.items():
        md = md.replace("{{" + k + "}}", val)
    md = number_labels(md)
    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", md)))
    if left:
        print("UNFILLED:", left, file=sys.stderr)
    (PAPER / "proposal.filled.md").write_text(md)
    return left


FONTS = ('<link rel="preconnect" href="https://fonts.gstatic.com">'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&display=block">')

PDF_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const [html, out, mode] = process.argv.slice(2);
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + html, { waitUntil: 'networkidle' });
  await p.evaluate(() => document.fonts.ready);
  await p.evaluate(async () => { if (window.MathJax && MathJax.startup) { await MathJax.startup.promise; } });
  if (mode === 'cover') {
    await p.pdf({ path: out, format: 'A4', printBackground: true, margin: { top: 0, bottom: 0, left: 0, right: 0 } });
  } else {
    await p.pdf({ path: out, format: 'A4', printBackground: true, preferCSSPageSize: true, displayHeaderFooter: true,
      headerTemplate: '<div></div>',
      footerTemplate: '<div style="width:100%;font-family:Inter,sans-serif;font-size:7pt;color:#8A8E98;padding:0 18mm;display:flex;justify-content:space-between;"><span>Rényi-Doob Policy Optimization · ADB Quantum Challenge, Phase 1</span><span class="pageNumber"></span></div>' });
  }
  await b.close();
})();
"""


def build(cv=None):
    art.main()
    art2.main()
    (R / "market_model.json").write_text(json.dumps(market.model(), indent=2))
    art3.main()
    paper_figs.main()
    left = fill()
    html = PAPER / "proposal.html"
    (PAPER / "fonts.html").write_text(FONTS)
    mj = (PAPER / "mathjax-config.html")
    mj.write_text('<script>window.MathJax={tex:{packages:{"[+]":["ams"]}},svg:{fontCache:"global",scale:0.98},'
                  'startup:{typeset:true}};</script>')
    subprocess.run(["pandoc", str(PAPER / "proposal.filled.md"), "-f", "markdown", "-t", "html5", "-s",
                    "--mathjax=node_modules/mathjax-full/es5/tex-svg-full.js", "-H", str(mj),
                    "--citeproc", "--bibliography", str(PAPER / "references.bib"), "--css", "paper.css",
                    "-H", str(PAPER / "fonts.html"), "-o", str(html)], check=True)
    js = PAPER / "pdf.js"; js.write_text(PDF_JS)
    body, cover = PAPER / "body.pdf", PAPER / "cover.pdf"
    subprocess.run(["node", str(js), str(html.resolve()), str(body), "body"], check=True)
    cov_html = PAPER / "cover.html"
    cover_svg = (ROOT / "docs/art/cover.svg").read_text().replace('width="794" height="1123"', 'width="100%" height="100%"')
    cov_html.write_text('<html><head><style>@page{size:A4;margin:0}html,body{margin:0;background:#F6F1E7;'
                        'width:210mm;height:297mm;overflow:hidden}svg{display:block;width:210mm;height:297mm}</style>'
                        + FONTS + '</head><body>' + cover_svg + '</body></html>')
    subprocess.run(["node", str(js), str(cov_html.resolve()), str(cover), "cover"], check=True)
    out = ROOT / "docs/RDPO_Proposal.pdf"
    subprocess.run(["pdfunite", str(cover), str(body), str(out)], check=True)
    if cv:
        Path("submission").mkdir(exist_ok=True)
        subprocess.run(["pdfunite", str(cover), str(body), cv, "submission/RDPO_Proposal_with_CV_annex.pdf"], check=True)
    print("built", out, "unfilled:", left)


if __name__ == "__main__":
    cv = sys.argv[sys.argv.index("--cv") + 1] if "--cv" in sys.argv else None
    build(cv)
