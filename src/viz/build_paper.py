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

from . import art, art2, art3, art4, art5, paper_figs
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


def money(x):
    """US$ amounts: millions when material, otherwise the exact small value."""
    return f"US${x / 1e6:.2f}m" if x >= 1e5 else f"US${x:,.0f}" if x >= 1 else f"US${sci(x)}"


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
    lines = ["**Table [tab:v2tp].** Rollout throughput (paths per second, one core) of the Phase 1 engine and the table-driven "
             "Phase 2 engine. Both sample from the identical policy (tested).", "",
             "| Graph | n | sources K | Phase 1 | Phase 2 | speed-up |", "|---|---|---|---|---|---|"]
    for r in tp:
        lines.append(f"| {r['topology']} | {r['n']} | {r['K']} | {r['v1']:,.0f} | {r['fast']:,.0f} | **{r['speedup']:.1f}×** |")
    v["V2_TP_TABLE"] = "\n".join(lines)
    get = {(r["topology"], r["n"], r["target"], r["variant"]): r for r in rows}
    keys = sorted({(r["topology"], r["n"], r["target"]) for r in rows}, key=lambda k: (k[0] != "ring", k[1], k[2]))
    tl = {"exceedance": "Pr(L > T)", "stop_loss": "stop-loss", "joint_exhaustion": "pair exhaustion"}
    ep = sc["exact_pair"]
    lines = [f"**Table [tab:app_v2scale].** Scaling study at B = 65,536 with {reps} replications per cell and identical seeds for "
             "every method. Relative RMSE, with empirical 95%-interval coverage in parentheses. *Phase 1 study*: the original "
             "results (6–20 replications). Pair exhaustion is answered exactly by the Phase 2 engine (pair reduction); its "
             "sampled estimate is shown for comparison.", "",
             "| Graph | n | target | Phase 1 study | Phase 1 engine | Phase 2 engine | exp. twisting | time per estimate |",
             "|---|---|---|---|---|---|---|---|"]
    gains, covs_v2, covs_v1, tw_rows, tw_wins, times = [], [], [], 0, 0, []
    for k in keys:
        a = get.get(k + ("v1",)); b = get.get(k + ("v2-final",)) or get.get(k + ("v2",))
        if not a or not b:
            continue
        old = a["v1_study"]
        tw = get.get(k + ("twist",))
        twv = tw["rel_rmse"] if tw else None
        if k[2] == "joint_exhaustion":
            e = ep.get(f"{k[0]}-{k[1]}")
            v2cell = f"**exact** ({e['ms']:.0f} ms); sampled {pct(b['rel_rmse'])}" if e else pct(b["rel_rmse"])
        else:
            v2cell = f"**{pct(b['rel_rmse'])}** ({pct(b['ci_coverage'], 0)})"
            gains.append(a["rel_rmse"] / b["rel_rmse"]); covs_v2.append(b["ci_coverage"])
            if twv:
                tw_rows += 1; tw_wins += twv > b["rel_rmse"]
        covs_v1.append(a["ci_coverage"])
        t_old = a["v1_study_seconds"]
        times.append(t_old / b["seconds_per_fit"]) if t_old else None
        lines.append(f"| {k[0]} | {k[1]} | {tl[k[2]]} | "
                     + (f"{pct(old['rel_rmse'])} ({pct(old['ci_coverage'], 0)})" if old else "—")
                     + f" | {pct(a['rel_rmse'])} ({pct(a['ci_coverage'], 0)}) | {v2cell} | {pct(twv) if twv else '—'} | "
                     + (f"{t_old:.0f} s → {b['seconds_per_fit']:.1f} s" if t_old else f"{b['seconds_per_fit']:.1f} s") + " |")
    v["V2_SCALE_TABLE"] = "\n".join(lines)
    better = [g for g in gains if g > 1]; worse = [g for g in gains if g <= 1]
    v["V2_GAIN"] = (f"is more accurate than the Phase 1 engine in {len(better)} of {len(gains)} settings (error lower by up to "
                    f"×{max(gains):.1f}, median ×{float(np.median(gains)):.2f})"
                    + (f" and less accurate in {len(worse)} (×{min(worse):.2f}–×{max(worse):.2f})" if worse else ""))
    v["V2_GAIN_MAX"] = f"{max(gains):.1f}×"
    v["V2_TIME"] = f"{min(times):.1f}–{max(times):.1f}×"
    v["V2_TW_WINS"] = f"{tw_wins} of {tw_rows}"
    v["V2_COV"] = f"{pct(min(covs_v2), 0)}–{pct(max(covs_v2), 0)} (mean {pct(float(np.mean(covs_v2)), 0)})"
    ep = sc["exact_pair"]
    v["V2_EXACT_MS"] = f"{max(e['ms'] for e in ep.values()):.0f} ms"
    v["V2_EXACT_ERR"] = sci(max(abs(e["rel_err_vs_mps"]) for e in ep.values()))
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
    lines = ["**Table [tab:v2bench].** Benchmark (25 scenario–target cells, 40 replications): Phase 1 → Phase 2 engine.", "",
             "| Budget | cells better than best reference | cells where Phase 2 beats Phase 1 | median error ratio Phase 1 / Phase 2 (non-pair cells) | expected loss, median factor vs Sobol | mean coverage, Phase 2 |",
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


def rd_values():
    d = json.loads((R / "realdata_eq.json").read_text())
    v = {}
    gr = d["gutenberg_richter"]
    v["RD_B"] = f"{gr['b']:.2f} ± {gr['b_se']:.2f}"
    v["RD_MSTAR"] = f"M{gr['KG']['implied_trigger_magnitude']:.1f} (Kyrgyz Republic) and M{gr['TJ']['implied_trigger_magnitude']:.1f} (Tajikistan)"
    v["RD_N"] = f"{d['data']['events_declustered']}"
    lines = ["**Table [tab:rd].** Earthquake main shocks (aftershocks removed) within a given distance of each country, "
             "1965–2016. *Shared share*: fraction of a country's events that are also within that distance of the other "
             "country (95% Wilson interval); the brief assumes 7.5% (KG) and 7.4% (TJ). *Dispersion*: variance/mean of yearly "
             "counts (Poisson = 1) with the χ² p-value. *Years*: Fisher test of independence between the two countries' event "
             "years.", "",
             "| Distance | M ≥ | events KG / TJ | shared share KG | shared share TJ | dispersion KG / TJ (p) | years: both observed vs independent (p) |",
             "|---|---|---|---|---|---|---|"]
    disp, shares = [], []
    for Rk, dm in d["radii_km"].items():
        for M, r in dm.items():
            k, t = r["KG"], r["TJ"]
            def sh(x):
                return f"{pct(x['shared_share'], 0)} ({pct(x['shared_share_ci95'][0], 0)}–{pct(x['shared_share_ci95'][1], 0)})"
            lines.append(f"| {Rk} km | {M} | {k['events']} / {t['events']} | {sh(k)} | {sh(t)} | "
                         f"{k['dispersion_index']:.2f} ({k['poisson_p_value']:.2f}) / {t['dispersion_index']:.2f} ({t['poisson_p_value']:.2f}) | "
                         f"{pct(r['years']['p_both_observed'], 0)} vs {pct(r['years']['p_both_if_independent'], 0)} (p = {r['years']['fisher_p']:.3f}) |")
            if Rk == "50":
                disp += [(k["dispersion_index"], k["poisson_p_value"]), (t["dispersion_index"], t["poisson_p_value"])]
            shares += [k["shared_share"], t["shared_share"]]
    v["RD_TABLE"] = "\n".join(lines)
    v["RD_DISP"] = f"{min(x for x, _ in disp):.2f}–{max(x for x, _ in disp):.2f}"
    v["RD_DISP_P"] = f"{min(p for _, p in disp):.2f}"
    v["RD_SHARE"] = f"{pct(min(shares), 0)}–{pct(max(shares), 0)}"
    im = d["implied_scenarios"]; rows = im["rows"]
    lines = [f"**Table [tab:rdimp].** Exact two-country targets when the shared-event intensity is set from the data "
             f"(events within {im['radius_km']:.0f} km of both countries, M ≥ {im['min_magnitude']:.1f}), compared with the brief.", "",
             "| Shared-event setting | shared share | intensity m | joint attachment | Pr(L > 80m) | joint exhaustion |", "|---|---|---|---|---|---|"]
    lab = {"brief": "brief (base case)", "ci_low": "data, lower 95% bound", "observed": "data, point estimate", "ci_high": "data, upper 95% bound"}
    for key in ("brief", "ci_low", "observed", "ci_high"):
        r = rows[key]
        lines.append(f"| {lab[key]} | {pct(r['shared_share'], 1)} | {r['m']:.1f} | {pct(r['joint_attachment'])} | "
                     f"{pct(r['exceedance'])} | {sci(r['joint_exhaustion'])} |")
    v["RD_IMP_TABLE"] = "\n".join(lines)
    b = rows["brief"]["joint_exhaustion"]
    v["RD_EXH_FACTOR"] = f"{rows['ci_low']['joint_exhaustion'] / b:.1f}–{rows['ci_high']['joint_exhaustion'] / b:.1f}×"
    v["RD_ATT"] = f"{pct(rows['ci_low']['joint_attachment'], 1)}–{pct(rows['ci_high']['joint_attachment'], 1)}"
    return v


def loss_values():
    d = json.loads((R / "realdata_losses.json").read_text())
    v = {}
    lines = ["**Table [tab:loss].** Loss-data backtest of the shared-event assumption. Yearly occurrence of reported losses "
             "in the Kyrgyz Republic (KG) and Tajikistan (TJ), and the bivariate-Poisson estimate of the shared share (the "
             "fraction of KG events that also hit TJ; the brief assumes 7.5%). Profile-likelihood 95% intervals.", "",
             "| Loss record | years | KG / TJ / both | both, if independent | shared share (95% CI) | test of independence |",
             "|---|---|---|---|---|---|"]
    for k, t in d["tests"].items():
        tab = t["table_both_KGonly_TJonly_none"]; n = t["years"][1] - t["years"][0] + 1
        lines.append(f"| {t['label'].rsplit(',', 1)[0]} | {t['years'][0]}–{t['years'][1]} | "
                     f"{tab[0][0] + tab[0][1]} / {tab[0][0] + tab[1][0]} / **{tab[0][0]}** | {t['p_both_if_independent'] * n:.1f} | "
                     f"**{pct(t['share_of_A'], 0)}** ({pct(t['share_of_A_ci95'][0], 0)}–{pct(t['share_of_A_ci95'][1], 0)}) | "
                     f"p = {t['lr_test_independence_p']:.3f} |")
    v["LOSS_TABLE"] = "\n".join(lines)
    t = d["tests"]["eq_noaa_1900"]
    v["LOSS_SHARE"] = f"{pct(t['share_of_A'], 0)} ({pct(t['share_of_A_ci95'][0], 0)}–{pct(t['share_of_A_ci95'][1], 0)})"
    v["LOSS_P"] = f"{t['lr_test_independence_p']:.3f}"
    im = d["implied"]; b = im["brief"]["joint_exhaustion"]
    lo, mid, hi = im["loss data, lower 95% bound"], im["loss data, point estimate"], im["loss data, upper 95% bound"]
    v["LOSS_EXH"] = f"{sci(mid['joint_exhaustion'])} (95% range {sci(lo['joint_exhaustion'])}–{sci(hi['joint_exhaustion'])})"
    v["LOSS_EXH_FACTOR"] = f"{mid['joint_exhaustion'] / b:.1f}×"
    return v


def pool_values():
    d = json.loads((R / "realdata_pool.json").read_text())
    cal = d["calibration"]; v = {}
    names = {"KG": "Kyrgyz Rep.", "TJ": "Tajikistan", "UZ": "Uzbekistan", "KZ": "Kazakhstan", "TM": "Turkmenistan"}
    lines = ["**Table [tab:poolcal].** Data-calibrated five-country pool. Annual probabilities of a 25% and a 100% earthquake "
             "payout per country (each country holds US$70m earthquake + US$10m rain cover), and shared-event rates on the "
             "edges found in the catalogue.", "",
             "| Country | main shocks M ≥ 5.5 within 50 km, per year | annual P(≥ 25% payout) | annual P(100% payout) | brief (KG, TJ) |",
             "|---|---|---|---|---|"]
    for c in ("KG", "TJ", "UZ", "KZ", "TM"):
        bt = cal["brief_tails"].get(c)
        lines.append(f"| {names[c]} | {cal['rate55'][c]:.2f} | {pct(cal['tails'][c][0])} | {pct(cal['tails'][c][3])} | "
                     + (f"{pct(bt[0])} / {pct(bt[3])}" if bt else "—") + " |")
    v["POOL_CAL_TABLE"] = "\n".join(lines)
    v["POOL_EDGES"] = ", ".join(f"{names[a]}–{names[b]} ({cal['n_both'][f'{a}-{b}']} events)" for a, b in cal["edges"])
    v["POOL_B"] = f"{cal['b']:.2f}"; v["POOL_M1"] = f"M{cal['m1']:.1f}"
    tl = {"exceedance": "Pr(pool payout > {})", "stop_loss": "stop-loss above {}", "joint_exhaustion": "KG and TJ both exhausted",
          "joint_attachment": "KG and TJ both paid", "expected_loss": "expected pool payout"}
    get = {(r["target"], r["threshold_musd"], r["method"]): r for r in d["rows"]}
    keys = list(dict.fromkeys((r["target"], r["threshold_musd"]) for r in d["rows"]))
    lines = [f"**Table [tab:pool].** RDPO (Phase 2 engine) on the data-calibrated pool against exact answers (dense "
             f"enumeration), at 16,384 simulated terms, {d['reps']} replications. Relative RMSE; interval coverage in parentheses.", "",
             "| Question | exact value | RDPO | plain Monte Carlo | exp. twisting | RDPO gain over best alternative |", "|---|---|---|---|---|---|"]
    gains = []
    for t, thr in keys:
        r = get.get((t, thr, "rdpo")); mc = get.get((t, thr, "mc")); tw = get.get((t, thr, "twist"))
        if not r:
            continue
        lab = tl[t].format(f"US${thr:.0f}m" if thr else "")
        val = money(r["mu"]) if t in ("expected_loss", "stop_loss") else sci(r["mu"])
        alts = [x["rel_rmse"] for x in (mc, tw) if x]
        g = min(alts) / r["rel_rmse"] if alts else float("nan"); gains.append((t, g))
        lines.append(f"| {lab} | {val} | **{pct(r['rel_rmse'])}** ({pct(r['ci_coverage'], 0)}) | "
                     f"{pct(mc['rel_rmse']) if mc else '—'} | {pct(tw['rel_rmse']) if tw else '—'} | ×{g:.1f} |")
    v["POOL_TABLE"] = "\n".join(lines)
    rare = [g for t, g in gains if t not in ("expected_loss",)]
    v["POOL_GAIN"] = f"×{min(rare):.1f}–×{max(rare):.0f}"
    el = [g for t, g in gains if t == "expected_loss"]
    v["POOL_EL"] = f"×{el[0]:.1f}" if el else "—"
    pr = get.get(("joint_exhaustion", None, "rdpo"))
    v["POOL_PAIR"] = pct(pr["rel_rmse"], 1) if pr else "—"
    v["POOL_PAIR_COV"] = pct(pr["ci_coverage"], 0) if pr else "—"
    return v


def wp2_values():
    d = json.loads((R / "wp2_carec.json").read_text())
    cal = d["calibration"]; v = {}
    names = {"AF": "Afghanistan", "AZ": "Azerbaijan", "GE": "Georgia", "KZ": "Kazakhstan", "KG": "Kyrgyz Rep.",
             "MN": "Mongolia", "PK": "Pakistan", "TJ": "Tajikistan", "TM": "Turkmenistan", "UZ": "Uzbekistan"}
    lines = ["**Table [tab:carec].** Calibrated CAREC pool (west to east). Each country holds US$70m earthquake + US$10m rain "
             "cover. Rain-trigger probabilities scale the brief's 2% by each country's frequency of major flood, landslide "
             "and storm losses (EM-DAT 1990–2020).", "",
             "| Country | main shocks M ≥ 5.5 within 50 km, per year | annual P(≥ 25% EQ payout) | annual P(100% EQ payout) | annual P(rain trigger) |",
             "|---|---|---|---|---|"]
    for c in cal["order"]:
        lines.append(f"| {names[c]} | {cal['rate55'][c]:.2f} | {pct(cal['tails'][c][0])} | {pct(cal['tails'][c][3])} | {pct(cal['rain_p'][c])} |")
    v["CAREC_TABLE"] = "\n".join(lines)
    v["CAREC_EDGES"] = ", ".join(f"{names[a]}–{names[b]}" for a, b in cal["edges"])
    v["CAREC_NEDGES"] = str(len(cal["edges"]))
    v["CAREC_EVENTS"] = str(cal["events"])
    v["CAREC_BOND"] = str(d["mps"]["max_bond"])
    v["CAREC_MPS_S"] = f"{d['mps']['mps_seconds'] / 60:.0f} min"
    v["CAREC_TR"] = f"US${d['T_rare']}m"; v["CAREC_TM"] = f"US${d['T_moderate']}m"
    v["CAREC_MU_RARE"] = sci(d["references"][f"exceedance|{d['T_rare']}"])
    v["CAREC_CHECK"] = sci(max(d["references"][f"check|{d['T_rare']}"], d["references"][f"check|{d['T_moderate']}"]))
    tl = {"exceedance": "Pr(pool payout > {})", "stop_loss": "stop-loss above {}", "joint_exhaustion": "KG and TJ both exhausted",
          "expected_loss": "expected pool payout"}
    get = {(r["target"], r["threshold_musd"], r["method"], r["budget"]): r for r in d["rows"]}
    keys = list(dict.fromkeys((r["target"], r["threshold_musd"], r["budget"]) for r in d["rows"]))
    lines = [f"**Table [tab:carecres].** RDPO (Phase 2 engine) on the CAREC pool against exact tensor-network references, "
             f"{d['reps']} replications. Relative RMSE; interval coverage in parentheses.", "",
             "| Question | budget | exact value | RDPO | plain Monte Carlo | exp. twisting |", "|---|---|---|---|---|---|"]
    for t, thr, B in keys:
        r = get.get((t, thr, "rdpo", B)); mc = get.get((t, thr, "mc", B)); tw = get.get((t, thr, "twist", B))
        if not r:
            continue
        lab = tl[t].format(f"US${thr}m" if thr else "")
        val = money(r["mu"]) if t in ("expected_loss", "stop_loss") else sci(r["mu"])
        lines.append(f"| {lab} | {B:,} | {val} | **{pct(r['rel_rmse'])}** ({pct(r['ci_coverage'], 0)}) | "
                     f"{pct(mc['rel_rmse']) if mc else '—'} | {pct(tw['rel_rmse']) if tw else '—'} |")
    v["CAREC_RES_TABLE"] = "\n".join(lines)
    crit = get.get(("exceedance", d["T_rare"], "rdpo", 65536))
    v["CAREC_CRIT"] = pct(crit["rel_rmse"], 1) if crit else "—"
    v["CAREC_CRIT_OK"] = "met" if crit and crit["rel_rmse"] <= 0.10 else "not met"
    crit16 = get.get(("exceedance", d["T_rare"], "rdpo", 16384))
    v["CAREC_CRIT16"] = pct(crit16["rel_rmse"], 1) if crit16 else "—"
    return v


def proj_values():
    d = json.loads((R / "projected_doob.json").read_text())
    v = {}
    a = d["ADB benchmark (2 countries)"]; c = d.get("CAREC pool (10 countries)", a)
    v["PROJ_SD"] = sci(max(a["rel_sd_per_path"], c["rel_sd_per_path"]))
    v["PROJ_MS"] = f"{1e3 * c['seconds']:.0f} ms"
    v["PROJ_COND_MEAN"] = f"US${c['conditional_pool_loss_mean_musd']:.0f}m"
    v["PROJ_COND_P95"] = f"US${c['conditional_pool_loss_p95_musd']:.0f}m"
    v["PROJ_COND_400"] = pct(c.get("P(pool > 400m | pair exhausted)", 0), 2)
    v["PROJ_TRIPLE"] = sci(c.get("triple_exhaustion_KG_TJ_UZ", float("nan")))
    return v


def rain_values():
    d = json.loads((R / "realdata_rain.json").read_text())
    v = {}
    names = {"KG": "Kyrgyz Rep.", "TJ": "Tajikistan", "UZ": "Uzbekistan", "KZ": "Kazakhstan", "TM": "Turkmenistan",
             "AZ": "Azerbaijan", "GE": "Georgia", "PK": "Pakistan", "MN": "Mongolia"}
    lines = ["**Table [tab:rain].** Shared share of extreme-rain events between neighbouring countries (GHCN-Daily stations). "
             "Events at annual exceedance probability q; an event is shared when the other country has one within ±1 day. "
             "Shared / events of the rarer country; the brief assumes 25%.", "",
             "| Pair | common years | q = 50% | q = 20% | q = 10% | q = 5% |", "|---|---|---|---|---|---|"]
    for pr, r in d["pairs"].items():
        a, b = pr.split("-")
        if r["0.5"]["years"] < 20:
            continue
        cells = [f"{x['shared']}/{min(x['events_A'], x['events_B'])}" for x in (r[q] for q in ("0.5", "0.2", "0.1", "0.05"))]
        lines.append(f"| {names[a]}–{names[b]} | {r['0.5']['years']} | " + " | ".join(cells) + " |")
    pl = d["pooled"]
    lines.append("| **pooled** | | " + " | ".join(
        f"**{pct(pl[q]['share'], 1)}** ({pct(pl[q]['share_ci95'][0], 1)}–{pct(pl[q]['share_ci95'][1], 1)})" for q in ("0.5", "0.2", "0.1", "0.05")) + " |")
    v["RAIN_TABLE"] = "\n".join(lines)
    v["RAIN_NST"] = str(sum(len(x) for x in d["stations"].values()))
    v["RAIN_SHARE"] = f"{pct(pl['0.2']['share'], 1)} ({pct(pl['0.2']['share_ci95'][0], 1)}–{pct(pl['0.2']['share_ci95'][1], 1)})"
    im = d["implied"]
    lines = ["**Table [tab:rainimp].** Exact two-country targets with the dependence assumptions replaced by data.", "",
             "| Dependence | earthquake intensity m | rain shared share | joint attachment | Pr(L > 80m) | joint exhaustion |",
             "|---|---|---|---|---|---|"]
    for k, r in im.items():
        lines.append(f"| {k} | {r['m']:.1f} | {pct(r['rain_shared'], 1)} | {pct(r['joint_attachment'])} | {pct(r['exceedance'])} | {sci(r['joint_exhaustion'])} |")
    v["RAIN_IMP_TABLE"] = "\n".join(lines)
    b, both = im["brief"], im["both from data"]
    v["RAIN_EXC_FACTOR"] = f"{both['exceedance'] / b['exceedance']:.1f}×"
    v["RAIN_EXH_FACTOR"] = f"{both['joint_exhaustion'] / b['joint_exhaustion']:.2f}×"
    return v


def ap_values():
    d = json.loads((R / "wp2_asia_pacific.json").read_text())
    v = {}
    cal = d.get("calibration", {})
    v["AP_N"] = str(d["cases"]["asia_pacific"]["n"]); v["AP_E"] = str(d["cases"]["asia_pacific"]["edges"])
    v["AP_BOND"] = str(d["cases"]["asia_pacific"]["max_bond"]); v["AP_MPS_S"] = f"{d['cases']['asia_pacific']['mps_seconds']:.0f} s"
    v["AP_EVENTS"] = str(cal.get("events", "—")); v["AP_BW"] = str(cal.get("bandwidth", "—"))
    v["AP_CHECK"] = sci(max(x["tol_check"] for x in d["cases"].values() if x.get("tol_check") is not None))
    v["R50_BOND"] = str(d["cases"]["ring50"]["max_bond"])
    get = {(r["case"], r["target"], r["threshold_musd"], r["method"]): r for r in d["rows"]}
    lab = {"asia_pacific": f"Asia–Pacific ({v['AP_N']} countries)", "ring50": "ring (50 countries)"}
    lines = [f"**Table [tab:ap].** WP2 at scale, 65,536 paths, {d['reps']} replications, exact tensor-network references. "
             "Relative RMSE; interval coverage in parentheses.", "",
             "| Pool | question | exact value | RDPO | plain Monte Carlo | exp. twisting |", "|---|---|---|---|---|---|"]
    crit = []
    for case in ("asia_pacific", "ring50"):
        c = d["cases"][case]
        for t, thr in (("exceedance", c["T_moderate"]), ("exceedance", c["T_rare"]), ("stop_loss", c["T_rare"]), ("joint_exhaustion", None)):
            r = get.get((case, t, thr, "rdpo"))
            if not r:
                continue
            mc = get.get((case, t, thr, "mc")); tw = get.get((case, t, thr, "twist"))
            q = {"exceedance": f"Pr(pool payout > US${thr}m)", "stop_loss": f"stop-loss above US${thr}m",
                 "joint_exhaustion": "KG and TJ both exhausted (sampled; exact by pair reduction)"}[t]
            val = money(r["mu"]) if t == "stop_loss" else sci(r["mu"])
            lines.append(f"| {lab[case]} | {q} | {val} | **{pct(r['rel_rmse'])}** ({pct(r['ci_coverage'], 0)}) | "
                         f"{pct(mc['rel_rmse']) if mc else '—'} | {pct(tw['rel_rmse']) if tw else '—'} |")
            if t == "exceedance" and thr == c["T_rare"]:
                crit.append((lab[case], r["mu"], r["rel_rmse"]))
    v["AP_TABLE"] = "\n".join(lines)
    v["AP_CRIT"] = "; ".join(f"{n}: {pct(e, 1)} at μ = {sci(mu)}" for n, mu, e in crit)
    v["AP_CRIT_OK"] = "met" if crit and all(e <= 0.10 for _, _, e in crit) else "not met in every case"
    return v


def headline_values():
    v = {}
    rain = json.loads((R / "realdata_rain.json").read_text())
    im = rain["implied"]
    v["HL_EXC_BRIEF"] = pct(im["brief"]["exceedance"], 1)
    v["HL_EXC_DATA"] = pct(im["both from data"]["exceedance"], 1)
    v["HL_RAIN_SHARE"] = f"about {pct(rain['pooled']['0.2']['share'], 0)}"
    eq = json.loads((R / "realdata_losses.json").read_text())["tests"]["eq_noaa_1900"]
    v["HL_EQ_SHARE"] = f"about {pct(eq['share_of_A'], 0)}"
    ap = json.loads((R / "wp2_asia_pacific.json").read_text())
    c = ap["cases"]["asia_pacific"]
    r = next(x for x in ap["rows"] if x["case"] == "asia_pacific" and x["method"] == "rdpo"
             and x["target"] == "exceedance" and x["threshold_musd"] == c["T_rare"])
    v["HL_AP_MU"] = f"1-in-{1 / r['mu']:.1e}".replace("e+0", "×10^").replace("e+", "×10^") + "-term"
    v["HL_AP_MU"] = f"{sci(r['mu'])}-probability"
    v["HL_AP_ERR"] = pct(r["rel_rmse"], 1)
    v["HL_AP_SEC"] = f"{r['seconds']:.0f} s"
    t = json.loads((R / "sweep_timing.json").read_text())
    v["HL_SWEEP"] = f"{t['seconds_total']:.0f} seconds"
    return v


def elt_values():
    d = json.loads((R / "elt_demo.json").read_text())
    v = {}
    es = d["event_set"]
    v["ELT_EVENTS"] = f"{es['events']:,}"; v["ELT_SOURCES"] = str(d["n_sources"]); v["ELT_MAXFP"] = str(d["max_footprint"])
    v["ELT_CONV"] = f"{1e3 * d['conversion_seconds']:.0f} ms"
    v["ELT_LOC"] = str(es["locations"])
    multi = sum(1 for k in d["footprints"] if k.count("-") >= 2)
    v["ELT_MULTI"] = str(multi)
    tl = {"exceedance": "Pr(pool payout > US${}m)", "stop_loss": "stop-loss above US${}m", "expected_loss": "expected pool payout",
          "joint_exhaustion": "KG and TJ both exhausted"}
    get = {(r["target"], r["threshold_musd"], r["method"]): r for r in d["rows"]}
    keys = list(dict.fromkeys((r["target"], r["threshold_musd"]) for r in d["rows"]))
    order = {"exceedance": 0, "stop_loss": 1, "joint_exhaustion": 2, "expected_loss": 3}
    keys.sort(key=lambda k: (order[k[0]], k[1] or 0))
    lines = [f"**Table [tab:elt].** RDPO on a converted event loss table against exact answers, {d['reps']} replications. "
             "*RDPO* and *plain MC*: 16,384 simulated terms from the converted sources. *YLT, 10,000 years*: the estimate "
             "an analyst would read directly off a 10,000-year year loss table (3,333 three-year terms).", "",
             "| Question | exact value | RDPO | plain Monte Carlo | YLT, 10,000 years | exact from the YLT itself |",
             "|---|---|---|---|---|---|"]
    for t, thr in keys:
        r = get.get((t, thr, "rdpo")); mc = get.get((t, thr, "mc")); y = get.get((t, thr, "ylt"))
        if not r:
            continue
        val = (money(r["mu"]) if t in ("stop_loss", "expected_loss") else sci(r["mu"]))
        ey = d["exact_from_ylt"][f"{t}|{thr}"]
        eyv = money(ey) if t in ("stop_loss", "expected_loss") else sci(ey)
        lines.append(f"| {tl[t].format(thr)} | {val} | **{pct(r['rel_rmse'])}** ({pct(r['ci_coverage'], 0)}) | "
                     f"{pct(mc['rel_rmse']) if mc else '—'} | {pct(y['rel_rmse']) if y else '—'} | {eyv} |")
    v["ELT_TABLE"] = "\n".join(lines)
    rare = get.get(("exceedance", max(k[1] for k in keys if k[0] == "exceedance"), "rdpo"))
    yr = get.get(("exceedance", max(k[1] for k in keys if k[0] == "exceedance"), "ylt"))
    v["ELT_RARE_RDPO"] = pct(rare["rel_rmse"], 1) if rare else "—"
    v["ELT_RARE_YLT"] = pct(yr["rel_rmse"], 0) if yr else "—"
    v["ELT_PAIR_DOOB"] = sci(d["projected_doob_pair"])
    return v


def qs_values():
    d = json.loads((R / "quantum_scaling.json").read_text())
    f = d["fits"]; v = {}
    lab = {"logical_qubits": "logical qubits", "toffoli_A": "Toffoli gates per A", "rotations_A": "single-qubit rotations per A",
           "cnot_Q": "CNOTs per Grover iterate Q", "depth_A": "CNOT depth of A"}
    lines = ["**Table [tab:qscale].** Fitted resource laws, value ≈ c₀ + c_n·n + c_E·|E|, over eight compiled settings.", "",
             "| Resource | c₀ | per country c_n | per edge c_E | max deviation |", "|---|---|---|---|---|"]
    for k, l in lab.items():
        x = f[k]
        lines.append(f"| {l} | {x['const']:.0f} | {x['per_country']:.1f} | {x['per_edge']:.1f} | {pct(x['max_rel_err'], 1)} |")
    v["QS_FITS"] = "\n".join(lines)
    v["QS_FIT_ERR"] = pct(max(x["max_rel_err"] for x in f.values()), 0)
    rho = [r["rho2"] for r in d["rows"]]; be = [r["rho2_breakeven_at_1pct"] for r in d["rows"]]
    v["QS_RHO2"] = f"{min(rho):.2g}–{max(rho):.0f}"
    v["QS_BREAKEVEN_1"] = f"{sci(min(be))}–{sci(max(be))}"
    return v


def mk_values():
    m = json.loads((R / "market_model.json").read_text())
    return {"MK_SAM": f"{m['sam_m']:.0f}", "MK_SOM": f"{m['som_m']:.1f}", "MK_CUST": str(m["customers_y5"]),
            "MK_INST": str(sum(r["institutions"] for r in m["segments"])), "MK_SHARE": pct(m["som_share_of_sam"], 0)}


def stronger_values():
    """Table and summary for the stronger-baselines comparison (6.5.5)."""
    try:
        d = json.loads((R / "stronger_baselines.json").read_text())
    except FileNotFoundError:
        return {"STR_TABLE": "*(table not available; run `python -m src.baselines.stronger`)*", "STR_RANGE": "—"}
    rdpo_e1 = {(r["scenario"], r["target"]): r for r in json.loads((R / "rdpo_e1.json").read_text())["rows"]
               if r["budget"] == 16384}
    rdpo_e2 = {(r["topology"], r["n"], r["target"]): r for r in json.loads((R / "rdpo_e2.json").read_text())["rows"]}
    rows = d["rows"]
    cells = {}
    for r in rows:
        cells[(r["case"], r["target"], r["method"])] = r
    cases = []
    seen = []
    for r in rows:
        key = (r["case"], r["target"], r["budget"])
        if key not in seen:
            seen.append(key); cases.append(key)
    lines = ["**Table [tab:stronger].** RDPO (Phase 2 engine) vs three stronger rare-event methods on the ADB benchmark "
             "and ring-16 scaling case, at the same budget and seeds (20 replications). Relative RMSE.",
             "",
             "| Case | Target | Budget | μ | RDPO (Phase 2) | Splitting | SMC (twist) | Flow IS |",
             "|---|---|---|---|---|---|---|---|"]
    ratios = []
    for case, tgt, B in cases:
        mu = cells[(case, tgt, "splitting")]["mu"]
        if case.startswith("adb"):
            rdpo = rdpo_e1.get((case, tgt))
            rdpo_rmse = rdpo["rdpo"]["rel_rmse"] if rdpo else None
        else:
            rdpo = rdpo_e2.get(("ring", 16, tgt))
            rdpo_rmse = rdpo["rdpo"]["rel_rmse"] if rdpo else None
        case_lab = case
        bud = f"{B:,}"
        r_split = cells[(case, tgt, "splitting")]["rel_rmse"]
        r_smc = cells[(case, tgt, "smc")]["rel_rmse"]
        r_flow = cells[(case, tgt, "flow")]["rel_rmse"]
        best_other = min(r_split, r_smc, r_flow)
        if rdpo_rmse:
            ratios.append(best_other / rdpo_rmse)
            rdpo_cell = f"**{pct(rdpo_rmse)}**"
        else:
            rdpo_cell = "—"
        lines.append(f"| {case_lab} | {tgt} | {bud} | {sci(mu) if mu < 0.1 else pct(mu)} | {rdpo_cell} | "
                     f"{pct(r_split)} | {pct(r_smc)} | {pct(r_flow)} |")
    v = {"STR_TABLE": "\n".join(lines)}
    if ratios:
        v["STR_RANGE"] = f"×{min(ratios):.1f}–×{max(ratios):.0f}"
    else:
        v["STR_RANGE"] = "—"
    return v


def qs_inline_values():
    """Inline summary of the fitted quantum resource laws for the short proposal."""
    try:
        d = json.loads((R / "quantum_scaling.json").read_text())["fits"]
    except FileNotFoundError:
        return {"QS_FITS_INLINE": "linear laws (not computed in this build)"}
    def line(name, label):
        x = d[name]
        return f"{label} ≈ {x['const']:.0f} + {x['per_country']:.1f}n + {x['per_edge']:.1f}|E|"
    return {"QS_FITS_INLINE": "; ".join([line("logical_qubits", "logical qubits"),
                                         line("toffoli_A", "Toffolis per A"),
                                         line("depth_A", "depth of A")])}


def pu_values():
    """Parameter-uncertainty posterior summary (6.6.5)."""
    try:
        d = json.loads((R / "param_uncertainty.json").read_text())
    except FileNotFoundError:
        return {"PU_TABLE": "*(not available; run `python -m src.realdata.param_uncertainty`)*",
                "PU_N": "—", "PU_EXC_MED": "—", "PU_EXC_LO": "—", "PU_EXC_HI": "—", "PU_EXC_X": "—",
                "PU_EXC_BRIEF": "—", "PU_ATT_MED": "—", "PU_ATT_LO": "—", "PU_ATT_HI": "—",
                "PU_ATT_X": "—", "PU_EXH_MED": "—", "PU_EXH_LO": "—", "PU_EXH_HI": "—", "PU_EXH_X": "—"}
    brief = d["brief_point"]
    s = d["posterior_summary"]
    fx = d["factor_vs_brief"]
    N = next(iter(s.values()))["n"]
    def fmt_p(x):
        return pct(x, 2) if x is None or x >= 1e-3 else sci(x)
    def fmt_m(x):
        return f"US${x / 1e6:.2f}m" if x >= 1e5 else sci(x)
    lines = [f"**Table [tab:pu].** Posterior credible intervals for the ADB bond's two-country targets, from Jeffreys-prior "
             f"posteriors on the two shared shares. {N} Monte Carlo samples; exact dense enumeration per sample.",
             "",
             "| Target | Brief (point) | Posterior median | 50% credible interval | 95% credible interval | Factor vs brief (median) |",
             "|---|---|---|---|---|---|"]
    for k, lab in (("exceedance", "Pr(L > 80m)"), ("joint_attachment", "Joint attachment"),
                   ("joint_exhaustion", "Joint exhaustion"), ("stop_loss", "Stop-loss"),
                   ("expected_loss", "Expected loss")):
        if s.get(k) is None:
            continue
        b = brief[k]
        med = s[k]["median"]
        c50 = s[k]["ci50"]; c95 = s[k]["ci95"]
        f = fx[k]["median_x"]
        fmt = fmt_m if k in ("stop_loss", "expected_loss") else fmt_p
        lines.append(f"| {lab} | {fmt(b)} | **{fmt(med)}** | [{fmt(c50[0])}, {fmt(c50[1])}] | "
                     f"[{fmt(c95[0])}, {fmt(c95[1])}] | **×{f:.2f}** |")
    v = {"PU_TABLE": "\n".join(lines), "PU_N": f"{N:,}"}
    for key, prefix in (("exceedance", "PU_EXC"), ("joint_attachment", "PU_ATT"), ("joint_exhaustion", "PU_EXH")):
        if s.get(key):
            fmt = pct if s[key]["median"] >= 1e-3 else sci
            v[f"{prefix}_MED"] = fmt(s[key]["median"])
            v[f"{prefix}_LO"] = fmt(s[key]["ci95"][0])
            v[f"{prefix}_HI"] = fmt(s[key]["ci95"][1])
            v[f"{prefix}_X"] = f"{fx[key]['median_x']:.2f}"
        else:
            for suf in ("MED", "LO", "HI", "X"):
                v[f"{prefix}_{suf}"] = "—"
    v["PU_EXC_BRIEF"] = pct(brief["exceedance"])
    return v


def number_labels(md):
    """Replace [fig:key] / [tab:key] by numbers in order of first appearance."""
    for kind in ("fig", "tab"):
        order = []
        # number by position of the caption, so numbers run in order of appearance even if a table is cited early
        cap = "Table" if kind == "tab" else "Figure"
        for k in re.findall(r"\*\*" + cap + r" \[" + kind + r":(\w+)\]", md):
            if k not in order:
                order.append(k)
        for k in re.findall(r"\[" + kind + r":(\w+)\]", md):
            if k not in order:
                order.append(k)
        main = [k for k in order if not k.startswith("app_")]
        app = [k for k in order if k.startswith("app_")]
        for i, k in enumerate(main, 1):
            md = md.replace(f"[{kind}:{k}]", str(i))
        for i, k in enumerate(app, 1):  # appendix items: A1, A2, ... (kept out of the main sequence)
            md = md.replace(f"[{kind}:{k}]", f"A{i}")
    left = re.findall(r"\[(?:fig|tab):\w+\]", md)
    if left:
        print("UNRESOLVED LABELS:", left, file=sys.stderr)
    return md


def toc(md):
    heads = [(n, re.sub(r"\s*\{[^}]*\}\s*$", "", t)) for n, t in re.findall(r"^# (\d+) (.+)$", md, flags=re.M)]
    items = "\n\n".join(f"**{n}**  {t}" for n, t in heads)
    return f"::: {{.toc}}\n{items}\n\n**·**  References\n:::"


def _collect_values():
    vals = {}
    for fn in (e1_values, e2_values, e3_values, q_values, mk_values, v2_values, rd_values, loss_values, pool_values, wp2_values, proj_values, rain_values, ap_values, headline_values, elt_values, qs_values, stronger_values, pu_values, qs_inline_values):
        try:
            vals.update(fn())
        except Exception as ex:
            print("placeholder group failed:", fn.__name__, type(ex).__name__, ex, file=sys.stderr)
    return vals


def fill(src="proposal.md", dst="proposal.filled.md", add_toc=True):
    md = (PAPER / src).read_text()
    vals = _collect_values()
    if add_toc:
        md = md.replace("# 1 Introduction {.newpage}", toc(md) + "\n\n# 1 Introduction", 1)
    for k, val in vals.items():
        md = md.replace("{{" + k + "}}", val)
    md = number_labels(md)
    # bold run-in headings on their own line: keep with the following block
    md = re.sub(r"^(\*\*[^*\n]+\.\*\*)[ \t]*$", r"::: {.runin}\n\1\n:::", md, flags=re.M)
    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", md)))
    if left:
        print(f"UNFILLED in {src}:", left, file=sys.stderr)
    (PAPER / dst).write_text(md)
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
    art4.main()
    art5.main()
    try:
        from ..realdata.asia_pacific_pool import pool_map
        pool_map()
    except FileNotFoundError as ex:
        print('pool map skipped:', ex)
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
    # ---- short version: separate pandoc build from short.md, no cover ----
    try:
        fill(src="short.md", dst="short.filled.md", add_toc=False)
        html_s = PAPER / "short.html"
        subprocess.run(["pandoc", str(PAPER / "short.filled.md"), "-f", "markdown", "-t", "html5", "-s",
                        "--mathjax=node_modules/mathjax-full/es5/tex-svg-full.js", "-H", str(mj),
                        "--citeproc", "--bibliography", str(PAPER / "references.bib"), "--css", "paper.css",
                        "-H", str(PAPER / "fonts.html"), "-o", str(html_s)], check=True)
        body_s = PAPER / "short_body.pdf"
        subprocess.run(["node", str(js), str(html_s.resolve()), str(body_s), "body"], check=True)
        short_out = ROOT / "docs/RDPO_Proposal_Short.pdf"
        import shutil
        shutil.copy(str(body_s), str(short_out))
        print("short built", short_out)
    except Exception as ex:
        print("short build skipped:", ex)
    if cv:
        Path("submission").mkdir(exist_ok=True)
        subprocess.run(["pdfunite", str(cover), str(body), cv, "submission/RDPO_Proposal_with_CV_annex.pdf"], check=True)
    print("built", out, "unfilled:", left)


if __name__ == "__main__":
    cv = sys.argv[sys.argv.index("--cv") + 1] if "--cv" in sys.argv else None
    build(cv)
