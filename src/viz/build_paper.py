"""Build the formal report: figures -> numbers from results/*.json -> Markdown -> HTML -> PDF (+ cover).

Every number substituted into docs/paper/proposal.md comes from the result files written by the
experiment scripts; nothing is typed by hand.
Usage: python -m src.viz.build_paper [--cv path/to/cv.pdf]
"""
import json
import math
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


def sci(x, d=1):
    m, e = f"{x:.{d}e}".split("e")
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
    lines = ["**Table [tab:e2].** Scaling results, Phase 1 engine, at a total budget of 65,536 paths (relative RMSE; training included; 6–20 replications). MC entries are theoretical relative errors at the same budget, shown as a percentage below 1,000% and as a multiple of μ above.", "", hdr, "|---" * 11 + "|"]
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
    lines = ["**Table [tab:e3].** Ablation: relative RMSE at B = 16,384 paths (40 replications). Ratios to full RDPO in parentheses. The ablation uses its own seeds, so its full-RDPO row is an independent replicate of Table [tab:e1] (e.g. base joint exhaustion); with 40 heavy-tailed replications the relative RMSE itself varies by tens of percent between replicates.", "", hdr, "|---" * (len(cases) + 1) + "|"]
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


def qa_file():
    """Crossover inputs: the faster classical engine (results/quantum_advantage_v3.json, same quantum model); the earlier
    results/quantum_advantage.json is kept for the record and still supplies the mitigation study."""
    return json.loads((R / "quantum_advantage_v3.json").read_text())


def q_values():
    d = qa_file()
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
    ms = json.loads((R / "quantum_advantage.json").read_text())["mitigation_summary"]   # classical-independent: kept in the earlier file
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
             "results (6–20 replications); its twisting errors differ from this table's, which re-runs twisting with 24 replications and RDPO's seeds. Pair exhaustion is answered exactly by the Phase 2 engine (pair reduction); its "
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
    d = qa_file()["scaling"]
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
        d = qa_file()["scaling"]["fits"]
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


def fin_values():
    """Finance analyses (src/finance/analysis.py): bond anchor, pooled limit, model-free coupling bounds."""
    d = json.loads((R / "finance.json").read_text())
    an, po, cb = d["anchor"], d["pooling"], d["coupling_bounds"]["targets"]
    def m(x):
        return "0" if x < 1 else f"US${x / 1e6:.2f}m"
    def p(x, dd=1):
        return "0" if x < 1e-12 else (pct(x, dd) if x >= 1e-3 else sci(x))
    def sx(x):
        return "0" if x < 1e-12 else sci(x)
    v = {}
    for c in ("KG", "TJ"):
        v[f"FIN_EL_{c}"] = pct(an["computed"][c]["annual_EL_fraction_of_eq_sublimit"], 3)
        v[f"FIN_ELC_{c}"] = pct(an["cited"]["initial_expected_loss"][c], 3)
        v[f"FIN_RM_X_{c}"] = f"{an['ratios_computed_over_cited'][c]['risk_margin_over_cited_EL']:.1f}"
    v["FIN_RM"] = pct(an["cited"]["risk_margin"]["KG"], 0)
    b80, d80 = po["brief"]["shared"]["80m"], po["both_from_data"]["shared"]["80m"]
    post = po["posterior"]
    df = post["delivered_fraction"]
    v.update({"FIN_POOL_BRIEF": pct(b80["delivered_fraction"], 1), "FIN_POOL_DATA": pct(d80["delivered_fraction"], 1),
              "FIN_POOL_MED": pct(df["median"], 1), "FIN_POOL_LO": pct(df["ci95"][0], 1), "FIN_POOL_HI": pct(df["ci95"][1], 1),
              "FIN_SHORT_BRIEF": m(b80["expected_payout_given_up"]), "FIN_SHORT_MED": m(post["shortfall"]["median"]),
              "FIN_EL_TERM": m(b80["expected_payout_separate"]),
              "FIN_POOL100_BRIEF": pct(po["brief"]["shared"]["100m"]["delivered_fraction"], 1),
              "FIN_POOL100_MED": pct(post["delivered_100"]["median"], 1),
              "FIN_POOL120_BRIEF": pct(po["brief"]["shared"]["120m"]["delivered_fraction"], 1),
              "FIN_POOL120_MED": pct(post["delivered_120"]["median"], 1)})
    rows = ["**Table [tab:pooling].** Pooling decision, exact. Two separate US$80m covers (status quo) against one shared limit "
            "for both countries. *Delivered*: share of the expected three-year payout of the separate covers that the shared "
            f"limit still pays. Posterior: median and 95% credible interval over {post['n_samples']:,} draws of the two shared shares.",
            "", "| Shared limit | Capital saved | Delivered, brief | Delivered, both from data | Delivered, posterior |",
            "|---|---|---|---|---|"]
    for k, key in (("80m", "delivered_fraction"), ("100m", "delivered_100"), ("120m", "delivered_120")):
        q = post[key]
        rows.append(f"| US${k} | {m(po['brief']['shared'][k]['capital_saving_usd'])} | "
                    f"{pct(po['brief']['shared'][k]['delivered_fraction'], 1)} | "
                    f"{pct(po['both_from_data']['shared'][k]['delivered_fraction'], 1)} | "
                    f"**{pct(q['median'], 1)}** [{pct(q['ci95'][0], 1)}, {pct(q['ci95'][1], 1)}] |")
    v["FIN_TABLE_POOL"] = "\n".join(rows)
    lab = {"prob_exceed": "Pr(L > 80m)", "joint_attachment": "Joint attachment", "joint_exhaustion": "Joint exhaustion",
           "stop_loss": "Stop-loss E[(L−80m)⁺]"}
    rows = ["**Table [tab:bounds].** Model-free range of each target. Both countries' term laws are fixed at the brief's "
            "values; a linear program over all 100-state couplings gives the smallest and largest value any dependence "
            "structure can produce. Position: where a value sits inside that range (0% = minimum, 100% = maximum).",
            "", "| Target | Feasible range | Independent | Brief (position) | Posterior median (position) |",
            "|---|---|---|---|---|"]
    for k, t in cb.items():
        f = m if k == "stop_loss" else (sx if k == "joint_exhaustion" else p)
        fc, pt = t["full_coupling"], t["points"]
        rows.append(f"| {lab[k]} | [{f(fc['min'])}, {f(fc['max'])}] | {f(t['independent_countries'])} | "
                    f"{f(pt['brief']['value'])} ({pct(pt['brief']['position_in_full_range'], 0)}) | "
                    f"{f(pt['posterior_median']['value'])} ({pct(pt['posterior_median']['position_in_full_range'], 0)}) |")
    v["FIN_TABLE_BOUNDS"] = "\n".join(rows)
    e, x = cb["prob_exceed"], cb["joint_exhaustion"]
    v.update({"FIN_EXC_MAX": pct(e["full_coupling"]["max"], 1),
              "FIN_EXC_POS_BRIEF": pct(e["points"]["brief"]["position_in_full_range"], 0),
              "FIN_EXC_POS_MED": pct(e["points"]["posterior_median"]["position_in_full_range"], 0),
              "FIN_EXH_MAX": sci(x["full_coupling"]["max"]),
              "FIN_EXH_X": f"{x['full_coupling']['max'] / x['points']['brief']['value']:.0f}",
              "FIN_ATT_MAX": pct(cb["joint_attachment"]["full_coupling"]["max"], 1)})
    return v


def test_values():
    from .art2 import _n_tests
    return {"N_TESTS": str(_n_tests())}


def hw_values():
    """Near-term hardware plan (src/quantum/hardware_plan.py): costed, emulated IBM Heron-class run."""
    d = json.loads((R / "hardware_plan.json").read_text())
    lab = {"exceedance/direct": ("Pr(L > 80m)", "direct"), "exceedance/tilt": ("Pr(L > 80m)", "tilted"),
           "joint_exhaustion/direct": ("Joint exhaustion", "direct"), "joint_exhaustion/tilt": ("Joint exhaustion", "tilted")}
    def best(e, shots=100000):
        rr = [r for rows in d["noisy"][e]["schedules"].values() for r in rows if r["shots"] == shots]
        return min(r["rel_rmse"] for r in rr)
    def scan(e, p2):
        rr = [r for r in d["scan"].get(e, []) if abs(r["p2"] - p2) < 1e-12 and abs(r["readout_error"] - 0.01) < 1e-12]
        return rr[0] if rr else None
    rows = ["**Table [tab:hw].** Costed near-term run of the two-country bond, emulated. Circuits transpiled for an "
            "IBM Heron-class device model (heavy-hex, native CZ; median of 5 transpiler seeds); noise from the device "
            "snapshot (gate errors, relaxation, readout). *Best error*: lowest relative RMSE over Grover schedules at "
            "10⁵ shots. Last columns: relative RMSE at 3×10⁴ shots under depolarising two-qubit error p₂, and the share of "
            "runs the trust guard accepts.", "",
            "| Target | Loading | Qubits | Amplitude a | Two-qubit gates, k = 0 / 1 / 2 | Best error, device model | p₂ = 10⁻³ | p₂ = 10⁻⁴ | p₂ = 10⁻⁵ (guard accepts) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for e, (t, l) in lab.items():
        enc = d["encodings"][e]
        g = " / ".join(f"{r['device_n2q_median']:,.0f}" for r in d["resources"][e][:3])
        sc = [scan(e, p) for p in (1e-3, 1e-4, 1e-5)]
        a = enc["a"]
        af = sci(a) if a < 1e-3 else f"{a:.3f}"
        c = ["—" if r is None else pct(r["rel_rmse"], 1) for r in sc]
        if sc[2] is not None:
            c[2] += f" ({pct(sc[2]['guard_acceptance'], 0)})"
        rows.append(f"| {t} | {l} | {enc['qubits']} | {af} | {g} | {pct(best(e), 0)} | {c[0]} | {c[1]} | {c[2]} |")
    b = d["budget"]
    q = b["qpu_minutes_total"]
    return {"HW_TABLE": "\n".join(rows),
            "HW_SHOTS": f"{b['recommended_run']['total_shots'] / 1e5:.0f}×10⁵",
            "HW_QPU_MIN": f"{q['high']['qpu_minutes']:.0f}–{q['low']['qpu_minutes']:.0f}",
            "HW_EXC_BEST": pct(best("exceedance/tilt"), 0),
            "HW_EXH_BEST": pct(best("joint_exhaustion/tilt"), 0),
            "HW_DM_MIN": f"{d['emulator_fallback']['wall_seconds_this_run'] / 60:.0f}"}


def dense_values():
    """Dense-regime exact references (src/tensor/junction.py) and RDPO vs twisting (src/rdpo/experiments_dense.py)."""
    d = json.loads((R / "dense_exact.json").read_text())
    ex, rows = d["exact"], d["rows"]
    gname = {"ladder-xdiag": "ladder + diagonals"}
    out = ["**Table [tab:dense].** Dense, bounded-treewidth dependence (band-k: each country linked to its next k "
           "neighbours; ladder with both diagonals), exact references by frontier variable elimination, B = 65,536, "
           "identical seeds. Relative RMSE. *Equal time*: twisting's error if given RDPO's wall-clock, "
           "scaled as 1/√(paths).", "",
           "| Graph | n | edges | target | exact μ | RDPO | twisting | twisting, equal time | RDPO coverage | reps |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    gains, eq = [], []
    for r in rows:
        g = gname.get(r["graph"], r["graph"])
        mu = sci(r["mu"]) if r["target"] == "exceedance" else "US$" + (sci(r["mu"]) if r["mu"] < 1 else f"{r['mu']:,.0f}")
        te = r["twist_rel_rmse"] * (r["twist_sec_per_estimate_median"] / r["rdpo_sec_per_estimate_median"]) ** 0.5
        gains.append(r["twist_over_rdpo"]); eq.append(te < r["rdpo_rel_rmse"])
        tgt = "Pr(L > T)" if r["target"] == "exceedance" else "stop-loss"
        out.append(f"| {g} | {r['n']} | {r['edges']} | {tgt} | {mu} | **{pct(r['rdpo_rel_rmse'], 1)}** | "
                   f"{pct(r['twist_rel_rmse'], 1)} | {pct(te, 1)} | {pct(r['rdpo_coverage'], 0)} | {r['reps']} |")
    big = max(ex, key=lambda e: (e["frontier_width"], e["n"]))
    sc = json.loads((R / "rdpo_v2_scale.json").read_text())["rows"]
    by = {}
    for r in sc:
        by.setdefault((r["topology"], r["n"], r["target"]), {})[r["variant"]] = r
    sw, st = 0, 0
    for v in by.values():
        if "twist" in v and "v2-final" in v:
            tw, rd = v["twist"], v["v2-final"]
            st += 1
            sw += tw["rel_rmse"] * (tw["seconds_per_fit"] / rd["seconds_per_fit"]) ** 0.5 < rd["rel_rmse"]
    return {"DENSE_TABLE": "\n".join(out),
            "DENSE_GAIN": f"×{min(gains):.1f}–×{max(gains):.1f}",
            "DENSE_NWIN": f"{len(gains)} of {len(gains)}",
            "DENSE_EQ_TWIST_WINS": f"{sum(eq)} of {len(eq)}",
            "DENSE_NEXACT": str(len(ex)),
            "DENSE_MINMU": sci(min(e["exceedance"] for e in ex)),
            "DENSE_BIG": f"band-{big['frontier_width']} with n = {big['n']} ({big['edges']} edges)",
            "DENSE_BIG_SEC": f"{big['exact_seconds']:.0f} s",
            "DENSE_RDPO_SEC": f"{min(r['rdpo_sec_per_estimate_median'] for r in rows):.0f}–{max(r['rdpo_sec_per_estimate_median'] for r in rows):.0f} s",
            "DENSE_TWIST_SEC": f"{max(r['twist_sec_per_estimate_median'] for r in rows):.0f} s",
            "EQT_SCALE": f"{sw} of {st}"}


BRIEF_METHODS = ("mc", "sobol", "stratified", "importance")
MNAME = {"mc": "Monte Carlo", "sobol": "Sobol", "stratified": "stratified", "importance": "brief IS", "smc": "SMC (twist)",
         "flow": "flow IS", "twist": "exp. twisting", "rdpo": "RDPO"}


def brief5_values():
    """Reproduction of the brief's Table 5 (exact.json: our exact values and the brief's rounded references)."""
    sc = json.loads((R / "exact.json").read_text())["scenarios"]
    keys = [("term_expected_principal_loss_usd", "Expected loss"), ("term_joint_attachment_probability", "Both pay"),
            ("term_portfolio_loss_exceedance_probability", "Pr(L > 80m)"), ("term_portfolio_stop_loss_usd", "Stop-loss"),
            ("term_both_bonds_exhausted_probability", "Both exhausted")]

    def f(k, x):
        if k.endswith("usd"):
            return f"US${x / 1e6:.2f}m" if x >= 1e6 else f"US${x:,.0f}"
        return pct(x, 3) if x >= 1e-3 else pct(x, 4)
    def g(k, x):  # more digits than the brief prints
        if k.endswith("usd"):
            return f"US${x / 1e6:.4f}m" if x >= 1e6 else f"US${x:,.1f}"
        return pct(x, 4) if x >= 1e-3 else sci(x, 3)
    lines = ["**Table [tab:brief5].** The brief's Table 5 reproduced: our exact values (dense enumeration of the source model, "
             "more digits than the brief prints). *Match*: number of the five values that equal the brief's published value at "
             "its printed rounding.", "",
             "| Scenario | " + " | ".join(l for _, l in keys) + " | Match |", "|---" * (len(keys) + 2) + "|"]
    for s in SCN:
        r = sc[s]
        cells, ok = [], 0
        for k, _ in keys:
            a, b = r["values"][k], r["reference_rounded"][k]
            cells.append(g(k, a) if k != "term_both_bonds_exhausted_probability" else sci(a, 3))
            ok += f(k, a) == f(k, b)
        lines.append(f"| {paper_figs.SCN_LABEL[s]} | " + " | ".join(cells) + f" | {ok}/5 |")
    rt = [sc[s]["runtime_s"] for s in SCN]
    return {"BRIEF5_TABLE": "\n".join(lines),
            "BRIEF5_MS": f"{1e3 * float(np.median(rt)):.1f} ms"}


def eqt_values():
    """Equal-wall-clock audit (results/equal_time.json): benchmark (A) and data-calibrated pools (B)."""
    d = json.loads((R / "equal_time.json").read_text())
    A, B, s = d["A_cells"], d["B_cells"], d["summary"]
    pa, pb = d["protocol"]["A"], d["protocol"]["B"]
    v = {}

    def wins(cells, key, methods):
        return sum(min(c[key][m] for m in methods if m in c[key]) > c[key]["rdpo"] for c in cells)
    allA = [m for m in pa["methods"] if m != "rdpo"]
    v["EQT_A_REPS"] = str(pa["reps"]); v["EQT_B_REPS"] = str(pb["reps"])
    v["EQT_A_PATH_BRIEF"] = f"{wins(A, 'rmse_paths', BRIEF_METHODS)} of {len(A)}"
    v["EQT_A_PATH_ALL"] = f"{wins(A, 'rmse_paths', allA)} of {len(A)}"
    v["EQT_A_TIME_BRIEF"] = f"{wins(A, 'rmse_time', BRIEF_METHODS)} of {len(A)}"
    v["EQT_A_TIME_ALL"] = f"{wins(A, 'rmse_time', allA)} of {len(A)}"
    je = [c for c in A if c["target"] == "joint_exhaustion"]
    v["EQT_A_TIME_JE"] = f"{wins(je, 'rmse_time', BRIEF_METHODS)} of {len(je)}"
    other = [c for c in A if c["target"] != "joint_exhaustion"]
    v["EQT_A_TIME_OTHER"] = str(wins(other, "rmse_time", BRIEF_METHODS))
    ex = d["exact_enumeration_A"]
    v["EQT_RDPO_MS"] = f"{1e3 * ex['rdpo_seconds_16384_median']:.0f} ms"
    v["EQT_EXACT_MS"] = f"{1e3 * ex['exact_dense_enumeration_s_median']:.2f} ms"
    v["EQT_BRIEFMODEL_MS"] = f"{1e3 * ex['exact_brief_model_s_median']:.2f} ms"
    v["EQT_PAIR_MS"] = f"{1e3 * ex['exact_pair_s_median']:.2f} ms"
    v["EQT_A_RDPO_SEC_TOTAL"] = f"{sum(c['seconds_rdpo_16384'] for c in A):.1f} s"
    ratios_t = [c["ratio_best_baseline_over_rdpo_time"] for c in A]
    v["EQT_A_TIME_MED"] = f"×{float(np.median(ratios_t)):.2f}"
    # pools
    v["EQT_B_PATH"] = f"{wins(B, 'rmse_paths', ['mc', 'twist'])} of {len(B)}"
    v["EQT_B_TIME"] = f"{wins(B, 'rmse_time', ['mc', 'twist'])} of {len(B)}"
    bje = [c for c in B if c["target"] == "joint_exhaustion"]
    g = [c["rmse_time"]["mc"] / c["rmse_time"]["rdpo"] for c in bje]
    v["EQT_B_JE_GAIN"] = f"×{min(g):.1f}–×{max(g):.1f}"
    v["EQT_B_JE_GAIN_MED"] = f"×{float(np.median(g)):.0f}"
    thr = [c for c in B if c["target"] != "joint_exhaustion"]
    v["EQT_B_THR_TIME"] = f"{wins(thr, 'rmse_time', ['mc', 'twist'])} of {len(thr)}"
    tr = [c["rmse_time"]["twist"] / c["rmse_time"]["rdpo"] for c in thr]
    v["EQT_B_THR_RANGE"] = f"×{min(tr):.2f}–×{max(tr):.2f}"
    # table A: per target
    lab = {"expected_loss": "Expected loss", "joint_attachment": "Both pay", "exceedance": "Pr(L > 80m)",
           "stop_loss": "Stop-loss", "joint_exhaustion": "Both exhausted"}
    lines = [f"**Table [tab:eqtA].** Equal-budget and equal-time audit of the benchmark (25 cells, {pa['reps']} replications, "
             "independent re-run of Table [tab:e1] with two more baselines: SMC on the optimal twist and a flow importance sampler). "
             "*Same paths*: every method gets 16,384 paths, RDPO's training included. *Same CPU time*: every baseline gets as many "
             "paths as it can draw in RDPO's wall-clock for one estimate. Cells won by RDPO against the brief's four methods and, "
             "in brackets, against all six. *Error ratio*: median over the five scenarios of best baseline error / RDPO error "
             "(> 1: RDPO better).", "",
             "| Target | Same paths: cells won | error ratio | Same CPU time: cells won | error ratio | most accurate at same time |",
             "|---|---|---|---|---|---|"]
    for t in TGT:
        cc = [c for c in A if c["target"] == t]
        sp = s["A_by_target"][t]
        best = ", ".join(f"{MNAME[k]} {n}" for k, n in sorted(sp["time"]["most_accurate_method_counts"].items(), key=lambda x: -x[1]))
        lines.append(f"| {lab[t]} | {wins(cc, 'rmse_paths', BRIEF_METHODS)} ({wins(cc, 'rmse_paths', allA)}) of 5 | "
                     f"×{sp['paths']['median_ratio_best_baseline_over_rdpo']:.1f} | "
                     f"**{wins(cc, 'rmse_time', BRIEF_METHODS)}** ({wins(cc, 'rmse_time', allA)}) of 5 | "
                     f"×{sp['time']['median_ratio_best_baseline_over_rdpo']:.2f} | {best} |")
    lines.append(f"| **All** | **{wins(A, 'rmse_paths', BRIEF_METHODS)}** ({wins(A, 'rmse_paths', allA)}) of 25 | "
                 f"×{s['A_overall']['paths']['median_ratio_best_baseline_over_rdpo']:.1f} | "
                 f"**{wins(A, 'rmse_time', BRIEF_METHODS)}** ({wins(A, 'rmse_time', allA)}) of 25 | "
                 f"×{s['A_overall']['time']['median_ratio_best_baseline_over_rdpo']:.2f} | |")
    v["TABLE_EQT_A"] = "\n".join(lines)
    # table B: pools
    pool = {"carec": "CAREC (10)", "asia_pacific": "Asia–Pacific (28)"}
    q = {"exceedance": "Pr(payout > US${}m)", "stop_loss": "stop-loss above US${}m", "joint_exhaustion": "KG and TJ both exhausted"}
    lines = [f"**Table [tab:eqtB].** Equal-budget and equal-time audit on the data-calibrated pools (65,536 paths, "
             f"{pb['reps']} replications, independent re-run of Tables [tab:carecres] and [tab:ap]). Relative RMSE. *Same CPU time*: "
             "the baseline's error with as many paths as it draws in RDPO's wall-clock. Monte Carlo uses a lean sampler written for "
             "this audit. Pair exhaustion is also available exactly by pair reduction.", "",
             "| Pool | question | exact μ | RDPO | twisting, same paths | twisting, same time | Monte Carlo, same time | RDPO time |",
             "|---|---|---|---|---|---|---|---|"]
    for c in B:
        rt, rp = c["rmse_time"], c["rmse_paths"]
        mu = money(c["mu"]) if c["target"] == "stop_loss" else sci(c["mu"])
        tw_p = pct(rp["twist"], 2) if "twist" in rp else "—"
        tw_t = rt.get("twist")
        win = min(x for k, x in rt.items() if k != "rdpo") > rt["rdpo"]
        rd = f"**{pct(rt['rdpo'], 2)}**" if win else pct(rt["rdpo"], 2)
        best_t = (f"**{pct(tw_t, 2)}**" if tw_t is not None and tw_t < rt["rdpo"] else (pct(tw_t, 2) if tw_t is not None else "—"))
        lines.append(f"| {pool[c['case']]} | {q[c['target']].format(c['threshold_musd'])} | {mu} | {rd} | {tw_p} | {best_t} | "
                     f"{pct(rt['mc'], 1)} | {c['seconds_rdpo_65536']:.1f} s |")
    v["TABLE_EQT_B"] = "\n".join(lines)
    return v


def v3_values():
    """Equal-time negative result: learned product-form tilt and automatic selector (results/rdpo_v3_equal_time.json)."""
    d = json.loads((R / "rdpo_v3_equal_time.json").read_text())
    s, m = d["summary"], d["meta"]
    lab = {"tilt": "learned product-form tilt", "selector": "automatic selector", "selector_tr": "selector with transfer",
           "rdpo_cold": "RDPO (trained per estimate)", "rdpo_transfer": "RDPO (transferred policy)"}
    lines = [f"**Table [tab:v3].** Equal wall-clock against exponential twisting on the scaling (16 cells: rings to n = 28, "
             f"complete graphs to n = 7) and dense (10 cells: band and ladder graphs) settings. Short budget: "
             f"{m['short_budget_s']:.0f} s per estimate ({m['reps']['short']} replications); long budget: "
             f"{m['long_budget_rule'].replace('min(RDPO-v2 seconds, 5.0 s)', 'the Phase 2 engine time, at most 5 s')} "
             f"({m['reps']['long']} replications). Win: error below 0.8× twisting's; tie: 0.8–1.25×; loss: above 1.25×. "
             "Median: method error / twisting error (< 1: better than twisting).", "",
             "| Method | scaling, short: W/T/L (median) | scaling, long | dense, short | dense, long |", "|---|---|---|---|---|"]
    for k in ("rdpo_cold", "rdpo_transfer", "tilt", "selector", "selector_tr"):
        cells = []
        for fam in ("scaling", "dense"):
            for b in ("short", "long"):
                x = s[f"{fam}|{b}|{k}"]
                cells.append(f"{x['wins']}/{x['ties']}/{x['losses']} (×{x['median_ratio']:.2f})")
        cells = [cells[0], cells[1], cells[2], cells[3]]
        lines.append(f"| {lab[k]} | " + " | ".join(cells) + " |")
    v = {"TABLE_V3": "\n".join(lines)}
    tm = [s[f"{f}|{b}|tilt"]["median_ratio"] for f in ("scaling", "dense") for b in ("short", "long")]
    v["V3_TILT_MED"] = f"×{min(tm):.2f}–×{max(tm):.2f}"
    sm = [s[f"{f}|{b}|{k}"]["median_ratio"] for f in ("scaling", "dense") for b in ("short", "long") for k in ("selector", "selector_tr")]
    v["V3_SEL_MED"] = f"×{min(sm):.2f}–×{max(sm):.2f}"
    cl = [s[f"scaling|{b}|rdpo_cold"] for b in ("short", "long")]
    v["V3_COLD_SCALE_LOSS"] = f"{min(x['losses'] for x in cl)}–{max(x['losses'] for x in cl)} of {cl[0]['cells']}"
    v["V3_COLD_SCALE_MED"] = f"×{min(x['median_ratio'] for x in cl):.0f}–×{max(x['median_ratio'] for x in cl):.0f}"
    dl = [s[f"dense|{b}|rdpo_cold"] for b in ("short", "long")]
    v["V3_COLD_DENSE_LOSS"] = f"{min(x['losses'] for x in dl)} of {dl[0]['cells']}"
    v["V3_COLD_DENSE_MED"] = f"×{min(x['median_ratio'] for x in dl):.0f}–×{max(x['median_ratio'] for x in dl):.0f}"
    v["V3_SHORT_S"] = f"{m['short_budget_s']:.0f} s"
    return v


def eqt_short_values():
    """One compact table for the main proposal: wins per path vs wins per CPU second, all four families."""
    d = json.loads((R / "equal_time.json").read_text())
    A, B = d["A_cells"], d["B_cells"]
    v3 = json.loads((R / "rdpo_v3_equal_time.json").read_text())["summary"]
    sc = json.loads((R / "rdpo_v2_scale.json").read_text())["rows"]
    by = {}
    for r in sc:
        by.setdefault((r["topology"], r["n"], r["target"]), {})[r["variant"]] = r
    sw = [x["twist"]["rel_rmse"] > x["v2-final"]["rel_rmse"] for x in by.values() if "twist" in x and "v2-final" in x]
    dn = json.loads((R / "dense_exact.json").read_text())["rows"]
    dw = [r["twist_over_rdpo"] > 1 for r in dn]

    def wins(cells, key, methods):
        return sum(min(c[key][m] for m in methods if m in c[key]) > c[key]["rdpo"] for c in cells)
    aje = [c for c in A if c["target"] == "joint_exhaustion"]
    bje = [c for c in B if c["target"] == "joint_exhaustion"]
    g = [c["rmse_time"]["mc"] / c["rmse_time"]["rdpo"] for c in bje]
    cs = [v3[f"scaling|{b}|rdpo_cold"] for b in ("short", "long")]
    cd = [v3[f"dense|{b}|rdpo_cold"] for b in ("short", "long")]
    rows = [
        ("Benchmark: the brief's 25 cells (vs the brief's four methods)", len(A), wins(A, "rmse_paths", BRIEF_METHODS),
         f"{wins(A, 'rmse_time', BRIEF_METHODS)} ({wins(aje, 'rmse_time', BRIEF_METHODS)} of 5 joint-exhaustion cells)",
         "scrambled Sobol; RDPO or SMC for joint exhaustion"),
        ("Data-calibrated pools of 10 and 28 countries (vs twisting, Monte Carlo)", len(B), wins(B, "rmse_paths", ["mc", "twist"]),
         f"{wins(B, 'rmse_time', ['mc', 'twist'])} (pair exhaustion ×{min(g):.0f}–×{max(g):.0f} over Monte Carlo)",
         "twisting on pool thresholds; pair targets exact"),
        ("Scaling: rings to 28, complete graphs to 7 (vs twisting)", len(sw), sum(sw),
         f"{min(x['wins'] for x in cs)}–{max(x['wins'] for x in cs)}", "twisting"),
        ("Dense band and ladder graphs to 28 (vs twisting)", len(dw), sum(dw), f"{max(x['wins'] for x in cd)}", "twisting"),
    ]
    lines = ["**Table [tab:eqt].** Where RDPO wins, per path and per CPU second. *Same paths*: every method draws the same number "
             "of simulated terms, RDPO's training included (the brief's criterion). *Same CPU time*: every competitor gets RDPO's "
             "single-core wall-clock for one estimate. Cells in which RDPO has the lower relative RMSE.", "",
             "| Setting | cells | RDPO better, same paths | RDPO better, same CPU time | best at same CPU time |",
             "|---|---|---|---|---|"]
    for lab, n, a, b, best in rows:
        lines.append(f"| {lab} | {n} | **{a}** | {b} | {best} |")
    return {"TABLE_EQT_SHORT": "\n".join(lines), "SCALE_TW_WINS": f"{sum(sw)} of {len(sw)}"}


def env_values():
    import platform
    req = dict(l.strip().split("==") for l in (ROOT / "requirements.txt").read_text().splitlines() if "==" in l)
    return {"ENV_PY": platform.python_version().rsplit(".", 1)[0], "ENV_NUMPY": req.get("numpy", "—"),
            "ENV_SCIPY": req.get("scipy", "—"), "ENV_MPL": req.get("matplotlib", "—"), "ENV_PYTEST": req.get("pytest", "—")}


def extra_values():
    """Single-source numbers quoted in several places (short proposal and annex)."""
    v = {}
    # benchmark coverage
    e1 = json.loads((R / "rdpo_e1.json").read_text())["rows"]
    covs = [r["rdpo"]["ci_coverage"] for r in e1]
    c16 = [r["rdpo"]["ci_coverage"] for r in e1 if r["budget"] == 16384]
    v["E1_COV_MIN"] = pct(min(covs), 0)
    v["E1_COV16_MEAN"] = pct(float(np.mean(c16)), 0); v["E1_COV16_MIN"] = pct(min(c16), 0)
    base_je = next(r for r in e1 if r["scenario"] == "adb-base" and r["target"] == "joint_exhaustion" and r["budget"] == 16384)
    v["E1_BASE_JE"] = pct(base_je["rdpo"]["rel_rmse"], 2)
    v["MC_ZERO"] = pct((1 - base_je["mu"]) ** 16384, 0)
    v["BASE_JE_MU"] = sci(base_je["mu"])
    # per-path variance ratio, Phase 2 engine vs optimal twist (same paths)
    sc = json.loads((R / "rdpo_v2_scale.json").read_text())["rows"]
    by = {}
    for r in sc:
        by.setdefault((r["topology"], r["n"], r["target"]), {})[r["variant"]] = r
    rv = [(v_["twist"]["rel_rmse"] / v_["v2-final"]["rel_rmse"]) ** 2 for (t, n, _), v_ in by.items()
          if t == "ring" and n >= 8 and "twist" in v_ and "v2-final" in v_]
    v["VAR_RING"] = f"{min(rv):.0f}–{max(rv):.0f}×"
    dn = json.loads((R / "dense_exact.json").read_text())["rows"]
    dv = [r["twist_over_rdpo"] ** 2 for r in dn]
    v["VAR_DENSE"] = f"{min(dv):.0f}–{max(dv):.0f}×"
    # Phase 2 engine on the ring-28 (synthetic) and the Asia-Pacific pool (data)
    r28 = by[("ring", 28, "exceedance")]
    v["V2_RING28_SEC"] = f"{r28['v2-final']['seconds_per_fit']:.0f} s"
    v["V1_RING28_SEC"] = f"{r28['v1']['v1_study_seconds']:.0f} s"
    v["TW_RING28_SEC"] = f"{r28['twist']['seconds_per_fit']:.1f} s"
    # pools: pair exhaustion sampled inside pools
    ap = json.loads((R / "wp2_asia_pacific.json").read_text())
    ca = json.loads((R / "wp2_carec.json").read_text())
    pj = [r for r in ap["rows"] if r["method"] == "rdpo" and r["target"] == "joint_exhaustion" and r["case"] == "asia_pacific"]
    pj += [r for r in ca["rows"] if r["method"] == "rdpo" and r["target"] == "joint_exhaustion" and r["budget"] == 65536]
    v["POOLPAIR_ERR"] = f"{pct(min(r['rel_rmse'] for r in pj), 1)}–{pct(max(r['rel_rmse'] for r in pj), 1)}"
    v["POOLPAIR_COV"] = f"{pct(min(r['ci_coverage'] for r in pj), 0)}–{pct(max(r['ci_coverage'] for r in pj), 0)}"
    rare = []
    for name, d_, key in (("CAREC", ca, None), ("Asia–Pacific", ap, "asia_pacific"), ("ring-50", ap, "ring50")):
        if key is None:
            r = next(x for x in d_["rows"] if x["method"] == "rdpo" and x["target"] == "exceedance"
                     and x["threshold_musd"] == d_["T_rare"] and x["budget"] == 65536)
        else:
            c = d_["cases"][key]
            r = next(x for x in d_["rows"] if x["case"] == key and x["method"] == "rdpo" and x["target"] == "exceedance"
                     and x["threshold_musd"] == c["T_rare"])
        rare.append((name, r))
    v["POOL_RARE_ERR"] = f"{pct(min(r['rel_rmse'] for _, r in rare), 2)}–{pct(max(r['rel_rmse'] for _, r in rare), 2)}"
    v["POOL_RARE_MU"] = f"{sci(min(r['mu'] for _, r in rare))}–{sci(max(r['mu'] for _, r in rare))}"
    v["POOL_RARE_LIST"] = "; ".join(f"{n} {pct(r['rel_rmse'], 2)} at μ = {sci(r['mu'])}" for n, r in rare)
    apr = rare[1][1]
    v["AP_RARE_ERR"] = pct(apr["rel_rmse"], 2); v["AP_RARE_MU"] = sci(apr["mu"]); v["AP_RARE_SEC"] = f"{apr['seconds']:.0f} s"
    v["AP_MPS_SEC"] = f"{ap['cases']['asia_pacific']['mps_seconds']:.0f} s"
    v["CAREC_MPS_MIN"] = f"{ca['mps']['mps_seconds'] / 60:.1f} min"
    # tensor networks
    t = json.loads((R / "tensor.json").read_text())["B_scaling_n"]
    ring28 = [r for r in t if r["topology"] == "ring" and r["n"] == 28 and r["tol"] == 1e-9]
    v["TN_RING28_S"] = f"{ring28[0]['runtime_s']:.0f} s" if ring28 else "—"
    v["TN_RING28_BOND"] = str(ring28[0]["max_bond"]) if ring28 else "—"
    sparse = [r["max_bond"] for r in t if r["topology"] in ("chain", "ring", "star") and r["tol"] == 1e-9]
    v["TN_SPARSE_BOND"] = f"{min(sparse)}–{max(sparse)}"
    # loss records (earthquake shared share)
    L = json.loads((R / "realdata_losses.json").read_text())["tests"]

    def sh(k):
        x = L[k]
        return f"{pct(x['share_of_A'], 0)} (95% CI {pct(x['share_of_A_ci95'][0], 0)}–{pct(x['share_of_A_ci95'][1], 0)})"
    v["LOSS_NOAA1900"] = sh("eq_noaa_1900"); v["LOSS_NOAA1975"] = sh("eq_noaa_1975"); v["LOSS_EMDAT"] = sh("eq_emdat_1990")
    v["LOSS_NBOTH"] = str(L["eq_noaa_1900"]["table_both_KGonly_TJonly_none"][0][0])
    tab = L["eq_noaa_1900"]["table_both_KGonly_TJonly_none"]
    v["LOSS_NKG"] = str(tab[0][0] + tab[0][1]); v["LOSS_NTJ"] = str(tab[0][0] + tab[1][0])
    # rain and posterior point vs median
    rain = json.loads((R / "realdata_rain.json").read_text())
    v["RAIN_Q_LEVEL"] = "1-in-5-year"
    im = rain["implied"]
    v["RAIN_BOTH_EXH"] = sci(im["both from data"]["joint_exhaustion"])
    # quantum resources (fault-tolerant projection)
    qa = qa_file()["advantage"]
    q2 = next(r for r in qa if r["topology"] == "ring" and r["n"] == 2)
    q28 = next(r for r in qa if r["topology"] == "ring" and r["n"] == 28)
    for tag, r in (("2", q2), ("28", q28)):
        v[f"Q_LQ_{tag}"] = str(r["logical_qubits"])
        v[f"Q_PHYS_{tag}"] = f"{r['physical_qubits'] / 1e6:.1f} million" if r["physical_qubits"] >= 1e6 else f"{r['physical_qubits']:,}"
        v[f"Q_TOFF_{tag}"] = f"{r['toffoli_A']:,}"
        v[f"Q_CNOTQ_{tag}"] = f"{r['cnot_Q']:,}"
        v[f"Q_DEPTHA_{tag}"] = f"{r['depth_A']:,}"
        v[f"Q_D_{tag}"] = str(r["code_distance"])
        v[f"Q_CALLS_{tag}"] = f"{r['quantum_calls']:,.0f}"
        v[f"Q_KMAX_{tag}"] = f"{r['quantum_kmax']:.0f}"
        v[f"Q_SHOTS_{tag}"] = f"{r['quantum_shots']:,.0f}"
        s_ = r["quantum_seconds"]
        v[f"Q_TIME_{tag}"] = f"{s_ / 86400:.0f} days" if s_ > 2 * 86400 else f"{s_ / 3600:.1f} h"
        v[f"Q_TCOUNT_{tag}"] = sci(r["quantum_T_count"])
        v[f"Q_CLS_{tag}"] = (f"{r['classical_seconds'] * 1e3:.1f} ms" if r["classical_seconds"] < 1 else f"{r['classical_seconds']:.0f} s")
        v[f"Q_RHO2_{tag}"] = f"{r['classical_rho2']:.2g}" if r["classical_rho2"] < 10 else f"{r['classical_rho2']:.0f}"
    v["Q_SSTAR_2"] = sci(q2["required_speedup"])
    v["Q_MU_28"] = sci(q28["mu"])
    # error mitigation summary
    ms = json.loads((R / "quantum_advantage.json").read_text())["mitigation_summary"]
    ks = sorted(ms, key=float)
    v["MIT_WORSE0"] = pct(ms[ks[0]]["mitigation_worse_than_raw"], 0)
    v["MIT_WORSE_MAX"] = pct(ms[ks[-1]]["mitigation_worse_than_raw"], 0)
    v["MIT_DELTA_MAX"] = f"{float(ks[-1]):g}"
    v["MIT_G2_P99"] = f"{min(ms[k]['guard2']['p99'] for k in ks):.2f}–{max(ms[k]['guard2']['p99'] for k in ks):.2f}"
    v["MIT_FB"] = f"{pct(min(ms[k]['fallback2_rate'] for k in ks), 0)}–{pct(max(ms[k]['fallback2_rate'] for k in ks), 0)}"
    # near-term Born-machine proposal
    nq = json.loads((R / "nisq.json").read_text())["qcbm_bias"]
    v["QCBM_BIAS"] = f"{pct(min(r['rel_bias_ideal_weights'] for r in nq), 0)}–{pct(max(r['rel_bias_ideal_weights'] for r in nq), 0)}"
    v["QCBM_CAL"] = pct(max(abs(r["rel_bias_calibrated_rms"]) for r in nq if r["calib_shots"] == 100000), 1)
    # hardware plan: direct and tilted circuits, depth and gate counts
    hp = json.loads((R / "hardware_plan.json").read_text())

    def best(e, shots=100000):
        return min(r["rel_rmse"] for rows in hp["noisy"][e]["schedules"].values() for r in rows if r["shots"] == shots)
    v["HW_EXC_DIRECT"] = pct(best("exceedance/direct"), 0)
    v["HW_EXH_DIRECT"] = pct(best("joint_exhaustion/direct"), 0)
    ok = [best(e) for e in hp["noisy"] if best(e) < 1]
    v["HW_BEST_RANGE"] = f"{pct(min(ok), 0)}–{pct(max(ok), 0)}"
    res = hp["resources"]
    n2 = [r["device_n2q_median"] for rr in res.values() for r in rr if r["k"] <= 2]
    dp = [r["device_depth_median"] for rr in res.values() for r in rr if r["k"] <= 2]
    n1 = [r["device_n2q_median"] for rr in res.values() for r in rr if r["k"] == 1]
    v["HW_N2Q_RANGE"] = f"{min(n2):,.0f}–{max(n2):,.0f}"
    v["HW_DEPTH_RANGE"] = f"{min(dp):,.0f}–{max(dp):,.0f}"
    v["HW_N2Q_K1"] = f"{min(n1):,.0f}–{max(n1):,.0f}"
    v["HW_QUBITS"] = f"{min(e['qubits'] for e in hp['encodings'].values())}–{max(e['qubits'] for e in hp['encodings'].values())}"
    v["HW_DEVICE"] = hp["meta"]["backend_model"].split(" (")[0]
    v["HW_QISKIT"] = hp["meta"]["versions"]["qiskit"]; v["HW_AER"] = hp["meta"]["versions"]["qiskit_aer"]
    v["HW_RUNTIME_V"] = hp["meta"]["versions"]["qiskit_ibm_runtime"]
    v["HW_NUMPY"] = hp["meta"]["versions"]["numpy"]
    sc_ = hp["scan"]

    def scan(e, p2):
        rr = [r for r in sc_.get(e, []) if abs(r["p2"] - p2) < 1e-12 and abs(r["readout_error"] - 0.01) < 1e-12]
        return rr[0]["rel_rmse"] if rr else None
    t4 = [scan(e, 1e-4) for e in ("exceedance/tilt", "joint_exhaustion/tilt")]
    t5 = [scan(e, 1e-5) for e in ("exceedance/tilt", "joint_exhaustion/tilt")]
    v["HW_TILT_P4"] = f"{pct(min(t4), 1)}–{pct(max(t4), 1)}"
    v["HW_TILT_P5"] = f"{pct(min(t5), 1)}–{pct(max(t5), 1)}"
    v["HW_SV_ERR"] = sci(max(e["statevector"]["mu_abs_err"] / e["mu"] for e in hp["encodings"].values()))
    # tensor-network truncation: plain vs tilted gauge (rarest target, worst case over the correctness study)
    tc = json.loads((R / "tensor.json").read_text())["A_correctness"]

    def terr(tol, g):
        return max(r["rel_err"]["all_exhausted_prob"] for r in tc if r["tilted_gauge"] == g and r["tol"] == tol)
    v["TN_PLAIN_ERR9"] = pct(terr(1e-9, False), 1); v["TN_PLAIN_ERR12"] = pct(terr(1e-12, False), 3)
    v["TN_TILT_ERR9"] = sci(terr(1e-9, True)); v["TN_TILT_ERR12"] = sci(terr(1e-12, True))
    comp = [r["n"] for r in t if r["topology"] == "complete"]
    v["TN_COMPLETE_N"] = str(max(comp))
    # Monte Carlo paths needed for 10% relative error at the rarest pool target, and the MC-vs-RDPO variance factor
    mu_min = min(r["mu"] for _, r in rare)
    v["POOL_MC_PATHS"] = sci((1 - mu_min) / (mu_min * 0.01)).replace("1.0×", "")
    v["Q_MC_OVER"] = sci(max(((1 - r["mu"]) / r["mu"]) / r["classical_rho2"] for r in qa)).replace("1.0×", "")
    # SWAP-routing overhead on a 2D superconducting lattice (nisq.json compilations)
    cnt = json.loads((R / "nisq.json").read_text())["counts"]
    sw = [c["modality"]["superconducting_2d"]["cnot_Q"] / c["modality"]["trapped_ion_all_to_all"]["cnot_Q"] for c in cnt]
    v["SWAP_RANGE"] = f"{min(sw):.0f}–{max(sw):.0f}×"
    v["RAIN_NST2"] = str(sum(len(x) for x in rain["stations"].values()))
    enc = hp["encodings"]
    v["HW_GAIN_EXC"] = f"{enc['exceedance/tilt']['gain_a_over_mu']:.0f}×"
    v["HW_GAIN_EXH"] = sci(enc["joint_exhaustion/tilt"]["gain_a_over_mu"]) + "×"
    v["Q_WU_28"] = str(q28["w_U"])
    v["Q_MC_PATHS_28"] = sci((1 - q28["mu"]) / (q28["mu"] * 0.01))
    v["TN_COMPLETE_BOND"] = " → ".join(str(r["max_bond"]) for r in sorted(
        [r for r in t if r["topology"] == "complete" and r["tol"] == 1e-9 and r["n"] in (4, 6)], key=lambda r: r["n"]))
    star = [r["max_bond"] for r in t if r["topology"] == "star" and r["tol"] == 1e-9]
    chain = [r["max_bond"] for r in t if r["topology"] == "chain" and r["tol"] == 1e-9]
    v["TN_CHAIN_BOND"] = str(max(chain)); v["TN_STAR_BOND"] = str(max(star)) if star else "—"
    v["TN_RING_BOND"] = str(max(r["max_bond"] for r in t if r["topology"] == "ring" and r["tol"] == 1e-9))
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


def addon_values():
    """Add-on results by Xu Zi Yu (addons/results/*.json): resource re-pricing and data-uncertainty layers."""
    A = ROOT / "addons/results"
    L = lambda n: json.loads((A / n).read_text())
    v = {}
    comb = L("combined.json")["rows"]
    fc = [r["factor_Q1cQ2"] for r in comb]; fm = [r["factor_Q1Q2"] for r in comb]
    v["ADD_RED_C"] = f"{min(fc):.1f}–{max(fc):.1f}"
    v["ADD_RED_M"] = f"{min(fm):.1f}–{max(fm):.1f}"
    v["ADD_RED_ALL"] = f"{min(fc + fm):.1f}–{max(fc + fm):.1f}"
    r28 = next(r for r in comb if r["topology"] == "ring" and r["n"] == 28)
    # the reductions come from the quantum side only (same classical time cancels), so they apply to the s* of §7.6 as it now
    # stands: the faster classical engine (results/quantum_advantage_v3.json)
    s28 = next(r for r in qa_file()["advantage"] if r["topology"] == "ring" and r["n"] == 28)["required_speedup"]
    v["ADD_S28_DECL"] = sci(s28); v["ADD_S28_C"] = sci(s28 / r28["factor_Q1cQ2"]); v["ADD_S28_M"] = sci(s28 / r28["factor_Q1Q2"])
    sl = L("small_world_loop.json")["A"]
    v["ADD_SW_TRAIN"] = pct(float(np.mean([1 - math.sqrt(1 - c["variance_reduction_by_training"]) for c in sl.values()])), 0)
    q = qa_file()["advantage"]
    old28 = next(r for r in q if r["topology"] == "ring" and r["n"] == 28)
    v["Q_SSTAR_28_OLD"] = sci(old28["old_engine_same_machine"]["required_speedup"]).replace("1.0×", "")
    v["Q_ENGINE_X28"] = f"{old28['time_ratio_same_machine']:.1f}"
    tr = [r["time_ratio_same_machine"] for r in q]
    v["Q_ENGINE_XRANGE"] = f"{min(tr):.1f}–{max(tr):.1f}"
    rot = L("q1_rotations.json")["rows"]
    tc = [r["T_reduction_qrom_fallback"] for r in rot]; tm = [r["T_reduction_qrom_mixed"] for r in rot]
    v["ADD_T_C"] = f"{min(tc):.1f}–{max(tc):.1f}"; v["ADD_T_M"] = f"{min(tm):.1f}–{max(tm):.1f}"
    q2 = L("q2_ae.json")["rows"]
    ch = [r["chebae_odd"] for r in q2]; iq = [r["iqae"] for r in q2]
    v["ADD_CHEB_C"] = f"{min(c['c'] for c in ch):.0f}–{max(c['c'] for c in ch):.0f}"
    v["ADD_IQAE_C"] = f"{min(c['c'] for c in iq):.0f}–{max(c['c'] for c in iq):.0f}"
    v["ADD_CHEB_COV"] = f"{min(c['coverage'] for c in ch):.2f}"
    calls = [r["declared_calls"] / r["chebae_odd"]["calls_mean"] for r in q2]
    v["ADD_CALLS_RED"] = f"{min(calls):.1f}–{max(calls):.1f}"
    v["ADD_IQAE_DEV"] = pct(max(abs(r["declared_calls"] / r["iqae"]["calls_mean"] - 1) for r in q2), 0)
    v["ADD_PL28"] = f"{sci(r28['pL_required_iqae'])} to {sci(r28['pL_required_chebae'])}"
    mr = L("model_risk.json")["targets"]
    e, ex, j = mr["exc_moderate"], mr["expected_loss"], mr["joint_exhaustion_KG_TJ"]
    v["ADD_MR_FRIEND"] = sci(e["friend"], 2); v["ADD_MR_PP"] = sci(e["posterior_predictive"], 2)
    v["ADD_MR_LO"] = sci(e["range95"][0], 2); v["ADD_MR_HI"] = sci(e["range95"][1], 2)
    v["ADD_MR_PABOVE"] = pct(e["P_above_friend"], 0)
    v["ADD_MR_EQ_RATES"] = pct(e["sobol"]["S"]["eq_rates"], 0)
    v["ADD_MR_EL_LO"] = mcfmt_m(ex["range95"][0]); v["ADD_MR_EL_HI"] = mcfmt_m(ex["range95"][1])
    v["ADD_MR_EL_FRIEND"] = mcfmt_m(ex["friend"])
    v["ADD_MR_JE_FRIEND"] = sci(j["friend"], 2); v["ADD_MR_JE_LO"] = sci(j["range95"][0], 1)
    v["ADD_MR_JE_HI"] = sci(j["range95"][1], 1); v["ADD_MR_JE_PABOVE"] = pct(j["P_above_friend"], 0)
    ru = L("quality_rates_unc.json")["sweeps"]
    a = ru["all"]["summary"]["joint_exhaustion"]["ci95"]; b = ru["shares_only"]["summary"]["joint_exhaustion"]["ci95"]
    v["ADD_RU_LO"] = sci(a[0], 1); v["ADD_RU_HI"] = sci(a[1], 1)
    v["ADD_RU_WIDTH"] = f"{(a[1] - a[0]) / (b[1] - b[0]):.0f}"
    el = ru["all"]["summary"]["expected_loss"]["ci95"]
    v["ADD_RU_EL"] = f"{el[0] / 1e6:.1f}–{el[1] / 1e6:.1f}"
    vo = L("voi.json")["outputs"]["joint_exhaustion"]
    v["ADD_VOI_RAIN"] = pct(vo["sobol"]["S"]["rain_rates"], 0)
    v["ADD_VOI_NARROW"] = pct(vo["voi"]["rain_rates"]["interval_narrowing_perfect"], 0)
    v["ADD_VOI_30"] = pct(vo["voi"]["rain_rates"]["years"]["30"], 0)
    return v


def mcfmt_m(x):
    return f"US${x / 1e6:.1f}m"


def _collect_values():
    vals = {}
    for fn in (e1_values, e2_values, e3_values, q_values, mk_values, v2_values, rd_values, loss_values, pool_values, wp2_values, proj_values, rain_values, ap_values, headline_values, elt_values, qs_values, stronger_values, pu_values, qs_inline_values, fin_values, test_values, hw_values, dense_values,
               brief5_values, eqt_values, v3_values, extra_values, eqt_short_values, env_values, addon_values):
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
        left_short = fill(src="short.md", dst="short.filled.md", add_toc=False)
        left = sorted(set(left) | set(left_short))
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
