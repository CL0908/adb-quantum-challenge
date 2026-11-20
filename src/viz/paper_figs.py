"""Data figures for the formal report (one consistent style; see style.py)."""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm

from . import style as S

R = Path("results")
OUT = Path("figures/paper")
TGT = ["expected_loss", "joint_attachment", "exceedance", "stop_loss", "joint_exhaustion"]
TGT_LABEL = {"expected_loss": "Expected\nloss", "joint_attachment": "Joint\nattachment",
             "exceedance": "Pr(L > 80m)", "stop_loss": "Stop-loss\nE[(L−80m)⁺]", "joint_exhaustion": "Joint\nexhaustion"}
SCN = ["adb-base", "adb-independent", "adb-common-shock", "adb-rain-stress", "adb-rain-cover"]
SCN_LABEL = {"adb-base": "Base", "adb-independent": "Independent", "adb-common-shock": "Common shock",
             "adb-rain-stress": "Rain stress", "adb-rain-cover": "Rain cover"}
BL = {"mc": "Monte Carlo", "sobol": "Scrambled Sobol", "stratified": "Stratified", "importance": "Brief IS"}


def _load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def e1_heatmap(B=16384):
    d = _load("rdpo_e1.json")
    rows = {(r["scenario"], r["target"]): r for r in d["rows"] if r["budget"] == B}
    err = np.full((5, 5), np.nan); imp = np.full((5, 5), np.nan)
    for i, sc in enumerate(SCN):
        for j, t in enumerate(TGT):
            if (sc, t) in rows:
                err[i, j] = rows[(sc, t)]["rdpo"]["rel_rmse"]; imp[i, j] = rows[(sc, t)]["improvement_vs_best"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"wspace": 0.42})
    seq = LinearSegmentedColormap.from_list("terra", ["#FBF3EC", "#EFC2A9", S.TERRA, "#7E2F1B"])
    div = LinearSegmentedColormap.from_list("div", [S.BLUE, "#F2EEE6", S.TERRA])
    for ax, M, cmap, norm, fmt, title in [
        (axes[0], err, seq, LogNorm(1e-3, 3e-2), lambda v: f"{100 * v:.2f}%", "(a) RDPO relative RMSE"),
        (axes[1], imp, div, TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.6), lambda v: f"×{v:.1f}" if v < 10 else f"×{v:.0f}",
         "(b) Improvement over best reference method")]:
        Z = M if ax is axes[0] else np.log10(M)
        ax.imshow(Z, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(5)); ax.set_xticklabels([TGT_LABEL[t] for t in TGT], fontsize=7.5)
        ax.set_yticks(range(5)); ax.set_yticklabels([SCN_LABEL[s] for s in SCN])
        ax.grid(False); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(5):
            for j in range(5):
                if not np.isnan(M[i, j]):
                    v = Z[i, j]
                    dark = (v > 0.7) if ax is axes[1] else (M[i, j] > 8e-3)
                    ax.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=8,
                            color=S.PAGE if dark else S.INK, fontweight="semibold")
        ax.set_title(title, fontsize=9.5)
    axes[1].text(1.02, -0.18, "blue: reference method better · terracotta: RDPO better", transform=axes[1].transAxes,
                 fontsize=7, color=S.INK2, ha="right")
    S.save(fig, OUT / "e1_heatmap.png")


def e1_budget(scenario="adb-base"):
    d = _load("rdpo_e1.json"); base = _load("baselines.json")["scenarios"][scenario]["methods"]
    key = {"expected_loss": "term_expected_principal_loss_usd", "joint_attachment": "term_joint_attachment_probability",
           "exceedance": "term_portfolio_loss_exceedance_probability", "stop_loss": "term_portfolio_stop_loss_usd",
           "joint_exhaustion": "term_both_bonds_exhausted_probability"}
    fig, axes = plt.subplots(1, 5, figsize=(10.2, 2.9), sharey=False, gridspec_kw={"wspace": 0.38})
    Bs = [4096, 16384, 65536]
    for ax, t in zip(axes, TGT):
        for m, lab in BL.items():
            y = [base[m][key[t]][str(B)]["rel_rmse"] for B in Bs]
            ax.plot(Bs, y, marker=S.MARK[lab], color=S.METHOD[lab], lw=1.4, ms=4.5, label=lab, alpha=0.9)
        rr = {r["budget"]: r["rdpo"]["rel_rmse"] for r in d["rows"] if r["scenario"] == scenario and r["target"] == t}
        if rr:
            xb = sorted(rr)
            ax.plot(xb, [rr[b] for b in xb], marker="o", color=S.TERRA, lw=2.4, ms=6, label="RDPO (training included)", zorder=5)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(Bs); ax.set_xticklabels(["4k", "16k", "65k"])
        ax.minorticks_off()
        y0 = ax.get_ylim()
        ax.plot([4096, 65536], [y0[1] * 0.5, y0[1] * 0.5 / 4], ls=(0, (3, 3)), color=S.MUTED, lw=1)
        ax.set_title(TGT_LABEL[t].replace("\n", " "), fontsize=8.5)
    axes[0].set_ylabel("relative RMSE")
    h, l = axes[-1].get_legend_handles_labels()
    fig.supxlabel("total path budget B (RDPO: training + estimation)", fontsize=8, y=-0.02, color=S.INK2)
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.17), fontsize=8)
    S.save(fig, OUT / "e1_budget.png")


def e2_scaling():
    d = _load("rdpo_e2.json")
    rows = [r for r in d["rows"] if r["topology"] == "ring"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"wspace": 0.28})
    for ax, t, title in [(axes[0], "exceedance", "(a) Pool exceedance Pr(L > Tₙ)"),
                         (axes[1], "joint_exhaustion", "(b) Joint exhaustion of one pair inside the pool")]:
        rr = sorted([r for r in rows if r["target"] == t], key=lambda r: r["n"])
        if not rr:
            continue
        n = [r["n"] for r in rr]
        ax.plot(n, [r["mc_theory_rel_rmse"] for r in rr], ls="--", marker="s", color=S.BLUE, lw=1.6, label="Monte Carlo (theoretical)")
        if t == "exceedance":
            ax.plot(n, [r["twisted"]["rel_rmse"] for r in rr], marker="^", color=S.OCHRE, label="Exponential twisting")
        ax.plot(n, [r["rdpo"]["rel_rmse"] for r in rr], marker="o", color=S.TERRA, lw=2.4, ms=6, label="RDPO")
        ax.set_yscale("log"); ax.set_xticks(n)
        ax.set_xticklabels([f"{k}\nμ={r['mu']:.0e}".replace("e-0", "e-").replace("e+0", "e") for k, r in zip(n, rr)], fontsize=7)
        ax.set_xlabel("countries on a ring (n)  ·  exact target value μ below", fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.axhline(0.1, color=S.MUTED, lw=0.8, ls=":")
        ax.text(n[0], 0.112, "10% relative error", fontsize=7, color=S.INK2)
        S.graph_glyph(ax, "ring", 0.93, 0.88, r=0.045, n=6, color=S.INK2)
    axes[0].set_ylabel("relative RMSE (B = 65,536, training included)")
    axes[0].legend(loc="upper left", fontsize=7.5)
    S.save(fig, OUT / "e2_scaling.png")


def e3_ablation():
    d = _load("rdpo_e3.json")
    cases = list(dict.fromkeys(r["case"] for r in d["rows"]))
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.6), gridspec_kw={"wspace": 0.75, "hspace": 0.45})
    for ax, c in zip(axes.ravel(), cases):
        from src.rdpo.experiments import ABLATIONS
        byab = {r["ablation"]: r for r in d["rows"] if r["case"] == c}
        order = [a for a in ABLATIONS if a in byab]
        rr = [byab[a] for a in order]
        labs = [r["ablation"] for r in rr][::-1]; vals = [r["rel_rmse"] for r in rr][::-1]
        cols = [S.TERRA if l == "full RDPO" else "#B9B2A4" for l in labs]
        ax.barh(range(len(vals)), vals, color=cols, height=0.62)
        ax.set_xscale("log"); ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7.5)
        for k, v in enumerate(vals):
            ax.text(v * 1.08, k, f"{v:.3f}" if v < 1 else f"{v:.2f}", va="center", fontsize=7, color=S.INK2)
        full = [r["rel_rmse"] for r in rr if r["ablation"] == "full RDPO"][0]
        ax.axvline(full, color=S.TERRA, lw=0.8, ls=":")
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("relative RMSE (log scale, lower is better)", fontsize=7.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(right=max(vals) * 3)
        from matplotlib.ticker import LogLocator, NullFormatter
        ax.xaxis.set_major_locator(LogLocator(base=10)); ax.xaxis.set_minor_formatter(NullFormatter())
    S.save(fig, OUT / "e3_ablation.png")


def tn_bond():
    t = _load("tensor.json")["B_scaling_n"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    cols = {"chain": S.BLUE, "ring": S.TEAL, "star": S.GREEN, "complete": S.TERRA}
    for kind in ["chain", "ring", "star", "complete"]:
        rr = sorted([r for r in t if r.get("topology") == kind and r.get("tol") == 1e-9 and "max_bond" in r], key=lambda r: r["n"])
        x = [r["n"] for r in rr]; y = [r["max_bond"] for r in rr]
        ax.plot(x, y, marker="o", color=cols[kind], ms=4)
        xe, ye = x[-1], y[-1]
        ax.text(xe + 0.6, ye, kind, color=cols[kind], fontsize=8.5, va="center", fontweight="semibold")
    ax.set_yscale("log"); ax.set_xlabel("countries n"); ax.set_ylabel("max bond dimension χ")
    ax.set_xlim(1, 31)
    ax.set_title("Exact tilted-gauge tensor network: representation size vs portfolio size", fontsize=9.5)
    S.save(fig, OUT / "tn_bond.png")


def exact_scaling():
    """Exact engine: compute time and representation size versus portfolio size (sparse MPS, dense frontier elimination,
    data-calibrated pools)."""
    t = _load("tensor.json")["B_scaling_n"]
    dn = _load("dense_exact.json")["exact"]
    ap = _load("wp2_asia_pacific.json")["cases"]
    ca = _load("wp2_carec.json")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    cols = {"chain": S.BLUE, "ring": S.TEAL, "complete": S.TERRA}  # legend placed below both panels
    for kind in ("chain", "ring", "complete"):
        rr = sorted([r for r in t if r["topology"] == kind and r["tol"] == 1e-9], key=lambda r: r["n"])
        ax.plot([r["n"] for r in rr], [r["runtime_s"] for r in rr], "o-", color=cols[kind], ms=4, label=f"{kind} (tilted MPS)")
    dcol = {"band2": S.OCHRE, "band3": S.PLUM, "band4": S.NAVY, "ladder-xdiag": S.GREEN}
    dlab = {"band2": "band-2", "band3": "band-3", "band4": "band-4", "ladder-xdiag": "ladder + diagonals"}
    for g in ("band2", "band3", "band4", "ladder-xdiag"):
        rr = sorted([e for e in dn if e["graph"] == g], key=lambda e: e["n"])
        ax.plot([e["n"] for e in rr], [e["exact_seconds"] for e in rr], "s--", color=dcol[g], ms=4, lw=1.5,
                label=f"{dlab[g]} (frontier elim.)")
    pts = [(10, ca["mps"]["mps_seconds"], "CAREC pool (10, data)"),
           (ap["asia_pacific"]["n"], ap["asia_pacific"]["mps_seconds"], "Asia–Pacific pool (28, data)"),
           (ap["ring50"]["n"], ap["ring50"]["mps_seconds"], "ring (50)")]
    for n, sec, lab in pts:
        ax.plot(n, sec, "*", color=S.INK, ms=12, zorder=6)
        off = {"ring (50)": (-30, 10), "CAREC pool (10, data)": (8, -3)}.get(lab, (8, -10))
        ax.annotate(lab, (n, sec), xytext=off, textcoords="offset points", fontsize=8.5, color=S.INK)
    ax.set_yscale("log"); ax.set_xlabel("countries n"); ax.set_ylabel("seconds for the exact joint law (one core)")
    ax.set_title("(a) Exact computation time", fontsize=10)
    ax.set_xlim(0, 56)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.2), columnspacing=1.2)
    ax.set_ylim(0.05, 2e3)
    ax = axes[1]
    for kind in ("chain", "ring", "complete"):
        rr = sorted([r for r in t if r["topology"] == kind and r["tol"] == 1e-9], key=lambda r: r["n"])
        ax.plot([r["n"] for r in rr], [r["max_bond"] for r in rr], "o-", color=cols[kind], ms=4)
        ax.text(rr[-1]["n"] + 0.8, rr[-1]["max_bond"], kind, color=cols[kind], fontsize=8.5, va="center", fontweight="semibold")
    for n, b, lab in ((10, ca["mps"]["max_bond"], "CAREC (data)"), (ap["asia_pacific"]["n"], ap["asia_pacific"]["max_bond"], "Asia–Pacific (data)")):
        ax.plot(n, b, "*", color=S.INK, ms=12)
        ax.annotate(lab, (n, b), xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=S.INK)
    ns = np.arange(2, 29)
    ax.plot(ns, 10.0 ** (ns / 2), ":", color=S.MUTED, lw=1.5)
    ax.text(9.5, 10 ** 5.4, "√(10ⁿ): bond dimension of a\ngeneric n-country law", fontsize=8, color=S.INK2)
    ax.set_yscale("log"); ax.set_ylim(5, 1e7); ax.set_xlim(0, 33)
    ax.set_xlabel("countries n"); ax.set_ylabel("max bond dimension χ")
    ax.set_title("(b) Representation size", fontsize=10)
    S.save(fig, OUT / "exact_scaling.png")


def quantum_queries():
    tw = [r for r in _load("twisted.json")["rows"] if r["topology"] == "ring"]
    tw = sorted(tw, key=lambda r: r["n"])
    n = [r["n"] for r in tw]
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    series = [("Monte Carlo", [r["mc_paths"] for r in tw], S.BLUE, "s", "--"),
              ("Exponential twisting (classical, bound)", [r["classical_twisted_paths_upper"] for r in tw], S.OCHRE, "^", "-"),
              ("Amplitude estimation, untwisted (projected)", [r["qae_calls_plain"] for r in tw], S.PLUM, "v", ":"),
              ("Amplitude estimation, twisted loading (projected)", [r["qae_calls_twisted"] for r in tw], S.NAVY, "*", "-")]
    for lab, y, c, m, ls in series:
        ax.plot(n, y, marker=m, color=c, ls=ls, label=lab, ms=6 if m == "*" else 4.5)
    ax.set_yscale("log"); ax.set_xlabel("countries on a ring (n)"); ax.set_ylabel("queries for 10% relative error")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title("Query complexity on Pr(L > Tₙ): quantum stage as a projected future feature", fontsize=9.5)
    S.graph_glyph(ax, "ring", 0.93, 0.12, r=0.045, n=6, color=S.INK2)
    S.save(fig, OUT / "quantum_queries.png")


def gantt():
    """Deep Development work plan: 30 Nov 2026 - 22 Jan 2027 (8 weeks), pitch 25 Jan 2027."""
    wps = [("WP1  Wall-clock: GPU engine, tensor-network value functions", 1, 3, S.BLUE, "M1"),
           ("WP2  Dependence at trigger level: gridded rain, cat-in-a-box", 2, 5, S.TEAL, "M2"),
           ("WP3  Quantum run on IBM hardware; decision package, pitch", 5, 8, S.PLUM, "M3, M4")]
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    ax.axvspan(3.5, 5.5, color=S.RULE, alpha=0.35, lw=0)
    ax.text(4.5, -0.85, "holiday period: reduced load, compute jobs run", fontsize=7, color=S.INK2, ha="center")
    for k, (lab, s0, e, c, m) in enumerate(wps):
        y = len(wps) - 1 - k
        ax.barh(y, e - s0 + 1, left=s0 - 0.5, color=c, height=0.5, alpha=0.9)
        ax.text(0.2, y, lab, va="center", ha="right", fontsize=8, color=S.INK)
        ax.plot(e + 0.5, y, marker="D", color=S.NAVY, ms=6)
        ax.text(e + 0.75, y, m, va="center", fontsize=7.5, color=S.NAVY, fontweight="bold")
    starts = ["30 Nov", "7 Dec", "14 Dec", "21 Dec", "28 Dec", "4 Jan", "11 Jan", "18 Jan"]
    ax.set_xlim(0.4, 9.3); ax.set_ylim(-1.1, len(wps) - 0.4); ax.set_yticks([])
    ax.set_xticks(range(1, 9)); ax.set_xticklabels([f"W{w}\n{d}" for w, d in zip(range(1, 9), starts)], fontsize=7)
    ax.axvline(8.5, color=S.NAVY, lw=1, ls=":")
    ax.text(8.55, len(wps) - 0.7, "final package\n22 Jan · pitch 25 Jan", fontsize=7, color=S.NAVY)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    S.save(fig, OUT / "gantt.png")


def quantum_advantage_fig():
    d = _load("quantum_advantage_v3.json")["advantage"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7), gridspec_kw={"wspace": 0.32})
    for kind, m, c in (("ring", "o", S.TERRA), ("complete", "s", S.PLUM)):
        rr = sorted([r for r in d if r["topology"] == kind], key=lambda r: r["n"])
        n = [r["n"] for r in rr]
        axes[0].plot(n, [r["required_speedup"] for r in rr], marker=m, color=c, label=f"{kind} dependence")
        axes[1].plot(n, [r["pL_required"] for r in rr], marker=m, color=c, label=f"{kind} dependence")
    ax = axes[0]
    ax.set_yscale("log"); ax.axhline(1, color=S.INK, lw=1)
    ax.text(2, 1.4, "break-even: quantum = classical wall-clock", fontsize=7.5, color=S.INK)
    ax.set_xlabel("countries n"); ax.set_ylabel("required hardware speed-up  s* = T_Q / T_RDPO")
    ax.set_title("(a) Wall-clock criterion (10% relative error)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper right")
    ax = axes[1]
    ax.set_yscale("log")
    for y, lab in ((1e-3, "today's physical two-qubit error"), (1e-6, "early fault tolerance"), (1e-9, "mature fault tolerance")):
        ax.axhline(y, color=S.MUTED, lw=0.9, ls=":")
        ax.text(27.5, y * 1.35, lab, fontsize=7, color=S.INK2, ha="right")
    ax.set_xlabel("countries n"); ax.set_ylabel("logical error rate needed per two-qubit gate")
    ax.set_title("(b) Noise criterion for the deepest IQAE circuit", fontsize=9.5)
    ax.set_ylim(1e-10, 3e-3)
    S.save(fig, OUT / "quantum_advantage.png")


def mitigation_fig():
    q = _load("quantum_advantage.json")
    rows = q["mitigation_instances"]
    dms = sorted({r["delta_max"] for r in rows})
    fig, axes = plt.subplots(1, len(dms), figsize=(9.8, 3.2), sharey=True, gridspec_kw={"wspace": 0.12})
    lab = {"raw": ("no mitigation", S.BLUE, "--"), "mit": ("noise-aware mitigation", S.OCHRE, "-"),
           "guard": ("+ goodness-of-fit guard", S.PLUM, ":"), "guard2": ("+ fit & pilot-disagreement guard", S.TERRA, "-")}
    for ax, dm in zip(axes, dms):
        sub = [r for r in rows if r["delta_max"] == dm]
        for k, (l, c, ls) in lab.items():
            e = np.sort(np.maximum([r["err_" + k] for r in sub], 1e-5))
            ccdf = 1 - np.arange(len(e)) / len(e)
            ax.step(e, ccdf, where="post", color=c, ls=ls, lw=2.2 if k == "guard2" else 1.5, label=l)
        s = q["mitigation_summary"][str(dm)]
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"control offset |δ| ≤ {dm:g} rad / iterate", fontsize=9)
        ax.text(0.03, 0.05, f"mitigation worse than none: {100 * s['mitigation_worse_than_raw']:.0f}%\nguard falls back: {100 * s['fallback2_rate']:.0f}%",
                transform=ax.transAxes, fontsize=7.2, color=S.INK2)
        ax.set_xlabel("relative error of the estimate")
    axes[0].set_ylabel("fraction of instances with error ≥ x")
    axes[0].legend(fontsize=7.2, loc="lower left", bbox_to_anchor=(0.0, 0.18))
    S.save(fig, OUT / "mitigation.png")


def v2_fig():
    d = _load("rdpo_v2_scale.json")["rows"]
    get = {(r["topology"], r["n"], r["target"], r["variant"]): r for r in d}
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), gridspec_kw={"wspace": 0.34})
    for ax, t, title in ((axes[0], "exceedance", "(a) Pool exceedance Pr(L > Tₙ)"), (axes[1], "stop_loss", "(b) Pool stop-loss")):
        for kind, mk in (("ring", "o"), ("complete", "s")):
            ns = sorted({r["n"] for r in d if r["topology"] == kind})
            for var, c, ls, lab in (("v1", S.BLUE, "--", "Phase 1 engine"), ("v2-final", S.TERRA, "-", "Phase 2 engine"),
                                    ("twist", S.MUTED, ":", "exponential twisting")):
                y = [get[(kind, n, t, var)]["rel_rmse"] if (kind, n, t, var) in get else np.nan for n in ns]
                ax.plot(ns, y, ls=ls, marker=mk, color=c, ms=4.5, lw=1.6,
                        label=f"{lab}" if kind == "ring" else None)
        ax.set_yscale("log"); ax.set_xlabel("countries n (circles: ring, squares: every pair)")
        ax.set_ylabel("relative RMSE (B = 65,536)"); ax.set_title(title, fontsize=9.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
        ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:g}%" if f"{100 * v:.2g}"[0] in "25" else ""))
    axes[0].legend(fontsize=7.5, loc="upper left")
    ax = axes[2]
    for kind, mk in (("ring", "o"), ("complete", "s")):
        ns = sorted({r["n"] for r in d if r["topology"] == kind})
        old = [get[(kind, n, "exceedance", "v1")]["v1_study_seconds"] for n in ns]
        new = [get[(kind, n, "exceedance", "v2-final")]["seconds_per_fit"] for n in ns]
        ax.plot(ns, old, ls="--", marker=mk, color=S.BLUE, ms=4.5, label="Phase 1 engine" if kind == "ring" else None)
        ax.plot(ns, new, ls="-", marker=mk, color=S.TERRA, ms=4.5, label="Phase 2 engine" if kind == "ring" else None)
    ax.set_yscale("log"); ax.set_xlabel("countries n"); ax.set_ylabel("seconds per estimate (one core)")
    ax.set_title("(c) Wall-clock, Pr(L > Tₙ)", fontsize=9.5); ax.legend(fontsize=7.5)
    S.save(fig, OUT / "v2_results.png")

def quantum_phase_fig():
    """Phase diagram: where amplitude estimation on the loaded measure beats classical RDPO in wall-clock."""
    d = _load("quantum_advantage_v3.json")["scaling"]["rows"]
    adv = {(r["topology"], r["n"]): r for r in _load("quantum_advantage_v3.json")["advantage"]}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), gridspec_kw={"wspace": 0.3, "width_ratios": [1.35, 1]})
    ax = axes[0]
    eps = np.logspace(-4, -0.7, 50)
    K = [r["c_Q"] * r["overhead"] / np.sqrt(r["a"]) for r in d]
    Kmid = float(np.exp(np.mean(np.log(K))))
    ax.fill_between(eps, Kmid * eps, 1e14, color=S.TEAL, alpha=0.13, lw=0)
    ax.fill_between(eps, 1e-2, Kmid * eps, color=S.TERRA, alpha=0.07, lw=0)
    ax.fill_between(eps, min(K) * eps, max(K) * eps, color=S.NAVY, alpha=0.12, lw=0)
    for h, ls, lab in ((1, "-", "break-even (stated fault-tolerant baseline)"), (1e2, "--", "break-even, hardware 100× faster"),
                       (1e4, ":", "break-even, hardware 10⁴× faster")):
        ax.plot(eps, Kmid * eps / h, ls=ls, color=S.NAVY, lw=1.6, label=lab)
    for r in d:
        mk = "o" if r["topology"] == "ring" else "s"
        ax.plot(0.1, r["rho2"], mk, color=S.TERRA, ms=6, zorder=5, mec="white", mew=0.6)
        mu = adv[(r["topology"], r["n"])]["mu"]
        ax.plot(0.1, (1 - mu) / mu, mk, mfc="none", color=S.INK2, ms=6, zorder=5)
    ax.annotate("RDPO (measured,\n2–28 countries)", (0.095, 3), (0.0045, 0.08), fontsize=7.5, color=S.TERRA,
                arrowprops=dict(arrowstyle="->", color=S.TERRA, lw=0.8))
    ax.annotate("plain Monte Carlo\n(same targets)", (0.095, 1e9), (0.003, 1e7), fontsize=7.5, color=S.INK2,
                arrowprops=dict(arrowstyle="->", color=S.INK2, lw=0.8))
    ax.text(1.4e-4, 2e12, "quantum faster", fontsize=9, color=S.TEAL, fontweight="bold")
    ax.text(1.4e-4, 0.04, "classical RDPO faster", fontsize=9, color=S.TERRA, fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-4, 0.2); ax.set_ylim(1e-2, 1e14)
    ax.set_xlabel("target relative error ε"); ax.set_ylabel("classical relative variance per path ρ²")
    ax.set_title("(a) Wall-clock phase diagram", fontsize=9.5)
    ax.legend(fontsize=6.8, loc="upper left", bbox_to_anchor=(0.0, 0.9), frameon=False)
    ax = axes[1]
    n = np.array([r["n"] for r in d]); ring = np.array([r["topology"] == "ring" for r in d])
    for k, c, lab, f in (("logical_qubits", S.BLUE, "logical qubits", 1), ("toffoli_A", S.TERRA, "Toffolis per A (÷10)", 10),
                         ("depth_A", S.TEAL, "depth of A", 1)):
        y = np.array([r[k] for r in d], float) / f
        ax.plot(n[ring], y[ring], "o-", color=c, ms=4.5, label=lab)
        ax.plot(n[~ring], y[~ring], "s", color=c, ms=5, mfc="none")
    ax.set_xlabel("countries n (circles: ring, squares: every pair)")
    ax.set_title("(b) Circuit resources: linear in n and |E|", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left")
    S.save(fig, OUT / "quantum_phase.png")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S.apply()
    done = []
    for f in (e1_heatmap, e1_budget, e2_scaling, e3_ablation, tn_bond, exact_scaling, quantum_queries, gantt, quantum_advantage_fig, mitigation_fig, v2_fig, quantum_phase_fig):
        try:
            f(); done.append(f.__name__)
        except Exception as ex:   # partial results while experiments are running
            print("skip", f.__name__, type(ex).__name__, ex)
    print("ok", done)


if __name__ == "__main__":
    main()


