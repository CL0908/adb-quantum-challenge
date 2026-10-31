"""Static report figures from results/*.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
F = Path("figures"); F.mkdir(exist_ok=True)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
EXH = "term_both_bonds_exhausted_probability"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "legend.frameon": False, "lines.linewidth": 2})


def fig_rmse(scenario="adb-base", target=EXH, name="rmse_both_exhausted.png"):
    b = json.loads((R / "baselines.json").read_text())["scenarios"][scenario]["methods"]
    l = json.loads((R / "learned.json").read_text())["scenarios"][scenario][target]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = [("Monte Carlo", b["mc"][target]), ("Scrambled Sobol", b["sobol"][target]),
              ("Stratified", b["stratified"][target]), ("PDF importance sampling", b["importance"][target]),
              ("Learned Doob, factorized Ṽ", l["factorized"]["curve"]),
              ("Learned Doob, pairwise Ṽ", l["pairwise"]["curve"])]
    for i, (lab, c) in enumerate(series):
        N = sorted(int(k) for k in c)
        ax.plot(N, [c[str(n)]["rel_rmse"] for n in N], marker="o", ms=5, color=SERIES[i], label=lab)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Path budget N (evaluation paths)"); ax.set_ylabel("Relative RMSE")
    ax.set_title(f"{target.replace('term_', '').replace('_', ' ')} — {scenario}", color=INK, loc="left")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(F / name, dpi=150); plt.close(fig)


def fig_bond():
    t = json.loads((R / "tensor.json").read_text())["B_scaling_n"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, kind in enumerate(["chain", "ring", "star", "complete"]):
        rows = [r for r in t if r.get("topology") == kind and r.get("tol") == 1e-9 and "max_bond" in r]
        if rows:
            ax.plot([r["n"] for r in rows], [r["max_bond"] for r in rows], marker="o", ms=5, color=SERIES[i], label=kind)
    ax.set_yscale("log"); ax.set_xlabel("Countries n"); ax.set_ylabel("Max MPS bond dimension χ (tol 1e-9, tilted gauge)")
    ax.set_title("Exact tensor-network baseline: bond dimension vs portfolio size", color=INK, loc="left")
    ax.legend(title="Dependence graph", fontsize=8)
    fig.tight_layout(); fig.savefig(F / "bond_dimension.png", dpi=150); plt.close(fig)


def fig_latent():
    c = json.loads((R / "causal.json").read_text())["cases"]
    fig, axes = plt.subplots(1, len(c), figsize=(4 * len(c), 3.8), sharey=True)
    for ax, (name, d) in zip(axes, c.items()):
        k = [r["k"] for r in d["curve"]]
        ax.plot(k, [r["nmf_rel_err_envelope"] for r in d["curve"]], marker="o", ms=5, color=SERIES[0],
                label="classical latent (NMF)")
        kp = [r["k"] for r in d["curve"] if r.get("psd_rel_err_envelope") is not None]
        ax.plot(kp, [r["psd_rel_err_envelope"] for r in d["curve"] if r.get("psd_rel_err_envelope") is not None],
                marker="o", ms=5, color=SERIES[1], label="quantum latent (PSD)")
        ax.set_yscale("log"); ax.set_xlabel("Latent dimension k"); ax.set_title(name, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("Relative error (tilted cut matrix)")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "latent_dimension.png", dpi=150); plt.close(fig)


def fig_structured():
    s = json.loads((R / "structured.json").read_text())["scaling"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for i, kind in enumerate(["chain", "ring", "complete"]):
        rows = [r for r in s if r.get("topology") == kind and "A" in r]
        ax.plot([r["n"] for r in rows], [r["A"]["toffoli"] for r in rows], marker="o", ms=5, color=SERIES[i], label=kind)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Countries n")
    from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
    ax.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50])); ax.xaxis.set_major_formatter(FixedFormatter(["2", "5", "10", "20", "50"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Toffoli gates per A (structured loading + oracle)")
    ax.set_title("(a) Quantum A-operator cost: polynomial for every graph", color=INK, loc="left", fontsize=10)
    ax.legend(title="Dependence graph", fontsize=8)
    ax = axes[1]
    rows = [r for r in s if r.get("topology") == "ring" and "exceedance" in r]
    n = [r["n"] for r in rows]
    ax.plot(n, [r["exceedance"]["mc_samples"] for r in rows], marker="o", ms=5, color=SERIES[0], label="MC samples")
    ax.plot(n, [r["exceedance"]["qae_calls"] for r in rows], marker="o", ms=5, color=SERIES[1], label="IQAE calls to A")
    ax.set_yscale("log"); ax.set_xlabel("Countries n (ring)"); ax.set_ylabel("Queries for 10% rel. error on Pr(L > T_n)")
    ax.set_title("(b) Queries: QAE ∝ 1/√μ vs MC ∝ 1/μ", color=INK, loc="left", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "structured_scaling.png", dpi=150); plt.close(fig)


def fig_policies():
    import numpy as np
    r = json.loads((R / "hybrid.json").read_text())["policies"]
    t = EXH
    pols = [("classical_factorized", "classical factorized (30p)"), ("classical_pairwise", "classical pairwise (45p)"),
            ("hybrid_vqc_L3", "hybrid VQC critic (36p)")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (p, lab) in enumerate(pols):
        xs = [x for x in r if x["target"] == t and x["policy"] == p]
        jit = np.linspace(-0.12, 0.12, len(xs))
        ax.scatter(i - 0.18 + jit, [x["rel_rmse_ce"] for x in xs], s=28, color=SERIES[0], alpha=0.85,
                   label="CE (KL)" if i == 0 else None, edgecolors=SURF, linewidths=1.5)
        ax.scatter(i + 0.18 + jit, [x["rel_rmse_ce_renyi2"] for x in xs], s=28, color=SERIES[1], alpha=0.85,
                   label="CE + Rényi-2 (χ²) refinement" if i == 0 else None, edgecolors=SURF, linewidths=1.5)
    ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in pols], fontsize=9)
    ax.axhline(0.05, color=INK2, lw=0.8, ls="--"); ax.text(2.45, 0.055, "failure line 5%", color=INK2, fontsize=8, ha="right")
    ax.set_ylabel("Relative RMSE @ N=4096 (both exhausted)")
    ax.set_title("Doob policy families: 5 scenarios × 3 seeds", color=INK, loc="left")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(F / "policy_comparison.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_rmse, fig_bond, fig_latent, fig_structured, fig_policies):
        try:
            f(); print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
