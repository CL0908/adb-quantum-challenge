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


if __name__ == "__main__":
    for f in (fig_rmse, fig_bond, fig_latent):
        try:
            f(); print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
