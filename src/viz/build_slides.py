"""Build the 16:9 presentation deck (docs/RDPO_Presentation.pdf) from docs/slides/slides.html.

Every number on the slides is a {{PLACEHOLDER}} filled from results/*.json through build_paper._collect_values(),
exactly as for the papers; a few slide-only values (week dates, table cells, bar positions) are derived from those
same values here, never typed by hand. Slide figures are re-drawn as single, large-type panels in figures/slides/.
Usage: PYTHONPATH=. python -m src.viz.build_slides
"""
import json
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import style as S
from .build_paper import _collect_values
from .paper_figs import SCN, SCN_LABEL, TGT, _load

ROOT = Path(".")
SLIDES = ROOT / "docs/slides"
FIG = ROOT / "figures/slides"
OUT = ROOT / "docs/RDPO_Presentation.pdf"
PHASE2_START = date(2026, 11, 30)   # Deep Development: 30 Nov 2026 - 22 Jan 2027 (challenge timeline)
PHASE2_END = date(2027, 1, 22)

TGT_SHORT = {"expected_loss": "Expected\nloss", "joint_attachment": "Both\npay", "exceedance": "Pr(L > 80m)",
             "stop_loss": "Stop-\nloss", "joint_exhaustion": "Both\nexhausted"}


# ------------------------------------------------------------------ figures (single panel, large type)
def _style(fs=16):
    S.apply()
    plt.rcParams.update({"font.size": fs, "axes.labelsize": fs, "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1,
                         "legend.fontsize": fs - 2, "svg.fonttype": "path", "lines.linewidth": 2.6,
                         "lines.markersize": 7, "axes.titlesize": fs})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def fig_exact():
    """Exact engine: seconds for the full joint law against pool size (sparse MPS, dense elimination, data pools)."""
    t = _load("tensor.json")["B_scaling_n"]
    dn = _load("dense_exact.json")["exact"]
    ap = _load("wp2_asia_pacific.json")["cases"]
    ca = _load("wp2_carec.json")
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    lines = [("chain", "chain", S.BLUE), ("ring", "ring", S.TEAL), ("complete", "complete (all linked)", S.TERRA)]
    for kind, lab, c in lines:
        rr = sorted([r for r in t if r["topology"] == kind and r["tol"] == 1e-9], key=lambda r: r["n"])
        x, y = [r["n"] for r in rr], [r["runtime_s"] for r in rr]
        ax.plot(x, y, "o-", color=c, ms=6)
        dy = {"chain": 0.75, "ring": 1.0}.get(kind, 1.0)
        ax.text(x[-1] + 0.9, y[-1] * dy, lab, color=c, fontsize=15, va="center", fontweight="semibold")
    rr = sorted([e for e in dn if e["graph"] == "band4"], key=lambda e: e["n"])
    x, y = [e["n"] for e in rr], [e["exact_seconds"] for e in rr]
    ax.plot(x, y, "s--", color=S.NAVY, ms=6, lw=2.2)
    ax.text(x[-1] + 0.9, y[-1] * 1.5, "dense: 4 neighbours each", color=S.NAVY, fontsize=15, va="center",
            fontweight="semibold")
    pts = [(10, ca["mps"]["mps_seconds"], "CAREC pool\n10 countries, data", (14, -14)),
           (ap["asia_pacific"]["n"], ap["asia_pacific"]["mps_seconds"], "Asia–Pacific pool (28, data)", (14, -12)),
           (ap["ring50"]["n"], ap["ring50"]["mps_seconds"], "ring of 50", (-34, 16))]
    for n, sec, lab, off in pts:
        ax.plot(n, sec, "*", color=S.INK, ms=19, zorder=6, mec=S.PAGE, mew=1)
        ax.annotate(lab, (n, sec), xytext=off, textcoords="offset points", fontsize=14, color=S.INK, linespacing=1.15)
    ax.set_yscale("log"); ax.set_ylim(0.1, 3e3); ax.set_xlim(0, 56)
    ax.set_xlabel("countries in the pool"); ax.set_ylabel("seconds, one CPU core")
    ax.set_yticks([0.1, 1, 10, 100, 1000]); ax.set_yticklabels(["0.1", "1", "10", "100", "1000"])
    _save(fig, "exact.svg")


def fig_heat(B=16384):
    """Improvement of RDPO over the best of the brief's four methods, 25 benchmark cells."""
    d = _load("rdpo_e1.json")
    rows = {(r["scenario"], r["target"]): r for r in d["rows"] if r["budget"] == B}
    imp = np.full((5, 5), np.nan)
    for i, sc in enumerate(SCN):
        for j, tg in enumerate(TGT):
            if (sc, tg) in rows:
                imp[i, j] = rows[(sc, tg)]["improvement_vs_best"]
    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    div = LinearSegmentedColormap.from_list("div", [S.BLUE, "#F2EEE6", S.TERRA])
    Z = np.log10(imp)
    ax.imshow(Z, cmap=div, norm=TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.6), aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels([TGT_SHORT[x] for x in TGT], fontsize=16)
    ax.xaxis.tick_top()
    ax.set_yticks(range(5)); ax.set_yticklabels([SCN_LABEL[s] for s in SCN], fontsize=16)
    ax.grid(False); ax.tick_params(length=0, pad=10)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(5):
        for j in range(5):
            v = imp[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"×{v:.1f}" if v < 10 else f"×{v:.0f}", ha="center", va="center", fontsize=19,
                    color=S.PAGE if Z[i, j] > 1.1 else S.INK, fontweight="semibold")
    for k in range(6):   # thin page-coloured gutters between cells
        ax.axhline(k - 0.5, color=S.PAGE, lw=3); ax.axvline(k - 0.5, color=S.PAGE, lw=3)
    _save(fig, "heat.svg")


def fig_phase():
    """Wall-clock phase diagram for amplitude estimation against RDPO (single panel)."""
    d = _load("quantum_scaling.json")["rows"]
    adv = {(r["topology"], r["n"]): r for r in _load("quantum_advantage.json")["advantage"]}
    fig, ax = plt.subplots(figsize=(10.8, 5.5))
    eps = np.logspace(-4, -0.7, 50)
    K = [r["c_Q"] * r["overhead"] / np.sqrt(r["a"]) for r in d]
    Kmid = float(np.exp(np.mean(np.log(K))))
    ax.fill_between(eps, Kmid * eps, 1e14, color=S.TEAL, alpha=0.13, lw=0)
    ax.fill_between(eps, 1e-2, Kmid * eps, color=S.TERRA, alpha=0.07, lw=0)
    ax.fill_between(eps, min(K) * eps, max(K) * eps, color=S.NAVY, alpha=0.12, lw=0)
    for h, ls, lab in ((1, "-", "break-even, projected fault-tolerant machine"), (1e2, "--", "… 100× faster hardware"),
                       (1e4, ":", "… 10⁴× faster hardware")):
        ax.plot(eps, Kmid * eps / h, ls=ls, color=S.NAVY, lw=2.2)
        ax.text(1.15e-4, Kmid * 1.15e-4 / h * 2.6, lab, fontsize=14, color=S.NAVY, rotation=8.5, va="bottom")
    for r in d:
        mk = "o" if r["topology"] == "ring" else "s"
        ax.plot(0.1, r["rho2"], mk, color=S.TERRA, ms=9, zorder=5, mec="white", mew=0.8)
        mu = adv[(r["topology"], r["n"])]["mu"]
        ax.plot(0.1, (1 - mu) / mu, mk, mfc="none", color=S.INK2, ms=9, zorder=5, mew=1.6)
    ax.annotate("RDPO, measured\n(2–28 countries)", (0.085, 3), (0.0042, 0.25), fontsize=15, color=S.TERRA,
                fontweight="semibold", arrowprops=dict(arrowstyle="->", color=S.TERRA, lw=1.4))
    ax.annotate("plain Monte Carlo,\nsame targets", (0.085, 1e9), (0.004, 3e9), fontsize=15, color=S.INK2,
                arrowprops=dict(arrowstyle="->", color=S.INK2, lw=1.4))
    ax.text(1.3e-4, 1.5e12, "quantum faster", fontsize=17, color=S.TEAL, fontweight="bold")
    ax.text(1.3e-4, 1.6e-2, "classical RDPO faster", fontsize=17, color=S.TERRA, fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-4, 0.2); ax.set_ylim(1e-2, 1e14)
    ax.set_yticks([1e-2, 1e2, 1e6, 1e10, 1e14])
    ax.set_xlabel("target relative error ε"); ax.set_ylabel("classical variance per path ρ²")
    _save(fig, "phase.svg")


# ------------------------------------------------------------------ slide-only values derived from _collect_values()
def _num(s):
    """First number in a formatted value ('3.76%' -> 3.76)."""
    return float(re.search(r"[\d.]+", s.replace(",", "")).group())


def slide_values(v):
    s = {}
    # wall-clock table: the rows of the paper's Table (same paths / same CPU time), parsed, not retyped
    rows = [ln for ln in v["TABLE_EQT_SHORT"].splitlines() if ln.startswith("| ") and not ln.startswith("| Setting")]
    labels = ["The brief's benchmark", "Data-calibrated pools, 10 and 28 countries", "Rings to 28, complete graphs to 7",
              "Dense graphs to 28 countries"]
    vs = ["vs the brief's four methods", "vs twisting and Monte Carlo", "vs exponential twisting", "vs exponential twisting"]
    for i, ln in enumerate(rows):
        c = [x.strip() for x in ln.strip("|").split("|")]
        s[f"EQ{i}_LAB"], s[f"EQ{i}_VS"] = labels[i], vs[i]
        s[f"EQ{i}_N"], s[f"EQ{i}_PATH"] = c[1], c[2].strip("*")
        s[f"EQ{i}_TIME"] = re.sub(r"\s*\(.*\)", "", c[3])
    # Phase 2 calendar: week starts from the challenge dates
    for w in range(8):
        d = PHASE2_START + timedelta(weeks=w)
        s[f"W{w + 1}_DATE"] = f"{d.day} {d.strftime('%b')}"
    s["P2_END"] = f"{PHASE2_END.day} {PHASE2_END.strftime('%b')}"
    # model-risk bar: positions on the model-free range [0, FIN_EXC_MAX]
    hi = _num(v["FIN_EXC_MAX"])
    for k, key in (("BRIEF", "PU_EXC_BRIEF"), ("MED", "PU_EXC_MED"), ("LO", "PU_EXC_LO"), ("HI", "PU_EXC_HI")):
        s[f"POS_{k}"] = f"{100 * _num(v[key]) / hi:.2f}%"
    s["POS_W"] = f"{100 * (_num(v['PU_EXC_HI']) - _num(v['PU_EXC_LO'])) / hi:.2f}%"
    # dependence shares: the brief's model parameters against the public-data estimates
    from ..model.poisson_fit import P_TJ_GIVEN_KG
    from ..model.scenarios import SCENARIOS
    s["BRIEF_EQ_SHARE"] = f"{100 * P_TJ_GIVEN_KG:.1f}%"
    s["BRIEF_RAIN_SHARE"] = f"{100 * SCENARIOS['adb-base'].rain_shared_share:.0f}%"
    for key, name in (("LOSS_SHARE", "EQ"), ("RAIN_SHARE", "RAIN")):
        m = re.match(r"\s*(\S+)\s*\((.+)\)", v[key])
        s[f"DATA_{name}_SHARE"], s[f"DATA_{name}_CI"] = m.group(1), m.group(2)
    s["N_SLIDES"] = "12"
    return s


# ------------------------------------------------------------------ build
FONTS = ('<link rel="preconnect" href="https://fonts.gstatic.com">'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700'
         '&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&display=block">')

PDF_JS = r"""
const { chromium } = require('playwright');
(async () => {
  const [html, out] = process.argv.slice(2);
  const b = await chromium.launch(); const p = await b.newPage({ viewport: { width: 1920, height: 1080 } });
  await p.goto('file://' + html, { waitUntil: 'networkidle' });
  await p.evaluate(() => document.fonts.ready);
  const over = await p.evaluate(() => [...document.querySelectorAll('section')].map((s, i) =>
      (s.scrollHeight > s.clientHeight + 1 || s.scrollWidth > s.clientWidth + 1) ? i + 1 : 0).filter(Boolean));
  if (over.length) console.error('OVERFLOW on slides', over.join(', '));
  await p.pdf({ path: out, width: '1920px', height: '1080px', printBackground: true, preferCSSPageSize: true,
                margin: { top: 0, bottom: 0, left: 0, right: 0 } });
  await b.close();
})();
"""


SUP = "⁻⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ"
SUP_MAP = str.maketrans(SUP, "−0123456789n")


def fill(vals):
    html = (SLIDES / "slides.html").read_text()
    html = html.replace("<!--FONTS-->", FONTS)
    html = re.sub(r"<!--.*?-->\n?", "", html, flags=re.S)   # authoring notes stay out of the filled page
    for k, val in vals.items():
        html = html.replace("{{" + k + "}}", val)
    left = sorted(set(re.findall(r"\{\{(\w+)\}\}", html)))
    if left:
        print("UNFILLED in slides.html:", left, file=sys.stderr)
    n = len(re.findall(r"<section\b", html))
    if n != int(vals["N_SLIDES"]):
        print(f"WARNING: {n} sections but N_SLIDES = {vals['N_SLIDES']}", file=sys.stderr)
    # unicode superscripts (10⁻⁵) -> <sup>: the serif lacks some of these glyphs, and <sup> sets them consistently
    html = re.sub(f"[{SUP}]+", lambda m: "<sup>" + m.group().translate(SUP_MAP) + "</sup>", html)
    # slide numbers: <span class="pg"></span> -> "04 / 12", by section (the title slide counts as 01)
    parts = re.split(r"(?=<section\b)", html)
    for k in range(1, len(parts)):
        parts[k] = parts[k].replace('<span class="pg"></span>', f'<span class="pg">{k:02d} / {n:02d}</span>')
    html = "".join(parts)
    (SLIDES / "slides.filled.html").write_text(html)
    return left


def build():
    _style()
    for f in (fig_exact, fig_heat, fig_phase):
        f()
    vals = _collect_values()
    vals.update(slide_values(vals))
    left = fill(vals)
    js = SLIDES / "pdf.js"
    js.write_text(PDF_JS)
    subprocess.run(["node", str(js), str((SLIDES / "slides.filled.html").resolve()), str(OUT)], check=True)
    print("built", OUT, "unfilled:", left)
    return left


if __name__ == "__main__":
    sys.exit(1 if build() else 0)
