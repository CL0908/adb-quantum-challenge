"""Single figure style for the report: palette (validated for CVD / contrast on the off-white page),
typography, and helpers (method colours, dependence-graph glyphs, panel labels)."""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PAGE = "#F6F1E7"     # off-white page / figure background
CARD = "#FBF8F1"
INK = "#1E2430"
INK2 = "#4F5563"
MUTED = "#8A8E98"
RULE = "#DDD4C4"
NAVY = "#1F3557"     # headings (text only)

# categorical slots (validated: lightness band, chroma floor, CVD ΔE >= 9.8, contrast >= 3:1)
BLUE, TERRA, TEAL, OCHRE, PLUM, GREEN = "#2F6AB0", "#C8553A", "#00897B", "#AD7A00", "#8E57AE", "#559A3C"

METHOD = {   # colour follows the method in every figure
    "RDPO": TERRA, "Monte Carlo": BLUE, "Scrambled Sobol": TEAL, "Exponential twisting": OCHRE,
    "Brief IS": PLUM, "Stratified": GREEN, "QAE": NAVY,
}
MARK = {"RDPO": "o", "Monte Carlo": "s", "Scrambled Sobol": "D", "Exponential twisting": "^",
        "Brief IS": "v", "Stratified": "P", "QAE": "*"}


def apply():
    plt.rcParams.update({
        "font.family": "Inter", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.facecolor": PAGE, "axes.facecolor": PAGE, "savefig.facecolor": PAGE,
        "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": RULE,
        "grid.linewidth": 0.6, "grid.alpha": 0.9, "axes.axisbelow": True, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 5.5, "mathtext.fontset": "dejavusans",
        "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    })


def panel_label(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=NAVY, ha="right", va="bottom")


def graph_glyph(ax, kind, x, y, r=0.05, n=6, color=INK2, transform=None, node_r=None):
    """Draw a small dependence-graph icon (chain / ring / star / complete) centred at (x, y) in axes coords."""
    tr = transform or ax.transAxes
    node_r = node_r or r * 0.18
    if kind == "chain":
        pts = [(x - r + 2 * r * i / (n - 1), y) for i in range(n)]
        edges = [(i, i + 1) for i in range(n - 1)]
    else:
        pts = [(x + r * math.cos(2 * math.pi * i / n + math.pi / 2), y + r * math.sin(2 * math.pi * i / n + math.pi / 2)) for i in range(n)]
        if kind == "ring":
            edges = [(i, (i + 1) % n) for i in range(n)]
        elif kind == "star":
            pts = [(x, y)] + pts[: n - 1]
            edges = [(0, i) for i in range(1, n)]
        else:
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in edges:
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=color, lw=0.8, transform=tr, clip_on=False, zorder=5)
    for p in pts:
        ax.add_patch(Circle(p, node_r, color=color, transform=tr, clip_on=False, zorder=6))


def save(fig, path):
    fig.savefig(path)
    fig.savefig(str(path).replace(".png", ".svg"))
    plt.close(fig)
