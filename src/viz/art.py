"""Concept diagrams and illustrations (SVG), generated so that they share the report palette/fonts.

Run: python -m src.viz.art   -> docs/art/*.svg
All illustrations (including the cat mascots) are original, drawn from primitives here.
"""
import math
import re
from pathlib import Path

from .style import PAGE, CARD, INK, INK2, MUTED, RULE, NAVY, BLUE, TERRA, TEAL, OCHRE, PLUM, GREEN

OUT = Path("docs/art")
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"
SERIF = "'Source Serif 4', Charter, 'Bitstream Charter', Georgia, serif"


def svg(w, h, body, bg=None):
    rect = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="{FONT}">{rect}{body}</svg>')


def text(x, y, s, size=12, color=INK, weight=400, anchor="start", family=None, italic=False):
    s = re.sub(r"&(?!amp;|lt;|gt;)", "&amp;", s)
    fam = f' font-family="{family}"' if family else ""
    st = ' font-style="italic"' if italic else ""
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}"{fam}{st}>{s}</text>'


def box(x, y, w, h, fill=CARD, stroke=RULE, r=10, sw=1.2):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def arrow(x1, y1, x2, y2, color=INK2, sw=1.6, dash=None, head=7):
    a = math.atan2(y2 - y1, x2 - x1)
    hx1, hy1 = x2 - head * math.cos(a - 0.4), y2 - head * math.sin(a - 0.4)
    hx2, hy2 = x2 - head * math.cos(a + 0.4), y2 - head * math.sin(a + 0.4)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>'
            f'<polygon points="{x2},{y2} {hx1:.1f},{hy1:.1f} {hx2:.1f},{hy2:.1f}" fill="{color}"/>')


# ------------------------------------------------------------------------------------------------
# cats (original flat illustrations)
# ------------------------------------------------------------------------------------------------
def cat_sitting(x, y, s=1.0, body=NAVY, accent=TERRA, eyes=PAGE, flip=False):
    """Sitting cat, base of body at (x, y)."""
    t = f'transform="translate({x},{y}) scale({-s if flip else s},{s})"'
    return f'''<g {t}>
  <path d="M 26 -6 C 60 -10 66 -40 46 -52 C 40 -56 34 -50 40 -44 C 52 -34 44 -18 22 -16" fill="none" stroke="{body}" stroke-width="7" stroke-linecap="round"/>
  <path d="M -24 0 C -30 -30 -22 -58 0 -62 C 22 -58 30 -30 24 0 Z" fill="{body}"/>
  <circle cx="0" cy="-74" r="20" fill="{body}"/>
  <path d="M -17 -84 L -20 -104 L -5 -92 Z" fill="{body}"/>
  <path d="M 17 -84 L 20 -104 L 5 -92 Z" fill="{body}"/>
  <path d="M -15 -87 L -17 -98 L -9 -91 Z" fill="{accent}"/>
  <path d="M 15 -87 L 17 -98 L 9 -91 Z" fill="{accent}"/>
  <path d="M -11 -76 q 4 -4 8 0" fill="none" stroke="{eyes}" stroke-width="2" stroke-linecap="round"/>
  <path d="M 3 -76 q 4 -4 8 0" fill="none" stroke="{eyes}" stroke-width="2" stroke-linecap="round"/>
  <path d="M -2 -69 l 2 2 l 2 -2" fill="none" stroke="{accent}" stroke-width="1.6"/>
  <path d="M -6 -66 l -16 -2 M -6 -64 l -15 3 M 6 -66 l 16 -2 M 6 -64 l 15 3" stroke="{eyes}" stroke-width="0.9" opacity="0.8"/>
  <ellipse cx="-9" cy="-1" rx="9" ry="4" fill="{body}"/>
  <ellipse cx="9" cy="-1" rx="9" ry="4" fill="{body}"/>
</g>'''


def cat_peeking(x, y, s=1.0, body=NAVY, accent=TERRA, eyes=PAGE):
    """Cat head + paws peeking over a horizontal line at y."""
    return f'''<g transform="translate({x},{y}) scale({s})">
  <circle cx="0" cy="-14" r="22" fill="{body}"/>
  <path d="M -19 -24 L -22 -46 L -5 -33 Z" fill="{body}"/>
  <path d="M 19 -24 L 22 -46 L 5 -33 Z" fill="{body}"/>
  <path d="M -16 -28 L -18 -40 L -9 -32 Z" fill="{accent}"/>
  <path d="M 16 -28 L 18 -40 L 9 -32 Z" fill="{accent}"/>
  <circle cx="-8" cy="-16" r="3.2" fill="{eyes}"/><circle cx="8" cy="-16" r="3.2" fill="{eyes}"/>
  <circle cx="-7.2" cy="-16.8" r="1.2" fill="{body}"/><circle cx="8.8" cy="-16.8" r="1.2" fill="{body}"/>
  <path d="M -2 -9 l 2 2 l 2 -2" fill="none" stroke="{accent}" stroke-width="1.6"/>
  <ellipse cx="-14" cy="2" rx="9" ry="5" fill="{body}"/><ellipse cx="14" cy="2" rx="9" ry="5" fill="{body}"/>
</g>'''


def cat_in_box(x, y, s=1.0, body=NAVY, accent=TERRA, boxc=OCHRE):
    """Schrödinger-style cat peeking from a box (quantum / future-work sections)."""
    return f'''<g transform="translate({x},{y}) scale({s})">
  {cat_peeking(0, -38, 1.0, body, accent)}
  <path d="M -48 -38 L -62 -60 L -6 -52 L 0 -38 Z" fill="{boxc}" opacity="0.85"/>
  <path d="M 48 -38 L 62 -60 L 6 -52 L 0 -38 Z" fill="{boxc}" opacity="0.85"/>
  <rect x="-48" y="-38" width="96" height="56" rx="3" fill="{boxc}"/>
  <text x="0" y="-4" font-size="15" font-weight="700" fill="{PAGE}" text-anchor="middle" font-family="{FONT}">|0⟩+|1⟩</text>
</g>'''


def cat_sleeping(x, y, s=1.0, body=NAVY, accent=TERRA, eyes=PAGE):
    return f'''<g transform="translate({x},{y}) scale({s})">
  <ellipse cx="0" cy="-16" rx="40" ry="18" fill="{body}"/>
  <path d="M 36 -8 C 52 -2 34 10 -10 6 C -30 4 -40 -2 -38 -6" fill="none" stroke="{body}" stroke-width="8" stroke-linecap="round"/>
  <circle cx="-30" cy="-22" r="15" fill="{body}"/>
  <path d="M -42 -30 L -48 -46 L -34 -36 Z" fill="{body}"/><path d="M -24 -34 L -20 -50 L -16 -34 Z" fill="{body}"/>
  <path d="M -40 -33 L -43 -41 L -37 -36 Z" fill="{accent}"/>
  <path d="M -36 -22 q 3 3 6 0 M -27 -22 q 3 3 6 0" fill="none" stroke="{eyes}" stroke-width="1.6" stroke-linecap="round"/>
  <text x="10" y="-42" font-size="13" fill="{MUTED}" font-family="{FONT}">z</text>
  <text x="20" y="-54" font-size="16" fill="{MUTED}" font-family="{FONT}">z</text>
</g>'''


# ------------------------------------------------------------------------------------------------
def cover():
    W, H = 794, 1123   # A4 at 96 dpi
    b = []
    b.append(f'<rect width="{W}" height="{H}" fill="{PAGE}"/>')
    # soft colour field
    b.append(f'<circle cx="690" cy="170" r="250" fill="{TERRA}" opacity="0.10"/>')
    b.append(f'<circle cx="120" cy="980" r="300" fill="{TEAL}" opacity="0.08"/>')
    # rare-tail curve (log-density) with the joint tail highlighted
    pts = []
    for i in range(0, 101):
        xx = 70 + i * 6.5
        yy = 860 - 330 * math.exp(-((i - 18) / 16.0) ** 2) - 40 * math.exp(-i / 40)
        pts.append((xx, yy))
    path = "M " + " L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in pts)
    b.append(f'<path d="{path}" fill="none" stroke="{NAVY}" stroke-width="3"/>')
    tail = [p for p in pts if p[0] > 520]
    b.append('<path d="M ' + " L ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in tail)
             + f' L {tail[-1][0]:.1f},862 L {tail[0][0]:.1f},862 Z" fill="{TERRA}" opacity="0.35"/>')
    b.append(f'<line x1="60" y1="862" x2="740" y2="862" stroke="{INK2}" stroke-width="1.2"/>')
    b.append(f'<path d="M 640 850 C 650 800 640 760 610 740" fill="none" stroke="{TERRA}" stroke-width="1.6"/>')
    b.append(f'<circle cx="640" cy="852" r="4" fill="{TERRA}"/>')
    b.append(text(600, 724, "joint tail", 14, TERRA, 700, "end"))
    b.append(text(600, 742, "both bonds exhausted", 11, INK2, 400, "end"))
    b.append(text(600, 758, "≈ 3 × 10⁻⁵ per term", 11, INK2, 400, "end"))
    # network of countries above the curve
    nodes = [(150, 610), (250, 560), (360, 600), (470, 545), (580, 590), (660, 520)]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if j - i <= 2:
                b.append(f'<line x1="{nodes[i][0]}" y1="{nodes[i][1]}" x2="{nodes[j][0]}" y2="{nodes[j][1]}" stroke="{RULE}" stroke-width="1.6"/>')
    for k, (xx, yy) in enumerate(nodes):
        c = [BLUE, TEAL, OCHRE, PLUM, GREEN, TERRA][k]
        b.append(f'<circle cx="{xx}" cy="{yy}" r="13" fill="{c}"/><circle cx="{xx}" cy="{yy}" r="5" fill="{PAGE}"/>')
    b.append(cat_sitting(430, 860, 1.15, NAVY, TERRA))
    b.append(cat_peeking(200, 862, 0.9, TERRA, NAVY))
    # titles
    b.append(text(60, 110, "ADB QUANTUM CHALLENGE · PHASE 1 TECHNICAL PROPOSAL", 12, TERRA, 700))
    b.append(text(60, 170, "Information-Theoretic", 38, NAVY, 700, family=SERIF))
    b.append(text(60, 214, "Reinforcement Learning for", 38, NAVY, 700, family=SERIF))
    b.append(text(60, 258, "Joint Catastrophe Tails", 38, NAVY, 700, family=SERIF))
    b.append(text(60, 306, "Quantum Enhanced Rényi-Doob Policy Optimization", 20, INK2, 400, family=SERIF, italic=True))
    b.append(f'<rect x="60" y="336" width="90" height="4" fill="{TERRA}"/>')
    b.append(text(60, 380, "Joint-loss modelling for natural catastrophe risk", 13, INK2))
    b.append(text(60, 1010, "Tan Chun Loong", 15, INK, 700))
    b.append(text(60, 1032, "Nanyang Technological University, Singapore", 12, INK2))
    b.append(text(60, 1052, "November 2026", 12, INK2))
    b.append(text(734, 1052, "github.com/CL0908/adb-quantum-challenge", 11, MUTED, anchor="end"))
    return svg(W, H, "".join(b))


def problem_setup():
    W, H = 900, 380
    b = []
    for k, (name, cx, col) in enumerate([("Kyrgyz Republic", 170, BLUE), ("Tajikistan", 730, TEAL)]):
        x0 = cx - 150
        b.append(box(x0, 30, 300, 230, CARD))
        b.append(text(cx, 60, name, 15, NAVY, 700, "middle"))
        b.append(text(cx, 80, "bond principal USD 80m · 3-year term", 11, INK2, 400, "middle"))
        # EQ quarter steps
        b.append(text(x0 + 20, 112, "Earthquake  (USD 70m)", 11, INK, 600))
        for q in range(4):
            b.append(f'<rect x="{x0 + 20 + q * 64}" y="122" width="58" height="26" rx="4" fill="{col}" opacity="{0.35 + 0.16 * q}"/>')
            b.append(text(x0 + 49 + q * 64, 140, f"{25 * (q + 1)}%", 10, PAGE, 700, "middle"))
        b.append(text(x0 + 20, 168, "payout in quarter steps a ∈ {0,…,4}, capped over the term", 9.5, INK2))
        b.append(text(x0 + 20, 196, "Precipitation  (USD 10m)", 11, INK, 600))
        b.append(f'<rect x="{x0 + 20}" y="206" width="58" height="22" rx="4" fill="{OCHRE}"/>')
        b.append(text(x0 + 49, 221, "100%", 10, PAGE, 700, "middle"))
        b.append(text(x0 + 90, 221, "at most once per term: r ∈ {0,1}", 9.5, INK2))
        b.append(text(cx, 250, f"L{['₁','₂'][k]} = 70m · a/4 + 10m · r", 11, NAVY, 600, "middle"))
    # shared events
    b.append(f'<circle cx="450" cy="140" r="48" fill="{TERRA}" opacity="0.14"/>')
    b.append(text(450, 128, "shared", 12, TERRA, 700, "middle"))
    b.append(text(450, 144, "events", 12, TERRA, 700, "middle"))
    b.append(text(450, 160, "(common shocks)", 9.5, INK2, 400, "middle"))
    b.append(arrow(400, 140, 326, 140, TERRA, 2))
    b.append(arrow(500, 140, 574, 140, TERRA, 2))
    b.append(text(450, 210, "rate c = m · ½(0.075 Λ₁ + 0.074 Λ₂)", 10, INK2, 400, "middle"))
    # targets
    b.append(box(130, 290, 640, 74, "#F1EADB", RULE, 10))
    b.append(text(450, 314, "Portfolio loss  L = L₁ + L₂   ·   five targets  μ = E[f(L₁, L₂)]", 12, NAVY, 700, "middle"))
    b.append(text(450, 338, "E[L]   ·   Pr(L₁>0, L₂>0)   ·   Pr(L > 80m)   ·   E[(L − 80m)⁺]   ·   Pr(L₁ = L₂ = 80m)", 11, INK, 400, "middle"))
    b.append(text(450, 356, "the last one, joint exhaustion, has probability ≈ 3 × 10⁻⁵ per term", 10, TERRA, 600, "middle"))
    return svg(W, H, "".join(b))


def topologies():
    W, H = 900, 250
    b = []
    defs = [("chain", "Chain", "each country shares a fault zone with its two neighbours (e.g. a mountain belt)"),
            ("ring", "Ring", "a closed chain: the first and last countries are also linked"),
            ("star", "Star", "one hub country (shared basin / river) linked to all others"),
            ("complete", "Complete", "every pair of countries can be hit by a common event")]
    n = 6
    for k, (kind, name, desc) in enumerate(defs):
        cx, cy, r = 112 + k * 225, 90, 52
        if kind == "chain":
            pts = [(cx - 80 + 32 * i, cy) for i in range(n)]
            edges = [(i, i + 1) for i in range(n - 1)]
        else:
            pts = [(cx + r * math.cos(2 * math.pi * i / n - math.pi / 2), cy + r * math.sin(2 * math.pi * i / n - math.pi / 2)) for i in range(n)]
            if kind == "ring":
                edges = [(i, (i + 1) % n) for i in range(n)]
            elif kind == "star":
                pts = [(cx, cy)] + pts[: n - 1]
                edges = [(0, i) for i in range(1, n)]
            else:
                edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for i, j in edges:
            b.append(f'<line x1="{pts[i][0]:.1f}" y1="{pts[i][1]:.1f}" x2="{pts[j][0]:.1f}" y2="{pts[j][1]:.1f}" stroke="{TERRA}" stroke-width="2"/>')
        for p in pts:
            b.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="9" fill="{NAVY}"/>')
        b.append(text(cx, 175, name, 14, NAVY, 700, "middle"))
        # wrap description
        words, line, lines = desc.split(), "", []
        for w_ in words:
            if len(line + " " + w_) > 34:
                lines.append(line); line = w_
            else:
                line = (line + " " + w_).strip()
        lines.append(line)
        for li, l in enumerate(lines):
            b.append(text(cx, 195 + 14 * li, l, 10, INK2, 400, "middle"))
    b.append(text(450, 20, "nodes = countries   ·   edges = shared-event sources (common shocks)", 11, INK2, 400, "middle"))
    return svg(W, H, "".join(b))


def sources_fig():
    W, H = 900, 300
    b = []
    b.append(text(20, 24, "Source decomposition of a 3-country chain (K = 2n + 2|E| = 10 independent Poisson sources)", 12, NAVY, 700))
    xs = [220, 450, 680]
    for i, x in enumerate(xs):
        b.append(f'<circle cx="{x}" cy="150" r="30" fill="{NAVY}"/>')
        b.append(text(x, 155, f"country {i + 1}", 10, PAGE, 700, "middle"))
        b.append(box(x - 70, 210, 64, 40, "#E6EEF7", BLUE, 8))
        b.append(text(x - 38, 228, "EQ solo", 9.5, BLUE, 700, "middle")); b.append(text(x - 38, 243, "Δa ∈ 0..4", 9, INK2, 400, "middle"))
        b.append(box(x + 6, 210, 64, 40, "#F6EEDB", OCHRE, 8))
        b.append(text(x + 38, 228, "rain solo", 9.5, OCHRE, 700, "middle")); b.append(text(x + 38, 243, "Δr ∈ 0,1", 9, INK2, 400, "middle"))
        b.append(arrow(x - 38, 210, x - 14, 178, BLUE, 1.3)); b.append(arrow(x + 38, 210, x + 14, 178, OCHRE, 1.3))
    for k in range(2):
        mx = (xs[k] + xs[k + 1]) / 2
        b.append(box(mx - 108, 44, 104, 50, "#F7E7E1", TERRA, 8))
        b.append(text(mx - 56, 64, "shared EQ", 10, TERRA, 700, "middle"))
        b.append(text(mx - 56, 80, "(Δaᵢ, Δaⱼ) ∈ 0..4²", 9, INK2, 400, "middle"))
        b.append(box(mx + 4, 44, 104, 50, "#F3EDF7", PLUM, 8))
        b.append(text(mx + 56, 64, "shared rain", 10, PLUM, 700, "middle"))
        b.append(text(mx + 56, 80, "Δrᵢ = Δrⱼ ∈ 0,1", 9, INK2, 400, "middle"))
        b.append(arrow(mx - 56, 94, xs[k] + 22, 128, TERRA, 1.3)); b.append(arrow(mx + 56, 94, xs[k + 1] - 22, 128, PLUM, 1.3))
    b.append(text(450, 290, "Capping makes every source's term effect a finite law (≤ 25 values):  min(a + min(x,4), 4) = min(a + x, 4)", 11, INK, 600, "middle"))
    return svg(W, H, "".join(b))


def pipeline():
    W, H = 900, 270
    b = []
    stages = [("0  Warm start", "optimal exponential twist\n(large-deviation Doob\ntransform of the loss)", OCHRE),
              ("1  Cross-entropy", "multilevel policy search\nminimises KL(p* ‖ q)\nconvex updates", BLUE),
              ("2  Rényi-2 gradient", "REINFORCE on log E[(wf)²]\nminimises χ²(p* ‖ q)\nearly-stopped", TERRA),
              ("3  Estimation", "selected policy + RQMC\nall eligible batches recycled\nunbiased, with CI", TEAL)]
    for k, (title, body, col) in enumerate(stages):
        x = 20 + k * 222
        b.append(box(x, 40, 196, 140, CARD, col, 12, 2))
        b.append(f'<rect x="{x}" y="40" width="196" height="34" rx="12" fill="{col}"/>')
        b.append(f'<rect x="{x}" y="60" width="196" height="14" fill="{col}"/>')
        b.append(text(x + 98, 62, title, 13, PAGE, 700, "middle"))
        for li, l in enumerate(body.split("\n")):
            b.append(text(x + 98, 102 + 20 * li, l, 11, INK, 400, "middle"))
        if k < 3:
            b.append(arrow(x + 198, 110, x + 220, 110, INK2, 2))
    b.append(text(20, 24, "RDPO: one fixed path budget B covers training and estimation", 13, NAVY, 700))
    # budget bar
    segs = [(0.06, OCHRE, ""), (0.20, BLUE, "CE"), (0.24, TERRA, "χ² stage"), (0.50, TEAL, "estimation")]
    x = 20
    for frac, col, lab in segs:
        w = 862 * frac
        b.append(f'<rect x="{x:.1f}" y="206" width="{w:.1f}" height="20" fill="{col}" opacity="0.85"/>')
        if lab:
            b.append(text(x + w / 2, 220, lab, 10, PAGE, 700, "middle"))
        x += w
    b.append(text(20, 248, "budget B (illustrative split); every batch drawn by an already level-reaching policy also enters the final estimate", 10.5, INK2))
    return svg(W, H, "".join(b))


def quantum_circuit():
    W, H = 900, 300
    b = []
    b.append(text(20, 24, "Future work: structured (Poisson-source) loading for amplitude estimation", 13, NAVY, 700))
    regs = [("source 1  (3 qb)", BLUE), ("source 2  (6 qb)", TERRA), ("source K  (1 qb)", OCHRE),
            ("country sums  Sᵢ, Rᵢ", NAVY), ("objective qubit", TEAL)]
    for k, (lab, col) in enumerate(regs):
        y = 62 + k * 44
        b.append(text(20, y + 4, lab, 11, col, 700))
        b.append(f'<line x1="170" y1="{y}" x2="880" y2="{y}" stroke="{RULE}" stroke-width="2"/>')
    for k in range(3):
        y = 62 + k * 44
        b.append(box(190, y - 15, 120, 30, "#EFE6D3", OCHRE if k == 2 else (BLUE if k == 0 else TERRA), 6))
        b.append(text(250, y + 5, "prepare √q_k (tilted)", 10, INK, 600, "middle"))
    b.append(box(340, 47, 170, 160, CARD, NAVY, 8))
    b.append(text(425, 120, "in-place adders", 11, NAVY, 700, "middle"))
    b.append(text(425, 138, "Sᵢ += Δaᵢ ,  Rᵢ += Δrᵢ", 10, INK2, 400, "middle"))
    b.append(box(540, 47, 200, 208, CARD, TEAL, 8))
    b.append(text(640, 150, "payoff oracle", 11, TEAL, 700, "middle"))
    b.append(text(640, 168, "cap · L = Σ 7aᵢ + 4rᵢ", 10, INK2, 400, "middle"))
    b.append(text(640, 186, "rotate by g = f · e^(−η(U−T))", 10, INK2, 400, "middle"))
    b.append(box(770, 225, 100, 30, "#E0F0EE", TEAL, 6))
    b.append(text(820, 245, "IQAE / MLAE", 10, TEAL, 700, "middle"))
    b.append(text(20, 290, "O(n + |E|) qubits and gates for any dependence graph; exact (no truncation); bounded weights keep the amplitude at a = μ·e^(I(T)).", 10.5, INK2))
    b.append(cat_in_box(830, 30, 0.38))
    return svg(W, H, "".join(b))


def integration():
    W, H = 900, 330
    b = []
    b.append(text(20, 24, "Where RDPO sits in a catastrophe-risk workflow", 13, NAVY, 700))
    cols = [("Catastrophe model", ["hazard (events)", "vulnerability", "financial terms"], BLUE, 20),
            ("RDPO engine", ["source laws + payoff in", "learned change of measure", "unbiased estimate + CI out"], TERRA, 330),
            ("Decisions", ["structuring &amp; pricing", "pooling / trigger design", "stress tests &amp; solvency"], TEAL, 640)]
    for title, items, col, x in cols:
        b.append(box(x, 50, 240, 200, CARD, col, 12, 2))
        b.append(f'<rect x="{x}" y="50" width="240" height="38" rx="12" fill="{col}"/><rect x="{x}" y="72" width="240" height="16" fill="{col}"/>')
        b.append(text(x + 120, 75, title, 14, PAGE, 700, "middle"))
        for k, it in enumerate(items):
            b.append(box(x + 20, 104 + 46 * k, 200, 34, PAGE, RULE, 8, 1))
            b.append(text(x + 120, 126 + 46 * k, it, 11, INK, 500, "middle"))
    b.append(arrow(262, 150, 328, 150, INK2, 2.2)); b.append(arrow(572, 150, 638, 150, INK2, 2.2))
    b.append(text(295, 140, "event / source", 9.5, INK2, 400, "middle")); b.append(text(295, 172, "level output", 9.5, INK2, 400, "middle"))
    b.append(text(605, 140, "joint-tail", 9.5, INK2, 400, "middle")); b.append(text(605, 172, "metrics", 9.5, INK2, 400, "middle"))
    for k, (lab, col) in enumerate([("open Python library", INK2), ("Oasis LMF plug-in", INK2), ("hosted API", INK2), ("future: quantum back end", TEAL)]):
        x = 150 + k * 160
        b.append(box(x, 272, 150, 30, "#EFE7D6" if k < 3 else "#E0F0EE", RULE if k < 3 else TEAL, 15, 1))
        b.append(text(x + 75, 291, lab, 10.5, col, 600, "middle"))
    b.append(text(20, 292, "Delivery:", 11, NAVY, 700))
    return svg(W, H, "".join(b))


def mascots():
    out = {}
    out["cat_sitting"] = svg(140, 130, cat_sitting(70, 122, 1.0))
    out["cat_peek"] = svg(120, 70, f'<line x1="0" y1="62" x2="120" y2="62" stroke="{RULE}" stroke-width="2"/>' + cat_peeking(60, 62, 1.0))
    out["cat_box"] = svg(160, 150, cat_in_box(80, 130, 1.0))
    out["cat_sleep"] = svg(150, 90, cat_sleeping(80, 80, 1.0))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"cover": cover(), "problem_setup": problem_setup(), "topologies": topologies(),
             "sources": sources_fig(), "pipeline": pipeline(), "quantum_circuit": quantum_circuit(),
             "integration": integration(), **mascots()}
    for k, v in files.items():
        (OUT / f"{k}.svg").write_text(v)
    print("wrote", ", ".join(files))


if __name__ == "__main__":
    main()
