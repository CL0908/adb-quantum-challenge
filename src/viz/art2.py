"""Second set of diagrams: system architecture, RL loop, concept cards, gate-level quantum circuit,
product/agent architecture. Run: python -m src.viz.art2 -> docs/art/*.svg"""
import math
from pathlib import Path

from .art import svg, text, box, arrow, cat_in_box, cat_peeking, FONT, SERIF, OUT
from .style import PAGE, CARD, INK, INK2, MUTED, RULE, NAVY, BLUE, TERRA, TEAL, OCHRE, PLUM, GREEN

LB = "#DCE8F5"      # light-blue block (architecture style)
LB2 = "#C9DBEF"
GOLD = "#C9A15A"
DARK = "#24426A"


def blk(x, y, w, h, label, sub=None, fill=LB, stroke=DARK, tc=NAVY, size=12, r=8, bold=600, sw=1.4):
    out = [box(x, y, w, h, fill, stroke, r, sw)]
    lines = label.split("\n")
    total = len(lines) + (len(sub.split("\n")) if sub else 0)
    y0 = y + h / 2 - (total - 1) * (size * 0.62) + size * 0.35
    for i, l in enumerate(lines):
        out.append(text(x + w / 2, y0 + i * size * 1.25, l, size, tc, bold, "middle"))
    if sub:
        for j, l in enumerate(sub.split("\n")):
            out.append(text(x + w / 2, y0 + (len(lines) + j) * size * 1.25, l, size - 2.5, INK2, 400, "middle"))
    return "".join(out)


def path_arrow(d, color=DARK, sw=1.8, head=7, end=None, dash=None):
    """Polyline arrow through points d=[(x,y),...]."""
    pts = " ".join(f"{x},{y}" for x, y in d)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    (x1, y1), (x2, y2) = d[-2], d[-1]
    a = math.atan2(y2 - y1, x2 - x1)
    hx1, hy1 = x2 - head * math.cos(a - 0.45), y2 - head * math.sin(a - 0.45)
    hx2, hy2 = x2 - head * math.cos(a + 0.45), y2 - head * math.sin(a + 0.45)
    return (f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}"{da} stroke-linejoin="round"/>'
            f'<polygon points="{x2},{y2} {hx1:.1f},{hy1:.1f} {hx2:.1f},{hy2:.1f}" fill="{color}"/>')


def frame(x, y, w, h, title, color=GOLD, dashed=False):
    da = ' stroke-dasharray="7 5"' if dashed else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="none" stroke="{color}" stroke-width="2.2"{da}/>'
            + text(x + w / 2, y - 12, title, 20, INK, 600, "middle"))


def dashed_hline(x1, x2, y, color=GOLD):
    return f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="1.6" stroke-dasharray="6 5"/>'


# ------------------------------------------------------------------------------------------------
def architecture():
    W, H = 1280, 760
    b = []
    # ---- inputs
    b.append(text(95, 92, "Inputs", 17, INK, 600, "middle"))
    b.append(blk(20, 110, 150, 70, "Hazard\nparameters", "tail probs · triggers"))
    b.append(blk(20, 195, 150, 62, "Contract\nterms", "limits · threshold T"))
    b.append(blk(20, 272, 150, 62, "Dependence\ngraph", "shared sources"))
    b.append(blk(20, 349, 150, 62, "Stress\nscenarios", "m, rain ×2, …"))
    # ---- model layer (dark, like 'Active Phase')
    b.append(box(210, 150, 170, 210, DARK, DARK, 10))
    b.append(text(295, 180, "Exact", 18, PAGE, 700, "middle"))
    b.append(text(295, 202, "source model", 18, PAGE, 700, "middle"))
    for i, l in enumerate(["compound Poisson fit", "capped CTMC  e^{τG}", "K = 2n + 2|E| sources", "finite laws p_k (≤ 25)"]):
        b.append(text(295, 238 + 24 * i, l, 11.5, "#DCE6F2", 400, "middle"))
    b.append(text(295, 340, "Props 1–2", 11, GOLD, 700, "middle"))
    for y in (145, 226, 303, 380):
        b.append(path_arrow([(170, y), (190, y), (190, 255), (207, 255)], DARK, 1.5))
    # ---- RDPO engine panel
    X0, Y0, WW, HH = 420, 70, 560, 620
    b.append(frame(X0, Y0, WW, HH, "RDPO engine  (one fixed path budget B)"))
    b.append(text(X0 + WW / 2, Y0 + 30, "Learning loop · soft-RL policy optimisation", 15, INK, 600, "middle"))
    b.append(dashed_hline(X0 + 10, X0 + WW - 10, Y0 + 300))
    # learning loop row
    xs = [X0 + 25, X0 + 160, X0 + 295, X0 + 430]
    labs = [("Warm start", "optimal twist η*"), ("Cross-entropy", "min KL(p*‖q)"), ("Rényi-2 PG", "min χ²(p*‖q)"), ("Policy q_θ", "selected by χ̂²")]
    for x, (l, s) in zip(xs, labs):
        b.append(blk(x, Y0 + 95, 115, 64, l, s, LB, DARK, NAVY, 12.5))
    for i in range(3):
        b.append(path_arrow([(xs[i] + 115, Y0 + 127), (xs[i + 1], Y0 + 127)], DARK, 1.6))
    # rollout feedback loop
    b.append(text(X0 + 130, Y0 + 70, "rollouts", 11, INK2, 400, "middle"))
    b.append(path_arrow([(X0 + 490, Y0 + 95), (X0 + 490, Y0 + 62), (X0 + 210, Y0 + 62), (X0 + 210, Y0 + 92)], DARK, 1.4))
    b.append(path_arrow([(X0 + 350, Y0 + 162), (X0 + 350, Y0 + 185), (X0 + 220, Y0 + 185), (X0 + 220, Y0 + 162)], DARK, 1.4))
    b.append(text(X0 + 285, Y0 + 200, "repair heavy-tailed fits", 10.5, INK2, 400, "middle"))
    # policy class strip
    b.append(box(X0 + 25, Y0 + 215, 520, 70, "#F4ECDD", GOLD, 8, 1.2))
    b.append(text(X0 + 285, Y0 + 238, "Soft value  log Ṽ_θ(s) = θ·Φ(s)   ·   Boltzmann policy  q_k(x|s) ∝ p_k(x)·e^{θ·ΔΦ_k}", 12, NAVY, 600, "middle"))
    b.append(text(X0 + 285, Y0 + 262, "Φ: country thresholds · dependence edges · target progress   |   irrelevant sources frozen at p_k", 11, INK2, 400, "middle"))
    # estimation row
    b.append(text(X0 + WW / 2, Y0 + 330, "Estimation · unbiased by construction", 15, INK, 600, "middle"))
    xs2 = [X0 + 25, X0 + 160, X0 + 295, X0 + 430]
    labs2 = [("Sequential\nsampler", "K draws / path"), ("Weights", "w = Π p_k / q_k"), ("Recycle +\nRQMC", "8 Sobol scrambles"), ("Estimate", "μ̂ ± 95% CI")]
    for x, (l, s) in zip(xs2, labs2):
        b.append(blk(x, Y0 + 360, 115, 74, l, s, LB, DARK, NAVY, 12.5))
    for i in range(3):
        b.append(path_arrow([(xs2[i] + 115, Y0 + 397), (xs2[i + 1], Y0 + 397)], DARK, 1.6))
    b.append(path_arrow([(X0 + 487, Y0 + 159), (X0 + 487, Y0 + 205)], GOLD, 2))
    b.append(path_arrow([(X0 + 82, Y0 + 285), (X0 + 82, Y0 + 357)], GOLD, 2))
    # validation strip
    b.append(box(X0 + 25, Y0 + 470, 520, 120, "#EEF3F9", LB2, 8, 1.2))
    b.append(text(X0 + 285, Y0 + 494, "Validation layer", 13, NAVY, 700, "middle"))
    for i, (l, s) in enumerate([("Exact enumeration", "n ≤ 6 · all 25 ADB values"), ("Tilted-gauge MPS", "exact for sparse graphs"),
                                 ("44 tests", "gradients · unbiasedness")]):
        b.append(blk(X0 + 40 + i * 170, Y0 + 510, 150, 64, l, s, PAGE, LB2, NAVY, 12))
    # ---- outputs
    b.append(f'<line x1="1040" y1="143" x2="1040" y2="{Y0 + 397}" stroke="{DARK}" stroke-width="1.8"/>')
    for yy in (143, 228, 313, 398):
        b.append(path_arrow([(1040, yy), (1058, yy)], DARK, 1.6))
    b.append(f'<line x1="{X0 + WW}" y1="{Y0 + 397}" x2="1040" y2="{Y0 + 397}" stroke="{DARK}" stroke-width="1.8"/>')
    b.append(text(1160, 92, "Outputs", 17, INK, 600, "middle"))
    for i, (l, s) in enumerate([("Joint tails", "exhaustion · attachment"), ("Pool metrics", "Pr(L>T) · stop-loss"),
                                 ("Stress tests", "per scenario"), ("Uncertainty", "CI · coverage")]):
        b.append(blk(1060, 110 + 85 * i, 200, 66, l, s))
    # ---- quantum stage (future)
    b.append(f'<rect x="1000" y="470" width="270" height="270" rx="6" fill="#F7F1E4" stroke="{TEAL}" stroke-width="2" stroke-dasharray="7 5"/>')
    b.append(text(1135, 497, "Quantum stage (future)", 15, TEAL, 700, "middle"))
    for i, (l, s) in enumerate([("Twisted source loader", "exact · O(n+|E|)"), ("Bitwise oracle", "e^{−η(U−T)}"),
                                 ("IQAE / MLAE + guard", "fallback if misfit")]):
        b.append(blk(1020, 512 + 72 * i, 230, 58, l, s, "#E1F1EE", TEAL, NAVY, 12))
    b.append(path_arrow([(X0 + 545, Y0 + 270), (992, Y0 + 270), (992, 600), (1000, 600)], TEAL, 1.6, dash="5 4"))
    b.append(text(997, Y0 + 330, "q_θ", 12, TEAL, 700, "start"))
    return svg(W, H, "".join(b))


# ------------------------------------------------------------------------------------------------
def rl_loop():
    W, H = 1180, 470
    b = [f'<rect x="0" y="0" width="{W}" height="{H}" rx="18" fill="#FAF6EE"/>']
    b.append(text(40, 52, "THE REINFORCEMENT-LEARNING VIEW", 12, MUTED, 600, family="'JetBrains Mono', monospace"))
    b.append(text(40, 92, "One loop.", 34, TERRA, 600, family="'JetBrains Mono', monospace"))
    b.append(text(40, 122, "Sampling a catastrophe scenario is a sequence of K decisions;", 14, INK2))
    b.append(text(40, 142, "the zero-variance sampler is the optimal soft policy.", 14, INK2))
    # loop diagram
    cx = 380
    b.append(blk(cx - 150, 185, 300, 64, "Policy  q_θ(x | s)", "Boltzmann over p_k(x)·e^{θ·ΔΦ}", "#F2E3DC", TERRA, NAVY, 14))
    b.append(blk(cx - 150, 360, 300, 64, "Environment: source k", "draw x ~ q, state s ⊕ x (capped)", LB, DARK, NAVY, 14))
    b.append(path_arrow([(cx + 150, 217), (cx + 230, 217), (cx + 230, 392), (cx + 150, 392)], INK, 1.8))
    b.append(text(cx + 240, 300, "action x_k", 12, INK2, 400))
    b.append(path_arrow([(cx - 150, 392), (cx - 230, 392), (cx - 230, 217), (cx - 150, 217)], INK, 1.8))
    b.append(text(cx - 310, 300, "state s_{k+1}", 12, INK2, 400))
    b.append(text(cx, 300, "k = 1 … K", 13, NAVY, 600, "middle"))
    b.append(text(cx, 318, "then payoff f, weight w", 11.5, INK2, 400, "middle"))
    # right: MDP table
    x0 = 700
    b.append(f'<line x1="{x0 - 30}" y1="40" x2="{x0 - 30}" y2="{H - 40}" stroke="{RULE}" stroke-width="1"/>')
    b.append(text(x0, 52, "MDP DEFINITION", 12, MUTED, 600, family="'JetBrains Mono', monospace"))
    rows = [("State", "capped levels (a_i, r_i) of all countries"), ("Action", "increment x of source k (≤ 25 values)"),
            ("Transition", "s ⊕ x  (deterministic, capped)"), ("Reference", "true source law p_k"),
            ("Reward", "log f(s_K) − KL(q ‖ p) per step"), ("Optimum", "Doob transform: w·f = μ on every path"),
            ("Learner 1", "cross-entropy: weighted MLE (KL)"), ("Learner 2", "REINFORCE on log E[(wf)²] (χ²)")]
    for i, (k, v) in enumerate(rows):
        y = 92 + 42 * i
        b.append(box(x0, y - 22, 430, 34, "#FFFFFF" if i % 2 == 0 else "#F6EFE3", RULE, 4, 0.8))
        b.append(text(x0 + 12, y, k, 12.5, TERRA if k.startswith("Learner") else NAVY, 700))
        b.append(text(x0 + 120, y, v, 12.5, INK, 400))
    return svg(W, H, "".join(b))


# ------------------------------------------------------------------------------------------------
def card(x, y, w, h, tag, head, hi, hicolor, take, graphic):
    b = [box(x, y, w, h, "#FAF6EE", RULE, 16, 1.2)]
    b.append(box(x + 24, y + 22, len(tag) * 8.4 + 26, 30, "none", INK, 8, 1.4))
    b.append(text(x + 37, y + 43, tag, 14, INK, 600))
    yy = y + 92
    for i, line in enumerate(head):
        parts = line.split("|")
        xx = x + 24
        tspans = "".join(f'<tspan fill="{hicolor if j % 2 else INK}">{p}</tspan>' for j, p in enumerate(parts))
        b.append(f'<text x="{xx}" y="{yy + 32 * i}" font-size="25" font-weight="800" font-family="{FONT}">{tspans}</text>')
    b.append(graphic)
    b.append(f'<rect x="{x + 18}" y="{y + h - 70}" width="{w - 36}" height="50" rx="10" fill="{NAVY}"/>')
    b.append(text(x + w / 2, y + h - 39, take, 14, PAGE, 700, "middle"))
    return "".join(b)


def concept_cards():
    W, H = 1260, 560
    w, h = 400, 540
    parts = []
    # 01 Doob transform
    x = 10
    base = 400
    g = [f'<line x1="{x + 30}" y1="{base}" x2="{x + 370}" y2="{base}" stroke="{INK}" stroke-width="2"/>']
    pts_p = [(x + 30 + i * 3.4, base - 110 * math.exp(-((i - 22) / 12) ** 2)) for i in range(101)]
    g.append('<path d="M ' + " L ".join(f"{a:.1f},{b_:.1f}" for a, b_ in pts_p) + f' L {x + 370},{base} L {x + 30},{base} Z" fill="{BLUE}" opacity="0.25" stroke="{BLUE}" stroke-width="1.6"/>')
    pts_q = [(x + 30 + i * 3.4, base - 150 * math.exp(-((i - 80) / 7) ** 2)) for i in range(101)]
    g.append('<path d="M ' + " L ".join(f"{a:.1f},{b_:.1f}" for a, b_ in pts_q) + f' L {x + 370},{base} L {x + 30},{base} Z" fill="{TERRA}" opacity="0.85"/>')
    g.append(text(x + 105, base + 22, "true law p", 13, BLUE, 700, "middle"))
    g.append(text(x + 302, base + 22, "learned q ∝ p·V", 13, TERRA, 700, "middle"))
    g.append(path_arrow([(x + 150, base - 120), (x + 255, base - 155)], INK, 1.8))
    g.append(text(x + 200, base - 168, "change of measure", 12, INK2, 400, "middle"))
    parts.append(card(x, 10, w, h, "01  Doob transform", ["Sample where the", "|loss| is, then reweight."], None, TERRA,
                      "Zero variance is attainable: q* ∝ p·V", "".join(g)))
    # 02 Poisson sources
    x = 430
    g = [f'<line x1="{x + 30}" y1="{base}" x2="{x + 370}" y2="{base}" stroke="{INK}" stroke-width="2"/>']
    import random
    rnd = random.Random(3)
    for k in range(70):
        xx = x + 32 + rnd.random() * 330
        yy = base - 6 - rnd.random() * 70
        g.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="2.6" fill="{NAVY}" opacity="0.55"/>')
    for k, (xx, hh, c) in enumerate([(x + 250, 150, TERRA), (x + 264, 120, TERRA), (x + 120, 60, OCHRE)]):
        g.append(f'<rect x="{xx}" y="{base - hh}" width="11" height="{hh}" fill="{c}"/>')
    g.append(text(x + 257, base - 182, "shared shock", 12, TERRA, 700, "middle"))
    g.append(text(x + 257, base - 166, "hits two countries", 11, INK2, 400, "middle"))
    g.append(text(x + 120, base + 22, "solo event", 12, OCHRE, 700, "middle"))
    parts.append(card(x, 10, w, h, "02  Poisson sources", ["Catastrophes arrive as", "|independent shocks|."], None, TERRA,
                      "A portfolio is K small independent draws", "".join(g)))
    # 03 Rényi-2
    x = 850
    g = [f'<line x1="{x + 30}" y1="{base}" x2="{x + 370}" y2="{base}" stroke="{INK}" stroke-width="2"/>']
    for k in range(30):
        hh = 8 + 4 * math.sin(k)
        g.append(f'<rect x="{x + 34 + k * 9}" y="{base - hh}" width="6" height="{hh}" fill="#B9B2A4"/>')
    for k, hh in enumerate([60, 120, 175]):
        g.append(f'<rect x="{x + 314 + k * 16}" y="{base - hh}" width="11" height="{hh}" fill="{OCHRE}"/>')
    g.append(text(x + 140, base - 40, "most paths: tiny weight", 12, INK2, 400, "middle"))
    g.append(text(x + 330, base - 190, "few paths dominate χ²", 12, OCHRE, 700, "middle"))
    parts.append(card(x, 10, w, h, "03  Rényi-2 (χ²)", ["Rare paths decide", "|the error|, not the average."], None, OCHRE,
                      "Train on χ², not only on KL", "".join(g)))
    return svg(W, H, "".join(parts))


# ------------------------------------------------------------------------------------------------
def quantum_circuit_v2():
    W, H = 1240, 560
    b = []
    b.append(text(20, 26, "Q-RDPO state-preparation operator A and Grover iterate Q = A S₀ A† S_χ  (future feature)", 16, NAVY, 700))
    regs = [("|0⟩ source 1   (3 qb)", BLUE), ("|0⟩ source 2   (6 qb)", TERRA), ("   ⋮", INK2), ("|0⟩ source K   (1 qb)", OCHRE),
            ("|0⟩ country sums Sᵢ, Rᵢ", NAVY), ("|0⟩ loss U, D = U − T", NAVY), ("|0⟩ flag 1{L > T}", NAVY),
            ("|0⟩ bit ancillas (w_U)", TEAL), ("|0⟩ objective", TEAL)]
    ys = [100 + 46 * i for i in range(len(regs))]
    for (lab, c), y in zip(regs, ys):
        b.append(text(20, y + 4, lab, 12, c, 700))
        if "⋮" not in lab:
            b.append(f'<line x1="200" y1="{y}" x2="1000" y2="{y}" stroke="{INK2}" stroke-width="1.3"/>')
    # stage 1: twisted source prep
    for i in (0, 1, 3):
        b.append(box(215, ys[i] - 15, 110, 30, "#EFE6D3", OCHRE, 5, 1.3))
        b.append(text(270, ys[i] + 5, "R_y tree √q_k", 11, INK, 600, "middle"))
    b.append(text(270, ys[-1] + 34, "① tilted source laws", 11.5, INK2, 600, "middle"))
    # stage 2: adders (vertical bars with dots)
    for j, (src, x) in enumerate([(0, 360), (1, 395), (3, 430)]):
        b.append(f'<line x1="{x}" y1="{ys[src]}" x2="{x}" y2="{ys[4]}" stroke="{NAVY}" stroke-width="1.6"/>')
        b.append(f'<circle cx="{x}" cy="{ys[src]}" r="5" fill="{NAVY}"/>')
        b.append(box(x - 14, ys[4] - 13, 28, 26, PAGE, NAVY, 4, 1.4))
        b.append(text(x, ys[4] + 5, "+=", 11, NAVY, 700, "middle"))
    b.append(text(395, ys[-1] + 56, "② in-place adders", 11.5, INK2, 600, "middle"))
    # stage 3: build U and D, comparator -> flag
    b.append(f'<line x1="480" y1="{ys[4]}" x2="480" y2="{ys[5]}" stroke="{NAVY}" stroke-width="1.6"/><circle cx="480" cy="{ys[4]}" r="5" fill="{NAVY}"/>')
    b.append(box(458, ys[5] - 13, 44, 26, PAGE, NAVY, 4, 1.4)); b.append(text(480, ys[5] + 5, "Σ, −T", 10.5, NAVY, 700, "middle"))
    b.append(f'<line x1="540" y1="{ys[4]}" x2="540" y2="{ys[6]}" stroke="{NAVY}" stroke-width="1.6"/><circle cx="540" cy="{ys[4]}" r="5" fill="{NAVY}"/>')
    b.append(box(518, ys[6] - 13, 44, 26, PAGE, NAVY, 4, 1.4)); b.append(text(540, ys[6] + 5, "> T", 10.5, NAVY, 700, "middle"))
    b.append(text(510, ys[-1] + 34, "③ loss & threshold", 11.5, INK2, 600, "middle"))
    # stage 4: bitwise controlled rotations
    for k, x in enumerate([600, 640, 680]):
        b.append(f'<line x1="{x}" y1="{ys[5]}" x2="{x}" y2="{ys[7]}" stroke="{TEAL}" stroke-width="1.6"/><circle cx="{x}" cy="{ys[5]}" r="5" fill="{TEAL}"/>')
        b.append(box(x - 17, ys[7] - 13, 34, 26, "#E0F0EE", TEAL, 4, 1.4)); b.append(text(x, ys[7] + 5, f"R{k}", 10.5, TEAL, 700, "middle"))
    b.append(text(640, ys[-1] + 56, "④ bitwise e^{−ηD}", 11.5, INK2, 600, "middle"))
    # stage 5: multi-controlled X onto objective
    x = 740
    b.append(f'<line x1="{x}" y1="{ys[6]}" x2="{x}" y2="{ys[8]}" stroke="{TEAL}" stroke-width="1.6"/>')
    for yy in (ys[6], ys[7]):
        b.append(f'<circle cx="{x}" cy="{yy}" r="5" fill="{TEAL}"/>')
    b.append(f'<circle cx="{x}" cy="{ys[8]}" r="10" fill="none" stroke="{TEAL}" stroke-width="1.8"/>'
             f'<line x1="{x - 10}" y1="{ys[8]}" x2="{x + 10}" y2="{ys[8]}" stroke="{TEAL}" stroke-width="1.8"/>'
             f'<line x1="{x}" y1="{ys[8] - 10}" x2="{x}" y2="{ys[8] + 10}" stroke="{TEAL}" stroke-width="1.8"/>')
    b.append(text(745, ys[-1] + 34, "⑤ mark good states", 11.5, INK2, 600, "middle"))
    # bracket A
    b.append(f'<path d="M 205 82 L 205 74 L 770 74 L 770 82" fill="none" stroke="{INK2}" stroke-width="1.4"/>')
    b.append(text(487, 66, "A  (prepares Σ √q(x) |x⟩ |g(x)⟩;  Pr[objective = 1] = a = μ · e^{ηT − Λ(η)})", 12, INK2, 600, "middle"))
    # amplitude estimation + guard panel
    b.append(box(820, 90, 400, 420, "#F7F1E4", TEAL, 10, 1.8))
    b.append(text(1020, 118, "Estimation with a trust guard", 14, TEAL, 700, "middle"))
    items = [("Grover powers k = 0, 1, 2, 4, …", "depth capped by circuit fidelity"),
             ("Noise-aware likelihood", "fits a and decay f jointly"),
             ("Goodness-of-fit test across depths", "detects coherent miscalibration"),
             ("Accept → quantum estimate", "reject → classical RDPO estimate")]
    for i, (l, s) in enumerate(items):
        b.append(blk(845, 140 + 88 * i, 350, 66, l, s, "#E1F1EE" if i < 3 else "#F2E3DC", TEAL if i < 3 else TERRA, NAVY, 12.5))
        if i < 3:
            b.append(path_arrow([(1020, 206 + 88 * i), (1020, 227 + 88 * i)], TEAL, 1.5))
    b.append(path_arrow([(770, 330), (842, 330)], TEAL, 1.8))
    b.append(text(20, 545, "All registers and gates scale as O(n + |E|) for any dependence graph; loading is exact (no truncation, no learned quantum model).", 12, INK2))
    return svg(W, H, "".join(b))


# ------------------------------------------------------------------------------------------------
def product_agent():
    W, H = 1180, 380
    b = []
    b.append(text(20, 30, "Product roadmap: a web application with an AI agent that calls the RDPO engine through a tool protocol", 15, NAVY, 700))
    cols = [("Web application", ["portfolio & scenario editor", "joint-tail dashboard", "report export"], BLUE, 20),
            ("AI analyst agent", ["natural-language questions", "plans stress tests", "explains results + caveats"], PLUM, 310),
            ("Tool protocol (MCP)", ["estimate_joint_tail()", "stress_test()", "validate_exact()"], OCHRE, 600),
            ("Engines", ["RDPO (CPU / GPU)", "tilted MPS (exact checks)", "quantum back end (future)"], TEAL, 890)]
    for title, items, col, x in cols:
        b.append(box(x, 60, 270, 250, CARD, col, 12, 2))
        b.append(f'<rect x="{x}" y="60" width="270" height="40" rx="12" fill="{col}"/><rect x="{x}" y="84" width="270" height="16" fill="{col}"/>')
        b.append(text(x + 135, 86, title, 14, PAGE, 700, "middle"))
        for k, it in enumerate(items):
            b.append(box(x + 18, 118 + 58 * k, 234, 42, PAGE, RULE, 8, 1))
            fam = "'JetBrains Mono', monospace" if "()" in it else None
            b.append(text(x + 135, 144 + 58 * k, it, 12 if not fam else 11.5, INK, 500, "middle", family=fam))
    for x in (290, 580, 870):
        b.append(path_arrow([(x, 175), (x + 20, 175)], INK2, 2))
        b.append(path_arrow([(x + 20, 205), (x, 205)], INK2, 2))
    b.append(text(20, 345, "Example: “What is the probability that both bonds are exhausted if the common-shock rate doubles?” → agent calls "
                      "estimate_joint_tail(scenario=common_shock, target=joint_exhaustion)", 12, INK2))
    b.append(text(20, 365, "→ returns μ̂ ± CI, the policy diagnostics (χ̂², ESS) and an exact cross-check where available.", 12, INK2))
    return svg(W, H, "".join(b))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"architecture": architecture(), "rl_loop": rl_loop(), "concept_cards": concept_cards(),
             "quantum_circuit_v2": quantum_circuit_v2(), "product_agent": product_agent()}
    for k, v in files.items():
        (OUT / f"{k}.svg").write_text(v)
    print("wrote", ", ".join(files))


if __name__ == "__main__":
    main()
