"""Q-RDPO figure: hybrid pipeline (top) and one Grover iterate as a proper circuit (bottom), with typeset maths.
Run: python -m src.viz.art5 -> docs/art/quantum_v3.svg"""
from .art import svg, box, FONT, SERIF, OUT, cat_in_box
from .style import PAGE, CARD, INK, INK2, MUTED, RULE, NAVY, BLUE, TERRA, TEAL, OCHRE, PLUM
from .art2 import path_arrow

LB, DARK, GOLD2 = "#DCE8F5", "#24426A", "#F3E7CF"
MATH = "'Source Serif 4', 'STIX Two Math', Georgia, serif"


def rich(x, y, parts, size=14, color=INK, weight=400, anchor="start", family=MATH):
    """Typeset text: parts is a list of (string, kind) with kind in {'n', 'sub', 'sup', 'b', 'i'}."""
    spans = []
    for s, k in parts:
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if k == "sub":
            spans.append(f'<tspan baseline-shift="sub" font-size="{size * 0.68:.1f}">{s}</tspan>')
        elif k == "sup":
            spans.append(f'<tspan baseline-shift="super" font-size="{size * 0.68:.1f}">{s}</tspan>')
        elif k == "b":
            spans.append(f'<tspan font-weight="700">{s}</tspan>')
        elif k == "i":
            spans.append(f'<tspan font-style="italic">{s}</tspan>')
        else:
            spans.append(f"<tspan>{s}</tspan>")
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}" '
            f'font-family="{family}">{"".join(spans)}</text>')


def t(x, y, s, size=13, color=INK, weight=400, anchor="start", family=FONT):
    s = s.replace("&", "&amp;")
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}" font-family="{family}">{s}</text>'


def gate(x, y, w, h, parts, fill=CARD, stroke=NAVY, size=13, color=INK):
    return box(x, y, w, h, fill, stroke, 6, 1.6) + rich(x + w / 2, y + h / 2 + size * 0.36, parts, size, color, 400, "middle")


def quantum_v3():
    W, H = 1320, 840
    b = []
    # ------------------------------------------------------------------ top: hybrid pipeline
    b.append(t(30, 34, "Q-RDPO: hybrid pipeline", 19, NAVY, 700))
    stages = [
        ("1", "Classical RDPO", [("learns θ: KL → χ", "n"), ("2", "sup")], "CPU / GPU", LB, NAVY),
        ("2", "Loadable measure", [("stage-wise twist η", "n"), ("b", "sub")], "bounded weights", LB, NAVY),
        ("3", "Compile A and Q", [("Q = A S", "n"), ("0", "sub"), (" A", "n"), ("†", "sup"), (" S", "n"), ("χ", "sub")], "fault-tolerant gates", GOLD2, NAVY),
        ("4", "Amplitude estimation", [("Grover powers k ≤ k", "n"), ("max", "sub")], "QPU", DARK, PAGE),
        ("5", "Trust guard", [("fit test + RDPO pilot", "n")], "accept or fall back", "#F6E2DC", TERRA),
    ]
    xs = [30 + 258 * k for k in range(5)]
    for x, (num, title, mth, sub, fill, tc) in zip(xs, stages):
        b.append(box(x, 52, 222, 112, fill, NAVY if fill != "#F6E2DC" else TERRA, 12, 1.8))
        b.append(f'<circle cx="{x + 22}" cy="{74}" r="12" fill="{TERRA if num == "5" else (PAGE if fill == DARK else NAVY)}"/>')
        b.append(t(x + 22, 79, num, 13, NAVY if fill == DARK else PAGE, 700, "middle"))
        b.append(t(x + 42, 80, title, 15, tc, 700))
        b.append(rich(x + 111, 116, mth, 15, PAGE if fill == DARK else INK, 400, "middle"))
        b.append(t(x + 111, 146, sub, 12, "#C9DBEF" if fill == DARK else INK2, 400, "middle"))
    labels = [[("θ", "n")], [("η", "n"), ("b", "sub")], [("A, Q", "n")], [("n", "n"), ("k", "sub")]]
    for k in range(4):
        x0 = xs[k] + 222
        b.append(path_arrow([(x0 + 2, 108), (x0 + 34, 108)], NAVY, 2.2))
        b.append(rich(x0 + 18, 98, labels[k], 13, TERRA, 700, "middle"))
    b.append(path_arrow([(xs[4] + 111, 164), (xs[4] + 111, 190)], TERRA, 2.2))
    b.append(rich(xs[4] + 111, 208, [("estimate ± CI", "n")], 15, TERRA, 700, "middle"))
    b.append(path_arrow([(xs[4] + 20, 164), (xs[4] + 20, 184), (xs[0] + 111, 184), (xs[0] + 111, 166)], MUTED, 1.4, dash="5 4"))
    b.append(t(xs[2] - 20, 200, "reject → classical RDPO estimate", 11.5, MUTED))

    # ------------------------------------------------------------------ bottom: one Grover iterate
    top = 262
    b.append(t(30, top, "One Grover iterate Q (applied k times between measurements)", 19, NAVY, 700))
    regs = [("source registers", [("4n + 7|E|", "n")], BLUE), ("country sums", [("n(w + 1)", "n")], NAVY),
            ("loss U, D = U − T", [("w", "n"), ("U", "sub")], NAVY),
            ("flag 1{L > T}", [("1", "n")], NAVY), ("objective", [("1", "n")], TERRA)]
    ys = [top + 112 + 62 * k for k in range(len(regs))]
    x_line0, x_line1 = 250, 1290
    for (name, cnt, col), y in zip(regs, ys):
        b.append(t(30, y + 5, name, 14, col, 700))
        b.append(f'<line x1="{x_line0}" y1="{y}" x2="{x_line1}" y2="{y}" stroke="{INK2}" stroke-width="1.4"/>')
        b.append(f'<line x1="{x_line0 + 14}" y1="{y + 8}" x2="{x_line0 + 26}" y2="{y - 8}" stroke="{INK2}" stroke-width="1.4"/>')
        b.append(rich(x_line0 + 20, y - 12, cnt, 11, INK2, 400, "middle"))
    # A block
    ax0, ax1 = 300, 820
    b.append(f'<rect x="{ax0}" y="{ys[0] - 34}" width="{ax1 - ax0}" height="{ys[-1] - ys[0] + 62}" rx="10" fill="{LB}" opacity="0.55" stroke="{BLUE}" stroke-dasharray="6 4"/>')
    b.append(rich((ax0 + ax1) / 2, ys[0] - 42, [("A", "b"), (": exact structured loading of the twisted measure", "n")], 14, BLUE, 400, "middle"))
    # (1) source loading
    b.append(gate(316, ys[0] - 20, 110, 40, [("R", "n"), ("y", "sub"), (" tree ", "n"), ("√q", "n"), ("k", "sub")], CARD, BLUE))
    # (2) adders
    for k, x in enumerate((452, 492, 532)):
        b.append(f'<circle cx="{x}" cy="{ys[0]}" r="5" fill="{NAVY}"/><line x1="{x}" y1="{ys[0]}" x2="{x}" y2="{ys[1] - 16}" stroke="{NAVY}" stroke-width="1.6"/>')
        b.append(gate(x - 17, ys[1] - 16, 34, 32, [("+=", "n")], CARD, NAVY, 12))
    # (3) sum and subtract T
    b.append(f'<circle cx="{588}" cy="{ys[1]}" r="5" fill="{NAVY}"/><line x1="588" y1="{ys[1]}" x2="588" y2="{ys[2] - 16}" stroke="{NAVY}" stroke-width="1.6"/>')
    b.append(gate(560, ys[2] - 16, 56, 32, [("Σ, −T", "n")], CARD, NAVY, 12))
    # (4) comparator
    b.append(f'<circle cx="{650}" cy="{ys[2]}" r="5" fill="{NAVY}"/><line x1="650" y1="{ys[2]}" x2="650" y2="{ys[3] - 16}" stroke="{NAVY}" stroke-width="1.6"/>')
    b.append(gate(626, ys[3] - 16, 48, 32, [("> 0", "n")], CARD, NAVY, 12))
    # (5) bitwise weight rotations on objective, controlled by bits of D and the flag
    for k, x in enumerate((700, 746, 792)):
        b.append(f'<circle cx="{x}" cy="{ys[2]}" r="5" fill="{TEAL}"/><circle cx="{x}" cy="{ys[3]}" r="5" fill="{TEAL}"/>')
        b.append(f'<line x1="{x}" y1="{ys[2]}" x2="{x}" y2="{ys[4] - 18}" stroke="{TEAL}" stroke-width="1.6"/>')
        b.append(gate(x - 20, ys[4] - 18, 40, 36, [("R", "n"), ("y", "sub")], "#DDF0EE", TEAL, 12))
    b.append(rich(746, ys[4] + 40, [("angle 2", "n"), ("b", "sup"), ("η per bit b of D", "n")], 11.5, TEAL, 400, "middle"))
    # S_chi, A dagger, S_0
    b.append(gate(842, ys[4] - 18, 46, 36, [("Z", "n")], "#F6E2DC", TERRA, 14))
    b.append(rich(865, ys[4] + 40, [("S", "n"), ("χ", "sub")], 13, TERRA, 700, "middle"))
    b.append(f'<rect x="910" y="{ys[0] - 22}" width="120" height="{ys[-1] - ys[0] + 44}" rx="10" fill="{BLUE}" opacity="0.85"/>')
    b.append(rich(970, (ys[0] + ys[-1]) / 2 + 8, [("A", "n"), ("†", "sup")], 24, PAGE, 700, "middle"))
    b.append(f'<rect x="1050" y="{ys[0] - 22}" width="120" height="{ys[-1] - ys[0] + 44}" rx="10" fill="{DARK}"/>')
    b.append(rich(1110, (ys[0] + ys[-1]) / 2 - 4, [("S", "n"), ("0", "sub")], 22, PAGE, 700, "middle"))
    b.append(t(1110, (ys[0] + ys[-1]) / 2 + 20, "reflect about |0⟩", 11.5, "#C9DBEF", 400, "middle"))
    # measurement
    b.append(box(1200, ys[4] - 18, 46, 36, CARD, NAVY, 6, 1.6))
    b.append(f'<path d="M 1208 {ys[4] + 10} A 15 15 0 0 1 1238 {ys[4] + 10}" fill="none" stroke="{NAVY}" stroke-width="1.6"/>')
    b.append(f'<line x1="1223" y1="{ys[4] + 10}" x2="1236" y2="{ys[4] - 10}" stroke="{NAVY}" stroke-width="1.6"/>')
    # repetition bracket
    yb = ys[0] - 62
    b.append(f'<path d="M 300 {yb + 8} L 300 {yb} L 1170 {yb} L 1170 {yb + 8}" fill="none" stroke="{NAVY}" stroke-width="1.6"/>')
    b.append(rich(735, yb - 8, [("× k,  ", "n"), ("k = 0, 1, 2, 4, …, k", "n"), ("max", "sub"), ("  (k", "n"), ("max", "sub"),
                                (" set by circuit fidelity)", "n")], 13, NAVY, 700, "middle"))
    # step labels
    for x, s_ in ((371, "①"), (492, "②"), (588, "③"), (650, "④"), (746, "⑤")):
        b.append(t(x, ys[-1] + 74, s_, 15, INK2, 600, "middle"))
    b.append(t(300, ys[-1] + 98, "① load each source in its twisted law   ② add increments per country   ③ total loss minus threshold   "
               "④ threshold flag   ⑤ weight one bit at a time", 12, INK2))
    # ------------------------------------------------------------------ formula strip
    yf = H - 40
    b.append(box(30, yf - 30, 1260, 62, CARD, RULE, 10, 1.2))
    b.append(rich(50, yf - 4, [("A|0⟩ = Σ", "n"), ("x", "sub"), (" √q", "n"), ("η", "sub"), ("(x) |x⟩ ( √(1 − g(x)) |0⟩ + √g(x) |1⟩ ),   ", "n"),
                               ("g(x) = 1{L > T} e", "n"), ("−η(U − T)", "sup"), (",   ", "n"),
                               ("a = Σ", "n"), ("x", "sub"), (" q", "n"), ("η", "sub"), ("(x) g(x) = μ e", "n"), ("ηT − Λ(η)", "sup")], 16, INK))
    b.append(t(50, yf + 20, "Every register and gate count grows linearly in n and |E|; loading is exact (no truncation, no trained quantum model); "
               "the weight needs one controlled rotation per bit of D.", 12, INK2))
    return svg(W, H, "".join(b))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "quantum_v3.svg").write_text(quantum_v3())
    print("wrote quantum_v3")


if __name__ == "__main__":
    main()
