"""Section mascots with props, and the 'brief vs data' infographic, in the style of the existing cats (art.py).
Run: python -m src.viz.art4 -> docs/art/*.svg"""
import json
import math
from pathlib import Path

from .art import svg, text, box, cat_sitting, cat_peeking, FONT, SERIF, OUT
from .style import PAGE, CARD, INK, INK2, MUTED, RULE, NAVY, BLUE, TERRA, TEAL, OCHRE, PLUM, GREEN

W, H = 220, 150


def _scene(body, w=W, h=H):
    return svg(w, h, body)


def magnifier():
    """Cat with a magnifying glass over a scatter of points: data, real-world checks."""
    pts = "".join(f'<circle cx="{x}" cy="{y}" r="3" fill="{BLUE if x % 3 else TERRA}"/>' for x, y in
                  [(140, 110), (156, 96), (170, 118), (186, 88), (150, 128), (196, 112), (176, 104)])
    return _scene(f'''{cat_sitting(70, 140, 0.95)}{pts}
<circle cx="168" cy="104" r="26" fill="none" stroke="{OCHRE}" stroke-width="6"/>
<circle cx="168" cy="104" r="22" fill="{PAGE}" opacity="0.35"/>
<line x1="149" y1="123" x2="118" y2="146" stroke="{OCHRE}" stroke-width="8" stroke-linecap="round"/>''')


def umbrella():
    """Cat under an umbrella with rain: rainfall calibration."""
    drops = "".join(f'<line x1="{x}" y1="{y}" x2="{x - 3}" y2="{y + 9}" stroke="{BLUE}" stroke-width="2" stroke-linecap="round"/>'
                    for x, y in [(20, 20), (40, 50), (175, 30), (200, 60), (190, 100), (30, 95), (60, 18), (160, 70), (205, 15)])
    return _scene(f'''{drops}
<path d="M 52 54 Q 110 -6 168 54 Q 154 46 139 54 Q 124 46 110 54 Q 96 46 81 54 Q 66 46 52 54 Z" fill="{TEAL}"/>
<line x1="110" y1="22" x2="110" y2="104" stroke="{INK}" stroke-width="3"/>
<path d="M 110 104 q 0 10 -9 10" fill="none" stroke="{INK}" stroke-width="3" stroke-linecap="round"/>
{cat_sitting(110, 146, 0.78)}''')


def seismo():
    """Cat peeking over a seismograph trace: earthquakes."""
    xs = list(range(0, 221, 6))
    ys = [104 - (math.sin(i * 1.7) * (3 + 26 * math.exp(-((i - 18) / 4.5) ** 2))) for i in range(len(xs))]
    pts = " ".join(f"{x},{y:.1f}" for x, y in zip(xs, ys))
    return _scene(f'''<rect x="6" y="80" width="208" height="54" rx="8" fill="{CARD}" stroke="{RULE}"/>
<polyline points="{pts}" fill="none" stroke="{TERRA}" stroke-width="2.4" stroke-linejoin="round"/>
{cat_peeking(60, 82, 1.05)}''')


def stopwatch():
    """Cat with a stopwatch: speed, stress tests."""
    return _scene(f'''{cat_sitting(68, 142, 0.95)}
<circle cx="160" cy="92" r="38" fill="{CARD}" stroke="{NAVY}" stroke-width="5"/>
<rect x="152" y="44" width="16" height="10" rx="2" fill="{NAVY}"/>
<line x1="160" y1="92" x2="160" y2="64" stroke="{TERRA}" stroke-width="4" stroke-linecap="round"/>
<line x1="160" y1="92" x2="180" y2="102" stroke="{NAVY}" stroke-width="3" stroke-linecap="round"/>
<circle cx="160" cy="92" r="4" fill="{NAVY}"/>
<path d="M 128 70 l -8 -6 M 192 70 l 8 -6" stroke="{NAVY}" stroke-width="4" stroke-linecap="round"/>
<text x="160" y="146" font-size="15" font-weight="700" fill="{TERRA}" text-anchor="middle" font-family="{FONT}">5 s</text>''')


def books():
    """Cat on a stack of books: related work, references."""
    stack = "".join(f'<rect x="{x}" y="{y}" width="{w}" height="16" rx="3" fill="{c}"/>'
                    for x, y, w, c in [(64, 128, 100, BLUE), (72, 112, 86, TEAL), (60, 96, 96, OCHRE), (70, 80, 84, PLUM)])
    return _scene(f'''{stack}{cat_sitting(112, 80, 0.62)}''')


def chart():
    """Cat beside a bar chart with one tall red bar: results."""
    bars = "".join(f'<rect x="{x}" y="{140 - h}" width="18" height="{h}" rx="3" fill="{c}"/>'
                   for x, h, c in [(118, 30, RULE), (142, 44, RULE), (166, 36, RULE), (190, 108, TERRA)])
    return _scene(f'''{bars}<line x1="110" y1="140" x2="214" y2="140" stroke="{INK2}" stroke-width="2"/>
{cat_sitting(58, 142, 0.95)}''')


def globe():
    """Cat with a globe: regional pools."""
    return _scene(f'''{cat_sitting(62, 142, 0.95)}
<circle cx="160" cy="84" r="44" fill="{BLUE}" opacity="0.18" stroke="{BLUE}" stroke-width="3"/>
<ellipse cx="160" cy="84" rx="18" ry="44" fill="none" stroke="{BLUE}" stroke-width="2"/>
<line x1="116" y1="84" x2="204" y2="84" stroke="{BLUE}" stroke-width="2"/>
<path d="M 122 62 Q 160 70 198 62 M 122 106 Q 160 98 198 106" fill="none" stroke="{BLUE}" stroke-width="2"/>
<path d="M 146 70 q 10 -8 22 0 q 6 8 -4 14 q -12 4 -18 -6 z" fill="{GREEN}"/>
<circle cx="170" cy="96" r="5" fill="{TERRA}"/><circle cx="150" cy="92" r="4" fill="{TERRA}"/>
<line x1="160" y1="128" x2="160" y2="142" stroke="{INK}" stroke-width="4"/><rect x="138" y="140" width="44" height="6" rx="3" fill="{INK}"/>''')


def table_sheet():
    """Cat holding a table (ELT/YLT)."""
    rows = "".join(f'<line x1="122" y1="{y}" x2="206" y2="{y}" stroke="{RULE}" stroke-width="2"/>' for y in range(56, 132, 14))
    return _scene(f'''<rect x="114" y="34" width="100" height="104" rx="8" fill="{CARD}" stroke="{NAVY}" stroke-width="3"/>
<rect x="114" y="34" width="100" height="18" rx="8" fill="{NAVY}"/>{rows}
<line x1="150" y1="52" x2="150" y2="136" stroke="{RULE}" stroke-width="2"/>
<text x="164" y="47" font-size="11" font-weight="700" fill="{PAGE}" text-anchor="middle" font-family="{FONT}">ELT</text>
{cat_sitting(62, 142, 0.95)}''')


def briefcase():
    """Cat with a briefcase: commercialisation."""
    return _scene(f'''{cat_sitting(70, 142, 0.95)}
<rect x="128" y="86" width="76" height="54" rx="8" fill="{OCHRE}"/>
<rect x="152" y="74" width="28" height="14" rx="4" fill="none" stroke="{OCHRE}" stroke-width="5"/>
<line x1="128" y1="108" x2="204" y2="108" stroke="{PAGE}" stroke-width="3"/>
<rect x="160" y="102" width="12" height="12" rx="2" fill="{PAGE}"/>''')


def calendar():
    """Cat with a calendar: work plan."""
    cells = "".join(f'<rect x="{126 + 20 * (k % 4)}" y="{72 + 18 * (k // 4)}" width="14" height="12" rx="2" fill="{TERRA if k == 6 else RULE}"/>'
                    for k in range(12))
    return _scene(f'''<rect x="116" y="40" width="94" height="100" rx="8" fill="{CARD}" stroke="{NAVY}" stroke-width="3"/>
<rect x="116" y="40" width="94" height="22" rx="8" fill="{TEAL}"/>{cells}
<circle cx="138" cy="40" r="4" fill="{NAVY}"/><circle cx="188" cy="40" r="4" fill="{NAVY}"/>
{cat_sitting(62, 142, 0.95)}''')


def laptop():
    """Cat on a laptop: platform, prototype."""
    return _scene(f'''<path d="M 40 128 L 180 128 L 196 142 L 24 142 Z" fill="{INK2}"/>
<rect x="56" y="58" width="108" height="70" rx="6" fill="{NAVY}"/>
<path d="M 72 112 l 16 -16 l 14 10 l 22 -26 l 18 12" fill="none" stroke="{TEAL}" stroke-width="3"/>
{cat_peeking(110, 60, 0.8)}''')


def team():
    """Two cats side by side: team."""
    return _scene(f'''{cat_sitting(80, 142, 0.95)}{cat_sitting(150, 142, 0.75, body=TERRA, accent=NAVY, flip=True)}''')


# ------------------------------------------------------------------------------------------------
def brief_vs_data():
    """Infographic: dependence assumptions of the brief versus public data, and the effect on the pool."""
    R = Path("results")
    eq = json.loads((R / "realdata_losses.json").read_text())["tests"]["eq_noaa_1900"]
    rain = json.loads((R / "realdata_rain.json").read_text())
    pr = rain["pooled"]["0.2"]; im = rain["implied"]
    W2, H2 = 1180, 432
    b = [text(30, 40, "What the data say about the bond's two dependence assumptions", 22, NAVY, 700)]

    def row(y, title, sub, brief, data, lo, hi, col, note):
        x0, x1 = 400, 960
        sc = lambda v: x0 + (x1 - x0) * v
        out = [text(30, y + 8, title, 17, INK, 700), text(30, y + 30, sub, 12.5, INK2)]
        out.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{RULE}" stroke-width="10" stroke-linecap="round"/>')
        for t in (0, 0.25, 0.5, 0.75, 1.0):
            out.append(text(sc(t), y + 34, f"{int(t * 100)}%", 11.5, MUTED, 400, "middle"))
        out.append(f'<line x1="{sc(lo)}" y1="{y}" x2="{sc(hi)}" y2="{y}" stroke="{col}" stroke-width="10" stroke-linecap="round" opacity="0.35"/>')
        out.append(f'<circle cx="{sc(brief)}" cy="{y}" r="11" fill="{PAGE}" stroke="{INK2}" stroke-width="3"/>')
        out.append(text(sc(brief), y - 18, f"brief {100 * brief:.1f}%", 12.5, INK2, 600, "middle"))
        out.append(f'<circle cx="{sc(data)}" cy="{y}" r="12" fill="{col}"/>')
        out.append(text(sc(data), y - 18, f"data {100 * data:.0f}%", 13.5, col, 700, "middle"))
        out.append(text(990, y + 6, note, 14, col, 700))
        return "".join(out)

    b.append(row(120, "Earthquakes hitting both countries", "share of events · NOAA loss records 1900–2020", 0.0745,
                 eq["share_of_A"], eq["share_of_A_ci95"][0], eq["share_of_A_ci95"][1], TERRA, "much higher"))
    b.append(row(230, "Extreme rain hitting both countries", "share of events · 48 weather stations (GHCN-Daily)", 0.25,
                 pr["share"], pr["share_ci95"][0], pr["share_ci95"][1], TEAL, "much lower"))
    pu = json.loads((R / "param_uncertainty.json").read_text())
    exb = pu["brief_point"]["exceedance"]
    ps = pu["posterior_summary"]["exceedance"]
    exd, lo, hi = ps["median"], ps["ci95"][0], ps["ci95"][1]
    fin = json.loads((R / "finance.json").read_text())["pooling"]
    pb = fin["brief"]["shared"]["80m"]["delivered_fraction"]
    pd = fin["posterior"]["delivered_fraction"]["median"]
    b.append(box(30, 300, 860, 110, CARD, RULE, 14, 1.5))
    b.append(text(52, 330, "Chance that the pool pays more than US$80m in a three-year term", 15, INK, 600))
    b.append(text(52, 368, f"{100 * exb:.1f}%", 30, INK2, 700))
    b.append(text(150, 364, "→", 28, INK2, 400))
    b.append(text(192, 368, f"{100 * exd:.1f}%", 34, TERRA, 700))
    b.append(text(310, 360, f"{exd / exb:.1f}× the brief (posterior median)", 18, TERRA, 700))
    b.append(text(310, 380, f"95% credible interval {100 * lo:.1f}%–{100 * hi:.1f}%", 13, TERRA, 600))
    b.append(text(52, 402, f"A shared US$80m limit for both countries would pay {100 * pb:.0f}% of expected payouts under the brief, "
                           f"{100 * pd:.0f}% under the data.", 12.5, INK, 600))
    b.append(text(52, 418, "Indicative: public data as proxies for payout triggers; not a re-pricing of the actual bond.", 11, MUTED))
    b.append(cat_sitting(1010, 400, 1.05))
    b.append(f'<path d="M 978 296 q 30 -26 62 0" fill="none" stroke="{TERRA}" stroke-width="3"/>')
    b.append(text(1010, 284, "!", 26, TERRA, 700, "middle"))
    return svg(W2, H2, "".join(b))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"cat_magnifier": magnifier(), "cat_umbrella": umbrella(), "cat_seismo": seismo(), "cat_stopwatch": stopwatch(),
             "cat_books": books(), "cat_chart": chart(), "cat_globe": globe(), "cat_table": table_sheet(),
             "cat_briefcase": briefcase(), "cat_calendar": calendar(), "cat_laptop": laptop(), "cat_team": team()}
    files["elt_pipeline"] = elt_pipeline()
    try:
        files["brief_vs_data"] = brief_vs_data()
    except FileNotFoundError:
        pass
    for k, v in files.items():
        (OUT / f"{k}.svg").write_text(v)
    print("wrote", ", ".join(files))



def elt_pipeline():
    """ELT/YLT -> footprints -> sources -> RDPO / exact."""
    from .art2 import blk, path_arrow, LB, DARK
    W2, H2 = 1180, 300
    b = []
    steps = [("Event loss table", "event · rate · loss per country", LB),
             ("Payouts", "loss → quarter steps,\nrain trigger", LB),
             ("Footprints", "events grouped by the set\nof countries they pay in", LB),
             ("Poisson sources", "exact capped 3-year law\nper footprint (Prop. 1)", "#F3E7CF"),
             ("RDPO / exact", "estimates ± CI,\nstress tests, pair exact", DARK)]
    x = 30
    for k, (t, sub, col) in enumerate(steps):
        dark = col == DARK
        b.append(box(x, 90, 190, 110, col, NAVY, 14, 1.8))
        b.append(text(x + 95, 130, t, 17, PAGE if dark else NAVY, 700, "middle"))
        for j, line in enumerate(sub.split("\n")):
            b.append(text(x + 95, 158 + 18 * j, line, 12.5, "#C9DBEF" if dark else INK2, 400, "middle"))
        if k < len(steps) - 1:
            b.append(path_arrow([(x + 192, 145), (x + 226, 145)], NAVY, 2.4))
        x += 230
    b.append(text(30, 50, "Catastrophe-model output in, joint-tail answers out", 20, NAVY, 700))
    b.append(text(30, 250, "Also accepts a year loss table: event rates = occurrences / simulated years.", 13, INK2))
    b.append(cat_peeking(1110, 92, 0.9))
    return svg(W2, H2, "".join(b))


if __name__ == "__main__":
    main()
