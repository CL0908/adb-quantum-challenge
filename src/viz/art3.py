"""Commercialisation exhibits in a financial-advisory style: end-to-end data flow, market sizing,
a day of the product in an analyst's workflow, and go-to-market phases.
Run: python -m src.viz.art3 -> docs/art/*.svg (market numbers from results/market_model.json)."""
import json
import math
from pathlib import Path

from .art import svg, text, box, OUT
from .art2 import path_arrow

NAVY = "#0B2341"      # deep advisory navy
NAVY2 = "#1F3A5F"
STEEL = "#5B7FA6"
STEEL2 = "#A9BED6"
ICE = "#E6EDF5"
GOLD = "#B8893B"
GOLD2 = "#F1E4CB"
INK = "#1E2430"
GREY = "#6B7280"
LINE = "#C9CED6"
PAPER = "#FBFAF7"


def header(W, label, title, sub=None):
    out = [text(30, 34, label, 12.5, GOLD, 700),
           text(30, 62, title, 21, NAVY, 700),
           f'<line x1="30" y1="76" x2="{W - 30}" y2="76" stroke="{NAVY}" stroke-width="2"/>']
    if sub:
        out.append(text(30, 98, sub, 13, GREY, 400))
    return "".join(out)


def footer(W, H, s):
    return (f'<line x1="30" y1="{H - 34}" x2="{W - 30}" y2="{H - 34}" stroke="{LINE}" stroke-width="1"/>'
            + text(30, H - 14, s, 11, GREY, 400))


def item(x, y, w, h, title, sub=None, planned=True, dark=False, dashed=False):
    fill = NAVY if dark else (PAPER if planned else ICE)
    stroke = NAVY if dark else (STEEL if not dashed else GOLD)
    da = ' stroke-dasharray="5 4"' if dashed else ""
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.3"{da}/>']
    if not dark and not planned:
        out.append(f'<rect x="{x}" y="{y}" width="4" height="{h}" fill="{GOLD}"/>')
    tc, sc = ("#FFFFFF", STEEL2) if dark else (NAVY, GREY)
    if sub:
        out.append(text(x + 12, y + h / 2 - 3, title, 13, tc, 600))
        out.append(text(x + 12, y + h / 2 + 14, sub, 11, sc, 400))
    else:
        out.append(text(x + 12, y + h / 2 + 5, title, 13, tc, 600))
    return "".join(out)


# ------------------------------------------------------------------------------------------------
def data_flow():
    W, H = 1340, 890
    b = [header(W, "EXHIBIT 1", "End-to-end data flow: from open hazard data and daily news to an advisory agent",
                "Six stages. Solid-edged items are built in the Phase 1 prototype; the rest are planned for Deep Development and beyond.")]
    cw, gap, x0 = 196, 26, 30
    xs = [x0 + k * (cw + gap) for k in range(6)]
    heads = [("1", "DATA", "open & licensed sources"), ("2", "INGEST", "AI agents, daily"),
             ("3", "CALIBRATE", "hazard → source laws"), ("4", "ESTIMATE", "RDPO core engine"),
             ("5", "ADVISE", "agent + MCP tools"), ("6", "DELIVER", "people & systems")]
    for x, (n, h, s) in zip(xs, heads):
        b.append(f'<circle cx="{x + 15}" cy="{136}" r="14" fill="{NAVY}"/>' + text(x + 15, 141, n, 13, "#FFFFFF", 700, "middle"))
        b.append(text(x + 38, 134, h, 14, NAVY, 700))
        b.append(text(x + 38, 151, s, 11, GREY))
        b.append(f'<rect x="{x}" y="166" width="{cw}" height="3" fill="{GOLD if n == "4" else STEEL2}"/>')

    # 1 data
    groups = [("Climate & rainfall", [("ERA5 reanalysis", "hourly, 1940–present"), ("GPM IMERG / CHIRPS", "satellite rainfall"),
                                       ("Google WeatherNext 2", "64-member ensemble, 15 d")]),
              ("Earth & terrain", [("USGS earthquake catalogue", "events, magnitudes"), ("GEM hazard models", "regional seismic hazard"),
                                   ("Copernicus DEM", "terrain, slope, basins")]),
              ("Exposure & contracts", [("Population & assets", "WorldPop, OpenStreetMap"), ("Bond & pool terms", "limits, triggers, term")]),
              ("Live signals", [("Daily news & reports", "wires, GDELT, ReliefWeb")])]
    y = 186
    for g, its in groups:
        b.append(text(xs[0], y + 12, g.upper(), 10.5, GOLD, 700))
        y += 20
        for t, s in its:
            built = t == "Bond & pool terms"
            b.append(item(xs[0], y, cw, 40, t, s, planned=not built))
            y += 45
        y += 6
    data_bottom = y

    # 2 ingest agents
    b.append(item(xs[1], 206, cw, 150, "", None))
    b.append(text(xs[1] + 12, 230, "News-scout agent", 14, NAVY, 700))
    for k, s in enumerate(["scrapes news every morning", "extracts events: type, place,", "  magnitude, damage", "geocodes + credibility score",
                           "flags portfolio-relevant events"]):
        b.append(text(xs[1] + 14, 254 + 19 * k, ("· " if not s.startswith("  ") else "  ") + s.strip(), 11.5, INK))
    b.append(item(xs[1], 376, cw, 150, "", None))
    b.append(text(xs[1] + 12, 400, "Data agent", 14, NAVY, 700))
    for k, s in enumerate(["pulls hazard & forecast feeds", "quality checks, gap filling", "versioned daily snapshots",
                           "licence & provenance log"]):
        b.append(text(xs[1] + 14, 424 + 19 * k, "· " + s, 11.5, INK))
    b.append(item(xs[1], 546, cw, 88, "Event & hazard store", "versioned, queryable"))

    # 3 calibrate
    cal = [("Event rates & severities", "per peril, per country"), ("Source laws p_k", "finite capped increments"),
           ("Dependence graph", "shared-event rates on edges"), ("Scenario generator", "stress + forecast-conditioned")]
    for k, (t, s) in enumerate(cal):
        b.append(item(xs[2], 206 + 82 * k, cw, 62, t, s, planned=k not in (1, 2)))
    b.append(text(xs[2], 556, "Phase 1: rates taken from", 11, GREY))
    b.append(text(xs[2], 572, "the ADB brief; Phase 2: fitted", 11, GREY))
    b.append(text(xs[2], 588, "to the open data in stage 1.", 11, GREY))

    # 4 estimate (core, navy)
    b.append(f'<rect x="{xs[3] - 6}" y="196" width="{cw + 12}" height="452" rx="4" fill="{GOLD2}" stroke="{GOLD}" stroke-width="1.5"/>')
    b.append(item(xs[3], 206, cw, 140, "", None, dark=True))
    b.append(text(xs[3] + 12, 232, "RDPO engine", 16, "#FFFFFF", 700))
    for k, s in enumerate(["learned change of measure", "KL → χ² policy training", "fixed path budget", "unbiased estimate + 95% CI"]):
        b.append(text(xs[3] + 14, 256 + 20 * k, "· " + s, 12, STEEL2))
    b.append(item(xs[3], 360, cw, 58, "Exact checks", "enumeration · tilted MPS", planned=False))
    b.append(item(xs[3], 430, cw, 58, "Diagnostics", "χ̂², ESS, coverage", planned=False))
    b.append(item(xs[3], 500, cw, 58, "GPU batch runner", "thousands of scenarios", planned=True))
    b.append(item(xs[3], 570, cw, 66, "Quantum back end", "future · trust-guarded", dashed=True))

    # 5 advise
    b.append(item(xs[4], 206, cw, 170, "", None))
    b.append(text(xs[4] + 12, 230, "Advisory agent", 14, NAVY, 700))
    for k, s in enumerate(["answers analyst questions", "plans stress tests", "compares to yesterday", "drafts recommendations",
                           "cites evidence + caveats", "never auto-executes"]):
        b.append(text(xs[4] + 14, 254 + 19 * k, "· " + s, 11.5, INK))
    b.append(text(xs[4], 404, "MCP TOOL SERVER", 10.5, GOLD, 700))
    for k, t in enumerate(["estimate_joint_tail()", "stress_test()", "validate_exact()", "event_impact()"]):
        b.append(f'<rect x="{xs[4]}" y="{414 + 44 * k}" width="{cw}" height="36" rx="3" fill="{PAPER}" stroke="{STEEL}" stroke-width="1.2"/>')
        b.append(text(xs[4] + 12, 437 + 44 * k, t, 12.5, NAVY, 500, family="'JetBrains Mono', Menlo, monospace"))

    # 6 deliver
    users = [("Joint-tail dashboard", "portfolio, scenarios, deltas"), ("Morning risk brief", "events + exposure impact"),
             ("Recommendation memo", "structuring, pricing, triggers"), ("API & MCP clients", "Excel, Oasis, other agents")]
    for k, (t, s) in enumerate(users):
        b.append(item(xs[5], 206 + 76 * k, cw, 58, t, s))
    b.append(text(xs[5], 532, "USERS", 10.5, GOLD, 700))
    for k, s in enumerate(["DRF officers (MDBs, pools)", "reinsurance & ILS analysts", "bond structurers, brokers", "regulators, rating agencies"]):
        b.append(text(xs[5], 554 + 19 * k, "· " + s, 11.5, INK))

    # flows
    mid = lambda k: xs[k] + cw
    b.append(path_arrow([(mid(0), 300), (xs[1], 300)], NAVY2, 2))
    b.append(path_arrow([(mid(0), 450), (xs[1], 450)], NAVY2, 2))
    b.append(path_arrow([(mid(0), data_bottom - 40), (mid(0) + 13, data_bottom - 40), (mid(0) + 13, 280), (xs[1], 280)], NAVY2, 1.6))
    b.append(path_arrow([(xs[1] + cw / 2, 526), (xs[1] + cw / 2, 546)], NAVY2, 2))
    b.append(path_arrow([(mid(1), 590), (mid(1) + 13, 590), (mid(1) + 13, 237), (xs[2], 237)], NAVY2, 2))
    for k in range(3):
        b.append(path_arrow([(xs[2] + cw / 2, 268 + 82 * k), (xs[2] + cw / 2, 288 + 82 * k)], NAVY2, 1.6))
    b.append(path_arrow([(mid(2), 400), (xs[3] - 6, 400)], NAVY2, 2))
    b.append(path_arrow([(xs[4], 470), (mid(3) + 6, 470)], NAVY2, 2))
    b.append(path_arrow([(mid(3) + 6, 300), (xs[4], 300)], NAVY2, 2))
    b.append(path_arrow([(mid(4), 290), (xs[5], 290)], NAVY2, 2))
    # news shortcut: event impact path
    b.append(path_arrow([(xs[1] + cw / 2, 206), (xs[1] + cw / 2, 186), (xs[4] + cw / 2, 186), (xs[4] + cw / 2, 206)], GOLD, 1.8, dash="6 4"))
    b.append(text(xs[2] + 30, 181, "breaking event → same-day impact estimate", 11.5, GOLD, 600))

    # feedback + governance band
    yb = 720
    b.append(path_arrow([(xs[5] + cw / 2, 640), (xs[5] + cw / 2, yb - 8), (xs[1] + cw / 2, yb - 8), (xs[1] + cw / 2, 648)], STEEL, 1.6, dash="4 4"))
    b.append(text(xs[3] - 40, yb - 14, "analyst feedback, new questions, approved decisions", 11.5, STEEL, 600))
    b.append(f'<rect x="30" y="{yb + 8}" width="{W - 60}" height="70" rx="3" fill="{NAVY}"/>')
    b.append(text(48, yb + 36, "GOVERNANCE", 12, GOLD, 700))
    gov = ["human approves every recommendation", "audit trail of data & model versions", "estimates with stated uncertainty",
           "data licences & provenance", "validation pack for regulators"]
    for k, s in enumerate(gov):
        x = 182 + k * 226
        b.append(f'<circle cx="{x}" cy="{yb + 32}" r="3.5" fill="{GOLD}"/>' + text(x + 10, yb + 37, s, 11, "#FFFFFF", 500))
    b.append(text(48, yb + 62, "Every number the agent reports comes from a tool call to the estimation engine; language models never generate risk numbers.", 11.5, STEEL2, 400))
    # legend
    ly = yb + 104
    b.append(f'<rect x="30" y="{ly - 12}" width="22" height="14" fill="{ICE}" stroke="{STEEL}"/><rect x="30" y="{ly - 12}" width="4" height="14" fill="{GOLD}"/>')
    b.append(text(60, ly, "built in Phase 1 prototype", 11.5, INK))
    b.append(f'<rect x="250" y="{ly - 12}" width="22" height="14" fill="{PAPER}" stroke="{STEEL}"/>' + text(280, ly, "planned", 11.5, INK))
    b.append(f'<rect x="370" y="{ly - 12}" width="22" height="14" fill="{PAPER}" stroke="{GOLD}" stroke-dasharray="4 3"/>' + text(400, ly, "future, gated by evidence", 11.5, INK))
    b.append(footer(W, H, "Data sources named are examples of open or licensable datasets; inclusion is subject to licence terms. "
                          "WeatherNext 2 data are distributed via Google Earth Engine / Cloud on request."))
    return svg(W, H, "".join(b), bg=PAPER)


# ------------------------------------------------------------------------------------------------
def market_sizing():
    m = json.loads(Path("results/market_model.json").read_text())
    W, H = 1340, 640
    b = [header(W, "EXHIBIT 3", "Market sizing: a focused, bottom-up serviceable market inside a US$2bn software category",
                "TAM is a third-party top-down estimate; SAM and SOM are bottom-up (institutions × annual contract value).")]
    cx, cy = 300, 370
    circles = [(230, ICE, STEEL2, "TAM", f"US${m['tam_bn']:.1f}bn", "cat-modelling & exposure software, 2025"),
               (140, STEEL2, STEEL, "SAM", f"US${m['sam_m']:.0f}m / yr", "joint-tail analytics buyers"),
               (66, NAVY, NAVY, "SOM", f"US${m['som_m']:.1f}m", f"ARR by year 5")]
    for r, f, s, *_ in circles:
        b.append(f'<circle cx="{cx}" cy="{cy + 230 - r}" r="{r}" fill="{f}" stroke="{s}" stroke-width="1.5"/>')
    b.append(text(cx, cy - 175, "TAM", 13, GOLD, 700, "middle"))
    b.append(text(cx, cy - 150, f"US${m['tam_bn']:.1f}bn", 26, NAVY, 700, "middle"))
    b.append(text(cx, cy - 128, "cat-modelling & exposure software, 2025", 11.5, GREY, 400, "middle"))
    b.append(text(cx, cy + 18, "SAM", 13, GOLD, 700, "middle"))
    b.append(text(cx, cy + 44, f"US${m['sam_m']:.0f}m / yr", 22, NAVY, 700, "middle"))
    b.append(text(cx, cy + 64, "joint-tail analytics buyers", 11.5, NAVY2, 500, "middle"))
    b.append(text(cx, cy + 162, "SOM", 12, GOLD, 700, "middle"))
    b.append(text(cx, cy + 186, f"US${m['som_m']:.1f}m", 19, "#FFFFFF", 700, "middle"))
    b.append(text(cx, cy + 204, f"ARR, year 5", 11, STEEL2, 400, "middle"))

    # segment table with bars
    x0, y0 = 590, 140
    b.append(text(x0, y0, "SERVICEABLE MARKET BY SEGMENT", 11.5, GOLD, 700))
    cols = [(x0, "Segment"), (x0 + 300, "Institutions"), (x0 + 390, "ACV"), (x0 + 460, "SAM (US$m)"), (x0 + 600, "Won by year 5")]
    for x, t in cols:
        b.append(text(x, y0 + 30, t, 11.5, GREY, 600))
    b.append(f'<line x1="{x0}" y1="{y0 + 40}" x2="{W - 30}" y2="{y0 + 40}" stroke="{NAVY}" stroke-width="1.2"/>')
    mx = max(r["sam_m"] for r in m["segments"])
    for k, r in enumerate(m["segments"]):
        y = y0 + 72 + 56 * k
        name = r["segment"]
        if len(name) > 34:
            cut = name.rfind(" ", 0, 34)
            b.append(text(x0, y - 7, name[:cut], 13, NAVY, 600)); b.append(text(x0, y + 10, name[cut + 1:], 13, NAVY, 600))
        else:
            b.append(text(x0, y, name, 13, NAVY, 600))
        b.append(text(x0 + 300, y, f"{r['institutions']}", 13, INK))
        b.append(text(x0 + 390, y, f"${r['acv_k']}k", 13, INK))
        bw = 100 * r["sam_m"] / mx
        b.append(f'<rect x="{x0 + 460}" y="{y - 13}" width="{bw:.1f}" height="16" fill="{STEEL}"/>')
        b.append(text(x0 + 466 + bw, y, f"{r['sam_m']:.1f}", 12.5, INK, 600))
        sw = 100 * r["som_m"] / mx
        b.append(f'<rect x="{x0 + 600}" y="{y - 13}" width="{max(sw, 2):.1f}" height="16" fill="{NAVY}"/>')
        b.append(text(x0 + 606 + sw, y, f"{r['won_y5']} · ${r['som_m']:.1f}m", 12.5, INK, 600))
        b.append(f'<line x1="{x0}" y1="{y + 22}" x2="{W - 30}" y2="{y + 22}" stroke="{LINE}" stroke-width="0.8"/>')
    yt = y0 + 72 + 56 * len(m["segments"]) - 6
    b.append(text(x0, yt, "Total", 13.5, NAVY, 700))
    b.append(text(x0 + 300, yt, f"{sum(r['institutions'] for r in m['segments'])}", 13.5, NAVY, 700))
    b.append(text(x0 + 460, yt, f"{m['sam_m']:.1f}", 13.5, NAVY, 700))
    b.append(text(x0 + 600, yt, f"{m['customers_y5']} · ${m['som_m']:.1f}m", 13.5, NAVY, 700))
    b.append(text(x0 + 600, yt + 18, f"({100 * m['som_share_of_sam']:.0f}% of SAM)", 11.5, GREY))
    b.append(text(x0, yt + 34, "Context (risk capital the analytics serve): cat bonds outstanding US$61.3bn (2025); parametric", 11.5, GREY))
    b.append(text(x0, yt + 52, "premiums ≈ US$19bn (2025); uninsured nat-cat losses (protection gap) US$181bn (2024).", 11.5, GREY))
    b.append(footer(W, H, "Sources: TrendX Insights / Market Intelo (software, 2025); Artemis.bm (cat bonds, end-2025); Global Market Insights (parametric, 2025); "
                          "Swiss Re Institute sigma 1/2025. ACVs and counts: author assumptions (src/business/market.py)."))
    return svg(W, H, "".join(b), bg=PAPER)


# ------------------------------------------------------------------------------------------------
def workflow_day():
    W, H = 1340, 560
    b = [header(W, "EXHIBIT 4", "A day with the product: agents prepare, the analyst decides",
                "How the tool fits the daily routine of a disaster-risk-financing or reinsurance analyst.")]
    x0, x1 = 170, W - 40
    times = ["07:00", "08:30", "09:30", "11:00", "14:00", "17:00"]
    xs = [x0 + (x1 - x0 - 150) * k / 5 for k in range(6)]
    lanes = [(160, "AGENTS", "automatic"), (330, "ANALYST", "human decision")]
    for y, l, s in lanes:
        b.append(f'<rect x="30" y="{y - 40}" width="{W - 60}" height="130" fill="{ICE if l == "AGENTS" else GOLD2}" opacity="0.55"/>')
        b.append(text(44, y, l, 13, NAVY, 700)); b.append(text(44, y + 18, s, 11.5, GREY))
    b.append(f'<line x1="{x0 - 10}" y1="{H - 76}" x2="{x1}" y2="{H - 76}" stroke="{NAVY}" stroke-width="1.5"/>')
    steps = [
        (0, 0, "News scan", ["overnight events found,", "geocoded, scored"]),
        (1, 0, "Hazard refresh", ["rainfall, quake catalogue,", "WeatherNext ensemble"]),
        (2, 0, "Portfolio re-estimate", ["RDPO joint tails ± CI;", "change vs yesterday"]),
        (2, 1, "Morning brief", ["reads 1-page summary,", "flags what matters"]),
        (3, 1, "Ask the agent", ["“both bonds exhausted if", "shared rate doubles?”"]),
        (3, 0, "Stress test", ["tool calls + exact", "cross-check"]),
        (4, 1, "Recommendation", ["pricing / trigger memo;", "approves or edits"]),
        (5, 0, "Trigger watch", ["monitor active events", "against bond triggers"]),
        (5, 1, "Sign-off", ["decision logged with", "data & model versions"]),
    ]
    for t, lane, head, lines in steps:
        x = xs[t]; y = lanes[lane][0] - 30
        dark = lane == 1
        b.append(f'<rect x="{x}" y="{y}" width="150" height="100" rx="3" fill="{NAVY if dark else PAPER}" stroke="{NAVY if dark else STEEL}" stroke-width="1.3"/>')
        b.append(text(x + 10, y + 24, head, 13.5, "#FFFFFF" if dark else NAVY, 700))
        for k, s in enumerate(lines):
            b.append(text(x + 10, y + 48 + 18 * k, s, 11.5, STEEL2 if dark else INK))
    for k, tm in enumerate(times):
        b.append(f'<line x1="{xs[k] + 75}" y1="{H - 82}" x2="{xs[k] + 75}" y2="{H - 70}" stroke="{NAVY}" stroke-width="1.5"/>')
        b.append(text(xs[k] + 75, H - 52, tm, 13, NAVY, 700, "middle"))
    # arrows between lanes
    b.append(path_arrow([(xs[2] + 75, 230), (xs[2] + 75, 300)], GOLD, 1.8))
    b.append(path_arrow([(xs[3] + 75, 300), (xs[3] + 75, 230)], GOLD, 1.8))
    b.append(path_arrow([(xs[3] + 150, 340), (xs[4], 340)], GOLD, 1.8))
    b.append(path_arrow([(xs[4] + 150, 340), (xs[5], 340)], GOLD, 1.8))
    for k in range(2):
        b.append(path_arrow([(xs[k] + 150, 170), (xs[k + 1], 170)], STEEL, 1.6))
    b.append(text(30, H - 14, "Illustrative workflow for the planned product; times indicative.", 11, GREY))
    return svg(W, H, "".join(b), bg=PAPER)


# ------------------------------------------------------------------------------------------------
def gtm():
    W, H = 1340, 400
    b = [header(W, "EXHIBIT 5", "Go-to-market: prove it with public institutions, scale it with private risk capital")]
    phases = [("PHASE A · 2027", "Prove", "Pilot with ADB / DRF teams", ["ADB bond portfolio analytics", "open-source library + paper", "validation pack"],
               "Revenue: grants, technical-assistance contracts"),
              ("PHASE B · 2028", "Productise", "SaaS + agent for analysts", ["web app, MCP tools, API", "first reinsurer & ILS clients", "Oasis LMF plug-in"],
               "Revenue: subscriptions (ACV US$50–250k)"),
              ("PHASE C · 2029+", "Scale", "Platform & partnerships", ["licences to model vendors/brokers", "new perils & regions", "quantum back end if s* met"],
               "Revenue: licences, enterprise, advisory")]
    w = (W - 60 - 2 * 30) / 3
    for k, (lab, verb, head, items, rev) in enumerate(phases):
        x = 30 + k * (w + 30)
        b.append(f'<polygon points="{x},{120} {x + w - 18},{120} {x + w},{148} {x + w - 18},{176} {x},{176} {x + (18 if k else 0)},{148}" fill="{[STEEL, NAVY2, NAVY][k]}"/>')
        b.append(text(x + 30, 143, lab, 11.5, GOLD2, 700)); b.append(text(x + 30, 166, verb, 17, "#FFFFFF", 700))
        b.append(text(x + 4, 208, head, 14, NAVY, 700))
        for j, s in enumerate(items):
            b.append(f'<circle cx="{x + 9}" cy="{232 + 22 * j}" r="3" fill="{GOLD}"/>' + text(x + 20, 237 + 22 * j, s, 12.5, INK))
        b.append(f'<line x1="{x}" y1="{312}" x2="{x + w}" y2="{312}" stroke="{LINE}"/>')
        b.append(text(x + 4, 334, rev, 12, NAVY2, 600))
    b.append(footer(W, H, "Phases are indicative and follow the Deep Development work plan (Section 9)."))
    return svg(W, H, "".join(b), bg=PAPER)


# ------------------------------------------------------------------------------------------------
def wake_sleep():
    W, H = 1340, 760
    b = [header(W, "EXHIBIT 2", "Wake–sleep operating cycle: serve by day, consolidate and self-improve by night",
                "Adapted from the wake–sleep framework for continual learning (Behrouz et al., 2026); what is consolidated here is the risk model, not the language model.")]
    # wake panel
    wx, wy, ww, wh = 30, 170, 520, 450
    b.append(f'<rect x="{wx}" y="{wy}" width="{ww}" height="{wh}" rx="4" fill="{ICE}" stroke="{STEEL}" stroke-width="1.3"/>')
    b.append(text(wx + 18, wy + 30, "WAKE PHASE", 13, GOLD, 700)); b.append(text(wx + 18, wy + 54, "Serving the day: frozen, validated policies", 16, NAVY, 700))
    ins = [("Daily news & reports", "news-scout agent"), ("Hazard & forecast feeds", "data agent"), ("Analyst questions", "advisory agent")]
    for k, (t, sub) in enumerate(ins):
        b.append(item(wx + 18, wy + 80 + 62 * k, 200, 50, t, sub))
        b.append(path_arrow([(wx + 218, wy + 105 + 62 * k), (wx + 268, wy + 105 + 62 * k)], NAVY2, 1.6))
    b.append(item(wx + 268, wy + 80, 232, 174, "", None, dark=True))
    b.append(text(wx + 282, wy + 106, "RDPO serving", 15, "#FFFFFF", 700))
    for k, t in enumerate(["today's policy library θ_t", "estimates ± CI on demand", "same-day event impact", "exact checks where possible",
                           "unbiased for any θ"]):
        b.append(text(wx + 284, wy + 130 + 21 * k, "· " + t, 12, STEEL2))
    b.append(text(wx + 18, wy + 300, "SHORT-TERM MEMORY (written all day)", 11, GOLD, 700))
    mem = [("Event log", "new events, credibility"), ("Query log", "targets & scenarios asked"), ("Path buffer", "sampled paths + weights")]
    for k, (t, sub) in enumerate(mem):
        b.append(item(wx + 18 + 166 * k, wy + 312, 152, 56, t, sub))
    b.append(path_arrow([(wx + 384, wy + 254), (wx + 384, wy + 312)], NAVY2, 1.6))
    b.append(text(wx + 18, wy + 410, "Nothing in the risk model changes during the day: every number", 12, INK))
    b.append(text(wx + 18, wy + 428, "served is reproducible from a versioned policy and source model.", 12, INK))
    # sleep panel
    sx, sy, sw, sh = 600, 170, 710, 450
    b.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" rx="4" fill="{NAVY}"/>')
    b.append(text(sx + 18, sy + 30, "SLEEP PHASE", 13, GOLD, 700)); b.append(text(sx + 18, sy + 54, "Overnight: consolidate, dream, then gate", 16, "#FFFFFF", 700))
    stages = [("1  Consolidation", ["event log → credibility-weighted", "rate updates (analyst-approved)", "new shared events → graph edges",
                                    "path buffer → warm start θ"]),
              ("2  Dreaming", ["synthetic stress scenarios where", "today's χ² was highest (RL curriculum)", "retrain KL → χ² from θ_t",
                               "grow features only if χ² stalls"]),
              ("3  Gate (no forgetting)", ["regression: 25 exact ADB values,", "tensor-network references,", "coverage on fresh replications",
                                           "promote θ_t+1 or roll back"])]
    cwid = (sw - 36 - 2 * 16) / 3
    for k, (t, lines) in enumerate(stages):
        x = sx + 18 + k * (cwid + 16)
        b.append(f'<rect x="{x}" y="{sy + 80}" width="{cwid}" height="180" rx="3" fill="{NAVY2}" stroke="{STEEL}" stroke-width="1"/>')
        b.append(text(x + 12, sy + 106, t, 14, "#FFFFFF", 700))
        for j, l in enumerate(lines):
            b.append(text(x + 12, sy + 134 + 22 * j, l, 11.2, STEEL2))
        if k < 2:
            b.append(path_arrow([(x + cwid, sy + 170), (x + cwid + 16, sy + 170)], GOLD, 1.8))
    b.append(f'<rect x="{sx + 18}" y="{sy + 280}" width="{sw - 36}" height="56" rx="3" fill="{GOLD2}"/>')
    b.append(text(sx + 34, sy + 304, "Long-term memory (versioned)", 13, NAVY, 700))
    b.append(text(sx + 34, sy + 324, "source laws p_k · dependence graph · policy library θ per target and scenario · validation record", 12, NAVY2))
    b.append(text(sx + 18, sy + 380, "Why this is safe: the estimator stays unbiased whatever θ is learned, so a weaker", 12, STEEL2))
    b.append(text(sx + 18, sy + 400, "overnight policy can only widen intervals; the gate rejects it before morning.", 12, STEEL2))
    # cycle arrows
    b.append(path_arrow([(wx + ww - 60, wy - 6), (wx + ww - 60, wy - 22), (sx + 60, wy - 22), (sx + 60, wy - 6)], GOLD, 2))
    b.append(text((wx + ww + sx) / 2, wy - 28, "night: short-term memory handed over", 12, GOLD, 600, "middle"))
    b.append(path_arrow([(sx + 60, sy + sh + 6), (sx + 60, sy + sh + 24), (wx + ww - 60, sy + sh + 24), (wx + ww - 60, sy + sh + 6)], GOLD, 2))
    b.append(text((wx + ww + sx) / 2, sy + sh + 42, "morning: promoted policies θ_t+1 + updated model + morning brief", 12, GOLD, 600, "middle"))
    b.append(footer(W, H, "Planned for WP3 (overnight retraining, capacity growth) and WP5 (agents). Phase 1 already provides the gate's exact references and the warm-start mechanism."))
    return svg(W, H, "".join(b), bg=PAPER)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"data_flow": data_flow(), "wake_sleep": wake_sleep(), "market_sizing": market_sizing(), "workflow_day": workflow_day(), "gtm": gtm()}
    for k, v in files.items():
        (OUT / f"{k}.svg").write_text(v)
    print("wrote", ", ".join(files))


if __name__ == "__main__":
    main()
