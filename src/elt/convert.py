"""ELT / YLT -> RDPO sources: run RDPO directly on catastrophe-model output.

Catastrophe models deliver an event loss table (ELT: one row per stochastic event with an annual rate and the
loss it causes) or a year loss table (YLT: simulated years listing the events that occurred). RDPO needs the
portfolio in source form (Proposition 2): independent Poisson sources with finite, capped increment laws.

Conversion.
  1. Payout per event and country. Each event's loss (or parametric index) in country i is mapped to the cover:
     earthquake quarter steps a ∈ {0..4} (US$ loss / (EQ limit / 4), floored and capped, or given directly as
     `eq_steps`), and the rain trigger r ∈ {0, 1}.
  2. Footprints. Events are grouped by the set of countries they pay in (their footprint). Events with the same
     footprint form one compound Poisson source (rate = sum of event rates; increment = the event's payout
     vector). Footprints may contain any number of countries, not only pairs.
  3. Term law. Over a T-year term, a source's capped total increment is computed exactly on its 10^k joint code
     space by uniformisation (k = footprint size; k <= 4 here), which is Proposition 1 applied per source.
  4. The result plugs into PortfolioProblem (custom_sources) and runs through RDPO, the projected Doob sampler
     and the exact references unchanged.

YLT input: event rates are estimated as occurrences / number of simulated years, and each event's payout is
taken from its YLT rows (mean over occurrences). Loss uncertainty within an event (secondary uncertainty) is not
modelled here: the mean loss is used.

File formats (CSV, long format, one row per event and country):
  ELT: event_id, rate, country, [eq_steps | loss_usd], [rain]
  YLT: year, event_id, country, [eq_steps | loss_usd], [rain]        (plus the number of simulated years)
"""
import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

UNIT = 2.5e6


@dataclass
class Event:
    rate: float
    payout: dict            # country index -> (eq_steps 0..4, rain 0/1)


def _steps(row, eq_limit):
    if row.get("eq_steps") not in (None, ""):
        a = int(float(row["eq_steps"]))
    elif row.get("loss_usd") not in (None, ""):
        a = int(math.floor(float(row["loss_usd"]) / (eq_limit / 4) + 1e-9))
    else:
        a = 0
    return min(max(a, 0), 4), min(max(int(float(row.get("rain") or 0)), 0), 1)


def read_elt(path, countries, eq_limit=70e6):
    """ELT CSV -> list of Event. `countries` fixes the site order (list of country codes)."""
    pos = {c: i for i, c in enumerate(countries)}
    rate, pay = {}, defaultdict(dict)
    for row in csv.DictReader(open(path)):
        if row["country"] not in pos:
            continue
        e = row["event_id"]
        rate[e] = float(row["rate"])
        a, r = _steps(row, eq_limit)
        pa, pr = pay[e].get(pos[row["country"]], (0, 0))
        pay[e][pos[row["country"]]] = (max(a, pa), max(r, pr))
    return [Event(rate[e], {c: v for c, v in pay[e].items() if v != (0, 0)}) for e in rate if any(v != (0, 0) for v in pay[e].values())]


def read_ylt(path, countries, n_years, eq_limit=70e6):
    """YLT CSV -> list of Event, with rates = occurrences / n_years and payouts averaged over occurrences."""
    pos = {c: i for i, c in enumerate(countries)}
    occ = defaultdict(set)
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    for row in csv.DictReader(open(path)):
        if row["country"] not in pos:
            continue
        e = row["event_id"]; occ[e].add(row["year"])
        a, r = _steps(row, eq_limit)
        s = acc[e][pos[row["country"]]]; s[0] += a; s[1] += r; s[2] += 1
    out = []
    for e, years in occ.items():
        pay = {}
        for c, (sa, sr, k) in acc[e].items():
            a, r = int(round(sa / k)), int(round(sr / k))
            if (a, r) != (0, 0):
                pay[c] = (min(a, 4), min(r, 1))
        if pay:
            out.append(Event(len(years) / n_years, pay))
    return out


def _term_law(events, sites, years):
    """Exact law of the capped total increment of one footprint source over the term (uniformisation)."""
    k = len(sites)
    S = 10 ** k
    codes = np.array(np.unravel_index(np.arange(S), (10,) * k)).T          # (S, k) codes per site
    a0, r0 = codes // 2, codes % 2
    lam = sum(e.rate for e in events)
    # one-event transition maps: state -> state for each event
    maps, w = [], []
    for e in events:
        a = np.minimum(a0 + np.array([e.payout.get(s, (0, 0))[0] for s in sites]), 4)
        r = np.minimum(r0 + np.array([e.payout.get(s, (0, 0))[1] for s in sites]), 1)
        maps.append(np.ravel_multi_index(tuple((2 * a + r).T), (10,) * k)); w.append(e.rate / lam)
    x = lam * years
    p = np.zeros(S); p[0] = 1.0
    term, coef = p.copy(), math.exp(-x)
    out = coef * term
    for n in range(1, 400):
        nxt = np.zeros(S)
        for m, wi in zip(maps, w):
            np.add.at(nxt, m, wi * term)
        term = nxt; coef *= x / n
        out += coef * term
        if coef < 1e-18 and n > x:
            break
    return out, codes


@dataclass
class ELTModel:
    """Portfolio model built from an ELT/YLT; usable by PortfolioProblem, RDPO and the exact references."""
    n: int
    events: list
    years: float = 3.0
    eq_limit: float = 70e6
    rain_limit: float = 10e6
    max_footprint: int = 4
    custom_sources: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    footprints: dict = field(default_factory=dict)

    def __post_init__(self):
        groups = defaultdict(list)
        for e in self.events:
            fp = tuple(sorted(e.payout))
            if len(fp) > self.max_footprint:      # keep the largest payouts within the footprint limit
                keep = sorted(fp, key=lambda c: -(e.payout[c][0] + e.payout[c][1]))[:self.max_footprint]
                e = Event(e.rate, {c: e.payout[c] for c in keep}); fp = tuple(sorted(keep))
            groups[fp].append(e)
        edges = set()
        for fp, evs in sorted(groups.items()):
            law, codes = _term_law(evs, fp, self.years)
            keep = law > 1e-18
            law = law[keep] / law[keep].sum(); codes = codes[keep]
            da = [(s, codes[:, t] // 2) for t, s in enumerate(fp)]
            dr = [(s, codes[:, t] % 2) for t, s in enumerate(fp)]
            self.custom_sources.append({"kind": f"elt{len(fp)}", "sites": fp, "p": law,
                                        "logp": np.log(np.maximum(law, 1e-300)), "da": da, "dr": dr})
            self.footprints[fp] = {"events": len(evs), "rate": sum(e.rate for e in evs), "values": int(len(law))}
            for i in range(len(fp)):
                for j in range(i + 1, len(fp)):
                    edges.add((fp[i], fp[j]))
        self.edges = sorted(edges)


def ylt_term_estimates(ylt_years, n_years, years=3, payoff=None):
    """Vendor-style estimate from a YLT: split the simulated years into consecutive terms and average the payoff."""
    vals = []
    for t0 in range(0, n_years - years + 1, years):
        vals.append(payoff(t0, t0 + years))
    return float(np.mean(vals)), len(vals)
