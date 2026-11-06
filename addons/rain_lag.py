"""Quality 1: chance-corrected rain shared share (method borrowed from QContagion's contagion_vs_common /
cds_highdim: same-day vs lagged co-occurrence, judged against a surrogate null). Additive; friend's src/ read-only.

The friend's rain_check counts an event of the rarer country as "shared" when the other country has an event within
±1 day, and reports shared / events. Two things are not separated there:
  * chance co-occurrence: two independent countries with seasonal extremes still overlap sometimes;
  * the window: ±1 day is a choice.
Null model: keep each country's events, but pair year y of the rarer country with a different year y' of the other
country (calendar dates kept, so seasonality is preserved and any same-storm link is destroyed). E0(w) = expected
matches under that null; corrected share = (observed − E0) / n_rare. Random year-derangements give the null spread
and a one-sided p-value. Window w = 0..3 days (friend: w = 1), plus w = 7 / 15 to show what pure seasonality gives.
Events, thresholds, valid years and the pair list are the friend's own (rain_check.events / shared / NEIGHBOURS).
"""
from datetime import date
from itertools import combinations

import numpy as np

from _src import in_repo, load, save
from src.realdata import rain_check as RC

WINDOWS = (0, 1, 2, 3, 7, 15)
LEVELS = (0.5, 0.2, 0.1)
N_PERM, SEED = 2000, 20261107


def country_data():
    with in_repo():
        st = RC.select_stations()
        return {c: dict(zip(("idx", "years", "stations"), RC.country_index(s))) for c, s in st.items()}


def _doy(d):
    return (d - date(d.year, 1, 1)).days


def match_matrix(rare, other, years, w):
    """M[i, j] = number of rare-country events of year years[i] matched (within w days, calendar-aligned) by an
    other-country event of year years[j]. The diagonal is the observed count."""
    yi = {y: i for i, y in enumerate(years)}
    R = [[] for _ in years]; O = [[] for _ in years]
    for s0, s1 in rare:
        R[yi[s0.year]].append((_doy(s0), _doy(s0) + (s1 - s0).days))
    for o0, o1 in other:
        O[yi[o0.year]].append((_doy(o0), _doy(o0) + (o1 - o0).days))
    M = np.zeros((len(years), len(years)), int)
    for i, rs in enumerate(R):
        for j, os_ in enumerate(O):
            if rs and os_:
                M[i, j] = sum(any(o0 - s1 <= w and s0 - o1 <= w for o0, o1 in os_) for s0, s1 in rs)
    return M


def pair_stats(ea, eb, years, rng):
    ys = set(years)
    a = [e for e in ea if e[0].year in ys]; b = [e for e in eb if e[0].year in ys]
    rare, other = (a, b) if len(a) <= len(b) else (b, a)
    n, Y = len(rare), len(years)
    out = {"n_rare": n, "n_other": len(other), "years": Y}
    perms = [rng.permutation(Y) for _ in range(N_PERM)]
    perms = [p for p in perms if not np.any(p == np.arange(Y))][:N_PERM] or perms   # derangements: y' != y
    for w in WINDOWS:
        M = match_matrix(rare, other, years, w)
        obs = int(np.trace(M))
        off = (M.sum() - obs) / max(Y - 1, 1)                    # E0 = Σ_y mean_{y'≠y} M[y, y']
        null = np.array([M[np.arange(Y), p].sum() for p in perms])
        out[f"w{w}"] = {"observed": obs, "E0": float(off), "null_q95": float(np.quantile(null, 0.95)),
                        "p_value": float((1 + (null >= obs).sum()) / (1 + len(null))),
                        "share_raw": obs / max(n, 1), "share_corrected": (obs - off) / max(n, 1)}
    return out


def run():
    ci = country_data()
    rng = np.random.default_rng(SEED)
    nb = {tuple(p) for p in RC.NEIGHBOURS}
    rows = {}
    for a, b in combinations(sorted(ci), 2):
        common = sorted(set(ci[a]["years"]) & set(ci[b]["years"]))
        if len(common) < 20:
            continue
        rows[f"{a}-{b}"] = {"neighbours": (a, b) in nb or (b, a) in nb, "levels": {}}
        for q in LEVELS:
            ea, _ = RC.events(ci[a]["idx"], ci[a]["years"], q)
            eb, _ = RC.events(ci[b]["idx"], ci[b]["years"], q)
            rows[f"{a}-{b}"]["levels"][str(q)] = pair_stats(ea, eb, common, rng)
    pooled = {}
    for grp, sel in (("neighbours", lambda r: r["neighbours"]), ("all_pairs", lambda r: True)):
        pooled[grp] = {}
        for q in LEVELS:
            pooled[grp][str(q)] = {}
            for w in WINDOWS:
                xs = [r["levels"][str(q)] for r in rows.values() if sel(r)]
                n = sum(x["n_rare"] for x in xs); obs = sum(x[f"w{w}"]["observed"] for x in xs)
                e0 = sum(x[f"w{w}"]["E0"] for x in xs)
                lo, hi = RC.wilson(obs, n)
                pooled[grp][str(q)][f"w{w}"] = {"n_rare": n, "observed": obs, "E0": e0, "share_raw": obs / max(n, 1),
                                                "share_corrected": (obs - e0) / max(n, 1),
                                                "corrected_ci95_approx": [lo - e0 / max(n, 1), hi - e0 / max(n, 1)]}
    friend = load("realdata_rain.json")["pooled"]
    for q in LEVELS:
        p = pooled["neighbours"][str(q)]["w1"]
        print(f"q={q}: friend pooled {friend[str(q)]['shared']}/{friend[str(q)]['events']} = {friend[str(q)]['share']:.1%};"
              f"  ours w=1 raw {p['observed']}/{p['n_rare']} = {p['share_raw']:.1%}, chance E0 = {p['E0']:.1f}, "
              f"corrected {p['share_corrected']:.1%}  " + "  ".join(
                  f"w{w}: {pooled['neighbours'][str(q)][f'w{w}']['share_raw']:.1%}->{pooled['neighbours'][str(q)][f'w{w}']['share_corrected']:.1%}"
                  for w in WINDOWS))
    save("quality_rain_lag.json", {"null": "year-derangement surrogate (calendar dates kept)", "n_perm": N_PERM,
                                   "seed": SEED, "windows_days": WINDOWS, "pairs": rows, "pooled": pooled})
    return pooled


def selftest():
    """Our reuse of the friend's functions must reproduce results/realdata_rain.json pair by pair."""
    ci = country_data()
    friend = load("realdata_rain.json")["pairs"]
    for a, b in RC.NEIGHBOURS:
        common = sorted(set(ci[a]["years"]) & set(ci[b]["years"]))
        for q in RC.LEVELS:
            ea, _ = RC.events(ci[a]["idx"], ci[a]["years"], q)
            eb, _ = RC.events(ci[b]["idx"], ci[b]["years"], q)
            s, f = RC.shared(ea, eb, common), friend[f"{a}-{b}"][str(q)]
            assert (s["shared"], s["events_A"], s["events_B"]) == (f["shared"], f["events_A"], f["events_B"]), (a, b, q)
            M = match_matrix(*(lambda x, y: (x, y) if len(x) <= len(y) else (y, x))(
                [e for e in ea if e[0].year in set(common)], [e for e in eb if e[0].year in set(common)]), common, 1)
            assert np.trace(M) == f["shared"], (a, b, q, np.trace(M), f["shared"])
    print("selftest ok: reproduces the friend's 10 pairs x 4 levels (shared counts and event counts), and the "
          "calendar-aligned matcher equals rain_check.shared at w = 1")


if __name__ == "__main__":
    import sys
    selftest()
    if "--selftest" not in sys.argv:
        run()
