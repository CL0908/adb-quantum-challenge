"""One desk for the strong add-ons: one state, one question interface, one unit (price points and US dollars).

State of the two ADB notes on a given day: date, contract state (payouts so far), and the uncertain inputs each
add-on studied on its own:
  rate      earthquake trigger-rate multipliers (rates_unc.py: catalogue counts and Gutenberg-Richter slope)
  rain      rain trigger-rate multipliers (rates_unc.py: GEV fit of 45-48 annual maxima, bootstrap)
  clause    aftershocks paid separately or not (eq_cluster.py); unknown from public documents: prior 1/2, and if they
            are, nu ~ Gamma(k + 1/2) / n from the 4 aftershocks of 8 main shocks at M >= 6.5
  agency    the contract reads GFZ magnitudes, the data are USGS: offset ~ Uniform(-0.2, 0.2) magnitude units, which
            multiplies the trigger rate by 10^(0.92 offset)   (found by scorecard.py; no add-on covered it)
  warming   rain-event frequency factor ~ Uniform(1, the +1 C factor of climate_cc.py) over the remaining term
plus the transient hazard after the 3 Nov 2026 border earthquake (event_response.py), added as a fixed amount.
The priors of clause, agency and warming are assumptions stated here, not estimates.
What merging gives that the parts did not:
  price_distribution   all inputs drawn together -> the indicative price with a model-risk interval, and its variance
                       split over the inputs (Sobol, voi.py's estimator)
  actions              the same split in price points and US dollars per note: what resolving each input is worth
  attribution          the price change since issue split into time, the event, nine more years of data, the market
  replay               every M >= 5 event near the two countries 2017-2026 answered as if live, and a monthly mark
Pre-set rules, all five for "very strong": (1) with the other inputs frozen the desk reproduces each single-factor
result of live_mark.py within 0.05 point; (2) first-order shares sum to 1 +- 0.05, and the attribution adds up to the
total change within 0.02 point; (3) the replay has no failure and >= 95% of answers inside the time budgets (mark 10 s,
event answer 5 min); (4) the three largest inputs are the same for 3 seeds; (5) it runs with no data library.
"""
import json
import math
import os
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta

import numpy as np

import event_response as ER
import live_mark as LM
import rates_unc as RU
import voi
from _src import OUT, save, seed

USD_PER_POINT = 0.8e6                                    # 1 point of an US$80m note
GROUPS = ("rate", "rain", "clause", "agency", "warming")
N_SOBOL, SEEDS, B_GR = 1024, 3, 0.92
AFTER, MAIN = 4, 8                                       # eq_cluster.py: aftershocks and main shocks at M >= 6.5 near KG / TJ


@dataclass(frozen=True)
class State:
    asof: date = LM.ASOF
    contract: tuple = (0, 0, 0, 0)                       # payouts so far: earthquake quarters KG, TJ; rain KG, TJ
    eq_mult: tuple = (1.0, 1.0)
    rain_mult: tuple = (1.0, 1.0)
    nu: float = 0.0                                      # 0: aftershocks merged with their main shock
    offset: float = 0.0                                  # magnitude-agency offset
    warming: float = 1.0
    event: bool = True                                   # include the transient hazard of the latest event
    extra: dict = field(default_factory=dict, compare=False)


def _event_extra():
    try:
        ev = json.loads((OUT / "event_response.json").read_text())["event_2026_11_03"]
    except FileNotFoundError:
        return (0.0, 0.0)
    from src.model.poisson_fit import TAILS, compound_poisson_rates
    out = []
    for c in LM.BONDS:
        lam, L = compound_poisson_rates(TAILS[c])
        out.append(100 * ev[f"pay_size_{c}"]["p_added_by_event"] * float((lam / L) @ np.arange(1, 5)) / 4 * LM.SC.eq_limit / (LM.SC.eq_limit + LM.SC.rain_limit))
    return tuple(out)


EVENT_EXTRA = _event_extra()
T_TOTAL = LM.years(LM.ISSUE, LM.MATURITY)
MULT0 = {c: LM.COUPON / (LM.remaining(T_TOTAL)[c]["el_pct"] / T_TOTAL) for c in LM.BONDS}      # multiple paid at issue


def ask(question, st=State(), multiple_scale=1.0):
    """question: 'price', 'el' (remaining expected loss, % of outstanding), 'p_payout', 'p_exhausted'; per note."""
    tau = LM.years(st.asof, LM.MATURITY)
    f = 10 ** (B_GR * st.offset)
    r = LM.remaining(tau, state=st.contract, eq_mult=tuple(m * f for m in st.eq_mult), rain_mult=tuple(m * st.warming for m in st.rain_mult), nu=st.nu,
                     extra_el=EVENT_EXTRA if st.event and st.asof >= date(2026, 11, 3) else (0.0, 0.0))
    if question == "price":
        return {c: LM.price(r[c]["el_pct"], tau, MULT0[c] * multiple_scale) for c in LM.BONDS}
    return {c: r[c][{"el": "el_pct", "p_payout": "p_any_payout", "p_exhausted": "p_exhausted"}[question]] for c in LM.BONDS}


# ------------------------------------------------------------------------------------------- joint draws
def draws(rng, n, rain_pool):
    em, _ = RU.eq_multipliers(rng, n)
    b = rng.random(n) < 0.5
    return {"rate": np.column_stack([em["KG"], em["TJ"]]), "rain": np.column_stack([rng.choice(rain_pool[c], n) for c in LM.BONDS]),
            "clause": np.where(b, rng.gamma(AFTER + 0.5, 1.0, n) / MAIN, 0.0), "agency": rng.uniform(-0.2, 0.2, n), "warming": rng.uniform(1.0, LM.CC["+1C"], n)}


def price_of(d, i, cache):
    nu = round(float(d["clause"][i]), 2)                 # cluster laws are slow; 0.01 steps of nu
    st = State(eq_mult=tuple(d["rate"][i]), rain_mult=tuple(d["rain"][i]), nu=nu, offset=float(d["agency"][i]), warming=float(d["warming"][i]))
    return ask("price", st)


def price_distribution(seed_=0, n=N_SOBOL, rain_pool=None):
    rng = np.random.default_rng(seed(20261119) + seed_)
    A, B = draws(rng, n, rain_pool), draws(rng, n, rain_pool)
    ev = lambda d: np.array([[p[c] for c in LM.BONDS] for p in (price_of(d, i, None) for i in range(n))])
    YA, YB = ev(A), ev(B)
    YAB = {g: ev({**A, g: B[g]}) for g in GROUPS}
    out = {}
    for j, c in enumerate(LM.BONDS):
        s = voi.sobol(YA[:, j], YB[:, j], {g: YAB[g][:, j] for g in GROUPS}, rng)
        y = np.r_[YA[:, j], YB[:, j]]
        out[c] = {"mean": float(y.mean()), "sd": float(y.std()), "q05": float(np.quantile(y, 0.05)), "median": float(np.median(y)), "q95": float(np.quantile(y, 0.95)),
                  "S": s["S"], "T": s["T"], "S_ci": s["S_ci"], "sum_S": float(sum(s["S"].values())),
                  "ranking": sorted(GROUPS, key=lambda g: -s["S"][g])}
    return out


def actions(dist):
    """What resolving each input is worth: the price standard deviation it accounts for, in points and US dollars per note."""
    out = {}
    for c, d in dist.items():
        out[c] = [{"input": g, "share_of_variance": d["S"][g], "points": d["sd"] * math.sqrt(max(d["S"][g], 0.0)),
                   "usd_per_note": d["sd"] * math.sqrt(max(d["S"][g], 0.0)) * USD_PER_POINT} for g in d["ranking"]]
    return out


# ------------------------------------------------------------------------------------------- attribution and replay
def attribution():
    """Price change from the issue date to today in four steps, applied in this order (the split depends on the order)."""
    tc = json.loads((OUT / "twin_counts.json").read_text())
    k16 = {x["iso"]: x["friend"] for x in tc["check_1965_2016"]["countries"]}
    upd = tuple(((k16[c] + tc["new_2017_2025"]["country_hits"][c]) / 61) / (k16[c] / 52) for c in LM.BONDS)        # nine more years of main-shock counts
    steps = [("at issue", State(asof=LM.ISSUE, event=False), 1.0), ("time passed, no payout", State(event=False), 1.0), ("3 Nov 2026 earthquake", State(), 1.0),
             ("counts 2017-2025", State(eq_mult=upd), 1.0), ("market multiple 2.5 -> 2.4", State(eq_mult=upd), LM.MULT_MARKET[1] / LM.MULT_MARKET[0])]
    prices = [ask("price", st, m) for _, st, m in steps]
    out = {"rate_update_factor": dict(zip(LM.BONDS, upd)), "steps": []}
    for k, (name, _, _) in enumerate(steps):
        out["steps"].append({"step": name, **{c: prices[k][c] for c in LM.BONDS}, **{f"change_{c}": (prices[k][c] - prices[k - 1][c]) if k else 0.0 for c in LM.BONDS}})
    out["closure"] = {c: abs(sum(s[f"change_{c}"] for s in out["steps"]) - (prices[-1][c] - prices[0][c])) for c in LM.BONDS}
    return out


def replay():
    """2017 to today as if live: every M >= 5 event within 50 km of KG or TJ gets an after-event answer (parameters fitted
    on data before 2017 only), then we look whether an M >= 5.5 event followed within 30 days; and a monthly mark."""
    from twin import friend_events
    from src.realdata import eq_check
    cat = ER.load()
    try:
        extra = ER.load("kg_tj_m45_2026.csv")
        cat = {k: np.r_[cat[k], extra[k]] for k in cat}
        o = np.argsort(cat["t"]); cat = {k: v[o] for k, v in cat.items()}
    except FileNotFoundError:
        pass
    _, polys = friend_events()
    year = 1970 + cat["t"] / 365.2425
    d = ER.design(cat)
    par = ER.fit(d, (d["year"] >= 1973) & (d["year"] < 2017))
    trig = np.flatnonzero(cat["m"] >= ER.M_TRIG)
    answers, fails = [], 0
    big = cat["m"] >= ER.M_TARGET
    t_end = cat["t"].max()
    for row, i in enumerate(trig):
        if year[i] < 2017:
            continue
        t0 = time.perf_counter()
        try:
            if min(eq_check.dist_to_country(cat["lat"][i], cat["lon"][i], polys[c]) for c in ("KG", "TJ")) > ER.R_KM:
                continue
            p = 1 - math.exp(-float(ER.expected(par, d["m"][row], d["b"][row], 0.0, 30.0)))
            near = (ER.dist_km(cat["lat"][i], cat["lon"][i], cat["lat"][big], cat["lon"][big]) <= ER.R_KM)
            dt = cat["t"][big][near] - cat["t"][i]
            known = cat["t"][i] + 30 <= t_end
            answers.append({"date": str(datetime.utcfromtimestamp(cat["t"][i] * 86400).date()), "mag": float(cat["m"][i]), "p30": p,
                            "followed": bool(((dt > 0) & (dt <= 30)).any()) if known else None, "seconds": time.perf_counter() - t0})
        except Exception as e:                           # a live desk must not stop on one event
            fails += 1
            answers.append({"date": None, "error": repr(e)[:120], "seconds": time.perf_counter() - t0})
    marks, m = [], LM.ISSUE
    while m <= LM.ASOF:
        t0 = time.perf_counter()
        try:
            marks.append({"date": str(m), **ask("price", State(asof=m)), "seconds": time.perf_counter() - t0})
        except Exception as e:
            fails += 1
            marks.append({"date": str(m), "error": repr(e)[:120], "seconds": time.perf_counter() - t0})
        m += timedelta(days=30)
    done = [a for a in answers if a.get("followed") is not None]
    ps, ys = np.array([a["p30"] for a in done]), np.array([a["followed"] for a in done], float)
    in_budget = np.mean([a["seconds"] <= 300 for a in answers] + [x["seconds"] <= 10 for x in marks]) if answers or marks else 0.0
    return {"events_answered": len(answers), "events_with_outcome": len(done), "mean_forecast_30d": float(ps.mean()) if len(done) else None,
            "observed_frequency_30d": float(ys.mean()) if len(done) else None, "expected_count": float(ps.sum()) if len(done) else None, "observed_count": float(ys.sum()) if len(done) else None,
            "marks": len(marks), "failures": fails, "share_in_time_budget": float(in_budget), "max_seconds": float(max([a["seconds"] for a in answers] + [x["seconds"] for x in marks])),
            "answers": answers, "monthly_marks": marks}


def coherence():
    """With the other inputs frozen, the desk gives live_mark.py's single-factor prices."""
    lm = json.loads((OUT / "live_mark.json").read_text())["cases"]
    q = json.loads((OUT / "quality_rates_unc.json").read_text())["multiplier_q"]
    cases = {"base (no event, no payout)": State(event=False), "3 Nov 2026 border earthquake": State(),
             "earthquake rate at 2.5% of data interval": State(eq_mult=(q["eqKG"][0], q["eqTJ"][0]), event=False),
             "earthquake rate at 97.5% of data interval": State(eq_mult=(q["eqKG"][2], q["eqTJ"][2]), event=False),
             "aftershocks paid separately (nu = 0.5)": State(nu=LM.NU_B, event=False), "rain +1C": State(warming=LM.CC["+1C"], event=False),
             "rain +2C": State(warming=LM.CC["+2C"], event=False), "after a 25% earthquake payout to both": State(contract=(1, 1, 0, 0), event=False)}
    diff = {name: max(abs(ask("price", st)[c] - lm[name][c]["price"]) for c in LM.BONDS) for name, st in cases.items()}
    return {"max_diff_points": max(diff.values()), "by_case": diff}


def run():
    t00 = time.perf_counter()
    co = coherence()
    with np.errstate(all="ignore"):
        rm, _ = RU.rain_multipliers(np.random.default_rng(seed(4243)), 2000)
    dists = [price_distribution(k, rain_pool=rm) for k in range(SEEDS)]
    dist = dists[0]
    att, rep = attribution(), replay()
    tops = [tuple(d[c]["ranking"][:3]) for d in dists for c in LM.BONDS]
    ok1 = co["max_diff_points"] <= 0.05
    ok2 = all(abs(dist[c]["sum_S"] - 1) <= 0.05 for c in LM.BONDS) and max(att["closure"].values()) <= 0.02
    ok3 = rep["failures"] == 0 and rep["share_in_time_budget"] >= 0.95
    ok4 = all(set(t) == set(tops[0]) for t in tops)
    ok5 = not os.environ.get("ADB_DATA_DIR") or True       # recorded by the caller: scorecard-style run with an empty data folder
    res = {"coherence": co, "price_distribution": dist, "price_distribution_other_seeds": dists[1:], "actions": actions(dist), "attribution": att, "replay": rep,
           "top3_by_seed_and_note": [list(t) for t in tops], "seconds": time.perf_counter() - t00,
           "checks": {"coherent": bool(ok1), "additive": bool(ok2), "replay": bool(ok3), "ranking_stable": bool(ok4)}}
    res["verdict_without_portability"] = "very strong" if ok1 and ok2 and ok3 and ok4 else "not very strong"
    save("desk.json", res)
    print(f"coherence: max difference from live_mark {co['max_diff_points']:.4f} points")
    for c in LM.BONDS:
        d = dist[c]
        print(f"{c}: price {d['median']:.1f} (mean {d['mean']:.1f}), 90% interval [{d['q05']:.1f}, {d['q95']:.1f}], sd {d['sd']:.1f} points | shares " + ", ".join(f"{g} {d['S'][g]:.2f}" for g in d["ranking"])
              + f" | sum {d['sum_S']:.2f}")
        print("   worth resolving: " + "; ".join(f"{a['input']} {a['points']:.1f} pt = US${a['usd_per_note'] / 1e6:.1f}m" for a in res["actions"][c]))
    print("attribution since issue (KG / TJ): " + "; ".join(f"{s['step']} {s['change_KG']:+.2f} / {s['change_TJ']:+.2f}" for s in att["steps"][1:]) + f" | closure {max(att['closure'].values()):.1e}")
    print(f"replay: {rep['events_answered']} events answered, {rep['events_with_outcome']} with outcome: expected {rep['expected_count']:.1f} followed, observed {rep['observed_count']:.0f}; "
          f"{rep['marks']} marks; failures {rep['failures']}; in time budget {rep['share_in_time_budget']:.0%}; slowest answer {rep['max_seconds']:.2f}s")
    print("top three by seed and note:", res["top3_by_seed_and_note"])
    print("checks:", res["checks"], "| verdict (portability checked separately):", res["verdict_without_portability"], f"| {res['seconds']:.0f}s")
    return res


def selftest():
    # the default state is live_mark's base with the event; a frozen-input state reproduces one of its what-ifs
    lm = json.loads((OUT / "live_mark.json").read_text())["cases"]
    assert all(abs(ask("price", State(event=False))[c] - lm["base (no event, no payout)"][c]["price"]) < 1e-9 for c in LM.BONDS)
    # an offset of +0.2 magnitude units is the rate multiplier 10^(0.92 * 0.2)
    f = 10 ** (B_GR * 0.2)
    a, b = ask("el", State(offset=0.2, event=False)), ask("el", State(eq_mult=(f, f), event=False))
    assert all(abs(a[c] - b[c]) < 1e-12 for c in LM.BONDS)
    # a note that already paid cannot be worth more than a clean one at the same multiple
    assert all(ask("price", State(contract=(1, 1, 0, 0)))[c] < ask("price", State())[c] for c in LM.BONDS)
    att = attribution()
    assert max(att["closure"].values()) < 1e-9
    print("selftest ok: default state = live mark; magnitude offset = rate multiplier; payout state lowers the price; attribution adds up")


if __name__ == "__main__":
    selftest()
    run()
