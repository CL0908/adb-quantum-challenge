"""Live mark of the two ADB notes: remaining-term loss and an indicative price, with what each add-on is worth in
price points. Additive, read-only. A calculation, not an experiment (no pre-set pass rule; not a strong item).

Model: the brief's two-country model (variant.py, which reproduces the friend's annual model). The earthquake payout
state (a, b) in quarter steps is a continuous-time Markov chain with yearly generator G, so the law after t years
from any state is exp(t G); each country's rain cover triggers once at rate mu, the shared part as in the brief.
For a note with contract state s (payouts so far) and t years left:
    remaining expected loss, as % of the principal still outstanding;  P(any further payout);  P(exhausted).
Indicative price (points of outstanding principal, no discounting):
    100 + (coupon - multiple * annualised remaining expected loss) * years left,
with the multiple fixed at the one paid at issue (coupon / the model's own annualised expected loss), so the price is
100 at issue by construction; a lower market multiple is shown as a sensitivity.
Dates: priced 24 Apr 2026 (trade press); a three-year term, maturity taken as 24 Apr 2029 (the exact date is not public).
What-ifs priced on the same day: the 3 Nov 2026 border earthquake (event_response.py), the earthquake trigger-rate
95% data interval (rates_unc.py), aftershocks paid separately (eq_cluster.py, nu = 0.5), +1 C and +2 C rain
(climate_cc.py), a 25% earthquake payout already made, and the market multiple moving from 2.5 to 2.4 (AM Best via
Artemis, first half of 2026 to 2026 to date).
"""
import json
from datetime import date

import numpy as np
from scipy.linalg import expm

import variant
from _src import OUT, save
from src.model.poisson_fit import TAILS, compound_poisson_rates
from src.model.scenarios import SCENARIOS

SC = SCENARIOS["adb-base"]
ISSUE, MATURITY, ASOF, COUPON = date(2026, 4, 24), date(2029, 4, 24), date(2026, 11, 18), 6.00
BONDS = ("KG", "TJ")
NU_B = 0.5                                              # eq_cluster.py: offspring per event at pay-size magnitudes
CC = {"+1C": 1.55, "+2C": 2.4}                          # climate_cc.py: 2% rain-event frequency factors (mid of its ranges)
MULT_MARKET = (2.5, 2.4)


def years(d0, d1):
    return (d1 - d0).days / 365.25


def remaining(tau, state=(0, 0, 0, 0), eq_mult=(1.0, 1.0), rain_mult=(1.0, 1.0), nu=0.0, extra_el=(0.0, 0.0)):
    """Per note, from contract state (a_KG, a_TJ, r_KG, r_TJ) with tau years left: outstanding principal (USD), expected
    further loss in % of it, P(any further payout), P(exhausted). extra_el adds expected loss in % of the full principal."""
    a0, b0, r1, r2 = state
    E = expm(tau * variant.eq_generator(SC, eq_mult, nu))[5 * a0 + b0].reshape(5, 5)
    mu = -np.log(1 - 0.02) * SC.rain_mult
    p_rain = [1.0 if r else 1 - np.exp(-mu * m * tau) for r, m in zip((r1, r2), rain_mult)]
    out = {}
    for i, c in enumerate(BONDS):
        pa = E.sum(1 - i)                               # law of this country's earthquake level at maturity
        a_now, r_now = (a0, b0)[i], (r1, r2)[i]
        principal = SC.eq_limit * (1 - a_now / 4) + SC.rain_limit * (1 - r_now)
        loss = SC.eq_limit * (pa @ np.arange(5) - a_now) / 4 + SC.rain_limit * (p_rain[i] - r_now) + extra_el[i] / 100 * (SC.eq_limit + SC.rain_limit)
        p_any = 1 - pa[a_now] * (1 - (0.0 if r_now else p_rain[i]))
        out[c] = {"principal_usd": float(principal), "el_pct": float(100 * loss / principal), "p_any_payout": float(p_any),
                  "p_exhausted": float(pa[4] * p_rain[i])}
    return out


def price(el_pct, tau, multiple):
    return 100 + (COUPON - multiple * el_pct / tau) * tau


def run():
    T, tau = years(ISSUE, MATURITY), years(ASOF, MATURITY)
    at_issue, now = remaining(T), remaining(tau)
    mult0 = {c: COUPON / (at_issue[c]["el_pct"] / T) for c in BONDS}
    q = json.loads((OUT / "quality_rates_unc.json").read_text())["multiplier_q"]
    try:
        ev = json.loads((OUT / "event_response.json").read_text())["event_2026_11_03"]
    except FileNotFoundError:
        ev = None
    sev = {c: float((compound_poisson_rates(TAILS[c])[0] / compound_poisson_rates(TAILS[c])[1]) @ np.arange(1, 5)) / 4 for c in BONDS}   # mean payout share of the earthquake limit per pay event
    extra = tuple(100 * ev[f"pay_size_{c}"]["p_added_by_event"] * sev[c] * SC.eq_limit / (SC.eq_limit + SC.rain_limit) if ev else 0.0 for c in BONDS)
    cases = {"base (no event, no payout)": now,
             "3 Nov 2026 border earthquake": remaining(tau, extra_el=extra),
             "earthquake rate at 2.5% of data interval": remaining(tau, eq_mult=(q["eqKG"][0], q["eqTJ"][0])),
             "earthquake rate at 97.5% of data interval": remaining(tau, eq_mult=(q["eqKG"][2], q["eqTJ"][2])),
             "aftershocks paid separately (nu = 0.5)": remaining(tau, nu=NU_B),
             **{f"rain {k}": remaining(tau, rain_mult=(v, v)) for k, v in CC.items()},
             "after a 25% earthquake payout to both": remaining(tau, state=(1, 1, 0, 0))}
    res = {"issue": str(ISSUE), "maturity_assumed": str(MATURITY), "as_of": str(ASOF), "years_total": T, "years_left": tau, "coupon_pct": COUPON,
           "multiple_at_issue": mult0, "at_issue": at_issue, "cases": {}, "path_monthly": []}
    for name, r in cases.items():
        res["cases"][name] = {c: {**r[c], "el_annualised_pct": r[c]["el_pct"] / tau, "fair_spread_pct": mult0[c] * r[c]["el_pct"] / tau,
                                  "price": price(r[c]["el_pct"], tau, mult0[c])} for c in BONDS}
    res["market_multiple_sensitivity"] = {c: price(now[c]["el_pct"], tau, mult0[c] * MULT_MARKET[1] / MULT_MARKET[0]) - price(now[c]["el_pct"], tau, mult0[c]) for c in BONDS}
    for k in range(0, 37, 3):                            # every three months from issue to maturity, no event
        t = max(T - k / 12, 1e-9)
        r = remaining(t)
        res["path_monthly"].append({"months_since_issue": k, **{c: {"el_pct": r[c]["el_pct"], "p_any_payout": r[c]["p_any_payout"], "price": price(r[c]["el_pct"], t, mult0[c])} for c in BONDS}})
    save("live_mark.json", res)
    print(f"as of {ASOF}: {tau:.2f} of {T:.2f} years left; multiple paid at issue " + ", ".join(f"{c} {mult0[c]:.2f}" for c in BONDS)
          + " | model annualised EL at issue " + ", ".join(f"{c} {at_issue[c]['el_pct'] / T:.3f}%" for c in BONDS))
    base = res["cases"]["base (no event, no payout)"]
    for name, r in res["cases"].items():
        print(f"{name:42s} " + " | ".join(f"{c}: P(payout) {r[c]['p_any_payout']:.4f}, EL left {r[c]['el_pct']:.3f}%, fair spread {r[c]['fair_spread_pct']:.2f}%, price {r[c]['price']:.2f} ({r[c]['price'] - base[c]['price']:+.2f})" for c in BONDS))
    print("market multiple 2.5 -> 2.4: price " + ", ".join(f"{c} {v:+.2f}" for c, v in res["market_multiple_sensitivity"].items()))
    return res


def selftest():
    T = 3.0
    r = remaining(T)
    t = variant.targets()
    # three years from the clean state reproduce the friend's term targets
    el = sum(r[c]["el_pct"] / 100 * r[c]["principal_usd"] for c in BONDS)
    assert abs(el / t["term_expected_principal_loss_usd"] - 1) < 1e-9, (el, t["term_expected_principal_loss_usd"])
    # Markov property: expected loss over 3 years = over the first year + the expected remainder from the state reached
    G = variant.eq_generator(SC)
    first = expm(1.0 * G)[0]
    lev = lambda i: np.arange(25) // 5 if i == 0 else np.arange(25) % 5
    for i, c in enumerate(BONDS):
        eq3 = expm(3.0 * G)[0] @ lev(i)
        chained = sum(first[s] * (expm(2.0 * G)[s] @ lev(i)) for s in range(25))
        assert abs(eq3 - chained) < 1e-12
    assert all(abs(price(r[c]["el_pct"], T, COUPON / (r[c]["el_pct"] / T)) - 100) < 1e-9 for c in BONDS)
    # a note that already paid 25% has less principal outstanding and cannot pay that quarter again
    s = remaining(T, state=(1, 1, 0, 0))
    assert all(s[c]["principal_usd"] == 80e6 - 17.5e6 and s[c]["p_exhausted"] > r[c]["p_exhausted"] for c in BONDS)
    print("selftest ok: three years from the clean state = the friend's term expected loss; Markov chaining; price 100 at issue; payout state")


if __name__ == "__main__":
    selftest()
    run()
