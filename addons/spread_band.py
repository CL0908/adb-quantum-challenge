"""From a data-based expected-loss interval to a spread band, with three published market curves. Additive; a
calculation, not an experiment (the inputs were known before it was written, so no pre-set pass rule is claimed).

The two ADB notes (Artemis deal directory): earthquake tranche initial expected loss 2.296% (Kyrgyz Republic) and
2.011% (Tajikistan) a year, risk margin 600 bp for both, price guidance 525-600 bp.
Market curves (spread as a function of the annual expected loss EL, both in % of principal):
  bodoff_gan   2.31 + 1.63 EL      Bodoff & Gan 2013, Variance 6(2), Exhibit 7a: non-peak earthquake, 93 bonds issued
                                   1998-2008, R^2 87.3% (multiplier 95% CI 1.41-1.85)
  ibrd         (a e^{-k EL} + b) EL  risk-multiple curve fitted to World Bank sovereign parametric bonds in
                                   CLIMADA-petals (premium_class.monoExp); a, k, b as printed by its tutorial
  am_best      m EL, m in 2.4-3.0   weighted-average loss multiple of 144A property cat bonds, 2025 to 2026 year
                                   to date (AM Best, reported by Artemis 2026-09-25); a range, not a curve
EL interval: rates_unc.py's 95% interval of the earthquake trigger-rate multiplier of each country (Gutenberg-Richter
extrapolation of the friend's own catalogue counts), applied to the issued EL. The tiers are small probabilities, so
EL is taken proportional to the trigger rate.
What it answers: is 600 bp what the market charges for this EL (level), and how wide a spread band the data
uncertainty of EL alone would justify (width). It replaces the ambiguity-premium attempt, which was not robust.
"""
import json
import math

from _src import OUT, save

ISSUED = {"KG": 2.296, "TJ": 2.011}                     # annual expected loss, % (Artemis; = mean of the brief's tiers)
SPREAD, GUIDANCE = 6.00, (5.25, 6.00)                   # risk margin and price guidance, %
BG = {"const": 2.31, "mult": 1.63, "mult_ci": (1.41, 1.85)}
IBRD = {"a": 4.0576, "k": 0.6462, "b": 1.5306}
AMB = (2.4, 3.0)


def bodoff_gan(el, mult=BG["mult"]):
    return BG["const"] + mult * el


def ibrd(el):
    return (IBRD["a"] * math.exp(-IBRD["k"] * el) + IBRD["b"]) * el


def run():
    q = json.loads((OUT / "quality_rates_unc.json").read_text())["multiplier_q"]
    res = {"issued_el_pct": ISSUED, "spread_pct": SPREAD, "guidance_pct": list(GUIDANCE), "bonds": {}}
    for c in ISSUED:
        lo, med, hi = (ISSUED[c] * m for m in q[f"eq{c}"])
        row = {"el_interval_pct": [lo, med, hi], "rate_multiplier_q": q[f"eq{c}"], "at_issued_el": {}, "band": {}}
        for name, f in (("bodoff_gan", bodoff_gan), ("ibrd", ibrd)):
            row["at_issued_el"][name] = f(ISSUED[c])
            row["band"][name] = [f(lo), f(hi)]
        row["at_issued_el"]["am_best"] = [m * ISSUED[c] for m in AMB]
        row["band"]["am_best"] = [AMB[0] * lo, AMB[1] * hi]
        row["multiple_paid"] = SPREAD / ISSUED[c]
        row["slope_bp_per_bp"] = {"bodoff_gan": BG["mult"], "ibrd": (ibrd(ISSUED[c] + 0.01) - ibrd(ISSUED[c] - 0.01)) / 0.02, "am_best": list(AMB)}
        res["bonds"][c] = row
    save("spread_band.json", res)
    for c, r in res["bonds"].items():
        a = r["at_issued_el"]
        print(f"{c}: EL {ISSUED[c]:.3f}% paid {SPREAD:.2f}% (multiple {r['multiple_paid']:.2f}) | curves at issued EL: Bodoff-Gan {a['bodoff_gan']:.2f}%, IBRD {a['ibrd']:.2f}%, AM Best {a['am_best'][0]:.2f}-{a['am_best'][1]:.2f}%"
              f" | EL 95% [{r['el_interval_pct'][0]:.2f}, {r['el_interval_pct'][2]:.2f}]% -> band Bodoff-Gan [{r['band']['bodoff_gan'][0]:.2f}, {r['band']['bodoff_gan'][1]:.2f}]%,"
              f" IBRD [{r['band']['ibrd'][0]:.2f}, {r['band']['ibrd'][1]:.2f}]%, AM Best [{r['band']['am_best'][0]:.2f}, {r['band']['am_best'][1]:.2f}]%"
              f" | slope {r['slope_bp_per_bp']['ibrd']:.2f}-{BG['mult']:.2f} bp per bp of EL (curves)")
    return res


def selftest():
    # the IBRD curve with the tutorial's parameters reproduces the tutorial's four printed premiums
    for el, prem in ((4.24, 7.6), (5.18, 8.7), (4.47, 7.8), (0.25, 1.3)):
        assert abs(ibrd(el) - prem) < 0.06, (el, ibrd(el))
    assert abs(bodoff_gan(2.296) - 6.05) < 0.005 and abs(SPREAD / ISSUED["KG"] - 2.61) < 0.005
    print("selftest ok: IBRD curve reproduces the CLIMADA-petals tutorial premiums; Bodoff-Gan arithmetic")


if __name__ == "__main__":
    selftest()
    run()
