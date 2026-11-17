"""Climate: Clausius–Clapeyron stress on the rain trigger. Additive; src/ read-only.

Extreme-precipitation intensities scale by about 7% per °C of warming (Clausius–Clapeyron). With the GEV fit of
rates_unc (GML, Martins–Stedinger prior) to each country's annual maxima of the friend's GHCN country index, the old
2% level x50 is exceeded with p' = P(X > x50 / 1.07^ΔT) after ΔT of warming; rain_mult = −log(1−p') / −log(0.98)
goes into the friend's annual model (variant.py reproduces it exactly) for every bond year, and the friend's term
convolution gives the five targets. ΔT = 0.5 / 1 / 1.5 / 2 °C: scenarios, not a forecast. Earthquakes unchanged.
Scenarios: the brief (adb-base), and the same with the friend's station-data rain share (pooled, q = 20%).
"""
import math

import numpy as np
from scipy import stats

import variant
from _src import load, save
from rates_unc import country_data, gev_gml
from src.model.annual import N_STATES, annual_distribution
from src.model.scenarios import SCENARIOS, Scenario
from src.model.targets import target_functions
from src.model.term import YEARS, step, term_distribution

DT = (0.5, 1.0, 1.5, 2.0)
J, A = "term_both_bonds_exhausted_probability", "term_joint_attachment_probability"


def scenarios():
    share = load("realdata_rain.json")["pooled"]["0.2"]["share"]
    return {"brief (adb-base)": SCENARIOS["adb-base"],
            "rain share from data": Scenario("adb-base-data", m=1.0, rain_mult=1.0, rain_shared_share=share,
                                             eq_limit=70e6, rain_limit=10e6, threshold=80e6)}


def targets_of(sc, term):
    return {k: float(term @ f) for k, f in target_functions(sc).items()}


def term_with(sc, mults):
    d = np.zeros(N_STATES); d[0] = 1.0
    for m in mults:
        d = step(d, variant.annual(sc, rain_mult=tuple(m)))
    return d


def annual_maxima(cd, c):
    ys, a = set(cd[c]["years"]), {}
    for key, v in cd[c]["idx"].items():
        y = int(key[:4])
        if y in ys:
            a[y] = max(a.get(y, 0.0), v)
    return np.array(list(a.values()))


def cc_mult(gev, dT, p0=0.02):
    sh, lo, s = gev
    p = stats.genextreme.sf(stats.genextreme.isf(p0, sh, lo, s) / 1.07 ** dT, sh, lo, s)
    return -math.log1p(-p) / -math.log1p(-p0)


def run():
    cd = country_data()
    gev = {c: gev_gml(annual_maxima(cd, c)) for c in ("KG", "TJ")}
    cc = {dT: [cc_mult(gev[c], dT) for c in ("KG", "TJ")] for dT in DT}
    res = {}
    for sn, sc in scenarios().items():
        r = {"friend": targets_of(sc, term_distribution(variant.annual(sc)))}
        for dT, m in cc.items():
            r[f"CC dT={dT}"] = {"point": targets_of(sc, term_with(sc, [m] * YEARS)), "mults": m}
        res[sn] = r
    out = {"gev": {c: list(map(float, g)) for c, g in gev.items()}, "cc_multipliers": {str(k): v for k, v in cc.items()},
           "targets": res}
    save("climate_cc.json", out)
    for sn, r in res.items():
        print(f"{sn}: friend joint-attach {r['friend'][A]:.4g} exhaust {r['friend'][J]:.3g}")
        for lab, v in r.items():
            if lab != "friend":
                print(f"  {lab}: mults {np.round(v['mults'], 3).tolist()}  joint-attach x{v['point'][A] / r['friend'][A]:.3f}  "
                      f"exhaust x{v['point'][J] / r['friend'][J]:.3f}")
    return out


def selftest():
    for name, sc in SCENARIOS.items():                      # no warming -> the friend's term law exactly
        assert np.allclose(term_with(sc, [(1.0, 1.0)] * YEARS), term_distribution(annual_distribution(sc)), atol=1e-13), name
    g = (0.1, 1.0, 0.3)
    assert abs(cc_mult(g, 0.0) - 1) < 1e-12 and cc_mult(g, 1.0) > cc_mult(g, 0.5) > 1
    print("selftest ok: ΔT = 0 equals the friend's term law; CC multiplier is 1 at ΔT = 0 and grows with ΔT")


if __name__ == "__main__":
    selftest()
    run()
