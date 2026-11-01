"""Loss-data backtest of the two-country dependence assumption.

Data (public, via the Our World in Data mirror on GitHub, cached in data/real/):
  * NOAA NGDC/WDS significant-earthquake database: significant earthquake events per country-year
    ("significant": at least one death, US$1m damage, M >= 7.5, MMI >= X or a tsunami), 1900–2020.
  * EM-DAT country-year losses by peril (earthquake; floods, landslides, mass movements and storms as the
    hydro-meteorological counterpart of the rain cover), 1990–2020.

Model being tested. In the ADB model, each country's events are a Poisson process whose rate is the sum of a
country-only rate and a shared rate (one event hits both). For yearly occurrence this is the bivariate
Poisson model
    P(no event in A) = exp(−λA),   P(no event in B) = exp(−λB),   P(neither) = exp(−(λA + λB − λS)),
so the 2×2 table of yearly occurrence identifies the shared rate λS. The shared share λS/λA is what the brief
sets to 7.5% (earthquakes) and 25% of the smaller rate (rain). We estimate it by maximum likelihood with a
profile-likelihood 95% interval, then recompute the exact joint targets of the bond with the data-based value.
Caveat: these are *reported loss* events, smaller than bond triggers; dependence at trigger level may differ.
Run: python -m src.realdata.loss_backtest -> results/realdata_losses.json
"""
import csv
import io
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2

DATA = Path("data/real")
OWID = "https://github.com/owid/owid-datasets"
FILES = {"noaa_events": "datasets/Significant earthquake events (NGDC-NASA)/Significant earthquake events (NGDC-NASA).csv",
         "noaa_deaths": "datasets/Earthquake deaths (NGDC-NOAA)/Earthquake deaths (NGDC-NOAA).csv",
         "emdat": "datasets/Natural disasters (EMDAT)/Natural disasters (EMDAT).csv"}
COUNTRIES = ["Kazakhstan", "Kyrgyzstan", "Tajikistan", "Turkmenistan", "Uzbekistan"]
HYDRO = ("flood", "landslide", "mass_movement", "storm")


def _fetch():
    """Cache the Central Asian rows of the three OWID files (sparse git checkout of the public repo)."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = {k: DATA / f"owid_{k}_central_asia.csv" for k in FILES}
    if all(p.exists() for p in out.values()):
        return out
    tmp = Path("/tmp/owid_ds")
    if not tmp.exists():
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--depth", "1", OWID, str(tmp)], check=True)
    subprocess.run(["git", "-C", str(tmp), "checkout", "-q", "HEAD", "--"] + list(FILES.values()), check=True)
    for k, rel in FILES.items():
        rows = list(csv.reader(open(tmp / rel)))
        keep = [rows[0]] + [r for r in rows[1:] if r[0] in COUNTRIES]
        buf = io.StringIO(); csv.writer(buf).writerows(keep)
        out[k].write_text(f"# source: {OWID} {rel}\n" + buf.getvalue())
    return out


def _read(path):
    lines = [l for l in Path(path).read_text().splitlines() if not l.startswith("#")]
    return list(csv.DictReader(lines))


def occurrence_years(years=(1975, 2020)):
    """Yearly occurrence per country: NOAA significant earthquakes, EM-DAT earthquake and hydro loss years."""
    f = _fetch()
    y0, y1 = years
    occ = {"eq_noaa": {c: set() for c in COUNTRIES}, "eq_emdat": {c: set() for c in COUNTRIES},
           "hydro_emdat": {c: set() for c in COUNTRIES}}
    for r in _read(f["noaa_events"]):
        if y0 <= int(r["Year"]) <= y1 and float(r["Significant earthquake events (NGDC-NASA)"] or 0) > 0:
            occ["eq_noaa"][r["Entity"]].add(int(r["Year"]))
    for r in _read(f["emdat"]):
        y = int(r["Year"])
        if not 1990 <= y <= y1:
            continue
        g = lambda k: float(r.get(k) or 0)
        if g("total_affected_earthquake") + g("deaths_earthquake") + g("total_damages_earthquake") > 0:
            occ["eq_emdat"][r["Entity"]].add(y)
        # hydro-meteorological losses large enough to matter: >= 10 deaths, >= 10,000 affected or >= US$10m damage
        if any(g(f"deaths_{h}") >= 10 or g(f"total_affected_{h}") >= 1e4 or g(f"total_damages_{h}") >= 1e4 for h in HYDRO):
            occ["hydro_emdat"][r["Entity"]].add(y)
    return occ


def bivariate_poisson_fit(table):
    """MLE of (λA, λB, λS) from a 2×2 yearly table [[both, A only], [B only, neither]]; returns the shared share
    λS/min(λA, λB)... reported both ways, with a profile-likelihood 95% interval for λS/λA."""
    n11, n10 = table[0]; n01, n00 = table[1]
    N = n11 + n10 + n01 + n00

    def probs(la, lb, ls):
        p00 = math.exp(-(la + lb - ls)); pa0 = math.exp(-la); pb0 = math.exp(-lb)
        p10 = pb0 - p00          # A happens, B does not
        p01 = pa0 - p00
        p11 = 1 - pa0 - pb0 + p00
        return p11, p10, p01, p00

    def nll(la, lb, ls):
        ps = probs(la, lb, ls)
        if min(ps) <= 0:
            return 1e18
        return -(n11 * math.log(ps[0]) + n10 * math.log(ps[1]) + n01 * math.log(ps[2]) + n00 * math.log(ps[3]))

    # marginals are sufficient for λA, λB (MLE), then profile over the shared share s = λS / min(λA, λB)
    la = -math.log((n01 + n00) / N); lb = -math.log((n10 + n00) / N)
    lmin = min(la, lb)
    prof = lambda s: nll(la, lb, s * lmin)
    res = minimize_scalar(prof, bounds=(0.0, 0.999), method="bounded")
    s_hat = float(res.x); best = res.fun
    cut = best + chi2.ppf(0.95, 1) / 2
    lo = 0.0 if prof(0.0) <= cut else brentq(lambda s: prof(s) - cut, 0.0, s_hat)
    hi = 0.999 if prof(0.999) <= cut else brentq(lambda s: prof(s) - cut, s_hat, 0.999)
    return {"lambda_A": la, "lambda_B": lb, "lambda_S": s_hat * lmin, "share_of_A": s_hat * lmin / la,
            "share_of_B": s_hat * lmin / lb, "share_of_min": s_hat, "share_of_min_ci95": [lo, hi],
            "share_of_A_ci95": [lo * lmin / la, hi * lmin / la], "lr_test_independence_p": float(chi2.sf(2 * (prof(0.0) - best), 1))}


def table_for(occ_a, occ_b, years):
    ys = range(years[0], years[1] + 1)
    a = [y in occ_a for y in ys]; b = [y in occ_b for y in ys]
    n11 = sum(x and y for x, y in zip(a, b)); n10 = sum(x and not y for x, y in zip(a, b))
    n01 = sum(y and not x for x, y in zip(a, b)); n00 = len(a) - n11 - n10 - n01
    return [[n11, n10], [n01, n00]]


def bond_targets(m=1.0, rain_shared=0.25):
    from src.tensor.synthetic import SyntheticModel, topology
    from src.rdpo.problem import PortfolioProblem
    model = SyntheticModel(2, topology("chain", 2), m=m, rain_shared=rain_shared)
    return {t: PortfolioProblem(model, t, threshold=80e6).exact_dense()
            for t in ("joint_attachment", "exceedance", "joint_exhaustion")}


def run():
    out = {"data": {"source": OWID, "files": FILES}, "tests": {}}
    brief_eq = 0.5 * (0.075 + 0.074)
    for key, years, label in (("eq_noaa", (1975, 2020), "NOAA significant earthquakes, 1975–2020"),
                              ("eq_noaa", (1900, 2020), "NOAA significant earthquakes, 1900–2020"),
                              ("eq_emdat", (1990, 2020), "EM-DAT earthquake losses, 1990–2020"),
                              ("hydro_emdat", (1990, 2020), "EM-DAT flood/landslide/storm losses, 1990–2020")):
        occ = occurrence_years(years)[key]
        tab = table_for(occ["Kyrgyzstan"], occ["Tajikistan"], years)
        fit = bivariate_poisson_fit(tab)
        n = years[1] - years[0] + 1
        out["tests"][f"{key}_{years[0]}"] = {
            "label": label, "years": years, "table_both_KGonly_TJonly_none": tab,
            "p_KG": (tab[0][0] + tab[0][1]) / n, "p_TJ": (tab[0][0] + tab[1][0]) / n, "p_both": tab[0][0] / n,
            "p_both_if_independent": (tab[0][0] + tab[0][1]) * (tab[0][0] + tab[1][0]) / n ** 2, **fit}
    # implied bond targets: earthquake shared share from the longest record (NOAA 1900–2020, the only test that
    # rejects independence); rain kept at the brief's 25% because the hydro loss data do not constrain it
    eq = out["tests"]["eq_noaa_1900"]
    rows = {"brief": {"eq_share": brief_eq, "rain_share": 0.25, **bond_targets(1.0, 0.25)}}
    for lab, s_eq, s_r in (("loss data, lower 95% bound", eq["share_of_A_ci95"][0], 0.25),
                           ("loss data, point estimate", eq["share_of_A"], 0.25),
                           ("loss data, upper 95% bound", eq["share_of_A_ci95"][1], 0.25)):
        m = s_eq / brief_eq
        try:
            rows[lab] = {"eq_share": s_eq, "rain_share": s_r, "m": m, **bond_targets(m, min(s_r, 1.0))}
        except AssertionError:
            rows[lab] = {"eq_share": s_eq, "rain_share": s_r, "m": m, "error": "shared rate exceeds total rate"}
    out["implied"] = rows
    return out


if __name__ == "__main__":
    o = run()
    Path("results/realdata_losses.json").write_text(json.dumps(o, indent=2))
    for k, t in o["tests"].items():
        print(f"{t['label']:48s} table={t['table_both_KGonly_TJonly_none']} both={t['p_both']:.3f} indep={t['p_both_if_independent']:.3f} "
              f"share(KG)={t['share_of_A']:.2f} [{t['share_of_A_ci95'][0]:.2f},{t['share_of_A_ci95'][1]:.2f}] "
              f"share(min)={t['share_of_min']:.2f} [{t['share_of_min_ci95'][0]:.2f},{t['share_of_min_ci95'][1]:.2f}] p_indep={t['lr_test_independence_p']:.3f}")
    for k, r in o["implied"].items():
        print(k, {a: (round(b, 6) if isinstance(b, float) else b) for a, b in r.items()})
