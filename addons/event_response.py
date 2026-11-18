"""Event-response mode of the risk twin: what does an earthquake that just happened do to the chance of the next one?
Additive, read-only; data from the field library (USGS ComCat, fetch_usgs.py).

The twin (twin.py) treats main shocks as a stationary Poisson background and is updated once a year. Within weeks of an
event the local rate is higher (Omori-Utsu aftershock decay, productivity growing with magnitude). Model for the
number of M >= 5.5 events within 50 km of a trigger event of magnitude m (M >= 5.0), in (t1, t2] days after it:
    Poisson( beta * b_i * (t2 - t1)  +  A * 10^(alpha (m - 5)) * [G(t2) - G(t1)] ),    G(t) = 1 - (c / (t + c))^(p - 1),
with b_i the long-run local rate of M >= 5.5 events within 50 km (Jeffreys estimate from the training years only).
(beta, A, alpha, c, p) are fitted by maximum likelihood on six time bins up to one year, on triggers of 1973-1999.
Baseline: stationary Poisson with the local rate b_i.
Out-of-sample test on triggers of 2000-2024 (ComCat M >= 4.5, the friend's CAREC window): forecast the probability of
at least one M >= 5.5 event within 50 km in the next 30 days and in the next year.
Pre-set rules, both for "strong": (1) the mean log-score gain over the Poisson baseline is positive with a 95%
year-block bootstrap interval that excludes 0, at both horizons; (2) calibration: with the test forecasts in five
equal-count bins, the 95% year-block bootstrap interval of (observed frequency - mean forecast) contains 0 in at least
four bins, at both horizons.
Then the event of 3 Nov 2026 on the Kyrgyz-Tajik border (USGS M 5.2; the Kyrgyz agency reported 5.5).
"""
import csv
import json
import math
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

from _src import DATA, save

CAT = DATA / "seismology" / "catalogs" / "usgs_comcat"
R_KM, M_TRIG, M_TARGET, BINS, HORIZONS = 50.0, 5.0, 5.5, (0, 1, 3, 10, 30, 100, 365), (30, 365)
TRAIN, TEST = (1973, 1999), (2000, 2024)
EVENT = {"time": "2026-11-03T21:34:08", "lat": 39.3453, "lon": 73.3772, "mag": 5.2}       # USGS ComCat, mww, depth 10 km
ASOF, MATURITY = "2026-11-18", "2029-04-30"
B_GR, M_PAY = 0.92, {"KG": 6.91, "TJ": 7.02}        # rates_unc.py: Gutenberg-Richter slope and implied trigger magnitudes


def load(name="carec_m45_1973_2025.csv"):
    t, la, lo, m = [], [], [], []
    with open(CAT / name, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["mag"]:
                t.append(datetime.fromisoformat(r["time"][:19]).timestamp() / 86400.0); la.append(float(r["latitude"])); lo.append(float(r["longitude"])); m.append(float(r["mag"]))
    o = np.argsort(t)
    return {k: np.asarray(v)[o] for k, v in (("t", t), ("lat", la), ("lon", lo), ("m", m))}


def dist_km(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def design(cat):
    """For every trigger: magnitude, year, local rate (training years), counts of M >= 5.5 followers per time bin."""
    year = lambda d: 1970 + d / 365.2425
    big = cat["m"] >= M_TARGET
    tb, lab, lob = cat["t"][big], cat["lat"][big], cat["lon"][big]
    in_train = (year(tb) >= TRAIN[0]) & (year(tb) < TRAIN[1] + 1)
    yrs = TRAIN[1] - TRAIN[0] + 1
    rows = []
    for i in np.flatnonzero(cat["m"] >= M_TRIG):
        t0 = cat["t"][i]
        near = dist_km(cat["lat"][i], cat["lon"][i], lab, lob) <= R_KM
        dt = tb[near] - t0
        cnt = np.histogram(dt[dt > 0], bins=BINS)[0]
        rows.append((cat["m"][i], year(t0), ((near & in_train).sum() + 0.5) / yrs / 365.2425, *cnt))
    a = np.array(rows)
    return {"m": a[:, 0], "year": a[:, 1], "b": a[:, 2], "cnt": a[:, 3:]}


def G(t, c, p):
    return 1.0 - (c / (np.asarray(t, float) + c)) ** (p - 1.0)


def expected(par, m, b, t1, t2):
    beta, A, alpha, c, p = par
    return beta * b * (t2 - t1) + A * 10.0 ** (alpha * (m - M_TRIG)) * (G(t2, c, p) - G(t1, c, p))


def fit(d, mask):
    edges = np.array(BINS, float)

    def nll(x):
        par = (math.exp(x[0]), math.exp(x[1]), x[2], math.exp(x[3]), 1.0 + math.exp(x[4]))
        mu = np.stack([expected(par, d["m"][mask], d["b"][mask], edges[k], edges[k + 1]) for k in range(len(edges) - 1)], 1)
        return float((mu - d["cnt"][mask] * np.log(mu)).sum())

    best = min((minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-6, "fatol": 1e-8}) for x0 in
                ([0.0, -2.0, 1.0, -2.0, -1.0], [-0.5, -1.0, 0.8, -1.0, -2.0], [0.0, -3.0, 1.2, -3.0, 0.0])), key=lambda r: r.fun)
    x = best.x
    return (math.exp(x[0]), math.exp(x[1]), float(x[2]), math.exp(x[3]), 1.0 + math.exp(x[4]))


def block_ci(values, years, rng, n=2000):
    """95% interval of a mean, resampling whole calendar years (events of one sequence stay together)."""
    ys = np.unique(np.floor(years))
    sums = np.array([values[np.floor(years) == y].sum() for y in ys]); cnts = np.array([(np.floor(years) == y).sum() for y in ys])
    idx = rng.integers(0, len(ys), (n, len(ys)))
    m = sums[idx].sum(1) / cnts[idx].sum(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def evaluate(d, par, mask, rng):
    out, ok1, ok2 = {}, True, True
    edges = np.array(BINS)
    for H in HORIZONS:
        y = (d["cnt"][mask][:, :int(np.searchsorted(edges, H))].sum(1) > 0).astype(float)
        pm = 1 - np.exp(-expected(par, d["m"][mask], d["b"][mask], 0.0, float(H)))
        p0 = 1 - np.exp(-d["b"][mask] * H)
        ls = lambda p: y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1))
        gain = ls(pm) - ls(p0)
        ci = block_ci(gain, d["year"][mask], rng)
        q = np.quantile(pm, [0.2, 0.4, 0.6, 0.8]); which = np.searchsorted(q, pm)
        bins = []
        for k in range(5):
            s = which == k
            dci = block_ci((y - pm)[s], d["year"][mask][s], rng)
            bins.append({"mean_forecast": float(pm[s].mean()), "observed": float(y[s].mean()), "n": int(s.sum()), "diff_ci95": dci, "ok": bool(dci[0] <= 0 <= dci[1])})
        out[str(H)] = {"n": int(mask.sum()), "base_rate": float(y.mean()), "mean_forecast": float(pm.mean()), "mean_poisson": float(p0.mean()),
                       "log_score_gain": float(gain.mean()), "gain_ci95": ci, "brier_model": float(((pm - y) ** 2).mean()), "brier_poisson": float(((p0 - y) ** 2).mean()),
                       "calibration": bins, "bins_ok": int(sum(b["ok"] for b in bins))}
        ok1 &= ci[0] > 0
        ok2 &= out[str(H)]["bins_ok"] >= 4
    return out, bool(ok1), bool(ok2)


def event_impact(cat, par):
    """The 3 Nov 2026 event: chance of a further M >= 5.5 event within 50 km, and of a pay-size event, to maturity."""
    yrs = TRAIN[1] - TRAIN[0] + 1
    big = cat["m"] >= M_TARGET
    year = 1970 + cat["t"][big] / 365.2425
    near = (dist_km(EVENT["lat"], EVENT["lon"], cat["lat"][big], cat["lon"][big]) <= R_KM) & (year >= TRAIN[0]) & (year < TRAIN[1] + 1)
    b = (near.sum() + 0.5) / yrs / 365.2425
    day = lambda s: datetime.fromisoformat(s).timestamp() / 86400.0
    t_now, t_end = day(ASOF) - day(EVENT["time"]), day(MATURITY) - day(EVENT["time"])
    beta, A, alpha, c, p = par
    trig = lambda t1, t2: A * 10.0 ** (alpha * (EVENT["mag"] - M_TRIG)) * (G(t2, c, p) - G(t1, c, p))
    res = {"event": EVENT, "as_of": ASOF, "maturity": MATURITY, "days_since_event": t_now, "local_rate_per_year": b * 365.2425, "m55_events_within_50km_in_training_years": int(near.sum())}
    for name, (t1, t2) in {"first_30_days_at_event_time": (0.0, 30.0), "first_year_at_event_time": (0.0, 365.0), "from_now_to_maturity": (t_now, t_end)}.items():
        mu_bg, mu_tr = beta * b * (t2 - t1), trig(t1, t2)
        res[name] = {"p_m55_background": 1 - math.exp(-mu_bg), "p_m55_with_event": 1 - math.exp(-(mu_bg + mu_tr)), "expected_triggered_m55": mu_tr}
    t1, t2 = t_now, t_end
    for cty, mp in M_PAY.items():
        s = 10.0 ** (-B_GR * (mp - M_TARGET))                       # Gutenberg-Richter share of M >= 5.5 events that reach the pay size
        res[f"pay_size_{cty}"] = {"magnitude": mp, "gr_share": s, "p_background": 1 - math.exp(-beta * b * (t2 - t1) * s),
                                  "p_added_by_event": 1 - math.exp(-trig(t1, t2) * s)}
    return res


def run():
    rng = np.random.default_rng(20261118)
    cat = load()
    d = design(cat)
    tr = (d["year"] >= TRAIN[0]) & (d["year"] < TRAIN[1] + 1)
    te = (d["year"] >= TEST[0]) & (d["year"] < TEST[1] + 1)
    par = fit(d, tr)
    ev, ok1, ok2 = evaluate(d, par, te, rng)
    imp = event_impact(cat, par)
    res = {"catalogue": "USGS ComCat M>=4.5, 22-56N 38-122E, 1973-2025", "triggers_train": int(tr.sum()), "triggers_test": int(te.sum()),
           "parameters": dict(zip(("beta", "A", "alpha", "c_days", "p"), par)), "test": ev, "event_2026_11_03": imp,
           "checks": {"log_score_gain": ok1, "calibration": ok2}, "verdict": "strong" if ok1 and ok2 else "not strong"}
    save("event_response.json", res)
    print(f"triggers: train {tr.sum()} (1973-1999), test {te.sum()} (2000-2024) | parameters " + ", ".join(f"{k} {v:.3g}" for k, v in res["parameters"].items()))
    for H, r in ev.items():
        print(f"{H} d: base rate {r['base_rate']:.3f}, forecast {r['mean_forecast']:.3f}, Poisson {r['mean_poisson']:.3f} | log-score gain {r['log_score_gain']:+.4f} [{r['gain_ci95'][0]:+.4f}, {r['gain_ci95'][1]:+.4f}]"
              f" | Brier {r['brier_model']:.4f} vs {r['brier_poisson']:.4f} | calibration bins ok {r['bins_ok']}/5: " + "; ".join(f"{b['mean_forecast']:.3f}->{b['observed']:.3f}" for b in r["calibration"]))
    for k in ("first_30_days_at_event_time", "first_year_at_event_time", "from_now_to_maturity"):
        x = imp[k]
        print(f"3 Nov 2026 event, {k}: P(M>=5.5 within 50 km) background {x['p_m55_background']:.4f} -> with event {x['p_m55_with_event']:.4f}")
    for cty in M_PAY:
        x = imp[f"pay_size_{cty}"]
        print(f"  pay-size (M>={x['magnitude']}) to maturity, {cty}: background {x['p_background']:.5f}, added by the event {x['p_added_by_event']:.6f}")
    print("checks:", res["checks"], "| verdict:", res["verdict"])
    return res


def selftest():
    # the fit recovers the parameters of data simulated from the model
    rng = np.random.default_rng(3)
    n = 6000
    d = {"m": rng.uniform(5.0, 7.0, n), "b": rng.gamma(2.0, 2e-4, n), "year": rng.uniform(1973, 2000, n)}
    true = (0.7, 0.08, 0.9, 0.05, 1.15)
    e = np.array(BINS, float)
    d["cnt"] = np.stack([rng.poisson(expected(true, d["m"], d["b"], e[k], e[k + 1])) for k in range(len(e) - 1)], 1).astype(float)
    par = fit(d, np.ones(n, bool))
    tot = lambda q: float(expected(q, 6.0, 4e-4, 0.0, 365.0))
    assert abs(tot(par) / tot(true) - 1) < 0.08 and abs(par[2] - true[2]) < 0.15, par
    assert abs(dist_km(0, 0, 0, 1) - 111.19) < 0.1
    print("selftest ok: fit recovers a simulated model; great-circle distance")


if __name__ == "__main__":
    selftest()
    run()
