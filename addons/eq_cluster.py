"""Quality 2: earthquake trigger clustering (method borrowed from QContagion's Hawkes estimation). Additive.

The friend's eq_check tests "Poisson after removing aftershocks" (Gardner–Knopoff). The bond model is a compound
Poisson process calibrated to the brief's annual payout tails, so clustering *within* a year is already inside those
tails if they were set on all payable events. Two channels remain open, and both are measured here on the friend's
own cached USGS catalogue (M >= 5.5, 1965–2016, within 50 km of KG / TJ):
  1. Contract reading. If each aftershock above the trigger is a separate payable event, while the tails describe
     main shocks only, every event brings ν extra events. ν is measured directly as (raw − declustered) / declustered
     at trigger-near magnitudes, and pushed through the exact model (variant.py, nu = ν). The public deal reports do
     not state the event definition (hours clause), so both readings are shown.
  2. Dependence across years (the term convolution assumes independent years): lag-1 autocorrelation of yearly
     main-shock counts with a permutation p-value.
Also: dispersion of raw vs declustered yearly counts, and a temporal Hawkes fit (exponential kernel) on raw times:
branching ratio n̂ with a profile-likelihood 95% interval.
"""
import math

import numpy as np
from scipy import stats
from scipy.optimize import minimize

import variant
from _src import in_repo, save, seed
from src.realdata import eq_check as EQ

R_KM, MAGS = 50.0, (5.5, 6.0, 6.5, 6.9)


def catalogue():
    with in_repo():
        ev, polys = EQ.load()
    ev = [e for e in ev if EQ.YEARS[0] <= e[0].year <= EQ.YEARS[1]]
    dec = EQ.decluster(ev)
    near = lambda E: {c: np.array([EQ.dist_to_country(la, lo, polys[c]) <= R_KM for _, la, lo, _ in E]) for c in ("KG", "TJ")}
    return ev, dec, near(ev), near(dec)


def yearly(E, mask):
    ny = EQ.YEARS[1] - EQ.YEARS[0] + 1
    yrs = np.array([e[0].year for e in E])[mask] - EQ.YEARS[0]
    return np.bincount(yrs, minlength=ny)


def dispersion(c):
    m, v = c.mean(), c.var(ddof=1)
    d = v / m if m > 0 else float("nan")
    return d, float(stats.chi2.sf((len(c) - 1) * d, len(c) - 1)) if m > 0 else float("nan")


def lag1(c, rng, nperm=5000):
    if c.std() == 0:
        return float("nan"), float("nan")
    r = np.corrcoef(c[:-1], c[1:])[0, 1]
    null = []
    for _ in range(nperm):
        x = rng.permutation(c)
        null.append(np.corrcoef(x[:-1], x[1:])[0, 1])
    return float(r), float((1 + np.sum(np.array(null) >= r)) / (1 + nperm))


def hawkes_nll(params, t, T):
    p0, p1, p2 = (float(np.clip(x, -30, 30)) for x in params)    # keep the optimiser inside float range
    mu, n, beta = math.exp(p0), 1 / (1 + math.exp(-p1)), math.exp(p2)
    A, ll = 0.0, 0.0
    for i in range(len(t)):
        if i:
            A = math.exp(-beta * (t[i] - t[i - 1])) * (1 + A)
        ll += math.log(mu + n * beta * A)
    ll -= mu * T + n * np.sum(1 - np.exp(-beta * (T - t)))
    return -ll


def hawkes_fit(t, T):
    best = None
    for b0 in (1 / 3, 1 / 30, 1 / 300):
        for n0 in (-2.0, 0.0, 1.0):
            r = minimize(hawkes_nll, [math.log(len(t) / T), n0, math.log(b0)], args=(t, T), method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-8})
            if best is None or r.fun < best.fun:
                best = r
    n_hat = 1 / (1 + math.exp(-best.x[1]))
    prof = []
    for n in np.linspace(0.0, 0.95, 39):
        lg = math.log(max(n, 1e-9) / (1 - n))
        f = lambda p: hawkes_nll([p[0], lg, p[1]], t, T)
        rr = min((minimize(f, [math.log(len(t) / T), math.log(b0)], method="Nelder-Mead") for b0 in (1 / 3, 1 / 30, 1 / 300)),
                 key=lambda x: x.fun)
        prof.append((n, rr.fun))
    ok = [n for n, v in prof if 2 * (v - best.fun) <= stats.chi2.ppf(0.95, 1)]
    return {"n_hat": n_hat, "mu_per_year": math.exp(best.x[0]) * 365.25, "decay_days": 1 / math.exp(best.x[2]),
            "n_ci95": [min(ok), max(ok)] if ok else None, "events": len(t)}


def run():
    ev, dec, near_r, near_d = catalogue()
    rng = np.random.default_rng(seed(20261107))
    mag_r, mag_d = np.array([e[3] for e in ev]), np.array([e[3] for e in dec])
    out = {"radius_km": R_KM, "years": list(EQ.YEARS), "by_magnitude": {}}
    for M in MAGS:
        row = {}
        for c in ("KG", "TJ", "KG_or_TJ"):
            mr = (near_r["KG"] | near_r["TJ"] if c == "KG_or_TJ" else near_r[c]) & (mag_r >= M)
            md = (near_d["KG"] | near_d["TJ"] if c == "KG_or_TJ" else near_d[c]) & (mag_d >= M)
            raw, dc = int(mr.sum()), int(md.sum())
            cr, cd = yearly(ev, mr), yearly(dec, md)
            k_after = raw - dc
            row[c] = {"raw": raw, "declustered": dc, "aftershock_share": k_after / max(raw, 1),
                      "nu": k_after / max(dc, 1),
                      "nu_ci95": [float(stats.gamma.ppf(.025, k_after + .5) / max(dc, 1)),
                                  float(stats.gamma.ppf(.975, k_after + .5) / max(dc, 1))],
                      "dispersion_raw": dispersion(cr), "dispersion_declustered": dispersion(cd),
                      "lag1_declustered": lag1((cd > 0).astype(float), rng)}
        out["by_magnitude"][str(M)] = row
    T0 = np.datetime64(f"{EQ.YEARS[0]}-01-01")
    sel = (near_r["KG"] | near_r["TJ"]) & (mag_r >= 5.5)
    t = np.sort(np.array([(np.datetime64(e[0].date()) - T0).astype(int) for e, s in zip(ev, sel) if s], float))
    T = float((np.datetime64(f"{EQ.YEARS[1] + 1}-01-01") - T0).astype(int))
    out["hawkes_raw_m55"] = hawkes_fit(t, T)
    # impact under contract reading B (aftershocks paid separately), nu measured at trigger-near magnitudes
    base = variant.targets()
    imp = {"reading_A_aftershocks_merged": base}
    for M in ("6.5", "6.0"):
        x = out["by_magnitude"][M]["KG_or_TJ"]
        for lab, nu in (("point", x["nu"]), ("ci_high", x["nu_ci95"][1])):
            imp[f"reading_B_nu_M{M}_{lab}"] = {"nu": nu, **variant.targets(nu=nu)}
    out["impact_adb_base"] = imp
    K = "term_both_bonds_exhausted_probability"
    for M, r in out["by_magnitude"].items():
        x = r["KG_or_TJ"]
        print(f"M>={M}: raw {x['raw']}, declustered {x['declustered']}, aftershock share {x['aftershock_share']:.0%}, "
              f"ν={x['nu']:.2f} [{x['nu_ci95'][0]:.2f},{x['nu_ci95'][1]:.2f}]  dispersion raw {x['dispersion_raw'][0]:.2f} "
              f"(p={x['dispersion_raw'][1]:.3f}) vs decl {x['dispersion_declustered'][0]:.2f}  lag1 decl r={x['lag1_declustered'][0]:.2f} "
              f"(p={x['lag1_declustered'][1]:.2f})")
    print("Hawkes raw M>=5.5:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out["hawkes_raw_m55"].items()})
    for k, v in imp.items():
        print(f"{k:34s} ν={v.get('nu', 0):.2f}  joint exhaustion {v[K]:.3g} (x{v[K] / base[K]:.2f})  "
              f"joint attachment {v['term_joint_attachment_probability']:.4f}  EL {v['term_expected_principal_loss_usd'] / 1e6:.2f}m")
    save("quality_eq_cluster.json", out)
    return out


if __name__ == "__main__":
    variant.selftest()
    run()
