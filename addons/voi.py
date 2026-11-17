"""Global sensitivity + value of information for the friend's two-country bond. Additive; exact targets via
variant.py (reproduces the friend's model; checked in rates_unc's selftest).

Question: of the uncertainty in the bond's risk numbers, how much comes from each uncertain input, and how much would
new data on each remove? Inputs (independent groups), all from existing posteriors:
  s_eq        earthquake shared share: the friend's NOAA 1900–2020 posterior (param_uncertainty.eq_posterior)
  s_rain      rain shared share: the friend's station posterior (param_uncertainty.rain_posterior)
  eq_rates    KG / TJ earthquake trigger-rate multipliers (rates_unc: Gutenberg–Richter b and Poisson counts; the two
              countries share b, so they form one group)
  rain_rates  KG / TJ rain trigger-rate multipliers (rates_unc: GEV-GML bootstrap)
Outputs Y = log10 of joint exhaustion (headline), expected loss and joint attachment (multiplicative uncertainty).
Sobol indices by group: first order S_g = Var(E[Y|X_g]) / Var(Y) (Saltelli 2010 estimator), total T_g (Jansen),
N pick-freeze rows, bootstrap 95% intervals over rows.
Value of information: learning X_g exactly removes S_g of Var(Y) on average (the 95% interval narrows by about
1 − sqrt(1 − S_g)). With Δ more record years on top of the n_g behind a posterior that shrinks like 1/n, the removed
share is ≈ S_g·Δ/(n_g + Δ); for the two shares this is checked against the exact Beta-binomial preposterior analysis
(h(θ) = E[Y | θ] on an equal-probability grid of the current posterior, the same inner draws at every grid point).
Side line: the contract's event definition (eq_cluster reading A vs B, ν = 0.5) is not data uncertainty; its effect is
reported as the variance it would add if the reading were a coin flip (resolved for free by reading the contract).
Pre-set rule: "strong" if (1) the method checks pass: grid vs Saltelli first-order index of each share within 0.05,
Σ S_g <= 1.05, T_g >= S_g − 0.05, asymptotic vs exact preposterior share within 0.02 absolute or 20%; (2) on log10
joint exhaustion the bootstrap 95% intervals of the top two total indices do not overlap; (3) the top group has
S_g >= 0.5 (perfect information on one input narrows the 95% interval by >= 30%).
"""
import math

import numpy as np
from scipy import stats

import rates_unc as RU
import variant
from _src import in_repo, save
from src.realdata import param_uncertainty as PU

N, NB, SEED = 4096, 1000, 20261113
GRID, INNER = 64, 1000
OUT = {"joint_exhaustion": RU.KEYS["joint_exhaustion"], "expected_loss": RU.KEYS["expected_loss"],
       "joint_attachment": RU.KEYS["joint_attachment"]}
GROUPS = {"s_eq": ("s_eq",), "s_rain": ("s_rain",), "eq_rates": ("eqKG", "eqTJ"), "rain_rates": ("rnKG", "rnTJ")}
RECORD_YEARS = {"s_eq": 121, "s_rain": 45, "eq_rates": 52, "rain_rates": 46}   # years of record behind each posterior
DELTAS = (10, 30, 100)
NU_B = 0.5                                                                     # eq_cluster reading B (M >= 6.5 point)


def evaluate(x):
    try:
        t = variant.targets(m=x["s_eq"] / PU.BRIEF_EQ, eq_mult=(x["eqKG"], x["eqTJ"]), rain_mult=(x["rnKG"], x["rnTJ"]),
                            rain_share=min(x["s_rain"], 1.0), nu=x.get("nu", 0.0))
        y = [math.log10(t[k]) for k in OUT.values()]
        return y if all(np.isfinite(y)) else [np.nan] * len(OUT)
    except Exception:
        return [np.nan] * len(OUT)


def rows(X, idx=None):
    idx = range(len(X["s_eq"])) if idx is None else idx
    return [{k: float(v[i]) for k, v in X.items()} for i in idx]


def sobol(YA, YB, YAB, rng):
    """First-order (Saltelli 2010) and total (Jansen) indices with bootstrap 95% intervals; rows with any NaN dropped."""
    ok = np.isfinite(YA) & np.isfinite(YB) & np.all([np.isfinite(v) for v in YAB.values()], 0)
    m = np.mean(np.r_[YA[ok], YB[ok]])               # centre: same expectation, far less variance (log10 EL ≈ 7)
    A, B, AB = YA[ok] - m, YB[ok] - m, {g: v[ok] - m for g, v in YAB.items()}

    def est(i):
        V = np.var(np.r_[A[i], B[i]])
        return ({g: float(np.mean(B[i] * (AB[g][i] - A[i])) / V) for g in AB},
                {g: float(0.5 * np.mean((A[i] - AB[g][i]) ** 2) / V) for g in AB}, V)

    S, T, V = est(np.arange(len(A)))
    boot = [est(rng.integers(0, len(A), len(A))) for _ in range(NB)]
    ci = lambda which, g: [float(np.quantile([b[which][g] for b in boot], q)) for q in (0.025, 0.975)]
    return {"S": S, "T": T, "S_ci": {g: ci(0, g) for g in AB}, "T_ci": {g: ci(1, g) for g in AB}, "var": float(V),
            "rows_used": int(ok.sum())}


def preposterior(h, a, b, trials_per_year, V):
    """Exact share of Var(Y) removed by Δ more years of Beta-binomial data, from h on the equal-probability grid."""
    theta = stats.beta.ppf((np.arange(GRID) + 0.5) / GRID, a, b)
    out = {}
    for d in DELTAS:
        m = max(int(round(trials_per_year * d)), 1)
        k = np.arange(m + 1)
        pmf = stats.betabinom.pmf(k, m, a, b)
        logw = k[:, None] * np.log(theta)[None, :] + (m - k)[:, None] * np.log1p(-theta)[None, :]
        w = np.exp(logw - logw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        post_mean = w @ h
        mu = pmf @ post_mean
        out[d] = {"trials": m, "share_removed": float(pmf @ (post_mean - mu) ** 2 / V)}
    return out


def run(procs=14):
    from multiprocessing import Pool
    rng = np.random.default_rng(SEED)
    with in_repo():
        eqp, rnp = PU.eq_posterior(), PU.rain_posterior()
    em, _ = RU.eq_multipliers(rng, 2 * N)
    rm, _ = RU.rain_multipliers(rng, 2 * N)
    full = {"s_eq": rng.beta(eqp["a"], eqp["b"], 2 * N), "s_rain": rng.beta(rnp["a"], rnp["b"], 2 * N),
            "eqKG": em["KG"], "eqTJ": em["TJ"], "rnKG": rm["KG"], "rnTJ": rm["TJ"]}
    A = {k: v[:N] for k, v in full.items()}; B = {k: v[N:] for k, v in full.items()}
    AB = {g: {k: (B[k] if k in cols else A[k]) for k in A} for g, cols in GROUPS.items()}
    grids = {"s_eq": stats.beta.ppf((np.arange(GRID) + 0.5) / GRID, eqp["a"], eqp["b"]),
             "s_rain": stats.beta.ppf((np.arange(GRID) + 0.5) / GRID, rnp["a"], rnp["b"])}
    inner = rows(A, range(INNER))
    grid_rows = [{**r, g: float(th)} for g in grids for th in grids[g] for r in inner]
    nu_rows = [{**r, "nu": NU_B} for r in inner]
    with Pool(procs) as pool:
        ev = lambda R: np.array(pool.map(evaluate, R, chunksize=64))
        YA, YB = ev(rows(A)), ev(rows(B))
        YAB = {g: ev(rows(X)) for g, X in AB.items()}
        YG = ev(grid_rows).reshape(len(grids), GRID, INNER, len(OUT))
        YN = ev(nu_rows)
    res = {}
    for j, name in enumerate(OUT):
        s = sobol(YA[:, j], YB[:, j], {g: v[:, j] for g, v in YAB.items()}, rng)
        V = s["var"]
        grid = {}
        for gi, g in enumerate(grids):
            h = np.nanmean(YG[gi, :, :, j], 1)
            a, b = (eqp["a"], eqp["b"]) if g == "s_eq" else (rnp["a"], rnp["b"])
            n_trials = eqp["n_kg"] if g == "s_eq" else rnp["n_total"]
            exact = preposterior(h, a, b, n_trials / RECORD_YEARS[g], V)
            grid[g] = {"S_grid": float(np.var(h) / V), "h_range_log10": [float(h.min()), float(h.max())], "preposterior": exact}
        voi = {g: {"perfect": s["S"][g], "interval_narrowing_perfect": float(1 - math.sqrt(max(1 - s["S"][g], 0))),
                   "years": {d: float(s["S"][g] * d / (RECORD_YEARS[g] + d)) for d in DELTAS}} for g in GROUPS}
        y = np.r_[YA[:, j], YB[:, j]]; y = y[np.isfinite(y)]
        delta = YN[:, j] - YA[:INNER, j]
        dm = float(np.nanmean(delta))
        res[name] = {"sobol": s, "grid_check": grid, "voi": voi,
                     "log10_quantiles": {q: float(np.quantile(y, q)) for q in (0.025, 0.5, 0.975)},
                     "contract_reading_B": {"mean_log10_shift": dm, "factor": 10 ** dm,
                                            "share_if_coin_flip": float((dm ** 2 / 4) / (V + dm ** 2 / 4))}}
    je = res["joint_exhaustion"]
    s, g = je["sobol"], je["grid_check"]
    checks = {"grid_vs_saltelli": all(abs(g[k]["S_grid"] - s["S"][k]) <= 0.05 for k in g),
              "sum_S": sum(s["S"].values()) <= 1.05,
              "T_ge_S": all(s["T"][k] >= s["S"][k] - 0.05 for k in GROUPS),
              "preposterior": all(abs(g[k]["preposterior"][d]["share_removed"] - je["voi"][k]["years"][d]) <= max(0.02, 0.2 * g[k]["preposterior"][d]["share_removed"])
                                  for k in g for d in DELTAS)}
    order = sorted(GROUPS, key=lambda k: -s["T"][k])
    separated = s["T_ci"][order[0]][0] > s["T_ci"][order[1]][1]
    top_S = s["S"][order[0]]
    verdict = "strong" if all(checks.values()) and separated and top_S >= 0.5 else "not strong"
    out = {"n": N, "posteriors": {"eq_share": eqp, "rain_share": rnp}, "record_years": RECORD_YEARS, "outputs": res,
           "checks": checks, "ranking_T": order, "top_separated": bool(separated), "top_S": top_S, "verdict": verdict}
    save("voi.json", out)
    for name, r in res.items():
        q = r["log10_quantiles"]; s = r["sobol"]
        print(f"{name}: 95% range 10^[{q[0.025]:.2f}, {q[0.975]:.2f}] (median 10^{q[0.5]:.2f}), Var(log10) {s['var']:.3f}, rows {s['rows_used']}")
        for k in GROUPS:
            v = r["voi"][k]
            print(f"  {k:10s} S {s['S'][k]:.3f} [{s['S_ci'][k][0]:.3f}, {s['S_ci'][k][1]:.3f}]  T {s['T'][k]:.3f} [{s['T_ci'][k][0]:.3f}, {s['T_ci'][k][1]:.3f}]"
                  f"  | perfect info: interval -{v['interval_narrowing_perfect']:.0%}; +10/30/100 yr removes "
                  + "/".join(f"{v['years'][d]:.0%}" for d in DELTAS)
                  + (f"  | grid S {r['grid_check'][k]['S_grid']:.3f}, exact preposterior " + "/".join(f"{r['grid_check'][k]['preposterior'][d]['share_removed']:.0%}" for d in DELTAS)
                     if k in r["grid_check"] else ""))
        c = r["contract_reading_B"]
        print(f"  contract reading B: x{c['factor']:.2f}; as a coin flip it would carry {c['share_if_coin_flip']:.0%} of the variance")
    print("checks:", checks, "| ranking (T):", order, "| top separated:", separated, f"| top S {top_S:.3f} | verdict: {verdict}")
    return out


def selftest():
    # Sobol estimators recover the analytic indices of Y = X1 + 2 X2 (independent N(0,1)): S = (0.2, 0.8)
    rng = np.random.default_rng(0)
    n = 20000
    A, B = rng.standard_normal((n, 2)), rng.standard_normal((n, 2))
    f = lambda X: X[:, 0] + 2 * X[:, 1]
    AB = {0: np.column_stack([B[:, 0], A[:, 1]]), 1: np.column_stack([A[:, 0], B[:, 1]])}
    global NB
    nb, NB = NB, 50
    s = sobol(f(A), f(B), {g: f(X) for g, X in AB.items()}, rng)
    NB = nb
    assert abs(s["S"][0] - 0.2) < 0.03 and abs(s["S"][1] - 0.8) < 0.03 and abs(s["T"][1] - 0.8) < 0.03, s
    # preposterior: linear h(θ) = θ, removed share = Var(θ) · m / (a + b + m) / Var(θ) exactly
    a, b = 3.0, 7.0
    theta = stats.beta.ppf((np.arange(GRID) + 0.5) / GRID, a, b)
    V = float(np.var(theta))
    r = preposterior(theta, a, b, 1.0, V)
    for d in DELTAS:
        m = r[d]["trials"]
        assert abs(r[d]["share_removed"] - m / (a + b + m)) < 0.03, (d, r[d], m / (a + b + m))
    # the model evaluation reproduces the friend's brief point
    y = evaluate({"s_eq": PU.BRIEF_EQ, "s_rain": 0.25, "eqKG": 1, "eqTJ": 1, "rnKG": 1, "rnTJ": 1})
    assert abs(10 ** y[0] / variant.targets()[OUT["joint_exhaustion"]] - 1) < 1e-12
    print("selftest ok: Sobol estimators on an analytic case; Beta-binomial preposterior; brief point reproduced")


if __name__ == "__main__":
    selftest()
    run()
