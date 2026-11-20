"""Model-risk engine for the friend's CAREC pool: posterior predictive, P(μ > the friend's figure), global sensitivity
and value of information for pool-level targets. Additive; src/ read-only. This is V (voi.py) extended from the
two-country bond to the 10-country pool.

Uncertain inputs (independent groups), each the data behind one step of the friend's own calibration
(carec_pool.calibrate); every draw re-runs his formulas, so KG/TJ stay pinned to the brief exactly as in his design:
  eq_rates    per-country main-shock counts within 50 km (USGS 1965–2016, Jeffreys Gamma) + the regional
              Gutenberg–Richter b (Aki, normal with s.e. b/√N)
  eq_shared   per-edge counts of shocks reaching both countries (Jeffreys Gamma; edges stay the friend's set)
  rain_rates  per-country EM-DAT hydro-loss years 1990–2020 (Jeffreys Beta) -> his rain_p formula
  rain_share  the friend's station posterior (param_uncertainty.rain_posterior; his CAREC model uses the brief's 25%)
Shared earthquake rates that would exceed a country's total are scaled down to 95% of it (his Asia–Pacific rule).
Exact targets per draw: the dependence graph splits into independent components ({7 countries}, {GE, AZ}, {MN}); each
is enumerated densely with the full model's own kernels and the component loss distributions are convolved (the
friend's MPS takes ~130 s per draw; this reproduces his references). KG–TJ joint exhaustion by his exact pair reduction.
Targets (log10): pool exceedance at his moderate and rare thresholds (US$240m, US$500m), expected loss, KG–TJ joint
exhaustion. Headline: exceedance at US$240m.
Outputs: posterior predictive mean and 95% range per target, P(μ > the friend's calibrated value), Sobol indices
(voi.sobol, centred Saltelli first order + Jansen total, bootstrap intervals) and value of information (perfect
information S_g; Δ more record years ≈ S_g·Δ/(n_g + Δ)).
Pre-set rule (same as V): "strong" if, on the headline, the method checks pass (rain_share grid vs Saltelli within 0.05,
Σ S <= 1.05, T >= S − 0.05), the top two total indices have non-overlapping bootstrap intervals, and the top S >= 0.5.
"""
import json
import math

import numpy as np

import voi
from _src import in_repo, load, save, seed
from src.model.poisson_fit import TAILS, compound_poisson_rates
from src.rdpo.problem import PortfolioProblem
from src.realdata import carec_pool as CP
from src.realdata import eq_check as EQ
from src.realdata import param_uncertainty as PU
from src.realdata.central_asia_pool import DataModel, _load

N, GRID, INNER, PROCS, SEED = 512, 16, 64, 8, seed(20261116)
YEARS_EQ, YEARS_HY = EQ.YEARS[1] - EQ.YEARS[0] + 1, 31
RECORD_YEARS = {"eq_rates": YEARS_EQ, "eq_shared": YEARS_EQ, "rain_rates": YEARS_HY, "rain_share": 45}
UNIT = 2.5e6
TARGETS = ("exc_moderate", "exc_rare", "expected_loss", "joint_exhaustion_KG_TJ")
CAL = {}


def inputs():
    """The data behind the friend's calibration: counts, pair counts, b and its sample size, hydro years."""
    with in_repo():
        cal = CP.calibrate()
        cat, geo, hyd = CP._cache()
        ev, polys = _load(cat, geo)
        ev = EQ.decluster([e for e in ev if EQ.YEARS[0] <= e[0].year <= EQ.YEARS[1]])
        near = np.array([[EQ.dist_to_country(la, lo, polys[c]) <= CP.R_KM for c in cal["order"]] for _, la, lo, _ in ev])
        refs = load("wp2_carec.json")
    order = cal["order"]
    pairs = {tuple(sorted((order.index(a), order.index(b)))): k for key, k in cal["n_both"].items() for a, b in [key.split("-")]}
    return {"order": order, "edges": [tuple(e) for e in cal["model"].edges], "pair": cal["pair"], "model": cal["model"],
            "counts": {c: cal["rate55"][c] * YEARS_EQ for c in order}, "pairs": pairs, "b": float(cal["b"]),
            "n_b": int(near.any(1).sum()), "hydro_years": {c: round(cal["hydro_freq"][c] * YEARS_HY) for c in order},
            "T": (refs["T_moderate"], refs["T_rare"]), "refs": refs["references"]}


def model_from(inp, th):
    """The friend's calibration formulas (carec_pool.calibrate) applied to drawn data."""
    order, b, years = inp["order"], th["b"], inp.get("years_eq", YEARS_EQ)
    rate = {c: th["counts"][c] / years for c in order}
    m1 = np.mean([5.5 + math.log10(rate[c] / -math.log(1 - TAILS[c][0])) / b for c in ("KG", "TJ")])
    tier = np.mean([np.log10(TAILS[c][0] / TAILS[c]) / b for c in ("KG", "TJ")], axis=0)
    tails = [1 - np.exp(-rate[c] * 10 ** (-b * (m1 + tier - 5.5))) for c in order]
    scale = 10 ** (-b * (m1 - 5.5))
    shared = {e: th["pairs"][e] / years * scale for e in inp["edges"]}
    Lam = [compound_poisson_rates(t)[1] for t in tails]
    for i in range(len(order)):                                   # feasibility (friend's Asia–Pacific rule)
        tot = sum(v for e, v in shared.items() if i in e)
        if tot > 0.95 * Lam[i]:
            for e in shared:
                if i in e:
                    shared[e] *= 0.95 * Lam[i] / tot
    fh = th["fh"]; fref = 0.5 * (fh["KG"] + fh["TJ"])
    rain_p = [float(np.clip(0.02 * fh[c] / fref, 0.005, 0.08)) for c in order]
    return DataModel(len(order), inp["edges"], tails=tails, shared_eq=shared, rain_p=rain_p, rain_shared=th["s_rain"])


def components(n, edges):
    parent = list(range(n))
    find = lambda x: x if parent[x] == x else find(parent[x])
    for i, j in edges:
        parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def loss_distribution(model):
    """Exact distribution of the pool loss (units of US$2.5m): dense enumeration per independent component with the
    full model's kernels, then convolution."""
    units = np.asarray(model.site_loss_units(UNIT), dtype=np.int64)
    dist = np.ones(1)
    for sites in components(model.n, model.edges):
        k, pos = len(sites), {s: t for t, s in enumerate(sites)}
        P = np.zeros((10,) * k); P[(0,) * k] = 1.0
        for s in sites:
            P = np.moveaxis(np.tensordot(P, model.single_kernel(s), axes=([pos[s]], [0])), -1, pos[s])
        for i, j in model.edges:
            if i in pos:
                K = model.pair_kernel(i, j).reshape(10, 10, 10, 10)
                P = np.moveaxis(np.tensordot(P, K, axes=([pos[i], pos[j]], [0, 1])), [-2, -1], [pos[i], pos[j]])
        L = np.zeros((10,) * k, dtype=np.int64)
        for t in range(k):
            L += units.reshape([10 if a == t else 1 for a in range(k)])
        dist = np.convolve(dist, np.bincount(L.ravel(), weights=P.ravel()))
    return dist


def targets(model, inp):
    d = loss_distribution(model)
    L = np.arange(len(d))
    exc = lambda T: float(d[L > T * 1e6 / UNIT].sum())
    je = PortfolioProblem(model, "joint_exhaustion", pair=inp["pair"]).exact_pair()
    return {"exc_moderate": exc(inp["T"][0]), "exc_rare": exc(inp["T"][1]), "expected_loss": float(d @ L) * UNIT,
            "joint_exhaustion_KG_TJ": je}


def _init(inp):
    CAL.update(inp)


def evaluate(th):
    try:
        t = targets(model_from(CAL, th), CAL)
        return [math.log10(t[k]) for k in TARGETS]
    except Exception:
        return [np.nan] * len(TARGETS)


GROUPS = {"eq_rates": ("counts", "b"), "eq_shared": ("pairs",), "rain_rates": ("fh",), "rain_share": ("s_rain",)}


def draw(inp, rng, k, rnp):
    o = inp["order"]
    b_se = inp.get("b_se", inp["b"] / math.sqrt(inp["n_b"]))
    return [{"counts": {c: rng.gamma(inp["counts"][c] + 0.5) for c in o}, "b": max(rng.normal(inp["b"], b_se), 0.3),
             "pairs": {e: rng.gamma(v + 0.5) for e, v in inp["pairs"].items()},
             "fh": {c: rng.beta(inp["hydro_years"][c] + 0.5, YEARS_HY - inp["hydro_years"][c] + 0.5) for c in o},
             "s_rain": float(rng.beta(rnp["a"], rnp["b"]))} for _ in range(k)]


def run():
    from multiprocessing import Pool
    rng = np.random.default_rng(SEED)
    inp = inputs()
    with in_repo():
        rnp = PU.rain_posterior()
    obs = {"counts": inp["counts"], "b": inp["b"], "pairs": inp["pairs"], "fh": {c: inp["hydro_years"][c] / YEARS_HY for c in inp["order"]}}
    friend = targets(model_from(inp, dict(obs, s_rain=0.25)), inp)
    center = targets(model_from(inp, dict(obs, s_rain=float(np.median(rng.beta(rnp["a"], rnp["b"], 200_000))))), inp)
    A, B = draw(inp, rng, N, rnp), draw(inp, rng, N, rnp)
    AB = {g: [dict(a, **{k: b[k] for k in keys}) for a, b in zip(A, B)] for g, keys in GROUPS.items()}
    grid_s = np.quantile(rng.beta(rnp["a"], rnp["b"], 200_000), (np.arange(GRID) + 0.5) / GRID)
    grid_rows = [dict(a, s_rain=float(s)) for s in grid_s for a in A[:INNER]]
    slim = {k: v for k, v in inp.items() if k != "model"}
    with Pool(PROCS, initializer=_init, initargs=(slim,), maxtasksperchild=50) as pool:
        ev = lambda R: np.array(pool.map(evaluate, R, chunksize=4))
        YA, YB = ev(A), ev(B)
        YAB = {g: ev(R) for g, R in AB.items()}
        YG = ev(grid_rows).reshape(GRID, INNER, len(TARGETS))
    out = {"pool": inp["order"], "thresholds_musd": list(inp["T"]), "n": N, "friend_point": friend, "data_center_point": center,
           "inputs": {"b": inp["b"], "n_b": inp["n_b"], "counts": inp["counts"], "pairs": {f"{inp['order'][i]}-{inp['order'][j]}": v for (i, j), v in inp["pairs"].items()},
                      "hydro_years": inp["hydro_years"], "rain_share_posterior": rnp},
           "record_years": RECORD_YEARS, "targets": {}}
    for j, name in enumerate(TARGETS):
        s = voi.sobol(YA[:, j], YB[:, j], {g: v[:, j] for g, v in YAB.items()}, rng)
        y = np.r_[YA[:, j], YB[:, j]]; y = y[np.isfinite(y)]; mu = 10 ** y
        h = np.nanmean(YG[:, :, j], 1)
        out["targets"][name] = {
            "posterior_predictive": float(mu.mean()), "range95": [float(np.quantile(mu, 0.025)), float(np.quantile(mu, 0.975))],
            "median": float(np.median(mu)), "friend": friend[name], "P_above_friend": float(np.mean(mu > friend[name])),
            "sobol": s, "grid_S_rain_share": float(np.var(h) / s["var"]),
            "voi": {g: {"perfect": s["S"][g], "interval_narrowing": float(1 - math.sqrt(max(1 - s["S"][g], 0))),
                        "years": {d: float(s["S"][g] * d / (RECORD_YEARS[g] + d)) for d in (10, 30, 100)}} for g in GROUPS}}
    hd = out["targets"]["exc_moderate"]; s = hd["sobol"]
    checks = {"grid_vs_saltelli": abs(hd["grid_S_rain_share"] - s["S"]["rain_share"]) <= 0.05, "sum_S": sum(s["S"].values()) <= 1.05,
              "T_ge_S": all(s["T"][g] >= s["S"][g] - 0.05 for g in GROUPS)}
    rank = sorted(GROUPS, key=lambda g: -s["T"][g])
    sep = s["T_ci"][rank[0]][0] > s["T_ci"][rank[1]][1]
    out.update(checks=checks, ranking_T=rank, top_separated=bool(sep), top_S=s["S"][rank[0]],
               verdict="strong" if all(checks.values()) and sep and s["S"][rank[0]] >= 0.5 else "not strong")
    save("model_risk.json", out)
    print(f"CAREC pool {inp['order']}; thresholds US${inp['T'][0]}m / US${inp['T'][1]}m; b {inp['b']:.3f} (N {inp['n_b']})")
    for name, t in out["targets"].items():
        s = t["sobol"]
        print(f"{name:24s} friend {t['friend']:.4g} | data center {center[name]:.4g} | posterior predictive {t['posterior_predictive']:.4g} "
              f"95% [{t['range95'][0]:.3g}, {t['range95'][1]:.3g}] | P(above friend) {t['P_above_friend']:.2f}")
        print("    " + "  ".join(f"{g} S {s['S'][g]:.2f} T {s['T'][g]:.2f} [{s['T_ci'][g][0]:.2f},{s['T_ci'][g][1]:.2f}]" for g in GROUPS)
              + f" | grid S rain_share {t['grid_S_rain_share']:.3f}")
    print("checks:", checks, "| ranking:", rank, "| separated:", sep, f"| top S {out['top_S']:.2f} | verdict: {out['verdict']}")
    return out


def selftest():
    """The engine reproduces the friend's calibrated CAREC model and his MPS reference values."""
    inp = inputs()
    obs = {"counts": inp["counts"], "b": inp["b"], "pairs": inp["pairs"], "fh": {c: inp["hydro_years"][c] / YEARS_HY for c in inp["order"]}, "s_rain": 0.25}
    m, f = model_from(inp, obs), inp["model"]
    assert all(np.allclose(a, b_, rtol=1e-12) for a, b_ in zip(m.tails, f.tails))
    assert all(abs(m.c[e] / f.c[e] - 1) < 1e-12 for e in f.c) and np.allclose(m.rain_p, f.rain_p, rtol=1e-12)
    t, r = targets(m, inp), inp["refs"]
    T1, T2 = inp["T"]
    for mine, ref in ((t["exc_moderate"], r[f"exceedance|{T1}"]), (t["exc_rare"], r[f"exceedance|{T2}"]), (t["expected_loss"], r["expected_loss|None"]),
                      (t["joint_exhaustion_KG_TJ"], r["joint_exhaustion|None"])):
        assert abs(mine / ref - 1) < 1e-6, (mine, ref)
    assert sorted(len(c) for c in components(m.n, m.edges)) == [1, 2, 7]
    print("selftest ok: friend's CAREC calibration reproduced; exact targets equal his MPS references (rel < 1e-6)")


if __name__ == "__main__":
    selftest()
    run()
