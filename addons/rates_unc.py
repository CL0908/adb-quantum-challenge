"""Quality 4: put the base trigger rates' data uncertainty next to the shared shares' (lesson from QContagion:
parameter error dominates estimation error). Additive; the friend's param_uncertainty is reused, not changed.

The friend's param_uncertainty samples only the two shared shares (Jeffreys-Beta posteriors) and keeps the brief's
trigger rates fixed. Here the trigger rates get data-based uncertainty too:
  * earthquake, per country: λ_trig = r55 · 10^(−b (M_trig − 5.5)) with the friend's own inputs (eq_check): r55 =
    declustered M >= 5.5 rate within 50 km (Jeffreys Gamma on the count over 52 years), b = Aki estimate with its
    standard error (one b for both countries), M_trig = the friend's implied trigger magnitude (so the median
    multiplier is 1 by construction; only the spread is new);
  * rain, per country: the 1-in-50 threshold of the friend's daily country index, from a GEV fit to the annual
    maxima; bootstrap over years gives the spread of the exceedance probability of that fixed threshold. The GEV is
    fitted by generalized maximum likelihood with the Martins–Stedinger (2000, WRR 36(3)) geophysical shape prior
    (κ + 0.5 ~ Beta(6, 9)); plain MLE on ~60 maxima often returns a bounded upper tail and a zero 2% probability.
Three sweeps with the same draws: shares only (= friend), rates only, all. Targets are exact (variant.py, which
reproduces the friend's model; the share-only sweep is checked against param_uncertainty's own numbers).
"""
import math

import numpy as np
from scipy import stats

import variant
from _src import in_repo, load, save
from src.realdata import param_uncertainty as PU
from src.realdata import rain_check as PU_RC

N, NB, SEED = 2000, 400, 4243
KEYS = {"joint_exhaustion": "term_both_bonds_exhausted_probability", "exceedance": "term_portfolio_loss_exceedance_probability",
        "joint_attachment": "term_joint_attachment_probability", "expected_loss": "term_expected_principal_loss_usd"}


def country_data():
    """The friend's GHCN country index per country (rain_check.select_stations / country_index, read-only)."""
    with in_repo():
        st = PU_RC.select_stations()
        return {c: dict(zip(("idx", "years", "stations"), PU_RC.country_index(s))) for c, s in st.items()}


def eq_multipliers(rng, n):
    eq = load("realdata_eq.json")
    gr, r50 = eq["gutenberg_richter"], eq["radii_km"]["50"]["5.5"]
    years = eq["data"]["years"][1] - eq["data"]["years"][0] + 1
    b = rng.normal(gr["b"], gr["b_se"], n)
    out = {}
    for c in ("KG", "TJ"):
        k, mt = r50[c]["events"], gr[c]["implied_trigger_magnitude"]
        r55 = rng.gamma(k + 0.5, 1.0, n) / years
        out[c] = (r55 / gr[c]["rate_M55"]) * 10 ** (-(b - gr["b"]) * (mt - 5.5))
    return out, {"b": gr["b"], "b_se": gr["b_se"], "M_trig": {c: gr[c]["implied_trigger_magnitude"] for c in ("KG", "TJ")}}


def gev_gml(x):
    """GEV by generalized ML: log-likelihood + log Beta(6, 9) prior on κ + 0.5 (scipy's shape c is κ)."""
    from scipy.optimize import minimize
    c0, l0, s0 = stats.genextreme.fit(x, 0.0)
    def nll(p):
        c, l, ls = p
        if not -0.5 < c < 0.5:
            return np.inf
        ll = stats.genextreme.logpdf(x, c, l, math.exp(ls)).sum()
        return -(ll + stats.beta.logpdf(c + 0.5, 6, 9)) if np.isfinite(ll) else np.inf
    r = minimize(nll, [float(np.clip(c0, -0.45, 0.45)), l0, math.log(s0)], method="Nelder-Mead",
                 options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
    return r.x[0], r.x[1], math.exp(r.x[2])


def rain_multipliers(rng, n):
    ci = country_data()
    out, info = {}, {}
    for c in ("KG", "TJ"):
        ys = set(ci[c]["years"])
        amax = {}
        for key, v in ci[c]["idx"].items():
            y = int(key[:4])
            if y in ys:
                amax[y] = max(amax.get(y, 0.0), v)
        x = np.array(list(amax.values()))
        shape, loc, scale = gev_gml(x)
        thr = stats.genextreme.isf(0.02, shape, loc, scale)
        ps = []
        for _ in range(NB):
            xb = rng.choice(x, len(x), replace=True)
            ps.append(stats.genextreme.sf(thr, *gev_gml(xb)))
        ps = np.clip(np.array(ps), 1e-6, 0.5)
        mult = -np.log1p(-ps) / -math.log(0.98)
        out[c] = rng.choice(mult, n, replace=True)
        info[c] = {"annual_maxima": len(x), "gev_shape": float(shape), "threshold_2pct": float(thr),
                   "p_boot_q": [float(np.quantile(ps, q)) for q in (0.025, 0.5, 0.975)]}
    return out, info


def sweep(draws):
    vals = {k: [] for k in KEYS}
    bad = 0
    for d in draws:
        try:
            t = variant.targets(m=d["s_eq"] / PU.BRIEF_EQ, eq_mult=(d["eqKG"], d["eqTJ"]),
                                rain_mult=(d["rnKG"], d["rnTJ"]), rain_share=min(d["s_rain"], 1.0))
        except Exception:
            bad += 1
            continue
        if not all(np.isfinite(list(t.values()))) or t[KEYS["joint_exhaustion"]] < 0:
            bad += 1
            continue
        for k, kk in KEYS.items():
            vals[k].append(t[kk])
    return {k: PU.summarise(v) for k, v in vals.items()}, bad


def run():
    rng = np.random.default_rng(SEED)
    with in_repo():
        eqp, rnp = PU.eq_posterior(), PU.rain_posterior()
    s_eq, s_rn = rng.beta(eqp["a"], eqp["b"], N), rng.beta(rnp["a"], rnp["b"], N)
    em, einfo = eq_multipliers(rng, N)
    rm, rinfo = rain_multipliers(rng, N)
    brief = {"s_eq": PU.BRIEF_EQ, "s_rain": 0.25}
    one = {"eqKG": 1.0, "eqTJ": 1.0, "rnKG": 1.0, "rnTJ": 1.0}
    mk = lambda i, shares, rates: {**(dict(s_eq=s_eq[i], s_rain=s_rn[i]) if shares else brief),
                                   **({"eqKG": em["KG"][i], "eqTJ": em["TJ"][i], "rnKG": rm["KG"][i], "rnTJ": rm["TJ"][i]} if rates else one)}
    out = {"n": N, "posteriors": {"earthquake_share": eqp, "rain_share": rnp}, "eq_rate": einfo, "rain_rate": rinfo,
           "multiplier_q": {k: [float(np.quantile(v, q)) for q in (0.025, 0.5, 0.975)] for k, v in
                            {"eqKG": em["KG"], "eqTJ": em["TJ"], "rnKG": rm["KG"], "rnTJ": rm["TJ"]}.items()},
           "brief_point": {k: variant.targets()[kk] for k, kk in KEYS.items()}, "sweeps": {}}
    for lab, sh, ra in (("shares_only", True, False), ("rates_only", False, True), ("all", True, True)):
        s, bad = sweep([mk(i, sh, ra) for i in range(N)])
        out["sweeps"][lab] = {"summary": s, "invalid_draws": bad}
    friend = load("param_uncertainty.json")["posterior_summary"]["joint_exhaustion"]
    out["friend_shares_only_joint_exhaustion"] = friend
    b = out["brief_point"]["joint_exhaustion"]
    print(f"friend (shares only): median {friend['median']:.3g}, 95% [{friend['ci95'][0]:.3g}, {friend['ci95'][1]:.3g}]")
    for lab, r in out["sweeps"].items():
        s = r["summary"]["joint_exhaustion"]
        print(f"{lab:12s} joint exhaustion median {s['median']:.3g} ({s['median'] / b:.2f}x brief), 95% [{s['ci95'][0]:.3g}, "
              f"{s['ci95'][1]:.3g}] width x{(s['ci95'][1] - s['ci95'][0]) / (friend['ci95'][1] - friend['ci95'][0]):.2f} of friend's; "
              f"EL median {r['summary']['expected_loss']['median'] / 1e6:.2f}m 95% [{r['summary']['expected_loss']['ci95'][0] / 1e6:.2f}, "
              f"{r['summary']['expected_loss']['ci95'][1] / 1e6:.2f}]  invalid {r['invalid_draws']}")
    print("multipliers (2.5/50/97.5%):", {k: [round(x, 2) for x in v] for k, v in out["multiplier_q"].items()})
    save("quality_rates_unc.json", out)
    return out


def selftest():
    """variant.targets at a shared-share point must equal the friend's targets_at (source-form model, n = 2)."""
    with in_repo():
        f = PU.targets_at(0.06, 0.05)
    v = variant.targets(m=0.06 / PU.BRIEF_EQ, rain_share=0.05)
    for k, kk in KEYS.items():
        assert abs(v[kk] / f[k] - 1) < 1e-9, (k, v[kk], f[k])
    print("selftest ok: variant.targets == param_uncertainty.targets_at at (s_eq, s_rain) = (0.06, 0.05)")


if __name__ == "__main__":
    import sys
    selftest()
    if "--selftest" not in sys.argv:
        run()
