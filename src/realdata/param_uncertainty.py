"""Parameter-uncertainty propagation: credible intervals for the bond's joint tail that account for
the uncertainty in the dependence assumptions.

The data give *ranges* for the two shared shares (earthquake and rain). Reporting only a point estimate
or the endpoints hides how much of the uncertainty in the headline findings comes from the data itself.

Method.
    1. Posterior on each shared share. Use a Jeffreys Beta prior Beta(1/2, 1/2) and the counts from the
       bivariate-Poisson occurrence table (loss-data backtest for earthquakes, GHCN station analysis for
       extreme rain). This gives a Beta posterior for each share.
    2. Monte Carlo sweep. Draw N posterior samples of (s_eq, s_rain), recompute the exact two-country
       targets for each sample, and report the posterior median, 50% and 95% credible intervals.
    3. The targets are computed exactly (dense enumeration, two countries, deterministic), so there is
       no simulation noise and the full width of the credible interval comes from parameter uncertainty.
Run: python -m src.realdata.param_uncertainty -> results/param_uncertainty.json
"""
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from src.tensor.synthetic import SyntheticModel, topology
from src.rdpo.problem import PortfolioProblem

OUT = Path("results/param_uncertainty.json")
BRIEF_EQ = 0.5 * (0.075 + 0.074)


def _posterior_params(k, n, prior=(0.5, 0.5)):
    """Jeffreys-prior posterior Beta(a+k, b+n-k)."""
    return (prior[0] + k, prior[1] + max(n - k, 0))


def eq_posterior():
    """Earthquake shared share: NOAA 1900-2020 bivariate-Poisson occurrence table."""
    d = json.loads(Path("results/realdata_losses.json").read_text())
    t = d["tests"]["eq_noaa_1900"]
    tab = t["table_both_KGonly_TJonly_none"]
    k_both = tab[0][0]
    n_total = tab[0][0] + tab[0][1]        # years with KG event; share_of_A = both / KG
    a, b = _posterior_params(k_both, n_total)
    return {"a": a, "b": b, "point": a / (a + b), "ci95": [float(beta_dist.ppf(.025, a, b)), float(beta_dist.ppf(.975, a, b))],
            "k_both": k_both, "n_kg": n_total}


def rain_posterior():
    """Extreme-rain shared share: pooled over neighbouring pairs at q=20% (1-in-5-year level)."""
    d = json.loads(Path("results/realdata_rain.json").read_text())
    pl = d["pooled"]["0.2"]
    k = pl["shared"]; n = pl["events"]
    a, b = _posterior_params(k, n)
    return {"a": a, "b": b, "point": a / (a + b), "ci95": [float(beta_dist.ppf(.025, a, b)), float(beta_dist.ppf(.975, a, b))],
            "k_both": k, "n_total": n}


def targets_at(s_eq, s_rain):
    """Exact two-country targets with shared shares (s_eq, s_rain). s_eq is interpreted on the KG rate
    (as in the brief); it scales to m = s_eq / brief_eq."""
    m = s_eq / BRIEF_EQ
    try:
        model = SyntheticModel(2, topology("chain", 2), m=m, rain_shared=min(float(s_rain), 1.0))
        p = PortfolioProblem(model, "exceedance", threshold=80e6)
        return {"m": m,
                "exceedance": p.exact_dense(),
                "joint_attachment": PortfolioProblem(model, "joint_attachment", threshold=80e6).exact_dense(),
                "joint_exhaustion": PortfolioProblem(model, "joint_exhaustion", threshold=80e6).exact_dense(),
                "stop_loss": PortfolioProblem(model, "stop_loss", threshold=80e6).exact_dense(),
                "expected_loss": PortfolioProblem(model, "expected_loss", threshold=80e6).exact_dense()}
    except AssertionError:
        return {"m": m, "exceedance": None, "joint_attachment": None, "joint_exhaustion": None,
                "stop_loss": None, "expected_loss": None, "error": "shared rate exceeds total rate"}


def summarise(values, levels=(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)):
    v = np.array([x for x in values if x is not None and np.isfinite(x)])
    if v.size == 0:
        return None
    out = {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std(ddof=1)),
           "min": float(v.min()), "max": float(v.max()), "n": int(v.size),
           "quantiles": {f"q{int(100 * q)}": float(np.quantile(v, q)) for q in levels}}
    out["ci50"] = [out["quantiles"]["q25"], out["quantiles"]["q75"]]
    out["ci95"] = [out["quantiles"]["q2"], out["quantiles"]["q97"]]
    return out


def run(N=2000, seed=4242):
    eq = eq_posterior(); rn = rain_posterior()
    rng = np.random.default_rng(seed)
    s_eq = rng.beta(eq["a"], eq["b"], N)
    s_rn = rng.beta(rn["a"], rn["b"], N)
    # evaluate exact targets at each sample
    draws = {"exceedance": [], "joint_attachment": [], "joint_exhaustion": [], "stop_loss": [], "expected_loss": [], "m": []}
    for i in range(N):
        t = targets_at(float(s_eq[i]), float(s_rn[i]))
        for k in draws:
            draws[k].append(t.get(k))
    # brief point (for reference)
    brief = targets_at(BRIEF_EQ, 0.25)
    summary = {k: summarise(v) for k, v in draws.items() if k != "m"}
    # factor relative to brief
    def factor(k):
        s = summary.get(k)
        if s is None:
            return None
        return {"median_x": s["median"] / brief[k],
                "ci50_x": [s["ci50"][0] / brief[k], s["ci50"][1] / brief[k]],
                "ci95_x": [s["ci95"][0] / brief[k], s["ci95"][1] / brief[k]]}
    out = {
        "method": "Jeffreys-prior Beta posteriors on the two shared shares (bivariate-Poisson occurrence tables); "
                  f"{N} Monte Carlo samples; exact dense enumeration per sample.",
        "posteriors": {"earthquake": eq, "rain": rn},
        "brief_point": brief,
        "posterior_summary": summary,
        "factor_vs_brief": {k: factor(k) for k in summary},
        "shared_share_samples": {"earthquake_q50_q95": [float(np.quantile(s_eq, 0.5)), float(np.quantile(s_eq, 0.025)),
                                                        float(np.quantile(s_eq, 0.975))],
                                 "rain_q50_q95": [float(np.quantile(s_rn, 0.5)), float(np.quantile(s_rn, 0.025)),
                                                  float(np.quantile(s_rn, 0.975))]},
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    import sys
    t0 = time.perf_counter()
    o = run()
    print(f"done in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    for k, s in o["posterior_summary"].items():
        f = o["factor_vs_brief"][k]
        b = o["brief_point"][k]
        print(f"{k:17s}  brief={b:.3g}  median={s['median']:.3g} ({f['median_x']:.2f}x)  "
              f"50% CI=[{s['ci50'][0]:.3g},{s['ci50'][1]:.3g}]  95% CI=[{s['ci95'][0]:.3g},{s['ci95'][1]:.3g}]")
