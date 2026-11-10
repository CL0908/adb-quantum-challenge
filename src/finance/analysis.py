"""Finance-facing exact analyses for the two-country Kyrgyz-Tajik bond (no sampling).

1. Bond anchor: annual expected EQ loss implied by the brief's tier exceedance probabilities versus the
   publicly reported initial expected loss and risk margin (cited constants, not computed here).
2. Pooling decision: two separate US$80m covers versus one shared limit, per 3-year term, for the brief
   base case, the 'both from data' setting and the posterior of src/realdata/param_uncertainty.py.
3. Model-free coupling bounds: linear programs over ALL joint laws of the two countries' term states
   (a in 0..4, r in 0..1; 10 states each) that keep each marginal fixed at the brief base case.

All term laws come from the existing exact model (dense enumeration). The only randomness is the posterior
draw of the shared shares (same seed and sample size as param_uncertainty.py); everything else is exact.
Run: python -m src.finance.analysis -> results/finance.json
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from src.model.poisson_fit import TAILS
from src.tensor.synthetic import SyntheticModel, topology, dense_term_distribution
from src.realdata import param_uncertainty as PU

OUT = Path("results/finance.json")
EQ_LIMIT, RAIN_LIMIT = 70e6, 10e6
FULL = EQ_LIMIT + RAIN_LIMIT
THRESH = 80e6
RAIN_ANNUAL_P = 0.02
BRIEF_M, BRIEF_RAIN_SHARED = 1.0, 0.25
POST_N, POST_SEED = 2000, 4242

CITED = {
    "source": 'Artemis, "ADB prices first cat bonds" (public market report); figures quoted as reported',
    "initial_expected_loss": {"KG": 0.02296, "TJ": 0.02011},
    "risk_margin": {"KG": 0.06, "TJ": 0.06},
}

# state code s = 2*a + r (as in the tensor model), country losses in USD
_a, _r = np.arange(10) // 2, np.arange(10) % 2
LOSS = EQ_LIMIT * _a / 4 + RAIN_LIMIT * _r            # per-country term loss of state s
LSUM = LOSS[:, None] + LOSS[None, :]                  # total loss for (s1, s2)


# ---------------------------------------------------------------------------------------------------
def term_law(m=BRIEF_M, rain_shared=BRIEF_RAIN_SHARED):
    """Exact 10x10 joint law of the two countries' term states."""
    mdl = SyntheticModel(2, topology("chain", 2), m=m, rain_shared=rain_shared)
    return dense_term_distribution(mdl)


def data_setting():
    """The paper's 'both from data' setting (m, rain shared share), read from the stored rain check."""
    try:
        d = json.loads(Path("results/realdata_rain.json").read_text())["implied"]["both from data"]
        return float(d["m"]), float(d["rain_shared"]), "results/realdata_rain.json (implied: both from data)"
    except (FileNotFoundError, KeyError):
        return 6.405135311515741, 2.0 / 90.0, "fallback constants"


def metrics(P, limit=THRESH):
    """Pooling metrics for a joint law P (10x10): separate covers vs one shared limit `limit`."""
    EL = float((P * LSUM).sum())
    short = float((P * np.maximum(LSUM - limit, 0)).sum())
    shared_pay = EL - short
    return {"expected_loss": EL, "expected_payout_separate": EL, "expected_payout_shared": shared_pay,
            "delivered_fraction": shared_pay / EL, "shortfall": short,
            "prob_exceed": float((P * (LSUM > limit + 1e-6)).sum()),
            "joint_attachment": float(P[1:, 1:].sum()),
            "joint_exhaustion": float(P[9, 9])}


def pooling_block(P):
    out = {"separate_cap_total_usd": 2 * FULL, "shared": {}}
    for lim in (80e6, 100e6, 120e6):
        m = metrics(P, lim)
        out["shared"][f"{int(lim / 1e6)}m"] = {
            "limit_usd": lim, "capital_saving_usd": 2 * FULL - lim,
            "expected_payout_separate": m["expected_payout_separate"],
            "expected_payout_shared": m["expected_payout_shared"],
            "delivered_fraction": m["delivered_fraction"],
            "expected_payout_given_up": m["shortfall"],
            "shortfall_per_usd_capital_saved": m["shortfall"] / (2 * FULL - lim),
            "prob_exceed_limit": m["prob_exceed"]}
    m = metrics(P)
    out["joint_attachment"], out["joint_exhaustion"] = m["joint_attachment"], m["joint_exhaustion"]
    return out


# --- 1. anchor ---------------------------------------------------------------------------------
def anchor():
    res = {"computed": {}, "cited": CITED, "ratios_computed_over_cited": {}}
    for c in ("KG", "TJ"):
        el = float(np.mean(TAILS[c]))                     # sum_k 0.25 * Pr(payout >= k/4)
        full_usd = (EQ_LIMIT * el + RAIN_LIMIT * RAIN_ANNUAL_P)
        res["computed"][c] = {
            "tier_exceedance": TAILS[c].tolist(),
            "annual_EL_fraction_of_eq_sublimit": el,
            "annual_EL_eq_usd": EQ_LIMIT * el,
            "annual_EL_rain_usd": RAIN_LIMIT * RAIN_ANNUAL_P,
            "annual_EL_full_80m_usd": full_usd,
            "annual_EL_full_80m_fraction": full_usd / FULL}
        cited_el = CITED["initial_expected_loss"][c]
        res["ratios_computed_over_cited"][c] = {
            "risk_margin_over_cited_EL": CITED["risk_margin"][c] / cited_el,
            "risk_margin_over_computed_EL": CITED["risk_margin"][c] / el,
            "computed_minus_cited_EL": el - cited_el}
    return res


# --- 2. pooling over settings and posterior --------------------------------------------------
def posterior_pooling(N=POST_N, seed=POST_SEED):
    eq, rn = PU.eq_posterior(), PU.rain_posterior()
    rng = np.random.default_rng(seed)
    s_eq, s_rn = rng.beta(eq["a"], eq["b"], N), rng.beta(rn["a"], rn["b"], N)
    rows = {k: [] for k in ("delivered_fraction", "shortfall", "prob_exceed", "delivered_100", "delivered_120")}
    laws = []
    for i in range(N):
        try:
            P = term_law(float(s_eq[i]) / PU.BRIEF_EQ, min(float(s_rn[i]), 1.0))
        except AssertionError:
            continue
        laws.append(P)
        m, m100, m120 = metrics(P), metrics(P, 100e6), metrics(P, 120e6)
        rows["delivered_fraction"].append(m["delivered_fraction"]); rows["shortfall"].append(m["shortfall"])
        rows["prob_exceed"].append(m["prob_exceed"])
        rows["delivered_100"].append(m100["delivered_fraction"]); rows["delivered_120"].append(m120["delivered_fraction"])
    def summ(v):
        v = np.array(v)
        return {"median": float(np.median(v)), "ci95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))],
                "ci50": [float(np.quantile(v, .25)), float(np.quantile(v, .75))], "n": int(v.size)}
    return {"n_samples": int(len(laws)), "seed": seed, "posteriors": {"earthquake": eq, "rain": rn},
            **{k: summ(v) for k, v in rows.items()}}, rows


# --- 3. coupling LPs -----------------------------------------------------------------------------
OBJ = {
    "prob_exceed": (LSUM > THRESH + 1e-6).astype(float),
    "joint_attachment": np.outer(np.arange(10) > 1, np.arange(10) > 1).astype(float),
    "joint_exhaustion": np.zeros((10, 10)),
    "stop_loss": np.maximum(LSUM - THRESH, 0) / 1e6,   # in US$m for conditioning
}
OBJ["joint_attachment"] = np.outer(LOSS > 0, LOSS > 0).astype(float)
OBJ["joint_exhaustion"][9, 9] = 1.0


def _marginal_constraints(n):
    A = np.zeros((2 * n, n * n))
    for i in range(n):
        for j in range(n):
            A[i, i * n + j] = 1
            A[n + j, i * n + j] = 1
    return A


def lp_range(p1, p2, f):
    """min/max of E[f(S1,S2)] over all couplings of marginals p1, p2 (10 states each)."""
    A, b = _marginal_constraints(10), np.concatenate([p1, p2])
    c = f.reshape(-1)
    lo = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    hi = linprog(-c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    assert lo.status == 0 and hi.status == 0
    return float(lo.fun), float(-hi.fun)


def _split_marginals(p1, p2):
    """Marginals of the EQ level a and rain r for each country (they are independent within a country)."""
    out = []
    for p in (p1, p2):
        pa = np.array([p[2 * a:2 * a + 2].sum() for a in range(5)])
        pr = np.array([p[r::2].sum() for r in (0, 1)])
        out.append((pa, pr))
    return out


def lp_range_component(p1, p2, f, component):
    """Couple only one component (EQ levels or rain) arbitrarily; the other is independent across countries.
    Joint law = coupling(component) x product of the other component's marginals (linear in the coupling)."""
    (a1, r1), (a2, r2) = _split_marginals(p1, p2)
    if component == "eq":
        n, m1, m2 = 5, a1, a2
    else:
        n, m1, m2 = 2, r1, r2
    A, b = _marginal_constraints(n), np.concatenate([m1, m2])
    c = np.zeros(n * n)
    for i in range(n):
        for j in range(n):
            if component == "eq":     # (a1=i, a2=j) with independent rain
                c[i * n + j] = sum(f[2 * i + x, 2 * j + y] * r1[x] * r2[y] for x in (0, 1) for y in (0, 1))
            else:                     # (r1=i, r2=j) with independent EQ
                c[i * n + j] = sum(f[2 * x + i, 2 * y + j] * a1[x] * a2[y] for x in range(5) for y in range(5))
    lo = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    hi = linprog(-c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    assert lo.status == 0 and hi.status == 0
    return float(lo.fun), float(-hi.fun)


def position(v, lo, hi):
    return float((v - lo) / (hi - lo)) if hi > lo else None


def coupling_bounds(P_base, points):
    """LP ranges for the four targets with marginals fixed at the base case; `points` are exact values to place."""
    p1, p2 = P_base.sum(1), P_base.sum(0)
    out = {"marginal_max_deviation_of_points": {}, "targets": {}}
    for name, f in OBJ.items():
        scale = 1e6 if name == "stop_loss" else 1.0
        full = lp_range(p1, p2, f)
        eq_only = lp_range_component(p1, p2, f, "eq")
        rain_only = lp_range_component(p1, p2, f, "rain")
        t = {"unit": "USD" if name == "stop_loss" else "probability",
             "full_coupling": {"min": full[0] * scale, "max": full[1] * scale},
             "eq_coupled_rain_independent": {"min": eq_only[0] * scale, "max": eq_only[1] * scale},
             "rain_coupled_eq_independent": {"min": rain_only[0] * scale, "max": rain_only[1] * scale},
             "independent_countries": float((np.outer(p1, p2) * f).sum() * scale),
             "points": {}}
        for pname, (val, ) in {k: (float((v * f).sum() * scale),) for k, v in points.items() if not k.startswith("_")}.items():
            t["points"][pname] = {"value": val, "position_in_full_range": position(val, full[0] * scale, full[1] * scale)}
        out["targets"][name] = t
    return out


# ---------------------------------------------------------------------------------------------------
def run():
    t0 = time.perf_counter()
    m_d, s_d, src_d = data_setting()
    P_brief, P_data = term_law(), term_law(m_d, s_d)
    post, rows = posterior_pooling()
    # marginals must be identical in all settings (dependence changes the coupling only)
    dev = max(float(np.abs(P.sum(1) - P_brief.sum(1)).max()) for P in (P_data,)) 
    pool = {"brief": pooling_block(P_brief), "both_from_data": pooling_block(P_data), "posterior": post}
    # posterior median / 95% points inside the LP ranges, using the posterior summaries of each target
    pu = json.loads(Path("results/param_uncertainty.json").read_text())["posterior_summary"]
    lp = coupling_bounds(P_brief, {"brief": P_brief, "both_from_data": P_data})
    for name, key in (("prob_exceed", "exceedance"), ("joint_attachment", "joint_attachment"),
                      ("joint_exhaustion", "joint_exhaustion"), ("stop_loss", "stop_loss")):
        t = lp["targets"][name]; lo, hi = t["full_coupling"]["min"], t["full_coupling"]["max"]
        s = pu[key]
        for lab, v in (("posterior_median", s["median"]), ("posterior_ci95_low", s["ci95"][0]),
                       ("posterior_ci95_high", s["ci95"][1])):
            t["points"][lab] = {"value": v, "position_in_full_range": position(v, lo, hi)}
    lp["marginal_max_deviation_of_points"]["both_from_data_vs_brief"] = dev
    res = {
        "meta": {
            "description": "Exact finance-facing analyses: bond anchor, pooling decision, model-free coupling bounds.",
            "units": "USD, 3-year term, two countries; loss in a country = 70m*a/4 + 10m*r",
            "inputs": {"eq_limit": EQ_LIMIT, "rain_limit": RAIN_LIMIT, "threshold": THRESH,
                       "annual_rain_trigger_prob": RAIN_ANNUAL_P, "tier_exceedance": {k: v.tolist() for k, v in TAILS.items()},
                       "brief": {"m": BRIEF_M, "rain_shared": BRIEF_RAIN_SHARED},
                       "both_from_data": {"m": m_d, "rain_shared": s_d, "source": src_d},
                       "posterior": {"n": POST_N, "seed": POST_SEED, "prior": "Jeffreys Beta(1/2,1/2)",
                                     "code": "src/realdata/param_uncertainty.py (re-drawn with its seed; pooled values "
                                             "read from results/param_uncertainty.json)"}},
            "sources": {"computed": "src/model, src/tensor/synthetic.py exact dense enumeration; scipy linprog (HiGHS)",
                        "cited": CITED["source"]},
            "seed": POST_SEED, "runtime_s": round(time.perf_counter() - t0, 1)},
        "anchor": anchor(),
        "pooling": pool,
        "coupling_bounds": lp,
    }
    OUT.write_text(json.dumps(res, indent=2))
    try:
        figures(res, rows)
    except Exception as e:      # figures are optional
        print("figure skipped:", e)
    return res


def figures(res, rows):
    import matplotlib.pyplot as plt
    from src.viz import style as S
    S.apply()
    out = Path("figures/paper"); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    ax.hist(100 * np.array(rows["delivered_fraction"]), bins=40, color=S.BLUE, alpha=0.85)
    b = res["pooling"]["brief"]["shared"]["80m"]["delivered_fraction"] * 100
    ax.axvline(b, color=S.TERRA, lw=2); ax.text(b, ax.get_ylim()[1] * 0.92, " brief", color=S.TERRA, ha="right", fontsize=8)
    ax.set_xlabel("Expected payout delivered by shared US$80m limit (%)"); ax.set_ylabel("Posterior samples")
    ax.set_title("(a) Cost of pooling under parameter uncertainty")
    ax = axes[1]
    names = [("prob_exceed", "Pr(L > 80m)"), ("joint_attachment", "Joint attachment"), ("joint_exhaustion", "Joint exhaustion")]
    for y, (k, lab) in enumerate(names[::-1]):
        t = res["coupling_bounds"]["targets"][k]; lo, hi = t["full_coupling"]["min"], t["full_coupling"]["max"]
        ax.plot([0, 1], [y, y], color=S.RULE, lw=6, solid_capstyle="round")
        for pn, col, mk in (("brief", S.TERRA, "o"), ("both_from_data", S.BLUE, "s"), ("posterior_median", S.TEAL, "D")):
            ax.plot(t["points"][pn]["position_in_full_range"], y, mk, color=col, label=pn.replace("_", " ") if y == 0 else None)
        ax.text(0, y + 0.22, f"{100 * lo:.2g}%", fontsize=7, color=S.INK2, ha="left")
        ax.text(1, y + 0.22, f"{100 * hi:.2g}%", fontsize=7, color=S.INK2, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels([lab for _, lab in names[::-1]])
    ax.set_xlabel("Position within model-free [min, max] (fixed marginals)"); ax.set_xlim(-0.03, 1.03)
    ax.set_title("(b) Coupling bounds"); ax.legend(loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.28))
    fig.savefig(out / "finance_pooling_bounds.svg")
    plt.close(fig)


if __name__ == "__main__":
    r = run()
    a = r["anchor"]
    for c in ("KG", "TJ"):
        print(c, "EL %.4f%%" % (100 * a["computed"][c]["annual_EL_fraction_of_eq_sublimit"]),
              "RM/EL cited %.2fx" % a["ratios_computed_over_cited"][c]["risk_margin_over_cited_EL"])
    for k in ("brief", "both_from_data"):
        s = r["pooling"][k]["shared"]["80m"]
        print(k, "delivered %.2f%% shortfall %.3fm Pr %.2f%%" % (100 * s["delivered_fraction"], s["expected_payout_given_up"] / 1e6, 100 * s["prob_exceed_limit"]))
    p = r["pooling"]["posterior"]["delivered_fraction"]
    print("posterior delivered median %.2f%% CI [%.2f, %.2f]" % (100 * p["median"], 100 * p["ci95"][0], 100 * p["ci95"][1]))
    for k, t in r["coupling_bounds"]["targets"].items():
        f = t["full_coupling"]
        print(k, "[%.4g, %.4g]" % (f["min"], f["max"]), {n: round(v["position_in_full_range"], 4) for n, v in t["points"].items()})
