"""Quality 5: a risk-measure layer on the friend's exact pool loss distributions (idea from QContagion's
evar_vs_covar: a measure that reads the whole tail, plus who contributes to it). Additive; src/ read-only.

For the friend's data-calibrated pools (central_asia_pool: 5 countries; carec_pool: 10 countries) the term loss
distribution is exact (MPS from the friend's own builders). We add, per pool and level α ∈ {99%, 99.5%, 99.9%}:
  VaR, expected shortfall ES (Acerbi–Tasche form for discrete laws) and EVaR (Ahmadi-Javid 2012),
  Euler contributions of each country to ES and EVaR (they add up to the pool figure), and
  the stand-alone figures, so the pool's diversification benefit is explicit (pooled / sum of stand-alone).
One MPS sweep carries, besides P(L = l), the first-moment channels m_i(l) = E[L_i 1{L = l}] for every country,
which is all that Euler contributions of law-invariant tail measures need.
"""
import math

import numpy as np
from scipy.optimize import minimize_scalar

from _src import in_repo, save
from src.realdata import carec_pool as CP
from src.realdata import central_asia_pool as CA
from src.tensor.mps import build_mps
from src.tensor.synthetic import dense_term_distribution

UNIT, ALPHAS = 2.5e6, (0.99, 0.995, 0.999)


def loss_moments(model, psi):
    """P(L = l) and m[l, i] = E[L_i 1{L = l}] (units of 2.5m) by one left-to-right MPS transfer."""
    lu = model.site_loss_units(UNIT)
    n, Lmax = model.n, int(lu.max()) * model.n
    v = np.zeros((1, Lmax + 1)); v[0, 0] = 1.0
    w = np.zeros((1, Lmax + 1, n))
    for s, t in enumerate(psi.T):
        if psi.gauge is not None:
            t = t / psi.gauge[s][None, :, None]
        nv = np.zeros((t.shape[2], Lmax + 1)); nw = np.zeros((t.shape[2], Lmax + 1, n))
        for x in range(t.shape[1]):
            A, sh = t[:, x, :], int(lu[x])
            cv = np.einsum("lL,lr->rL", v, A); cw = np.einsum("lLi,lr->rLi", w, A)
            nv[:, sh:] += cv[:, :Lmax + 1 - sh]
            nw[:, sh:, :] += cw[:, :Lmax + 1 - sh, :]
            nw[:, sh:, s] += sh * cv[:, :Lmax + 1 - sh]
        v, w = nv, nw
    return v[0], w[0]


def measures(p, m, alpha):
    """VaR, ES, EVaR and their Euler contributions from P(L = l) and m[l, i] (units)."""
    l = np.arange(len(p)); cdf = np.cumsum(p)
    k = int(np.searchsorted(cdf, alpha - 1e-15))          # VaR index
    var = l[k]
    tail = l > var
    es = (p[tail] @ l[tail] + var * (cdf[k] - alpha)) / (1 - alpha)
    es_i = (m[tail].sum(0) + (m[k] / p[k]) * (cdf[k] - alpha)) / (1 - alpha) if p[k] > 0 else m[tail].sum(0) / (1 - alpha)
    def evar_t(t):
        lw = t * l + np.log(np.maximum(p, 1e-300))
        mx = lw.max()
        return (mx + math.log(np.exp(lw - mx).sum()) - math.log(1 - alpha)) / t
    r = minimize_scalar(evar_t, bounds=(1e-4, 5.0), method="bounded", options={"xatol": 1e-9})
    t = r.x
    e = np.exp(t * l - (t * l).max())
    evar_i = (e @ m) / (e @ p)                              # E[L_i e^{tL}] / E[e^{tL}]; m already carries P(L = l)
    return {"VaR": var, "ES": es, "EVaR": float(r.fun), "ES_contrib": es_i, "EVaR_contrib": evar_i, "t_star": t}


def run():
    out = {}
    with in_repo():
        ca, cr = CA.calibrate(), CP.calibrate()
    pools = [("central_asia_5", CA.ISO, ca["model"], build_mps(ca["model"])[0]),
             ("carec_10", cr["order"], cr["model"], CP.build_mps_data(cr["model"]))]
    for name, countries, model, psi in pools:
        p, m = loss_moments(model, psi)
        res = {"countries": list(countries), "EL_musd": float(np.arange(len(p)) @ p * 2.5), "P_loss": float(1 - p[0]),
               "levels": {}}
        for a in ALPHAS:
            r = measures(p, m, a)
            # stand-alone law of country i: marginal of L_i, from a one-site sweep with the other sites' losses zeroed
            sa = {}
            for i, c in enumerate(countries):
                pi = marginal(model, psi, i)
                ri = measures(pi, np.arange(len(pi))[:, None] * pi[:, None], a)
                sa[c] = {"ES_musd": ri["ES"] * 2.5, "EVaR_musd": ri["EVaR"] * 2.5}
            res["levels"][str(a)] = {
                "VaR_musd": float(r["VaR"] * 2.5), "ES_musd": float(r["ES"] * 2.5), "EVaR_musd": float(r["EVaR"] * 2.5),
                "ES_contrib_musd": {c: float(x * 2.5) for c, x in zip(countries, r["ES_contrib"])},
                "EVaR_contrib_musd": {c: float(x * 2.5) for c, x in zip(countries, r["EVaR_contrib"])},
                "standalone": sa,
                "diversification_ES": float(r["ES"] * 2.5 / sum(v["ES_musd"] for v in sa.values())),
                "diversification_EVaR": float(r["EVaR"] * 2.5 / sum(v["EVaR_musd"] for v in sa.values()))}
            lv = res["levels"][str(a)]
            print(f"{name} α={a}: VaR {lv['VaR_musd']:.1f}m  ES {lv['ES_musd']:.1f}m  EVaR {lv['EVaR_musd']:.1f}m  "
                  f"pooled/standalone ES {lv['diversification_ES']:.2f} EVaR {lv['diversification_EVaR']:.2f}  ES shares " +
                  " ".join(f"{c}:{x / lv['ES_musd']:.0%}" for c, x in lv["ES_contrib_musd"].items()), flush=True)
        out[name] = res
    save("quality_risk_measures.json", out)
    return out


def marginal(model, psi, i):
    """Law of L_i alone: same transfer with every other site's loss set to 0."""
    lu = model.site_loss_units(UNIT)
    F = int(lu.max())
    v = np.zeros((1, F + 1)); v[0, 0] = 1.0
    for s, t in enumerate(psi.T):
        if psi.gauge is not None:
            t = t / psi.gauge[s][None, :, None]
        nv = np.zeros((t.shape[2], F + 1))
        for x in range(t.shape[1]):
            sh = int(lu[x]) if s == i else 0
            cv = np.einsum("lL,lr->rL", v, t[:, x, :])
            nv[:, sh:] += cv[:, :F + 1 - sh]
        v = nv
    return v[0]


def selftest():
    """MPS moment sweep vs dense enumeration on the 5-country pool: same P(L = l) and same m[l, i]."""
    with in_repo():
        model = CA.calibrate()["model"]
    psi, _ = build_mps(model)
    p, m = loss_moments(model, psi)
    P = dense_term_distribution(model).reshape(-1)
    n = model.n
    idx = np.arange(10 ** n)
    codes = np.stack([(idx // 10 ** (n - 1 - i)) % 10 for i in range(n)], 1)
    Li = model.site_loss_units(UNIT)[codes]
    L = Li.sum(1)
    pd = np.bincount(L, weights=P, minlength=len(p))
    md = np.stack([np.bincount(L, weights=P * Li[:, i], minlength=len(p)) for i in range(n)], 1)
    assert np.allclose(p, pd[:len(p)], atol=1e-10) and np.allclose(m, md[:len(p)], atol=1e-9)
    r = measures(p, m, 0.995)
    assert abs(r["ES_contrib"].sum() - r["ES"]) < 1e-8 * r["ES"], (r["ES_contrib"].sum(), r["ES"])
    assert abs(r["EVaR_contrib"].sum() / r["EVaR"] - 1) < 1e-4, (r["EVaR_contrib"].sum(), r["EVaR"])   # optimum of t
    pi = marginal(model, psi, 0)
    assert np.allclose(pi, np.bincount(Li[:, 0], weights=P, minlength=len(pi))[:len(pi)], atol=1e-10)
    print("selftest ok: MPS moment sweep == dense enumeration (5 countries); Euler contributions add up; marginal ok")


if __name__ == "__main__":
    import sys
    selftest()
    if "--selftest" not in sys.argv:
        run()
