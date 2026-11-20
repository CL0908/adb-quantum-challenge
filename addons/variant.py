"""A parametrised copy of the brief's two-country annual model (src/model/annual.py), used by the quality
experiments. At default arguments it reproduces annual_distribution() exactly (selftest). Friend's src/ untouched.

Extra knobs (each 0 / 1 by default):
  eq_mult=(ρ1, ρ2)   scale each country's EQ trigger rate Λ_i (the shared rate follows the friend's formula)
  nu                 cluster offspring: every EQ event brings Poisson(ν) further payout events of the same
                     country(ies) in the same year, each with the same severity law (aftershocks paid separately)
  rain_mult=(κ1, κ2) scale each country's 2% rain-trigger rate; rain_share = shared share of the smaller rate
"""
import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson

from _src import REPO  # noqa: F401  (puts the friend repo on sys.path)
from src.model.annual import annual_distribution
from src.model.poisson_fit import TAILS, P_TJ_GIVEN_KG, P_KG_GIVEN_TJ, compound_poisson_rates
from src.model.scenarios import SCENARIOS, Scenario
from src.model.targets import target_functions
from src.model.term import term_distribution


def _cap(x, y):
    """Law of min(X + Y, 4) for independent X, Y on 0..4."""
    z = np.zeros(5)
    for i in range(5):
        for j in range(5):
            z[min(i + j, 4)] += x[i] * y[j]
    return z


def cluster_law(w, nu, nmax=40):
    """Payout-quarter law (index 0..4, capped) of one cluster: a parent with severity law w (k = 1..4) plus
    Poisson(ν) offspring with the same law."""
    one = np.r_[0.0, w]
    out, conv = np.zeros(5), one.copy()
    for n in range(nmax):
        out += poisson.pmf(n, nu) * conv
        conv = _cap(conv, one)
    return out / out.sum()


def shared_cluster_law(w1, w2, nu, nmax=40):
    """Joint (Δa, Δb) law of a shared cluster: parent hits both, every offspring hits both again."""
    one = np.outer(np.r_[0.0, w1], np.r_[0.0, w2])
    def cap2(X, Y):
        Z = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                if X[i, j]:
                    for k in range(5):
                        for l in range(5):
                            Z[min(i + k, 4), min(j + l, 4)] += X[i, j] * Y[k, l]
        return Z
    out, conv = np.zeros((5, 5)), one.copy()
    for n in range(nmax):
        p = poisson.pmf(n, nu)
        if p < 1e-16 and n > nu:
            break
        out += p * conv
        conv = cap2(conv, one)
    return out / out.sum()


def eq_generator(sc, eq_mult=(1.0, 1.0), nu=0.0):
    """Yearly generator of the earthquake payout state (a, b), index 5 a + b; expm(t G) is the law after t years."""
    lam1, L1 = compound_poisson_rates(TAILS["KG"]); lam2, L2 = compound_poisson_rates(TAILS["TJ"])
    w1, w2 = lam1 / L1, lam2 / L2
    L1, L2 = L1 * eq_mult[0], L2 * eq_mult[1]
    c = sc.m * 0.5 * (P_TJ_GIVEN_KG * L1 + P_KG_GIVEN_TJ * L2)
    J1, J2, JS = cluster_law(w1, nu), cluster_law(w2, nu), shared_cluster_law(w1, w2, nu)
    G = np.zeros((25, 25))
    for a in range(5):
        for b in range(5):
            i = a * 5 + b
            for k in range(1, 5):
                G[i, min(a + k, 4) * 5 + b] += (L1 - c) * J1[k]
                G[i, a * 5 + min(b + k, 4)] += (L2 - c) * J2[k]
                for j in range(1, 5):
                    G[i, min(a + k, 4) * 5 + min(b + j, 4)] += c * JS[k, j]
            G[i, i] -= G[i].sum()
    return G


def eq_annual(sc, eq_mult=(1.0, 1.0), nu=0.0):
    return expm(eq_generator(sc, eq_mult, nu))[0].reshape(5, 5)


def rain_annual(sc, rain_mult=(1.0, 1.0), rain_share=None):
    mu = -np.log(1 - 0.02) * sc.rain_mult
    mu1, mu2 = mu * rain_mult[0], mu * rain_mult[1]
    cr = (sc.rain_shared_share if rain_share is None else rain_share) * min(mu1, mu2)
    P00 = np.exp(-(mu1 + mu2 - cr)); P10 = np.exp(-mu2) - P00; P01 = np.exp(-mu1) - P00
    return np.array([[P00, P01], [P10, 1 - P00 - P10 - P01]])


def annual(sc, **kw):
    E = eq_annual(sc, kw.get("eq_mult", (1.0, 1.0)), kw.get("nu", 0.0))
    R = rain_annual(sc, kw.get("rain_mult", (1.0, 1.0)), kw.get("rain_share"))
    return np.einsum("ab,rs->abrs", E, R).reshape(-1)


def targets(sc=SCENARIOS["adb-base"], m=None, **kw):
    if m is not None:
        sc = Scenario(sc.name, m=m, rain_mult=sc.rain_mult, rain_shared_share=sc.rain_shared_share,
                      eq_limit=sc.eq_limit, rain_limit=sc.rain_limit, threshold=sc.threshold)
    term = term_distribution(annual(sc, **kw))
    return {k: float(term @ f) for k, f in target_functions(sc).items()}


def selftest():
    for name, sc in SCENARIOS.items():
        assert np.allclose(annual(sc), annual_distribution(sc), atol=1e-14, rtol=0), name
    t = targets()
    t2 = targets(nu=0.3)
    assert t2["term_both_bonds_exhausted_probability"] > t["term_both_bonds_exhausted_probability"]
    assert abs(sum(cluster_law(np.array([.4, .3, .2, .1]), 0.7)) - 1) < 1e-12
    print("selftest ok: variant model equals src.model.annual.annual_distribution for all 5 scenarios at defaults")


if __name__ == "__main__":
    selftest()
