"""End-to-end Q-RDPO: exponentially twisted Poisson-source loading + bounded-weight QAE.

Each source register is prepared in the *tilted* law  q_k(x) ∝ p_k(x) e^{η ℓ_k(x)}, where ℓ_k(x)
is the (uncapped) loss contribution of increment x in 2.5m units. This only changes the source
amplitudes, so the structured circuit loads it exactly at no extra gate cost. The likelihood ratio
depends on the uncapped total U = Σ_k ℓ_k only:
        w = Π_k p_k/q_k = exp(−η U + Λ(η)),     Λ(η) = Σ_k log Σ_x p_k(x) e^{η ℓ_k(x)}.
For the exceedance target f = 1{L > T} and U >= L (caps only reduce loss), w <= exp(−ηT + Λ(η)) on
{f = 1}. So the objective rotation can encode g = f · exp(−η(U − T)) ∈ [0, 1] and
        a = E_q[g] = μ · exp(ηT − Λ(η)).
Maximising over η gives a = μ · e^{I(T)}, with I the Cramér rate function of U: the amplitude is
the *large-deviation gap*, not μ itself. The oracle needs U (already in the uncapped sum registers)
and a lookup g(U) over at most U_max + 1 values.

Classically the same twisted IS has per-path relative variance 1/a − 1 at best (bounded weights),
so the quadratic QAE gain is in 1/√a vs 1/a, not in 1/√μ vs 1/μ.
"""
import math

import numpy as np
from scipy.optimize import minimize_scalar

from . import structured_loading as SL


def source_losses(model):
    unit = 2.5e6
    au, ru = round(model.eq_limit / 4 / unit), round(model.rain_limit / unit)
    out = []
    for s in SL.sources(model):
        x = np.arange(len(s.law))
        if s.kind == "eq1":
            ell = au * x
        elif s.kind == "eq2":
            ell = au * (x // 5 + x % 5)
        elif s.kind == "rain1":
            ell = ru * x
        else:
            ell = 2 * ru * x
        out.append((s, ell))
    return out


def cgf(model, eta):
    return sum(math.log(float(np.sum(s.law * np.exp(eta * ell)))) for s, ell in source_losses(model))


def best_twist(model, T_units):
    obj = lambda eta: -(eta * T_units - cgf(model, eta))
    r = minimize_scalar(obj, bounds=(0.0, 5.0), method="bounded")
    return float(r.x), float(-r.fun)        # η*, I(T) = η*T − Λ(η*)


def twisted_amplitude(model, mu, T_units):
    eta, rate = best_twist(model, T_units)
    a = mu * math.exp(rate)
    umax = int(sum(ell.max() for _, ell in source_losses(model)))
    return {"eta": eta, "rate_I": rate, "a_twisted": a, "a_untwisted": mu, "U_max": umax,
            "lookup_toffoli": umax + int(math.ceil(math.log2(umax + 1)))}


def simulate_twisted_A(model, threshold):
    """Exact sparse simulation of the twisted A operator (n small): returns a = Σ q·g."""
    eta, _ = best_twist(model, threshold / 2.5e6)
    T = threshold / 2.5e6
    srcs = source_losses(model)
    n = model.n
    amp2 = np.ones(1); S = np.zeros((1, n), int); Rr = np.zeros((1, n), int); U = np.zeros(1)
    lam = 0.0
    for s, ell in srcs:
        q = s.law * np.exp(eta * ell); Z = q.sum(); q = q / Z; lam += math.log(Z)
        vals = np.nonzero(q)[0]; K = len(amp2)
        amp2 = (amp2[:, None] * q[vals][None, :]).ravel()
        S = np.repeat(S, len(vals), 0); Rr = np.repeat(Rr, len(vals), 0); U = np.repeat(U, len(vals))
        v = np.tile(vals, K); U = U + ell[v]
        if s.kind == "eq1":
            S[:, s.sites[0]] += v
        elif s.kind == "eq2":
            S[:, s.sites[0]] += v // 5; S[:, s.sites[1]] += v % 5
        elif s.kind == "rain1":
            Rr[:, s.sites[0]] += v
        else:
            Rr[:, s.sites[0]] += v; Rr[:, s.sites[1]] += v
    a_cap, r_cap = np.minimum(S, 4), np.minimum(Rr, 1)
    L = (a_cap * 7 + r_cap * 4).sum(1)
    g = (L > T) * np.exp(-eta * (U - T))
    assert g.max() <= 1 + 1e-12
    a = float(amp2 @ g)
    mu_est = a * math.exp(-eta * T + lam)
    return {"eta": eta, "a": a, "mu_recovered": mu_est, "g_max": float(g.max())}
