"""How much can a state-dependent sampler (what RDPO learns) gain over the best state-independent twist? Exact, no
simulation, no training. Additive, read-only; the brief's two-country model as crisis.py's 100-state chain.

For each crisis event the relative variance rho2 = E_Q[w^2 1_A] / p^2 - 1 of an importance sampler is computed exactly
(crisis.second_moment). Samples needed for a given relative error are proportional to rho2. Samplers:
  one-eta twist   Q = R exp(eta * nominal loss of the event), the family crisis.cascade used
  rich twist      every source rate and every earthquake payout-level weight free (14 parameters), still ignoring the
                  state: the best any state-independent change of measure of this model can do
  projected Doob  the Doob transform averaged onto (time, running pool loss), the features RDPO learns on
Pre-set rule (decides whether training RDPO for this event is worth it): ceiling = rho2(rich twist) / rho2(projected
Doob) >= 10 for joint exhaustion -> go to phase 2; < 2 -> stop. Reported for all three events.
Limit: a per-sample ceiling. It says nothing about cost per sample or about how close a trained RDPO gets to the
projected Doob transform; those are phase 2.
"""
import numpy as np
from scipy.optimize import minimize

import crisis as C
from _src import save

S = C.S


def split_rates():
    """Rates with one entry per (source, nominal marks): needed because capped states merge marks into one transition."""
    R = []                                                   # list of (rate matrix, source, level a, level b)
    w1, w2 = C.W
    lam = C.LAM
    def mat(f):
        M = np.zeros((S, S))
        for s in range(S):
            M[s, f(C.A_[s], C.B_[s], C.R1_[s], C.R2_[s])] += 1.0
        return M
    for k in range(1, 5):
        R.append((lam[0] * w1[k - 1] * mat(lambda a, b, r1, r2: C.idx(min(a + k, 4), b, r1, r2)), 0, k, 0))
        R.append((lam[1] * w2[k - 1] * mat(lambda a, b, r1, r2: C.idx(a, min(b + k, 4), r1, r2)), 1, 0, k))
        for j in range(1, 5):
            R.append((lam[2] * w1[k - 1] * w2[j - 1] * mat(lambda a, b, r1, r2: C.idx(min(a + k, 4), min(b + j, 4), r1, r2)), 2, k, j))
    R.append((lam[3] * mat(lambda a, b, r1, r2: C.idx(a, b, 1, r2)), 3, 0, 0))
    R.append((lam[4] * mat(lambda a, b, r1, r2: C.idx(a, b, r1, 1)), 4, 0, 0))
    R.append((lam[5] * mat(lambda a, b, r1, r2: C.idx(a, b, 1, 1)), 5, 0, 0))
    return np.array([r[0] for r in R]), np.array([r[1:] for r in R])


RS, META = split_rates()
NOMINAL = 17.5 * (META[:, 1] + META[:, 2]) + np.where(META[:, 0] >= 3, np.where(META[:, 0] == 5, 20.0, 10.0), 0.0)


def log_tilt(x):
    """x = 6 source log multipliers, 4 KG level tilts, 4 TJ level tilts -> log multiplier of each split rate."""
    c, ua, ub = x[:6], np.concatenate([[0.0], x[6:10]]), np.concatenate([[0.0], x[10:14]])
    return c[META[:, 0]] + ua[META[:, 1]] + ub[META[:, 2]]


def rho2_static(logm, one, p):
    with np.errstate(all="ignore"):
        m2 = C.second_moment(RS, lambda t: RS * np.exp(logm)[:, None, None], one, steps=1)     # constant in time: one step is exact
    return m2 / p ** 2 - 1 if np.isfinite(m2) and m2 > 0 else np.inf


def projected(event, one, p, G):
    from scipy.linalg import expm
    lv = np.unique(C.LOSS)
    which = np.searchsorted(lv, C.LOSS)

    def Q(t, alpha=0.02):
        h, pi = expm((C.T - t) * G) @ one, expm(t * G)[0]
        num, den = np.bincount(which, pi * h, len(lv)), np.bincount(which, pi, len(lv))
        hb = np.maximum(np.where(den > 0, num / np.maximum(den, 1e-300), 0.0), 1e-12)[which]
        return RS * ((1 - alpha) * hb[None, None, :] / hb[None, :, None] + alpha)
    return C.second_moment(RS, Q, one) / p ** 2 - 1


def measure(event):
    one = C.EVENTS[event].astype(float)
    G = C.generator(RS)
    p = C.prob(event)
    f1 = lambda eta: rho2_static(eta * NOMINAL, one, p)
    grid = np.linspace(0.0, 0.08, 81)
    eta = grid[int(np.argmin([f1(e) for e in grid]))]
    x0 = np.zeros(14)                                      # start the rich family at the one-eta optimum
    x0[:6] = eta * np.array([0, 0, 0, 10.0, 10.0, 20.0])
    x0[6:10] = x0[10:14] = eta * 17.5 * np.arange(1, 5)
    obj = lambda x: np.log1p(min(rho2_static(log_tilt(x), one, p), 1e30))
    best = min((minimize(obj, x0 + d, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-9, "adaptive": True})
                for d in (0.0, 0.3 * np.ones(14))), key=lambda r: r.fun)
    best = minimize(obj, best.x, method="Powell", options={"xtol": 1e-7, "ftol": 1e-10})
    r_rich, r_proj = float(np.expm1(best.fun)), float(projected(event, one, p, G))
    return {"p": p, "rho2_plain": 1 / p - 1, "rho2_one_eta": float(f1(eta)), "eta": float(eta), "rho2_rich_static": r_rich,
            "rich_static_parameters": best.x.tolist(), "rho2_projected_doob": r_proj, "ceiling": r_rich / r_proj}


def run():
    res = {e: measure(e) for e in C.EVENTS}
    c = res["joint_exhaustion"]["ceiling"]
    res["verdict"] = "go to phase 2" if c >= 10 else ("stop" if c < 2 else "inconclusive")
    save("ceiling.json", res)
    for e, r in res.items():
        if isinstance(r, dict):
            print(f"{e}: p {r['p']:.3g} | relative variance: plain {r['rho2_plain']:.3g}, one-eta twist {r['rho2_one_eta']:.3g}, "
                  f"rich static twist {r['rho2_rich_static']:.3g}, projected Doob {r['rho2_projected_doob']:.3g} | ceiling x{r['ceiling']:.1f}")
    print("verdict:", res["verdict"])
    return res


def selftest():
    one, p = C.EVENTS["exceedance"].astype(float), C.prob("exceedance")
    assert np.allclose(RS.sum(0), C.rates().sum(0))                                   # split rates = the chain's rates
    assert abs(rho2_static(np.zeros(len(RS)), one, p) - (1 / p - 1)) < 1e-9           # no tilt = plain Monte Carlo
    x = np.zeros(14); x[:6] = 0.03 * np.array([0, 0, 0, 10.0, 10.0, 20.0]); x[6:10] = x[10:14] = 0.03 * 17.5 * np.arange(1, 5)
    assert np.allclose(log_tilt(x), 0.03 * NOMINAL)                                   # the rich family contains the one-eta twist
    # crisis.cascade's one-eta twist merges payout levels that end in the same capped state, so it is a slightly worse
    # sampler than the one here (12.2 against 8.6 at its eta); the split one is the fair baseline
    assert rho2_static(0.03 * NOMINAL, one, p) < C.cascade("exceedance", 1.0)["rho2_static"]
    print("selftest ok: split rates = chain; no tilt = plain Monte Carlo; rich family contains the one-eta twist, which is no worse than crisis.cascade's")


if __name__ == "__main__":
    selftest()
    run()
