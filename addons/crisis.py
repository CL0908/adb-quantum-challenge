"""The crisis potential Phi = log P(crisis): one function behind sensitivities, model risk, early warning and the
sampler. Additive, read-only; exact on the brief's two-country model, written as a 100-state Markov chain in time.

The model has six Poisson sources (earthquake KG / TJ / shared, rain KG / TJ / shared) with rates lam_k. Write
theta_k = log lam_k. Conditioning on a crisis A keeps the model an exponential family in theta, so
    dPhi/dtheta_k              = E[N_k | A] - lam_k T                (extra events of source k in a crisis)
    d2Phi/dtheta_k dtheta_l    = Cov(N_k, N_l | A) - delta_kl lam_k T   (which sources come together in a crisis)
(N_k = number of events of source k in the term, including events that change nothing because a cover is exhausted).
Uses, each checked against something that already exists:
  identities      both formulas against an independent simulation of event counts
  model risk      Phi(theta + d) ~ Phi + g d + d H d / 2 turns a parameter posterior into the posterior of the crisis
                  probability from one point; compared with rates_unc.py's 2000-draw sweep
  early warning   h(t, x) = P(A | state x at time t) is a martingale along the events; its jump at an event is the
                  factor by which that event multiplies the crisis probability
  cascade         kappa multiplies every earthquake rate once any earthquake payout has happened (kappa = 1: the
                  brief). Then rates depend on the state, the sources no longer commute, and a tilt that ignores the
                  state is no longer the Doob transform. The variance of an importance sampler is computed exactly
                  (no simulation): E_Q[w^2 1_A] solves a linear equation with rates p^2 / q.
                  Samplers: the best state-independent exponential twist; the Doob transform projected on (time,
                  running pool loss), the features RDPO learns on; the exact Doob transform (variance 0).
Pre-set rules: (1) first and second derivatives equal the simulated conditional mean and covariance of event counts
within simulation error (|z| <= 3 for every entry) and at kappa = 1 the chain equals the friend's term targets;
(2) the quadratic formula reproduces the median and both ends of the 95% interval of rates_unc's rates-only sweep
within 10% for the three crisis probabilities, from at most 1/20 of its evaluations; (3) h is a martingale to 1e-9;
(4) cascade: for some kappa inside the range implied by the measured aftershock productivity, the best static twist's
relative variance is >= 10x its kappa = 1 value while the projected Doob sampler's changes by < 2x.
"""
import json
import math

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

import variant
from _src import OUT, save, seed
from src.model.poisson_fit import P_KG_GIVEN_TJ, P_TJ_GIVEN_KG, TAILS, compound_poisson_rates
from src.model.scenarios import SCENARIOS

SC, T = SCENARIOS["adb-base"], 3.0
NAMES = ("eq KG", "eq TJ", "eq shared", "rain KG", "rain TJ", "rain shared")
S = 100                                                  # state ((a * 5 + b) * 2 + r1) * 2 + r2
KAPPAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def idx(a, b, r1, r2):
    return ((a * 5 + b) * 2 + r1) * 2 + r2


A_, B_, R1_, R2_ = (np.array([f(s) for s in range(S)]) for f in (lambda s: s // 20, lambda s: (s // 4) % 5, lambda s: (s // 2) % 2, lambda s: s % 2))
LOSS = (SC.eq_limit * (A_ + B_) / 4 + SC.rain_limit * (R1_ + R2_)) / 1e6            # pool loss, US$m
EVENTS = {"exceedance": LOSS > SC.threshold / 1e6, "joint_exhaustion": (A_ == 4) & (B_ == 4) & (R1_ == 1) & (R2_ == 1),
          "joint_attachment": ((A_ > 0) | (R1_ > 0)) & ((B_ > 0) | (R2_ > 0))}
KEY = {"exceedance": "term_portfolio_loss_exceedance_probability", "joint_exhaustion": "term_both_bonds_exhausted_probability", "joint_attachment": "term_joint_attachment_probability"}


def sources():
    """Rates per year and jump matrices J_k (rows sum to 1; an event at an exhausted cover stays put), loss per event."""
    l1, L1 = compound_poisson_rates(TAILS["KG"]); l2, L2 = compound_poisson_rates(TAILS["TJ"])
    w1, w2 = l1 / L1, l2 / L2
    c = SC.m * 0.5 * (P_TJ_GIVEN_KG * L1 + P_KG_GIVEN_TJ * L2)
    mu = -math.log(1 - 0.02) * SC.rain_mult
    cr = SC.rain_shared_share * mu
    lam = np.array([L1 - c, L2 - c, c, mu - cr, mu - cr, cr])
    J = np.zeros((6, S, S))
    for s in range(S):
        a, b, r1, r2 = A_[s], B_[s], R1_[s], R2_[s]
        for k in range(1, 5):
            J[0, s, idx(min(a + k, 4), b, r1, r2)] += w1[k - 1]
            J[1, s, idx(a, min(b + k, 4), r1, r2)] += w2[k - 1]
            for j in range(1, 5):
                J[2, s, idx(min(a + k, 4), min(b + j, 4), r1, r2)] += w1[k - 1] * w2[j - 1]
        J[3, s, idx(a, b, 1, r2)] = 1.0; J[4, s, idx(a, b, r1, 1)] = 1.0; J[5, s, idx(a, b, 1, 1)] = 1.0
    return lam, J, (w1, w2)


LAM, JUMP, W = sources()
EQ_SRC = np.array([1, 1, 1, 0, 0, 0], bool)


def rates(theta=None, kappa=1.0):
    """Event rates R[k, x, y] (per year): source k takes x to y. kappa scales earthquake sources after any earthquake payout."""
    lam = LAM * np.exp(np.zeros(6) if theta is None else np.asarray(theta, float))
    mult = np.ones((6, S))
    mult[np.ix_(EQ_SRC, (A_ + B_) > 0)] = kappa
    return lam[:, None, None] * mult[:, :, None] * JUMP


def generator(R):
    G = R.sum(0)
    return G - np.diag(G.sum(1))


def prob(event, theta=None, kappa=1.0, years=T):
    return float(expm(years * generator(rates(theta, kappa)))[0] @ EVENTS[event])


def gradient(event, theta=None):
    """dPhi/dtheta at kappa = 1, by the identity: E[N_k 1_A] is an integral of exp(sG) lam_k J_k exp((T - s)G) (Van Loan block)."""
    R = rates(theta)
    G, one = generator(R), EVENTS[event].astype(float)
    p = float(expm(T * G)[0] @ one)
    lam = LAM * np.exp(np.zeros(6) if theta is None else np.asarray(theta, float))
    g = np.empty(6)
    for k in range(6):
        blk = expm(T * np.block([[G, R[k]], [np.zeros((S, S)), G]]))[:S, S:]
        g[k] = float(blk[0] @ one) / p - lam[k] * T
    return g, p


def hessian(event, eps=1e-4):
    H = np.empty((6, 6))
    for k in range(6):
        d = np.zeros(6); d[k] = eps
        H[k] = (gradient(event, d)[0] - gradient(event, -d)[0]) / (2 * eps)
    return 0.5 * (H + H.T)


def simulate(event, n, rng):
    """Event counts and the crisis indicator by direct simulation at kappa = 1 (sources commute, so order is irrelevant)."""
    N = rng.poisson(LAM * T, (n, 6))
    lvl = np.zeros((n, 2), int)
    for k, cols in ((0, (0,)), (1, (1,)), (2, (0, 1))):
        for col in cols:
            marks = rng.choice(4, (n, 4), p=W[col]) + 1
            use = np.arange(4)[None, :] < N[:, k][:, None]            # only the first four events can matter under the cap
            lvl[:, col] += (marks * use).sum(1)
    a, b = np.minimum(lvl[:, 0], 4), np.minimum(lvl[:, 1], 4)
    r1, r2 = ((N[:, 3] + N[:, 5]) > 0).astype(int), ((N[:, 4] + N[:, 5]) > 0).astype(int)
    return N, EVENTS[event][idx(a, b, r1, r2)]


# ------------------------------------------------------------------------------------------- model risk from one point
def quadratic_model_risk(seed_offset=0):
    """rates_unc's rates-only sweep (four rate multipliers, 2000 draws) from one point: value, gradient and Hessian of the
    log target in the log multipliers by central differences (33 evaluations)."""
    import contextlib
    import io
    import rates_unc as RU
    rng = np.random.default_rng(RU.SEED)
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        from _src import in_repo
        with in_repo():
            eqp, rnp = RU.PU.eq_posterior(), RU.PU.rain_posterior()
        rng.beta(eqp["a"], eqp["b"], RU.N); rng.beta(rnp["a"], rnp["b"], RU.N)       # same stream as rates_unc.run
        em, _ = RU.eq_multipliers(rng, RU.N)
        rm, _ = RU.rain_multipliers(rng, RU.N)
    U = np.log(np.column_stack([em["KG"], em["TJ"], rm["KG"], rm["TJ"]]))
    f = lambda u: variant.targets(eq_mult=(math.exp(u[0]), math.exp(u[1])), rain_mult=(math.exp(u[2]), math.exp(u[3])))
    h, d, calls = 0.05, 4, 1
    f0 = f(np.zeros(d))
    plus, minus = [], []
    for i in range(d):
        e = np.zeros(d); e[i] = h
        plus.append(f(e)); minus.append(f(-e)); calls += 2
    cross = {}
    for i in range(d):
        for j in range(i + 1, d):
            e = np.zeros(d); e[i] = h; e[j] = h
            e2 = np.zeros(d); e2[i] = h; e2[j] = -h
            cross[(i, j)] = (f(e), f(-e), f(e2), f(-e2)); calls += 4
    ref = json.loads((OUT / "quality_rates_unc.json").read_text())["sweeps"]["rates_only"]["summary"]
    out = {"evaluations": calls, "sweep_evaluations": RU.N, "targets": {}}
    for name, key in {"joint_exhaustion": KEY["joint_exhaustion"], "exceedance": KEY["exceedance"], "joint_attachment": KEY["joint_attachment"], "expected_loss": "term_expected_principal_loss_usd"}.items():
        L = lambda t: math.log(t[key])
        g = np.array([(L(plus[i]) - L(minus[i])) / (2 * h) for i in range(d)])
        H = np.diag([(L(plus[i]) - 2 * L(f0) + L(minus[i])) / h ** 2 for i in range(d)])
        for (i, j), (pp, mm, pm, mp) in cross.items():
            H[i, j] = H[j, i] = (L(pp) + L(mm) - L(pm) - L(mp)) / (4 * h ** 2)
        y = np.exp(L(f0) + U @ g + 0.5 * np.einsum("ni,ij,nj->n", U, H, U))
        y1 = np.exp(L(f0) + U @ g)
        r = ref[name]
        q = lambda v: [float(np.quantile(v, x)) for x in (0.025, 0.5, 0.975)]
        want = [r["ci95"][0], r["median"], r["ci95"][1]]
        out["targets"][name] = {"sweep": want, "quadratic": q(y), "linear": q(y1), "rel_err_quadratic": [a / b - 1 for a, b in zip(q(y), want)],
                                "rel_err_linear": [a / b - 1 for a, b in zip(q(y1), want)], "gradient": g.tolist(), "hessian_diag": np.diag(H).tolist()}
    return out


# ------------------------------------------------------------------------------------------- early warning
def early_warning(event="exceedance"):
    G = generator(rates())
    one = EVENTS[event].astype(float)
    h = lambda t: expm((T - t) * G) @ one
    t, dt = 0.6, 0.25                                     # martingale: E[h(t + dt, X_{t+dt}) | X_t = x] = h(t, x) for every x
    err = float(np.max(np.abs(expm(dt * G) @ h(t + dt) - h(t))))
    h0 = h(0.57)                                          # 18 Nov 2026 is 0.57 years after pricing
    jumps = {"a 25% earthquake payout in KG": h0[idx(1, 0, 0, 0)] / h0[0], "a 100% earthquake payout in KG": h0[idx(4, 0, 0, 0)] / h0[0],
             "a rain payout in KG": h0[idx(0, 0, 1, 0)] / h0[0], "a 25% earthquake payout in both": h0[idx(1, 1, 0, 0)] / h0[0]}
    return {"martingale_error": err, "crisis_probability_now": float(h0[0]), "multiplier_after": {k: float(v) for k, v in jumps.items()}}


# ------------------------------------------------------------------------------------------- cascade: exact sampler variance
def second_moment(R, Q_of_t, one, steps=48):
    """E_Q[(dP/dQ)^2 1_A] for a sampler with event rates Q(t)[k, x, y] > 0 wherever R > 0, piecewise constant in time."""
    v = one.astype(float).copy()
    dt = T / steps
    px = R.sum((0, 2))
    for s in reversed(range(steps)):
        Q = Q_of_t((s + 0.5) * dt)
        with np.errstate(divide="ignore", invalid="ignore"):
            M = np.where(R > 0, R ** 2 / Q, 0.0).sum(0)
        M -= np.diag(2 * px - Q.sum((0, 2)))
        v = expm(dt * M) @ v
    return float(v[0])


def cascade(event, kappa):
    R = rates(kappa=kappa)
    G, one = generator(R), EVENTS[event].astype(float)
    p = float(expm(T * G)[0] @ one)
    dL = LOSS[None, :] - LOSS[:, None]                       # realised loss increment of a transition
    mark = np.zeros((6, S, S))                               # nominal (uncapped) loss of the event, as a state-free tilt uses
    for s in range(S):
        a, b, r1, r2 = A_[s], B_[s], R1_[s], R2_[s]
        for k in range(1, 5):
            mark[0, s, idx(min(a + k, 4), b, r1, r2)] = mark[1, s, idx(a, min(b + k, 4), r1, r2)] = 17.5 * k
            for j in range(1, 5):
                mark[2, s, idx(min(a + k, 4), min(b + j, 4), r1, r2)] = 17.5 * (k + j)
        mark[3, s, idx(a, b, 1, r2)] = mark[4, s, idx(a, b, r1, 1)] = 10.0; mark[5, s, idx(a, b, 1, 1)] = 20.0
    rho2 = lambda m2: m2 / p ** 2 - 1
    def static(eta):                                     # a strong tilt on fast rates overflows the matrix exponential: treat as unusable
        with np.errstate(all="ignore"):
            m2 = second_moment(R, lambda t: R * np.exp(eta * mark), one)
        return m2 if np.isfinite(m2) and m2 > 0 else np.inf
    grid = np.linspace(0.0, 0.08, 33)
    vals = [static(e) for e in grid]
    k0 = int(np.argmin(vals))
    best = minimize_scalar(static, bounds=(grid[max(k0 - 1, 0)], grid[min(k0 + 1, len(grid) - 1)]), method="bounded", options={"xatol": 1e-5})
    # Doob transform projected on (time, running pool loss)
    levels = np.unique(LOSS)
    which = np.searchsorted(levels, LOSS)

    def Q_proj(t, alpha=0.02):
        h = expm((T - t) * G) @ one
        pi = expm(t * G)[0]
        num, den = np.bincount(which, pi * h, len(levels)), np.bincount(which, pi, len(levels))
        hb = np.maximum(np.where(den > 0, num / np.maximum(den, 1e-300), 0.0), 1e-12)[which]
        return R * ((1 - alpha) * hb[None, None, :] / hb[None, :, None] + alpha)

    def Q_doob(t):
        h = np.maximum(expm((T - t) * G) @ one, 1e-300)
        return R * h[None, None, :] / h[None, :, None]

    return {"kappa": kappa, "p": p, "rho2_plain": 1 / p - 1, "rho2_static": rho2(best.fun), "eta": float(best.x),
            "rho2_projected": rho2(second_moment(R, Q_proj, one)), "rho2_exact_doob": rho2(second_moment(R, Q_doob, one))}


def run():
    rng = np.random.default_rng(seed(20261120))
    res = {"sources": dict(zip(NAMES, LAM.tolist())), "identities": {}}
    # (1) the chain is the friend's model; derivatives are conditional moments of event counts
    t = variant.targets()
    chain_err = max(abs(prob(e) / t[KEY[e]] - 1) for e in EVENTS)
    ok1 = chain_err < 1e-9
    for e in ("exceedance", "joint_attachment"):
        g, p = gradient(e)
        H = hessian(e)
        N, hit = simulate(e, 4_000_000, rng)
        Nc = N[hit].astype(float)
        m_sim, n_c = Nc.mean(0) - LAM * T, len(Nc)
        z1 = (g - m_sim) / (Nc.std(0) / math.sqrt(n_c))
        C = np.cov(Nc.T) - np.diag(LAM * T)
        boot = np.array([np.cov(Nc[rng.integers(0, n_c, n_c)].T) for _ in range(60)])
        z2 = (H - C) / np.maximum(boot.std(0), 1e-12)
        res["identities"][e] = {"p": p, "crisis_paths": int(n_c), "extra_events_in_crisis": dict(zip(NAMES, g.tolist())), "simulated": dict(zip(NAMES, m_sim.tolist())),
                                "max_abs_z_gradient": float(np.abs(z1).max()), "max_abs_z_hessian": float(np.abs(z2).max()), "hessian": H.tolist(), "simulated_cov_minus_mean": C.tolist()}
        ok1 &= np.abs(z1).max() <= 3 and np.abs(z2).max() <= 3
    g_je, _ = gradient("joint_exhaustion")
    res["identities"]["joint_exhaustion_extra_events"] = dict(zip(NAMES, g_je.tolist()))
    res["chain_vs_friend_max_rel_err"] = chain_err
    # (2) model risk from one point
    mr = quadratic_model_risk()
    res["model_risk"] = mr
    worst = max(abs(x) for n in ("joint_exhaustion", "exceedance", "joint_attachment") for x in mr["targets"][n]["rel_err_quadratic"])
    ok2 = worst <= 0.10 and mr["evaluations"] <= mr["sweep_evaluations"] / 20
    # (3) early warning
    ew = early_warning()
    res["early_warning"] = ew
    ok3 = ew["martingale_error"] < 1e-9
    # (4) cascade
    res["cascade"] = {e: [cascade(e, k) for k in KAPPAS] for e in ("exceedance", "joint_exhaustion")}
    nu = 0.5                                                 # eq_cluster.py: extra pay-size events per event; as a rate multiplier over the ~1.5 years left on average
    kappa_real = 1 + nu / (float(LAM[:3].sum()) * 1.5)
    res["kappa_from_aftershock_productivity"] = kappa_real
    ok4 = False
    for e, rows in res["cascade"].items():
        base = next(r for r in rows if r["kappa"] == 1.0)
        for r in rows:
            r["static_vs_kappa1"] = r["rho2_static"] / base["rho2_static"]; r["projected_vs_kappa1"] = r["rho2_projected"] / base["rho2_projected"]
            ok4 |= 1.0 < r["kappa"] <= kappa_real and r["static_vs_kappa1"] >= 10 and r["projected_vs_kappa1"] < 2
    res["checks"] = {"identities": bool(ok1), "model_risk_quadratic": bool(ok2), "early_warning_martingale": bool(ok3), "cascade_breaks_static_twist": bool(ok4)}
    res["verdict"] = "strong" if all(res["checks"].values()) else "not strong"
    save("crisis.json", res)
    print(f"chain vs the friend's term targets: max rel err {chain_err:.1e}")
    for e in ("exceedance", "joint_attachment"):
        r = res["identities"][e]
        print(f"{e}: p {r['p']:.4g}; extra events in a crisis " + ", ".join(f"{k} {v:+.3f}" for k, v in r["extra_events_in_crisis"].items())
              + f" | vs simulation ({r['crisis_paths']} crisis paths): max |z| gradient {r['max_abs_z_gradient']:.2f}, Hessian {r['max_abs_z_hessian']:.2f}")
    print("joint exhaustion, extra events: " + ", ".join(f"{k} {v:+.3f}" for k, v in res["identities"]["joint_exhaustion_extra_events"].items()))
    print(f"model risk from one point: {mr['evaluations']} evaluations instead of {mr['sweep_evaluations']}")
    for n, r in mr["targets"].items():
        print(f"  {n}: sweep 2.5/50/97.5% {['%.3g' % x for x in r['sweep']]} | quadratic err {['%+.1f%%' % (100 * x) for x in r['rel_err_quadratic']]} | linear err {['%+.1f%%' % (100 * x) for x in r['rel_err_linear']]}")
    print(f"early warning: martingale error {ew['martingale_error']:.1e}; crisis probability now {ew['crisis_probability_now']:.4g}; multiplied by " + ", ".join(f"{v:.1f} after {k}" for k, v in ew["multiplier_after"].items()))
    for e, rows in res["cascade"].items():
        print(f"cascade, {e} (kappa implied by aftershock productivity: {kappa_real:.1f}):")
        for r in rows:
            print(f"  kappa {r['kappa']:5.2f}: p {r['p']:.3g} | relative variance: plain {r['rho2_plain']:.3g}, best static twist {r['rho2_static']:.3g} (x{r['static_vs_kappa1']:.2f} of kappa=1), "
                  f"projected Doob {r['rho2_projected']:.3g} (x{r['projected_vs_kappa1']:.2f}), exact Doob {r['rho2_exact_doob']:.1e}")
    print("checks:", res["checks"], "| verdict:", res["verdict"])
    return res


def selftest():
    t = variant.targets()
    assert all(abs(prob(e) / t[KEY[e]] - 1) < 1e-9 for e in EVENTS)                 # kappa = 1 is the friend's model
    R, one = rates(), EVENTS["exceedance"].astype(float)
    p = prob("exceedance")
    assert abs(second_moment(R, lambda t_: R, one) / p - 1) < 1e-9                    # sampling from the model itself: E[1_A]
    g, _ = gradient("exceedance")
    d = np.zeros(6); d[2] = 1e-5
    assert abs((math.log(prob("exceedance", d)) - math.log(prob("exceedance", -d))) / 2e-5 - g[2]) < 1e-6    # identity vs finite difference
    assert abs(prob("exceedance", kappa=4.0) - p) > 1e-4 and prob("exceedance", kappa=4.0) > p               # a cascade raises the crisis probability
    print("selftest ok: chain = the friend's two-country model; second moment under the model itself; gradient identity = finite difference; cascade raises the crisis probability")


if __name__ == "__main__":
    selftest()
    run()
