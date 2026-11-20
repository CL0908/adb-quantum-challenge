"""The desk that learns: every payout and every quiet day is evidence about the model itself. Additive, read-only.

All earlier numbers (RDPO, the exact engines, the early-warning multipliers of crisis.py) hold the trigger rates fixed.
They are uncertain (desk.py's joint draws: catalogue counts and Gutenberg-Richter slope, magnitude agency, aftershock
clause, rain). A payout is more likely in worlds with high rates, so seeing one moves the posterior toward them:
    P(crisis | history) = sum_i  w_i(history) h_i(t, x),     w_i ~ prior_i * likelihood_i(history),
with h_i the exact Doob function of world i (crisis.py's chain) and the likelihood of the observed contract-state
path, prod over quiet spells of exp(G_i[x, x] dt) times prod over payouts of G_i[x, y].
Three consequences that a fixed-parameter model cannot show:
  contagion by learning   a payout in one country reprices the other note with no physical link, because both share
                          the slope, the magnitude agency and the contract clause
  rational after-shock    part of the price drop after an event is the state change, the rest is what the event taught
  quiet drift             every day without a payout moves the posterior down a little; an event gives it back at once
(Belief-updating contagion is known for credit, Benzoni et al. 2015; static parameter uncertainty for cat bonds,
Froot & Posner 2002. No event-by-event version for parametric cat bonds was found in a search.)
Pre-set rules: (1) with one world the learner equals the fixed-parameter chain; (2) updating step by step equals one
Bayes update on the whole history to 1e-10; (3) calibration on 20,000 simulated (world, path) pairs: mean forecast
within 3 standard errors of the realised crisis frequency, overall and among paths with a payout; (4) the effect is
large: after a 25% earthquake payout in KG the crisis-probability multiplier is >= 1.5x the fixed-parameter one, and the
TJ note falls >= 0.5 point where the fixed-parameter model gives < 0.1.
"""
import math

import numpy as np
from scipy.linalg import expm

import crisis as CR
import desk
import live_mark as LM
import rates_unc as RU
import variant
from _src import save, seed

M_WORLDS, T, NOW = 800, CR.T, LM.years(LM.ISSUE, LM.ASOF)
I4 = np.eye(4)
LOSS_NOTE = {"KG": (CR.SC.eq_limit * CR.A_ / 4 + CR.SC.rain_limit * CR.R1_), "TJ": (CR.SC.eq_limit * CR.B_ / 4 + CR.SC.rain_limit * CR.R2_)}
EVENTS = {"a 25% earthquake payout in KG": CR.idx(1, 0, 0, 0), "a 100% earthquake payout in KG": CR.idx(4, 0, 0, 0), "a rain payout in KG": CR.idx(0, 0, 1, 0),
          "a 25% earthquake payout in both": CR.idx(1, 1, 0, 0)}


def world_generator(eq_mult=(1.0, 1.0), rain_mult=(1.0, 1.0), nu=0.0):
    """100-state generator of one world: the earthquake part from variant.py (rates and aftershock clause), rain as in the brief."""
    Ge = variant.eq_generator(CR.SC, eq_mult, nu)
    mu = -math.log(1 - 0.02) * CR.SC.rain_mult
    m1, m2 = mu * rain_mult[0], mu * rain_mult[1]
    cr = CR.SC.rain_shared_share * min(m1, m2)
    Gr = np.zeros((4, 4))                                   # rain state r1 * 2 + r2
    for r1 in (0, 1):
        for r2 in (0, 1):
            s = r1 * 2 + r2
            for rate, (d1, d2) in ((m1 - cr, (1, 0)), (m2 - cr, (0, 1)), (cr, (1, 1))):
                t = max(r1, d1) * 2 + max(r2, d2)
                if t != s:
                    Gr[s, t] += rate
            Gr[s, s] = -Gr[s].sum()
    return np.kron(Ge, I4) + np.kron(np.eye(25), Gr)


def worlds(rng, n):
    with np.errstate(all="ignore"):
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            rm, _ = RU.rain_multipliers(np.random.default_rng(seed(4243)), 2000)
    d = desk.draws(rng, n, rm)
    f = 10 ** (desk.B_GR * d["agency"])
    return [world_generator(tuple(d["rate"][i] * f[i]), tuple(d["rain"][i] * d["warming"][i]), round(float(d["clause"][i]), 2)) for i in range(n)]


def log_lik(G, history):
    """history: list of (time, new state) payouts in time order, then the time now. Quiet spells and observed transitions."""
    x, t0, ll = 0, 0.0, 0.0
    for t, y in history["payouts"]:
        ll += G[x, x] * (t - t0) + math.log(max(G[x, y], 1e-300))
        x, t0 = y, t
    return ll + G[x, x] * (history["now"] - t0), x


class Learner:
    def __init__(self, Gs, prior=None):
        self.G = Gs
        self.prior = np.full(len(Gs), 1.0 / len(Gs)) if prior is None else prior

    def weights(self, history):
        ll = np.array([log_lik(G, history)[0] for G in self.G]) + np.log(self.prior)
        w = np.exp(ll - ll.max())
        return w / w.sum()

    def crisis(self, event, history, w=None):
        w = self.weights(history) if w is None else w
        x = log_lik(self.G[0], history)[1]
        one = CR.EVENTS[event].astype(float)
        return float(sum(wi * (expm((T - history["now"]) * G)[x] @ one) for wi, G in zip(w, self.G)))

    def price(self, note, history, w=None):
        """Indicative price of a note: desk.py's formula on the remaining expected loss, averaged over the worlds."""
        w = self.weights(history) if w is None else w
        x = log_lik(self.G[0], history)[1]
        tau = T - history["now"]
        full = CR.SC.eq_limit + CR.SC.rain_limit
        out = full - LOSS_NOTE[note][x]
        el = sum(wi * float(expm(tau * G)[x] @ LOSS_NOTE[note] - LOSS_NOTE[note][x]) for wi, G in zip(w, self.G)) / out * 100
        return LM.price(el, tau, desk.MULT0[note])


def simulate(G, rng, t_obs):
    """One path of a world: payouts up to t_obs (the observed history) and whether the crisis happens by T."""
    x, t, pay = 0, 0.0, []
    while True:
        rate = -G[x, x]
        t += rng.exponential(1 / rate) if rate > 0 else math.inf
        if t >= T:
            return pay, x
        p = np.maximum(G[x], 0.0).copy(); p[x] = 0.0
        x = int(rng.choice(CR.S, p=p / p.sum()))
        if t < t_obs:
            pay.append((t, x))


def run():
    rng = np.random.default_rng(seed(20261121))
    Gs = worlds(rng, M_WORLDS)
    L, fixed = Learner(Gs), Learner([world_generator()])
    quiet = {"payouts": [], "now": NOW}
    res = {"worlds": M_WORLDS, "now_years_after_issue": NOW, "events": {}}
    # (1) one world = the fixed-parameter chain; (2) step by step = all at once
    ok1 = abs(fixed.crisis("exceedance", {"payouts": [], "now": 0.0}) / CR.prob("exceedance") - 1) < 1e-9
    hist = {"payouts": [(0.3, CR.idx(1, 0, 0, 0)), (0.5, CR.idx(1, 0, 1, 0))], "now": NOW}
    w_all = L.weights(hist)
    w1 = L.weights({"payouts": hist["payouts"][:1], "now": 0.4})
    ll2 = np.array([G[CR.idx(1, 0, 0, 0), CR.idx(1, 0, 0, 0)] * (0.5 - 0.4) + math.log(max(G[CR.idx(1, 0, 0, 0), CR.idx(1, 0, 1, 0)], 1e-300))
                    + G[CR.idx(1, 0, 1, 0), CR.idx(1, 0, 1, 0)] * (NOW - 0.5) for G in Gs])
    w_seq = w1 * np.exp(ll2 - ll2.max()); w_seq /= w_seq.sum()
    seq_err = float(np.abs(w_seq - w_all).max())
    ok2 = seq_err < 1e-10
    # the effects
    base = {e: (L.crisis(e, quiet), fixed.crisis(e, quiet)) for e in ("exceedance", "joint_exhaustion")}
    p0 = {n: (L.price(n, quiet), fixed.price(n, quiet)) for n in LM.BONDS}
    prior_now = {n: L.price(n, quiet, w=L.prior) for n in LM.BONDS}
    res["quiet_drift_points"] = {n: p0[n][0] - prior_now[n] for n in LM.BONDS}
    res["now"] = {"crisis_learning": base["exceedance"][0], "crisis_fixed": base["exceedance"][1], "price_learning": {n: p0[n][0] for n in LM.BONDS}, "price_fixed": {n: p0[n][1] for n in LM.BONDS}}
    for name, y in EVENTS.items():
        h = {"payouts": [(NOW, y)], "now": NOW}
        row = {}
        for e in ("exceedance", "joint_exhaustion"):
            row[e] = {"multiplier_learning": L.crisis(e, h) / base[e][0], "multiplier_fixed": fixed.crisis(e, h) / base[e][1]}
            row[e]["ratio"] = row[e]["multiplier_learning"] / row[e]["multiplier_fixed"]
        for n in LM.BONDS:
            row[f"price_change_{n}"] = {"learning": L.price(n, h) - p0[n][0], "fixed": fixed.price(n, h) - p0[n][1]}
            row[f"price_change_{n}"]["taught_by_the_event"] = row[f"price_change_{n}"]["learning"] - row[f"price_change_{n}"]["fixed"]
        res["events"][name] = row
    # (3) calibration under the model with uncertain parameters
    n_sim, t_obs = 20000, 1.5
    pred, real, had = np.empty(n_sim), np.empty(n_sim), np.empty(n_sim, bool)
    one = CR.EVENTS["exceedance"]
    H = np.stack([expm((T - t_obs) * G) @ one.astype(float) for G in Gs])      # every world's Doob function at the observation time
    for k in range(n_sim):
        i = int(rng.integers(M_WORLDS))
        pay, xT = simulate(Gs[i], rng, t_obs)
        hist = {"payouts": pay, "now": t_obs}
        pred[k], real[k], had[k] = float(L.weights(hist) @ H[:, log_lik(Gs[0], hist)[1]]), one[xT], bool(pay)
    cal = {}
    for lab, m in (("all paths", np.ones(n_sim, bool)), ("paths with a payout before 1.5 years", had), ("quiet paths", ~had)):
        se = real[m].std() / math.sqrt(m.sum())
        cal[lab] = {"n": int(m.sum()), "mean_forecast": float(pred[m].mean()), "realised": float(real[m].mean()), "z": float((pred[m].mean() - real[m].mean()) / max(se, 1e-12))}
    res["calibration"] = cal
    ok3 = abs(cal["all paths"]["z"]) <= 3 and abs(cal["paths with a payout before 1.5 years"]["z"]) <= 3
    ev = res["events"]["a 25% earthquake payout in KG"]
    ok4 = ev["exceedance"]["ratio"] >= 1.5 and ev["price_change_TJ"]["learning"] <= -0.5 and abs(ev["price_change_TJ"]["fixed"]) < 0.1
    res["sequential_vs_batch_max_diff"] = seq_err
    res["checks"] = {"one_world_equals_fixed": bool(ok1), "sequential_equals_batch": bool(ok2), "calibrated": bool(ok3), "effect_is_large": bool(ok4)}
    res["verdict"] = "strong" if ok1 and ok2 and ok3 and ok4 else "not strong"
    save("learner.json", res)
    n = res["now"]
    print(f"today ({NOW:.2f} y after issue, no payout): crisis probability {n['crisis_learning']:.4f} with learning, {n['crisis_fixed']:.4f} fixed; "
          f"quiet drift since issue {res['quiet_drift_points']['KG']:+.2f} / {res['quiet_drift_points']['TJ']:+.2f} points (KG / TJ)")
    for name, r in res["events"].items():
        e, j = r["exceedance"], r["joint_exhaustion"]
        print(f"{name}: crisis x{e['multiplier_learning']:.1f} (fixed x{e['multiplier_fixed']:.1f}, ratio {e['ratio']:.2f}); joint exhaustion x{j['multiplier_learning']:.1f} (fixed x{j['multiplier_fixed']:.1f})"
              f" | KG note {r['price_change_KG']['learning']:+.1f} (fixed {r['price_change_KG']['fixed']:+.1f}) | TJ note {r['price_change_TJ']['learning']:+.2f} (fixed {r['price_change_TJ']['fixed']:+.2f})")
    for lab, c in cal.items():
        print(f"calibration, {lab}: n {c['n']}, forecast {c['mean_forecast']:.4f}, realised {c['realised']:.4f}, z {c['z']:+.2f}")
    print(f"sequential vs batch: {seq_err:.1e} | checks: {res['checks']} | verdict: {res['verdict']}")
    return res


def selftest():
    G = world_generator()
    assert np.abs(G - CR.generator(CR.rates())).max() < 1e-12            # one world at the brief's point is crisis.py's chain
    f = Learner([G])
    assert abs(f.crisis("exceedance", {"payouts": [], "now": 0.0}) / CR.prob("exceedance") - 1) < 1e-9
    # two worlds, one with doubled earthquake rates: a payout shifts weight to it, silence shifts weight away
    L = Learner([G, world_generator((2.0, 2.0))])
    assert L.weights({"payouts": [(0.5, CR.idx(1, 0, 0, 0))], "now": 0.5})[1] > 0.5 > L.weights({"payouts": [], "now": 2.0})[1]
    print("selftest ok: one world = the fixed-parameter chain; a payout moves belief to the high-rate world, silence to the low-rate world")


if __name__ == "__main__":
    selftest()
    run()
