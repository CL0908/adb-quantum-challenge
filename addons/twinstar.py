"""Twin and sampler as two halves of one transform: RDPO tilts paths inside a world, the twin tilts worlds.

Conditioning on a crisis A changes the path law inside a world (the Doob transform RDPO learns) and the weight of every
world, w_i -> w_i P_i(A): "if the crisis happens, which assumption was wrong". Three loops, each tested:
  why    fragility premium  E_worlds[P(A)] - P_point(A): the crisis probability is convex in the rates, so any
         uncertainty raises it (Taleb et al. 2012); split over the uncertain inputs. Lethal assumptions: how far each
         input sits from its prior in the worlds where the crisis happens. Tilted worlds: for the CAREC model-risk
         sweep, draw the catalogue counts from Gamma(k + 1/2 + g), g = dPhi/dlog(count) (crisis.py's gradient): the
         exact tilt if P(A) were a power law in the counts; fewer worlds for the same accuracy.
  watch  model alarm: the likelihood ratio of "rates uncertain" against "rates fixed at the calibration" on yearly
         main-shock counts is a nonnegative martingale under the fixed model, so P(it ever exceeds 1/alpha) <= alpha
         (Ville; Ramdas et al. 2023): a retraining trigger with a guaranteed false-alarm rate.
  how    crisis anatomy of the 28-country pool from weighted crisis paths (which countries pay in a pool crisis, and
         how much more likely the crisis is given a country paid), with paths from exponential twisting and from an
         RDPO policy, compared at equal CPU.
Pre-set rules: (1) premium: the quadratic expansion of Phi predicts the rate-and-agency premium within 30%; the lethal
ranking's top three are the same for 3 seeds; (2) tilted worlds need <= 1/5 of the worlds (variance ratio >= 5) on the
CAREC moderate threshold and agree with plain sampling within 3 standard errors; (3) alarm: false-alarm rate <= alpha
under the fixed model, and >= 50% detection within 3 years when all rates are 1.5x; (4) anatomy from weighted paths
equals the exact two-country values within 3 standard errors, and the 28-country top five is the same for 3 seeds.
"Binary star" needs both directions: (2) the twin makes the per-world engine cheaper, and (5) RDPO's crisis paths give
the anatomy at lower CPU than twisting's (efficiency ratio > 1).
"""
import contextlib
import io
import json
import math
import time

import numpy as np
from scipy.linalg import expm
from scipy.special import gammaln

import ap_risk
import crisis as CR
import desk
import learner as LE
import model_risk as MR
import rates_unc as RU
from _src import OUT, in_repo, save, seed
from src.rdpo.experiments_v2 import final_config
from src.rdpo.fast import FastPolicy
from src.rdpo.problem import PortfolioProblem
from src.rdpo.tilt import TiltSampler
from src.rdpo.train import _best_twist_sources, rdpo_estimate
from src.realdata import param_uncertainty as PU
from src.tensor.synthetic import SyntheticModel, topology

T, ALPHA = CR.T, 0.05
BASE = {"rate": (1.0, 1.0), "rain": (1.0, 1.0), "clause": 0.0, "agency": 0.0, "warming": 1.0}


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="ignore"):
        return f(*a, **k)


# ------------------------------------------------------------------------------------------- loop 1: why
def world(d, i, only=None):
    """Generator of world i of the joint draws d; with `only`, every other input stays at the brief's point."""
    g = lambda name: d[name][i] if only is None or name in only else BASE[name]
    f = 10 ** (desk.B_GR * float(g("agency")))
    return LE.world_generator(tuple(np.asarray(g("rate"), float) * f), tuple(np.asarray(g("rain"), float) * float(g("warming"))), round(float(g("clause")), 2))


def fragility(seed_=0, M=400, rain_pool=None):
    rng = np.random.default_rng(seed(20261122) + seed_)
    d = desk.draws(rng, M, rain_pool)
    P = lambda G, e: float(expm(T * G)[0] @ CR.EVENTS[e])
    G0 = LE.world_generator()
    out = {}
    for e in ("exceedance", "joint_exhaustion"):
        p0 = P(G0, e)
        full = np.array([P(world(d, i), e) for i in range(M)])
        parts = {g: float(np.mean([P(world(d, i, (g,)), e) for i in range(M)]) - p0) for g in desk.GROUPS}
        w = full / full.sum()                                   # the worlds seen from inside a crisis
        x = {"rate": np.log(d["rate"]).mean(1), "agency": d["agency"], "clause": (d["clause"] > 0).astype(float), "rain": np.log(d["rain"]).mean(1), "warming": d["warming"]}
        lethal = {g: {"prior_mean": float(v.mean()), "in_crisis_mean": float(w @ v), "shift_in_sd": float((w @ v - v.mean()) / v.std())} for g, v in x.items()}
        out[e] = {"p_point": p0, "p_with_uncertainty": float(full.mean()), "premium": float(full.mean() - p0), "premium_relative": float(full.mean() / p0 - 1),
                  "premium_by_input": parts, "interaction": float(full.mean() - p0 - sum(parts.values())), "lethal": lethal,
                  "lethal_ranking": sorted(lethal, key=lambda g: -abs(lethal[g]["shift_in_sd"]))}
    # rule: Phi's quadratic expansion predicts the premium of the rate and agency inputs (two log multipliers)
    U = np.log(d["rate"]) + desk.B_GR * math.log(10) * d["agency"][:, None]
    Phi = lambda u: math.log(P(LE.world_generator((math.exp(u[0]), math.exp(u[1]))), "exceedance"))
    h, f0 = 0.05, Phi((0.0, 0.0))
    e1, e2 = np.array([h, 0.0]), np.array([0.0, h])
    g = np.array([(Phi(e1) - Phi(-e1)) / (2 * h), (Phi(e2) - Phi(-e2)) / (2 * h)])
    H = np.array([[(Phi(e1) - 2 * f0 + Phi(-e1)) / h ** 2, 0.0], [0.0, (Phi(e2) - 2 * f0 + Phi(-e2)) / h ** 2]])
    H[0, 1] = H[1, 0] = (Phi(e1 + e2) + Phi(-e1 - e2) - Phi(e1 - e2) - Phi(e2 - e1)) / (4 * h ** 2)
    exact = float(np.mean([P(world(d, i, ("rate", "agency")), "exceedance") for i in range(M)]) - math.exp(f0))
    quad = float(np.mean(np.exp(f0 + U @ g + 0.5 * np.einsum("ni,ij,nj->n", U, H, U))) - math.exp(f0))
    lin = float(np.mean(np.exp(f0 + U @ g)) - math.exp(f0))
    out["quadratic_check"] = {"premium_exact": exact, "premium_quadratic": quad, "premium_linear": lin, "rel_err_quadratic": quad / exact - 1, "rel_err_linear": lin / exact - 1,
                              "gradient": g.tolist(), "hessian": H.tolist()}
    return out


def tilted_carec(n=256, procs=6):
    """Posterior-predictive exceedance of the CAREC pool with worlds drawn plainly and with counts tilted along dPhi/dlog(count)."""
    from multiprocessing import Pool
    rng = np.random.default_rng(seed(20261123))
    inp = MR.inputs()
    with in_repo():
        rnp = PU.rain_posterior()
    order, edges = inp["order"], list(inp["pairs"])
    obs = {"counts": dict(inp["counts"]), "b": inp["b"], "pairs": dict(inp["pairs"]), "fh": {c: inp["hydro_years"][c] / MR.YEARS_HY for c in order}, "s_rain": float(rnp["point"])}
    b_se = inp["b"] / math.sqrt(inp["n_b"])
    slim = {k: v for k, v in inp.items() if k != "model"}
    h = 0.1
    bump = []
    for c in order:
        for s in (h, -h):
            bump.append(dict(obs, counts={**obs["counts"], c: obs["counts"][c] * math.exp(s)}))
    for e in edges:
        for s in (h, -h):
            bump.append(dict(obs, pairs={**obs["pairs"], e: obs["pairs"][e] * math.exp(s)}))
    bump += [dict(obs, b=obs["b"] + 0.5 * b_se), dict(obs, b=obs["b"] - 0.5 * b_se)]
    plain = MR.draw(inp, rng, n, rnp)
    out = {"worlds": n, "targets": {}}
    with Pool(procs, initializer=MR._init, initargs=(slim,), maxtasksperchild=50) as pool:
        ev = lambda R: np.array(pool.map(MR.evaluate, R, chunksize=4))
        t0 = time.perf_counter()
        yb, yp = ev(bump), ev(plain)
        for name, col in (("exc_moderate", 0), ("exc_rare", 1)):
            ln = math.log(10) * yb[:, col]
            gc = {c: (ln[2 * i] - ln[2 * i + 1]) / (2 * h) for i, c in enumerate(order)}
            ge = {e: (ln[2 * len(order) + 2 * i] - ln[2 * len(order) + 2 * i + 1]) / (2 * h) for i, e in enumerate(edges)}
            gb = (ln[-2] - ln[-1]) / b_se
            sh = lambda a, g: max(a + g, 0.25 * a)             # keep the tilted shape positive
            worlds, lw = [], []
            for th in MR.draw(inp, rng, n, rnp):
                l = 0.0
                for c in order:
                    a, a2 = inp["counts"][c] + 0.5, sh(inp["counts"][c] + 0.5, gc[c])
                    x = rng.gamma(a2); th["counts"][c] = x
                    l += gammaln(a2) - gammaln(a) + (a - a2) * math.log(x)
                for e in edges:
                    a, a2 = inp["pairs"][e] + 0.5, sh(inp["pairs"][e] + 0.5, ge[e])
                    x = rng.gamma(a2); th["pairs"][e] = x
                    l += gammaln(a2) - gammaln(a) + (a - a2) * math.log(x)
                dlt = gb * b_se ** 2
                x = rng.normal(inp["b"] + dlt, b_se); th["b"] = max(x, 0.3)
                l += -dlt * (x - inp["b"]) / b_se ** 2 + dlt ** 2 / (2 * b_se ** 2)
                worlds.append(th); lw.append(l)
            yt = 10 ** ev(worlds)[:, col]
            y0, w = 10 ** yp[:, col], np.exp(np.array(lw))
            ok0, ok1 = np.isfinite(y0), np.isfinite(yt)
            m0, m1 = y0[ok0].mean(), (w * np.where(ok1, yt, 0.0)).mean()
            v0, v1 = y0[ok0].var() / m0 ** 2, (w * np.where(ok1, yt, 0.0)).var() / m1 ** 2
            se = math.sqrt(v0 / ok0.sum() * m0 ** 2 + v1 / n * m1 ** 2)
            out["targets"][name] = {"plain_mean": float(m0), "tilted_mean": float(m1), "z_difference": float((m1 - m0) / se), "rel_variance_plain": float(v0), "rel_variance_tilted": float(v1),
                                    "variance_ratio": float(v0 / v1), "largest_count_gradients": dict(sorted(gc.items(), key=lambda kv: -abs(kv[1]))[:4]), "gradient_b": float(gb),
                                    "weight_ess_fraction": float(w.sum() ** 2 / (w ** 2).sum() / n)}
        out["seconds"] = time.perf_counter() - t0
    return out


# ------------------------------------------------------------------------------------------- loop 2: watch
def log_e(Y, lam0, a=4.0):
    """log e-process after each year: fixed rates lam0 against rates lam0 * m, m ~ Gamma(a, a); half on one common m, half on one m per country."""
    cum, t = np.cumsum(Y, 0), np.arange(1, Y.shape[0] + 1)[:, None]
    le = lambda y, L: a * math.log(a) + gammaln(y + a) - gammaln(a) - (y + a) * np.log(L + a) + L
    common = le(cum.sum(1), (lam0[None, :] * t).sum(1))
    each = le(cum, lam0[None, :] * t).sum(1)
    return np.logaddexp(common, each) - math.log(2)


def alarm(rng, n_sim=4000):
    c = json.loads((OUT / "twin_counts.json").read_text())
    K = np.asarray(c["K_yearly"])
    friend = np.array([x["friend"] for x in c["check_1965_2016"]["countries"]])
    comcat = K[:52].sum(0)
    years, thr = 9, math.log(1 / ALPHA)
    lam = friend / 52
    sim = lambda mult, yrs: np.array([log_e(rng.poisson(lam * mult, (yrs, len(lam))), lam) for _ in range(n_sim)])
    h0 = sim(1.0, years)
    h1 = sim(1.5, years)
    one = np.ones(len(lam)); one[int(np.argmax(lam))] = 2.0
    h2 = sim(one, years)
    real = {"calibration = the friend's catalogue": log_e(K[52:], friend / 52), "calibration = ComCat 1965-2016": log_e(K[52:], np.maximum(comcat, 0.5) / 52)}
    return {"alpha": ALPHA, "false_alarm_rate_9y": float((h0.max(1) >= thr).mean()), "detect_1p5x_within_3y": float((h1[:, :3].max(1) >= thr).mean()),
            "detect_1p5x_within_9y": float((h1.max(1) >= thr).mean()), "detect_largest_country_2x_within_9y": float((h2.max(1) >= thr).mean()),
            "real_2017_2025": {k: {"e_by_year": np.exp(np.minimum(v, 50)).tolist(), "alarm": bool(v.max() >= thr), "max_e": float(np.exp(min(v.max(), 50)))} for k, v in real.items()}}


# ------------------------------------------------------------------------------------------- loop 3: how
def levels(P, acts):
    a = np.zeros((acts.shape[0], P.n), np.int64); r = np.zeros((acts.shape[0], P.n), np.int64)
    for k, s in enumerate(P.src):
        for site, inc in s["da"]:
            a[:, site] += np.asarray(inc)[acts[:, k]]
        for site, inc in s["dr"]:
            r[:, site] += np.asarray(inc)[acts[:, k]]
    return np.minimum(a, 4), np.minimum(r, 1)


def anatomy(P, acts, w, f, p_paid, batches=20):
    """In a pool crisis: each country's mean payout (in units of its own limit) and the lift P(A | country paid) / P(A)."""
    a, r = levels(P, acts)
    loss = (P.a_u * a + P.r_u * r) / P.F
    paid = (a + r) > 0
    g = (w * f)[:, None]

    def est(sl):
        pa = g[sl].mean()
        return (g[sl] * loss[sl]).mean(0) / pa, (g[sl] * paid[sl]).mean(0) / p_paid / pa
    full = est(slice(None))
    idx = np.array_split(np.arange(len(w)), batches)
    parts = [est(i) for i in idx]
    se = [np.std([p[j] for p in parts], 0, ddof=1) / math.sqrt(batches) for j in (0, 1)]
    return {"share": full[0], "share_se": se[0], "lift": full[1], "lift_se": se[1], "p": float(g.mean())}


def twist_paths(P, n, rng):
    t0 = time.process_time()
    S = TiltSampler(P); S.set_theta(S.twist_theta(_best_twist_sources(P)))
    o = S.sample(n, rng, record=True)
    return o["acts"].T.astype(np.int64), o["w"], o["f"].astype(float), time.process_time() - t0


def rdpo_paths(P, n, rng, b_fit=32768):
    t0 = time.process_time()
    cfg = final_config("exceedance")
    r = rdpo_estimate(P, b_fit, rng, cfg)
    pol = FastPolicy(P, cfg.alpha, cfg.edge_features, cfg.relevance, value=cfg.value, stages=cfg.stages)
    pol.theta = np.array(r["theta"], float)
    o = pol.rollout(n, rng, record=True)
    return o["acts"], o["w"], o["f"].astype(float), time.process_time() - t0


def paid_probability(P, rng, n=400000):
    S = TiltSampler(P); S.set_theta(S.twist_theta(0.0))
    o = S.sample(n, rng, record=True)
    a, r = levels(P, o["acts"].T.astype(np.int64))
    return ((a + r) > 0).mean(0)


def how(n_tw=1_000_000, n_rd=65536, b_fit=32768):
    rng = np.random.default_rng(seed(20261124))
    out = {}
    # exact check on the two-country model (crisis.py's chain gives the terminal law)
    P2 = PortfolioProblem(SyntheticModel(2, topology("chain", 2)), "exceedance")
    law = expm(T * CR.generator(CR.rates()))[0]
    A = CR.EVENTS["exceedance"]
    pa = law @ A
    ex_share = np.array([law[A] @ ((CR.SC.eq_limit * CR.A_[A] / 4 + CR.SC.rain_limit * CR.R1_[A]) / 80e6), law[A] @ ((CR.SC.eq_limit * CR.B_[A] / 4 + CR.SC.rain_limit * CR.R2_[A]) / 80e6)]) / pa
    paid2 = np.array([law @ ((CR.A_ + CR.R1_) > 0), law @ ((CR.B_ + CR.R2_) > 0)])
    ex_lift = np.array([law @ (A & ((CR.A_ + CR.R1_) > 0)), law @ (A & ((CR.B_ + CR.R2_) > 0))]) / paid2 / pa
    acts, w, f, _ = twist_paths(P2, n_tw, rng)
    an = anatomy(P2, acts, w, f, paid2)
    # in the two-country model a pool loss above US$80m needs both countries to pay, so the lift is exactly 1 / P(paid): no sampling error there
    z = np.r_[(an["share"] - ex_share) / an["share_se"], (an["lift"] - ex_lift) / np.maximum(an["lift_se"], 1e-9 * ex_lift)]
    out["two_country_check"] = {"share_exact": ex_share.tolist(), "share_paths": an["share"].tolist(), "lift_exact": ex_lift.tolist(), "lift_paths": an["lift"].tolist(), "max_abs_z": float(np.abs(z).max())}
    # the 28-country pool at the friend's point
    inp = ap_risk.inputs()
    P = PortfolioProblem(inp["model"], "exceedance", threshold=ap_risk.THRESH[0] * 1e6)
    order = inp["order"]
    p_paid = paid_probability(P, rng)
    tw = [twist_paths(P, n_tw, np.random.default_rng(seed(20261124) + 10 + k)) for k in range(3)]
    an_tw = [anatomy(P, a, w_, f_, p_paid) for a, w_, f_, _ in tw]
    rd = rdpo_paths(P, n_rd, np.random.default_rng(seed(20261124) + 50), b_fit)
    an_rd = anatomy(P, rd[0], rd[1], rd[2], p_paid)
    cpu_tw, cpu_rd = float(np.mean([t[3] for t in tw])), rd[3]
    eff_share = (an_tw[0]["share_se"] ** 2 * cpu_tw) / np.maximum(an_rd["share_se"] ** 2 * cpu_rd, 1e-300)
    eff_lift = (an_tw[0]["lift_se"] ** 2 * cpu_tw) / np.maximum(an_rd["lift_se"] ** 2 * cpu_rd, 1e-300)
    top = [list(np.array(order)[np.argsort(-a["share"])[:5]]) for a in an_tw]
    out["pool_28"] = {"p_crisis": an_tw[0]["p"], "p_crisis_rdpo_paths": an_rd["p"], "countries": order, "share": an_tw[0]["share"].tolist(), "share_se": an_tw[0]["share_se"].tolist(),
                      "lift": an_tw[0]["lift"].tolist(), "lift_se": an_tw[0]["lift_se"].tolist(), "p_paid": p_paid.tolist(), "top5_by_seed": top,
                      "top5_stable": bool(all(set(t) == set(top[0]) for t in top)), "cpu_twisting_1e6_paths": cpu_tw, "cpu_rdpo_65536_paths_with_training": cpu_rd,
                      "rdpo_vs_twisting_efficiency_share_median": float(np.median(eff_share)), "rdpo_vs_twisting_efficiency_lift_median": float(np.median(eff_lift)),
                      "share_rdpo_paths": an_rd["share"].tolist()}
    return out


def run():
    rng = np.random.default_rng(seed(20261125))
    t00 = time.perf_counter()
    rm, _ = quiet(RU.rain_multipliers, np.random.default_rng(seed(4243)), 2000)
    fr = [fragility(k, rain_pool=rm) for k in range(3)]
    q = fr[0]["quadratic_check"]
    tops = [tuple(f["exceedance"]["lethal_ranking"][:3]) for f in fr]
    ok1 = abs(q["rel_err_quadratic"]) <= 0.30 and all(set(t) == set(tops[0]) for t in tops)
    al = alarm(rng)
    ok3 = al["false_alarm_rate_9y"] <= ALPHA and al["detect_1p5x_within_3y"] >= 0.5
    tc = tilted_carec()
    m = tc["targets"]["exc_moderate"]
    ok2 = m["variance_ratio"] >= 5 and abs(m["z_difference"]) <= 3
    hw = how()
    ok4 = hw["two_country_check"]["max_abs_z"] <= 3 and hw["pool_28"]["top5_stable"]
    ok5 = hw["pool_28"]["rdpo_vs_twisting_efficiency_share_median"] > 1
    res = {"why": {"fragility": fr[0], "lethal_top3_by_seed": [list(t) for t in tops], "tilted_carec": tc}, "watch": al, "how": hw, "seconds": time.perf_counter() - t00,
           "checks": {"premium_and_lethal_ranking": bool(ok1), "tilted_worlds_save_5x": bool(ok2), "alarm_valid": bool(ok3), "anatomy_exact_and_stable": bool(ok4), "rdpo_paths_cheaper_than_twisting": bool(ok5)}}
    res["binary_star"] = bool(ok2 and ok5)
    res["verdict"] = "binary star" if ok2 and ok5 and ok1 and ok3 and ok4 else ("twin helps the sampler; the sampler need not be RDPO" if ok2 and not ok5 else "not a binary star")
    save("twinstar.json", res)
    for e in ("exceedance", "joint_exhaustion"):
        f = fr[0][e]
        print(f"{e}: point {f['p_point']:.3g} -> with uncertainty {f['p_with_uncertainty']:.3g} (premium {f['premium_relative']:+.0%}); by input " + ", ".join(f"{g} {v:+.2g}" for g, v in f["premium_by_input"].items())
              + f"; interaction {f['interaction']:+.2g}")
        print("   in a crisis: " + "; ".join(f"{g} {f['lethal'][g]['prior_mean']:.2f} -> {f['lethal'][g]['in_crisis_mean']:.2f} ({f['lethal'][g]['shift_in_sd']:+.2f} sd)" for g in f["lethal_ranking"]))
    print(f"quadratic check (rate + agency): premium exact {q['premium_exact']:.3g}, quadratic {q['premium_quadratic']:.3g} ({q['rel_err_quadratic']:+.0%}), linear {q['premium_linear']:.3g} ({q['rel_err_linear']:+.0%}); top three by seed {[list(t) for t in tops]}")
    for name, r in tc["targets"].items():
        print(f"tilted worlds, CAREC {name}: plain {r['plain_mean']:.4g}, tilted {r['tilted_mean']:.4g} (z {r['z_difference']:+.2f}); relative variance {r['rel_variance_plain']:.3g} -> {r['rel_variance_tilted']:.3g}, "
              f"ratio {r['variance_ratio']:.2f}; weight ESS {r['weight_ess_fraction']:.2f}")
    print(f"alarm (alpha {ALPHA}): false alarms {al['false_alarm_rate_9y']:.3f}; rates x1.5 detected within 3 y {al['detect_1p5x_within_3y']:.2f}, within 9 y {al['detect_1p5x_within_9y']:.2f}; "
          + "; ".join(f"{k}: max e {v['max_e']:.3g}, alarm {v['alarm']}" for k, v in al["real_2017_2025"].items()))
    c2, p28 = hw["two_country_check"], hw["pool_28"]
    o = np.argsort(-np.array(p28["share"]))[:6]
    print(f"anatomy check, two countries: max |z| {c2['max_abs_z']:.2f}; 28-country pool crisis p {p28['p_crisis']:.4g}: " + ", ".join(f"{p28['countries'][i]} pays {p28['share'][i]:.2f} of its limit (lift x{p28['lift'][i]:.1f})" for i in o))
    print(f"   top five by seed {p28['top5_by_seed']}; CPU twisting {p28['cpu_twisting_1e6_paths']:.0f}s per 1e6 paths, RDPO {p28['cpu_rdpo_65536_paths_with_training']:.0f}s; "
          f"RDPO / twisting efficiency: share {p28['rdpo_vs_twisting_efficiency_share_median']:.2f}, lift {p28['rdpo_vs_twisting_efficiency_lift_median']:.2f}")
    print("checks:", res["checks"], "| binary star:", res["binary_star"], "| verdict:", res["verdict"], f"| {res['seconds']:.0f}s")
    return res


def selftest():
    # e-process: no evidence when the data sit exactly at the calibration's mean rate over many years; strong evidence when rates double
    lam = np.array([1.0, 3.0, 2.0])
    rng = np.random.default_rng(0)
    assert log_e(np.tile(lam, (40, 1)), lam)[-1] < 0 < log_e(np.tile(2 * lam, (40, 1)), lam)[-1]
    # two-country levels from the friend's source form reproduce his exceedance probability with twisting weights
    P2 = PortfolioProblem(SyntheticModel(2, topology("chain", 2)), "exceedance")
    acts, w, f, _ = twist_paths(P2, 200000, np.random.default_rng(1))
    a, r = levels(P2, acts)
    hit = (P2.a_u * a + P2.r_u * r).sum(1) > P2.T
    assert np.array_equal(hit, f > 0)
    assert abs((w * f).mean() / CR.prob("exceedance") - 1) < 0.05
    # tilting a Gamma shape: the weight makes the mean of the untilted law come back
    x = rng.gamma(7.5, size=400000)
    wt = np.exp(gammaln(7.5) - gammaln(5.5) + (5.5 - 7.5) * np.log(x))
    assert abs((wt * x).mean() / 5.5 - 1) < 0.02
    print("selftest ok: e-process sign; levels from source increments reproduce the payoff and the exact probability; Gamma shape tilt is unbiased")


if __name__ == "__main__":
    selftest()
    run()
