"""Model-risk engine for the friend's Asia–Pacific pool with RDPO + multiple importance sampling (MIS). Additive.

Why RDPO here: the pool's dependence graph has a 15-country component, so dense enumeration is impossible; the friend's
tilted MPS takes ~50 s per parameter point and one RDPO run ~40 s, so a posterior of hundreds of points by either costs
hours. RDPO + MIS: train an RDPO policy at a few anchor points, pool their paths, and reweight the same paths to any
parameter point with the balance heuristic,
    μ̂(θ) = (1/N) Σ_i f(x_i) p_θ(x_i) / Σ_j (N_j/N) q_j(x_i),
where q_j(x) is policy j's path probability (replayed exactly from the friend's FastPolicy tables) and p_θ(x) the
product of the source laws at θ.
Uncertain inputs (the friend's calibration, asia_pacific_pool.calibrate): every country's tiers are pinned to its own
1-in-25.6-year event, so the marginals do not depend on the data; b and the record length cancel; what remains is
dependence: shared rate (i, j) = λ_t · K_ij / max(K_i, K_j) (capped at 95% of λ_t per country, his rule), with
  country counts  K_i  ~ Gamma(k_i + ½)      pair counts  K_ij ~ Gamma(k_ij + ½)      rain share ~ his station posterior.
Targets: pool exceedance at his moderate and rare thresholds (US$450m, US$850m) by RDPO + MIS; KG–TJ joint exhaustion
by his exact pair reduction (no sampling).
Anchors: 6 parameter points per threshold (5 quantiles of the total shared earthquake rate at the median rain share,
plus one at the 97.5% rain share); RDPO (final_config, B_FIT) trains each policy, then N_ANCHOR paths are drawn from it.
Pre-set rule: "strong" if (1) on 30 posterior points checked against the friend's MPS the median |relative error| is
<= 5% at US$450m and <= 15% at US$850m; (2) the CPU time of the whole MIS posterior (training, sampling, replay,
reweighting) is <= 1/10 of 500 MPS evaluations; (3) for >= 90% of 500 posterior points the effective sample size of the
weighted hits is >= 10% of the pooled paths (no weight collapse).
"""
import copy
import json
import math
import time

import numpy as np
from scipy import sparse
from scipy.special import logsumexp

import voi
from _src import OUT, in_repo, load, save
from src.realdata import asia_pacific_pool as AP
from src.realdata import param_uncertainty as PU
from src.realdata.carec_pool import build_mps_data
from src.realdata.central_asia_pool import DataModel
from src.rdpo.experiments_v2 import final_config
from src.rdpo.fast import FastPolicy
from src.rdpo.problem import PortfolioProblem
from src.rdpo.train import rdpo_estimate
from src.tensor.mps import targets_from_mps

THRESH = (450, 850)
N_ANCHOR, B_FIT, N_POST, N_CHECK, N_SOBOL, PROCS, SEED = 32768, 32768, 500, 30, 256, 6, 20261119
GROUPS = {"country_counts": ("K",), "pair_counts": ("Kp",), "rain_share": ("rs",)}
CAL = {}


# ------------------------------------------------------------------------------------------- the friend's model
def inputs():
    with in_repo():
        cal = AP.calibrate()
        rnp = PU.rain_posterior()
        refs = load("wp2_asia_pacific.json")["cases"]["asia_pacific"]
    order, edges = cal["order"], [tuple(e) for e in cal["model"].edges]
    years = AP.eq_check.YEARS[1] - AP.eq_check.YEARS[0] + 1
    pairs = {}
    for key, k in cal["n_both"].items():
        a, b = key.split("-")
        pairs[tuple(sorted((order.index(a), order.index(b))))] = k
    return {"order": order, "edges": edges, "pair": cal["pair"], "tails": [np.asarray(t) for t in cal["model"].tails],
            "lam_t": -math.log(1 - AP.T1), "K": np.array([cal["rate55"][c] * years for c in order]),
            "Kp": np.array([pairs[e] for e in edges], float), "rs": float(cal["rain_share"]), "rnp": rnp,
            "refs": refs, "model": cal["model"]}


def model_of(inp, th):
    K, Kp = th["K"], th["Kp"]
    shared = {e: inp["lam_t"] * Kp[m] / max(K[e[0]], K[e[1]]) for m, e in enumerate(inp["edges"])}
    for i in range(len(inp["order"])):                         # the friend's feasibility rule
        tot = sum(v for e, v in shared.items() if i in e)
        if tot > 0.95 * inp["lam_t"]:
            f = 0.95 * inp["lam_t"] / tot
            for e in shared:
                if i in e:
                    shared[e] *= f
    n = len(inp["order"])
    return DataModel(n, inp["edges"], tails=inp["tails"], shared_eq=shared, rain_p=[0.02] * n, rain_shared=th["rs"])


def observed(inp):
    return {"K": inp["K"].copy(), "Kp": inp["Kp"].copy(), "rs": inp["rs"]}


def draws(inp, rng, k):
    return [{"K": rng.gamma(inp["K"] + 0.5), "Kp": rng.gamma(inp["Kp"] + 0.5), "rs": float(rng.beta(inp["rnp"]["a"], inp["rnp"]["b"]))}
            for _ in range(k)]


# ------------------------------------------------------------------------------------------- exact (validation)
def exact_job(args):
    th, = args
    t0 = time.process_time()
    m = model_of(CAL, th)
    psi = build_mps_data(m, 1e-10)
    out = [targets_from_mps(m, psi, T * 1e6)["exceedance_prob"] for T in THRESH]
    return out, time.process_time() - t0


def je_job(th):
    return PortfolioProblem(model_of(CAL, th), "joint_exhaustion", pair=CAL["pair"]).exact_pair()


def logvec_job(th):
    t0 = time.process_time()
    P = PortfolioProblem(model_of(CAL, th), "exceedance", threshold=THRESH[0] * 1e6)   # source laws do not depend on T
    return np.concatenate([s["logp"] for s in P.src]), time.process_time() - t0


def _init(inp):
    CAL.update(inp)


# ------------------------------------------------------------------------------------------- RDPO + MIS
def replay_logq(pol, acts):
    """log q(path) under a FastPolicy for given increments: the friend's rollout loop with the actions fixed."""
    P = pol.P
    psi_t, chi_t, h_t = pol._tables()
    N = acts.shape[0]
    S, Lt, Li, Lj = pol._init_state(N)
    idx = np.arange(N); out = np.zeros(N); a = pol.alpha
    pi, pj = P.pair
    for k in range(P.K):
        plan, src = pol.plan[k], P.src[k]
        newS = {i: pol.next[k][i][S[:, i]] for i in plan["sites"]}
        dL = sum(pol.loss_tab[newS[i]] - pol.loss_tab[S[:, i]][:, None] for i in plan["sites"])
        Lt_new = Lt[:, None] + dL
        Li_new = pol.loss_tab[newS[pi]] if pi in newS else None
        Lj_new = pol.loss_tab[newS[pj]] if pj in newS else None
        v = acts[:, k]
        if plan["frozen"]:
            out += src["logp"][v]
        else:
            lg = src["logp"][None, :].repeat(N, 0)
            for i in plan["fsites"]:
                lg += psi_t[i][newS[i]] - psi_t[i][S[:, i]][:, None]
            for e in plan["edges"]:
                i, j = P.edges[e]
                si = newS[i] if i in newS else S[:, i][:, None]
                sj = newS[j] if j in newS else S[:, j][:, None]
                lg += chi_t[e][si, sj] - chi_t[e][S[:, i], S[:, j]][:, None]
            ht = h_t[pol.blk[k]]
            if pol.pair_target:
                Lin = Li_new if Li_new is not None else Li[:, None]
                Ljn = Lj_new if Lj_new is not None else Lj[:, None]
                lg += ht[Lin, Ljn] - ht[Li, Lj][:, None]
            else:
                lg += ht[Lt_new] - ht[Lt][:, None]
            lg -= lg.max(1, keepdims=True)
            qt = np.exp(lg); qt /= qt.sum(1, keepdims=True)
            out += np.log((1 - a) * qt[idx, v] + a * src["p"][v])
        for i in newS:
            S[:, i] = newS[i][idx, v]
        Lt = Lt_new[idx, v]
        if Li_new is not None:
            Li = Li_new[idx, v]
        if Lj_new is not None:
            Lj = Lj_new[idx, v]
    return out


def anchor_job(args):
    """Train an RDPO policy at an anchor, then draw N_ANCHOR recorded paths from it."""
    th, T, seed = args
    t0 = time.process_time()
    rng = np.random.default_rng(seed)
    P = PortfolioProblem(model_of(CAL, th), "exceedance", threshold=T * 1e6)
    cfg = final_config("exceedance")
    r = rdpo_estimate(P, B_FIT, rng, cfg)
    pol = FastPolicy(P, cfg.alpha, cfg.edge_features, cfg.relevance, value=cfg.value, stages=cfg.stages)
    pol.theta = np.array(r["theta"], dtype=float)
    out = pol.rollout(N_ANCHOR, rng, record=True)
    return {"theta": r["theta"], "acts": out["acts"].astype(np.int16), "f": out["f"].astype(np.float32), "cpu": time.process_time() - t0,
            "rdpo_estimate": r["estimate"], "paths_fit": r["paths_used"]}


def replay_job(args):
    th, T, theta, acts = args
    t0 = time.process_time()
    P = PortfolioProblem(model_of(CAL, th), "exceedance", threshold=T * 1e6)
    cfg = final_config("exceedance")
    pol = FastPolicy(P, cfg.alpha, cfg.edge_features, cfg.relevance, value=cfg.value, stages=cfg.stages)
    pol.theta = np.array(theta, dtype=float)
    return replay_logq(pol, acts.astype(np.int64)), time.process_time() - t0


def onehot(acts, sizes):
    """Sparse (paths × Σ V_k) indicator of each path's increment per source."""
    off = np.r_[0, np.cumsum(sizes)[:-1]]
    rows = np.repeat(np.arange(acts.shape[0]), acts.shape[1])
    cols = (acts + off[None, :]).ravel()
    return sparse.csr_matrix((np.ones(cols.size), (rows, cols)), shape=(acts.shape[0], int(np.sum(sizes))))


class MIS:
    def __init__(self, acts, f, logmix):
        self.f, self.logmix = f, logmix
        self.hit = np.flatnonzero(f > 0)
        self.C = None
        self.acts = acts

    def build(self, sizes):
        self.C = onehot(self.acts[self.hit], sizes)                # only hits contribute

    def estimate(self, LV, block=64):
        """μ̂ and ESS fraction for parameter points (columns of LV = log source laws), in column blocks to bound memory."""
        mu, ess = [], []
        for c in range(0, LV.shape[1], block):
            lw = np.asarray(self.C @ LV[:, c:c + block]) - self.logmix[self.hit][:, None]
            m = lw.max(0)
            w = np.exp(lw - m)
            mu.append(np.exp(m) * w.sum(0) / len(self.f))
            ess.append(w.sum(0) ** 2 / (w ** 2).sum(0) / len(self.f))
        return np.concatenate(mu), np.concatenate(ess)


def anchors(inp, rng):
    """5 quantiles of the total shared earthquake rate (median rain share) + 1 at the 97.5% rain share."""
    pool = draws(inp, rng, 4000)
    tot = np.array([sum(model_of(inp, th).c.values()) for th in pool])
    rs_med = float(np.median([th["rs"] for th in pool])); rs_hi = float(np.quantile([th["rs"] for th in pool], 0.975))
    pick = [pool[int(np.argmin(abs(tot - np.quantile(tot, q))))] for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    out = [dict(th, rs=rs_med) for th in pick] + [dict(pick[2], rs=rs_hi)]
    return out, np.quantile(tot, [0.05, 0.5, 0.95]).tolist()


def design(inp):
    rng = np.random.default_rng(SEED)
    anc, tot_q = anchors(inp, rng)
    post = draws(inp, rng, N_POST)
    A, B = draws(inp, rng, N_SOBOL), draws(inp, rng, N_SOBOL)
    AB = {g: [dict(a, **{k: b[k] for k in ks}) for a, b in zip(A, B)] for g, ks in GROUPS.items()}
    return anc, tot_q, post, A, B, A + B + [x for g in GROUPS for x in AB[g]]


def _pool():
    import os
    from multiprocessing import Pool
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):    # one thread per worker: CPU time comparable
        os.environ[v] = "1"
    inp = inputs()
    return inp, Pool(PROCS, initializer=_init, initargs=({k: v for k, v in inp.items() if k != "model"},), maxtasksperchild=4)


def exact_points():
    """Phase 1: the friend's MPS at N_CHECK posterior points and at his observed point (validation set, cached)."""
    inp, pool = _pool()
    post = design(inp)[2]
    with pool:
        exact = pool.map(exact_job, [(th,) for th in post[:N_CHECK] + [observed(inp)]])
    ex = np.array([x[0] for x in exact])
    out = {"seed": SEED, "n_check": N_CHECK, "thresholds_musd": list(THRESH), "exact": ex.tolist(), "cpu": [x[1] for x in exact]}
    save("ap_risk_exact.json", out)
    r = inp["refs"]["refs"]
    print(f"MPS at observed: {ex[-1].tolist()} vs friend's refs {[r[f'exceedance|{T}'] for T in THRESH]}; "
          f"rel diff {[float(ex[-1][k] / r[f'exceedance|{T}'] - 1) for k, T in enumerate(THRESH)]}")
    print(f"30 posterior points: exc|450 range [{ex[:-1, 0].min():.3g}, {ex[:-1, 0].max():.3g}], exc|850 range [{ex[:-1, 1].min():.3g}, {ex[:-1, 1].max():.3g}]; "
          f"MPS CPU per point median {np.median(out['cpu']):.1f}s")
    return out


def run():
    inp, pool = _pool()
    anc, tot_q, post, A, B, sobol_pts = design(inp)
    friend = observed(inp)
    try:
        c = json.loads((OUT / "ap_risk_exact.json").read_text())
        assert c["seed"] == SEED and c["n_check"] == N_CHECK
        exact = list(zip(c["exact"], c["cpu"]))
    except Exception:
        exact = None
    cpu = {"train_sample": 0.0, "replay": 0.0, "laws": 0.0, "reweight": 0.0}
    with pool:
        if exact is None:
            exact = pool.map(exact_job, [(th,) for th in post[:N_CHECK] + [friend]])
        je = np.array(pool.map(je_job, post + [friend], chunksize=8))
        lv = pool.map(logvec_job, post + sobol_pts + [friend], chunksize=8)
        LV = np.stack([x[0] for x in lv], 1)
        cpu["laws"] = float(sum(x[1] for x in lv))
        engines, anchor_info = {}, {}
        for T in THRESH:
            runs = pool.map(anchor_job, [(th, T, SEED + 1000 * T + j) for j, th in enumerate(anc)])
            cpu["train_sample"] += sum(r["cpu"] for r in runs)
            acts = np.concatenate([r["acts"] for r in runs]); f = np.concatenate([r["f"] for r in runs]).astype(float)
            reps = pool.map(replay_job, [(th, T, r["theta"], acts) for th, r in zip(anc, runs)])
            cpu["replay"] += sum(x[1] for x in reps)
            lq = np.stack([x[0] for x in reps], 1)
            logmix = logsumexp(lq + np.log(1.0 / len(anc)), axis=1)
            eng = MIS(acts, f, logmix)
            sizes = [len(s["p"]) for s in PortfolioProblem(model_of(inp, anc[0]), "exceedance", threshold=T * 1e6).src]
            eng.build(sizes)
            engines[T] = eng
            anchor_info[T] = {"rdpo_at_anchor": [r["rdpo_estimate"] for r in runs], "paths_fit": [r["paths_fit"] for r in runs],
                              "hit_rate": float(f.mean())}
    t0 = time.process_time()
    est = {T: engines[T].estimate(LV) for T in THRESH}
    cpu["reweight"] = time.process_time() - t0
    n_post, n_sob = len(post), len(sobol_pts)
    mu_post = {T: est[T][0][:n_post] for T in THRESH}; ess_post = {T: est[T][1][:n_post] for T in THRESH}
    mu_sob = {T: est[T][0][n_post:n_post + n_sob] for T in THRESH}
    mu_friend = {T: float(est[T][0][-1]) for T in THRESH}
    ex = np.array([x[0] for x in exact]); t_mps = float(np.median([x[1] for x in exact]))
    rel = {T: np.abs(mu_post[T][:N_CHECK] / ex[:N_CHECK, k] - 1) for k, T in enumerate(THRESH)}
    cpu_mis = float(sum(cpu.values())); cpu_mps = N_POST * t_mps
    res = {"thresholds_musd": list(THRESH), "n_post": N_POST, "n_check": N_CHECK, "anchors_total_shared_rate_q05_50_95": tot_q,
           "friend": {"refs": inp["refs"]["refs"], "exact_at_observed": ex[-1].tolist(), "mis_at_observed": mu_friend},
           "check": {str(T): {"median_rel_err": float(np.median(rel[T])), "max_rel_err": float(np.max(rel[T]))} for T in THRESH},
           "cpu_seconds": {"mis_total": cpu_mis, **cpu, "mps_per_point": t_mps, "mps_500_points": cpu_mps, "ratio": cpu_mis / cpu_mps},
           "ess_fraction": {str(T): {"q10": float(np.quantile(ess_post[T], 0.1)), "median": float(np.median(ess_post[T])),
                                     "share_ge_10pct": float(np.mean(ess_post[T] >= 0.10))} for T in THRESH},
           "anchors": anchor_info, "targets": {}}
    names = {450: "exc_450", 850: "exc_850"}
    rng2 = np.random.default_rng(SEED + 7)
    for T in THRESH:
        mu = mu_post[T]; fr = float(ex[-1][THRESH.index(T)])
        y = np.log10(np.maximum(mu_sob[T], 1e-300)); k = N_SOBOL
        s = voi.sobol(y[:k], y[k:2 * k], {g: y[2 * k + m * k: 2 * k + (m + 1) * k] for m, g in enumerate(GROUPS)}, rng2)
        res["targets"][names[T]] = {"friend": fr, "posterior_predictive": float(mu.mean()), "range95": [float(np.quantile(mu, 0.025)), float(np.quantile(mu, 0.975))],
                                    "P_above_friend": float(np.mean(mu > fr)), "S": s["S"], "T": s["T"], "T_ci": s["T_ci"]}
    jp, jf = je[:-1], float(je[-1])
    yA, yB = np.log10(np.array([je_job_local(inp, th) for th in A])), None
    res["targets"]["joint_exhaustion_KG_TJ"] = {"friend": jf, "posterior_predictive": float(jp.mean()),
                                                "range95": [float(np.quantile(jp, 0.025)), float(np.quantile(jp, 0.975))], "P_above_friend": float(np.mean(jp > jf))}
    ok1 = res["check"]["450"]["median_rel_err"] <= 0.05 and res["check"]["850"]["median_rel_err"] <= 0.15
    ok2 = cpu_mis <= cpu_mps / 10
    ok3 = all(res["ess_fraction"][str(T)]["share_ge_10pct"] >= 0.90 for T in THRESH)
    res.update(checks={"accuracy": bool(ok1), "cpu_tenth_of_mps": bool(ok2), "ess": bool(ok3)}, verdict="strong" if ok1 and ok2 and ok3 else "not strong")
    save("ap_risk.json", res)
    print(f"AP pool {len(inp['order'])} countries, {len(inp['edges'])} edges; anchors on total shared rate {np.round(tot_q, 5).tolist()}")
    print(f"friend refs { {k: f'{v:.4g}' for k, v in res['friend']['refs'].items()} } |exact at observed {np.round(ex[-1], 12).tolist()} | MIS at observed {mu_friend}")
    for T in THRESH:
        c = res["check"][str(T)]; e = res["ess_fraction"][str(T)]; t = res["targets"][names[T]]
        print(f"T={T}: check median rel err {c['median_rel_err']:.3f} (max {c['max_rel_err']:.3f}); ESS q10 {e['q10']:.3f}, median {e['median']:.3f}, share>=10% {e['share_ge_10pct']:.2f}; "
              f"posterior predictive {t['posterior_predictive']:.3g} 95% [{t['range95'][0]:.3g}, {t['range95'][1]:.3g}], P(above friend) {t['P_above_friend']:.2f}; "
              f"S " + ", ".join(f"{g} {v:.2f}" for g, v in t["S"].items()))
    j = res["targets"]["joint_exhaustion_KG_TJ"]
    print(f"KG–TJ joint exhaustion: friend {j['friend']:.3g}, posterior predictive {j['posterior_predictive']:.3g} 95% [{j['range95'][0]:.3g}, {j['range95'][1]:.3g}], P(above) {j['P_above_friend']:.2f}")
    print(f"CPU: MIS {cpu_mis:.0f}s ({ {k: round(v) for k, v in cpu.items()} }) vs 500 MPS {cpu_mps:.0f}s -> ratio {cpu_mis / cpu_mps:.3f}")
    print("checks:", res["checks"], "| verdict:", res["verdict"])
    return res


def je_job_local(inp, th):
    return PortfolioProblem(model_of(inp, th), "joint_exhaustion", pair=inp["pair"]).exact_pair()


def selftest():
    inp = inputs()
    m, f = model_of(inp, observed(inp)), inp["model"]
    assert all(np.allclose(a, b) for a, b in zip(m.tails, f.tails))
    assert all(abs(m.c[e] / f.c[e] - 1) < 1e-12 for e in f.c) and all(abs(m.cr[e] / f.cr[e] - 1) < 1e-12 for e in f.cr)
    ref_je = inp["refs"]["refs"]["joint_exhaustion|None"]
    assert abs(je_job_local(inp, observed(inp)) / ref_je - 1) < 1e-10
    # replayed path probability = the one rollout used (log q = Σ log p − log w)
    P = PortfolioProblem(f, "exceedance", threshold=THRESH[0] * 1e6)
    cfg = final_config("exceedance")
    pol = FastPolicy(P, cfg.alpha, cfg.edge_features, cfg.relevance, value=cfg.value, stages=cfg.stages)
    pol.theta = np.random.default_rng(0).normal(0, 0.3, pol.d)
    out = pol.rollout(2000, np.random.default_rng(1), record=True)
    sizes = [len(s["p"]) for s in P.src]
    logp = np.asarray(onehot(out["acts"], sizes) @ np.concatenate([s["logp"] for s in P.src])).ravel()
    assert np.allclose(replay_logq(pol, out["acts"]), logp - np.log(out["w"]), atol=1e-9)
    # one anchor at the evaluated point: MIS = that policy's own importance-sampling estimate
    eng = MIS(out["acts"], out["f"].astype(float), replay_logq(pol, out["acts"]))
    eng.build(sizes)
    mu, _ = eng.estimate(np.concatenate([s["logp"] for s in P.src])[:, None])
    assert abs(mu[0] - np.mean(out["w"] * out["f"])) <= 1e-12 + 1e-9 * abs(mu[0])
    print("selftest ok: parameterised AP model = friend's (tails, shared rates, exact pair); replayed log q exact; MIS reduces to IS")


if __name__ == "__main__":
    selftest()
    run()
