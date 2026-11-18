"""Risk twin of the friend's Asia-Pacific pool: the parameter posterior is the state, new earthquake records update it
every year, rare-event queries are answered by reweighting paths that already exist (ap_risk's RDPO + MIS engine).
Additive, read-only.
Phase 0: yearly country and pair counts from a catalogue, by the friend's own rules (Gardner-Knopoff declustering,
50 km, a shock reaching several countries counts for its two nearest).
State in year y: the friend's 1965-2016 counts plus ComCat main shocks 2017..y (Gamma(k + 1/2) posteriors); the rain share
keeps his station posterior (no new rain data here); the dependence graph and the return-period design stay as in 2016.
Queries each year: pool exceedance at US$450m / US$850m for 200 posterior draws plus a Sobol design (which data matter
now), all by reweighting one fixed set of paths. Anchors are fixed from the 2016 posterior before any update.
Three ways to answer:  A  RDPO anchors + reweighting;  B  exponential-twisting anchors + reweighting (8x the paths);
C  exponential twisting from scratch at every point, run until it reaches A's error (cost from its measured variance).
Pre-set rules, all four for "strong": (1) on 10 MPS-checked points in each of 2017, 2021, 2025 the median relative
error of A is <= 2% (US$450m) and <= 5% (US$850m); (2) total CPU of A <= 1/5 of C; (3) at equal total CPU (error scaled
by sqrt of the CPU ratio) A's RMS error <= 0.7 x B's, both thresholds; (4) PSIS k-hat < 0.7 for >= 90% of A's posterior
queries. If (1), (2), (4) hold but (3) fails, the twin works without RDPO and is reported as such: the same rules
(1), (2), (4) are then checked for B itself (error, cost against C at B's error, k-hat), and the yearly output uses B.
"""
import csv
import json
import math
import os
import time
from datetime import datetime

import numpy as np
from scipy.special import logsumexp

import ap_risk
import voi
from _src import OUT, in_repo, save, seed
from src.realdata import asia_pacific_pool as AP
from src.rdpo.problem import PortfolioProblem
from src.rdpo.tilt import TiltSampler, tilt_estimate
from src.rdpo.train import _best_twist_sources

Y0, Y_CAL, Y1 = 1965, 2016, 2025
THRESH = ap_risk.THRESH
N_DRAW, N_SOB, CHECK_YEARS, N_CHK, TW_MULT, PROCS, SEED = 200, 128, (2017, 2021, 2025), 10, 8, 6, seed(20261118)
GROUPS = ap_risk.GROUPS


def friend_events():
    with in_repo():
        return AP._load(*AP._cache())


def comcat_events(path):
    """ComCat csv (time, latitude, longitude, depth, mag, ...) -> the friend's (time, lat, lon, mag) tuples."""
    ev = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["mag"]:
                ev.append((datetime.fromisoformat(r["time"][:19]), float(r["latitude"]), float(r["longitude"]), float(r["mag"])))
    return sorted(ev)


def yearly(ev, polys, order, edges, y0=Y0, y1=Y_CAL):
    """Main shocks within 50 km per country and year, shared main shocks per edge and year; pairs outside `edges`."""
    ev = AP.eq_check.decluster([e for e in ev if y0 <= e[0].year <= y1])
    dist = np.stack([[AP.eq_check.dist_to_country(la, lo, polys[c]) for _, la, lo, _ in ev] for c in order], 1)
    near = dist <= AP.R_KM
    yr = np.array([e[0].year for e in ev]) - y0
    K = np.zeros((y1 - y0 + 1, len(order))); Kp = np.zeros((y1 - y0 + 1, len(edges)))
    np.add.at(K, (np.repeat(yr, near.shape[1]), np.tile(np.arange(near.shape[1]), len(ev))), near.ravel())
    eidx, new = {e: m for m, e in enumerate(edges)}, {}
    for e in range(len(ev)):
        hit = np.flatnonzero(near[e])
        if len(hit) >= 2:
            pr = tuple(sorted(int(x) for x in hit[np.argsort(dist[e, hit])[:2]]))
            if pr in eidx:
                Kp[yr[e], eidx[pr]] += 1
            else:
                key = f"{order[pr[0]]}-{order[pr[1]]}"
                new.setdefault(key, []).append(int(yr[e]) + y0)
    return {"years": [y0, y1], "main_shocks": len(ev), "K": K, "Kp": Kp, "new_pairs": new}


# ------------------------------------------------------------------------------------------- twin core
def state(inp, counts, year):
    k = year - Y_CAL
    r0 = Y_CAL - Y0 + 1
    return dict(inp, K=inp["K"] + np.asarray(counts["K_yearly"])[r0:r0 + k].sum(0), Kp=inp["Kp"] + np.asarray(counts["Kp_yearly"])[r0:r0 + k].sum(0))


def gpdfit(x):
    """GPD shape of sorted positive exceedances (Zhang & Stephens 2009 with the PSIS prior; Vehtari et al. 2024)."""
    n = len(x)
    m = 30 + int(math.sqrt(n))
    b = 1 - np.sqrt(m / (np.arange(1, m + 1) - 0.5))
    b = b / (3 * x[int(n / 4 + 0.5) - 1]) + 1 / x[-1]
    k = np.log1p(-b[:, None] * x).mean(axis=1)
    ll = n * (np.log(-(b / k)) - k - 1)
    wts = 1 / np.exp(ll - ll[:, None]).sum(axis=1)
    ok = wts >= 10 * np.finfo(float).eps
    wts, b = wts[ok] / wts[ok].sum(), b[ok]
    k_post = np.log1p(-(b * wts).sum() * x).mean()
    return (n * k_post + 5) / (n + 10)


def khat(w):
    S = len(w)
    M = int(math.ceil(min(0.2 * S, 3 * math.sqrt(S))))
    top = np.sort(np.partition(w, S - M - 1)[S - M - 1:])
    x = top[1:] - top[0]
    x = x[x > 0]
    return float(gpdfit(x)) if len(x) >= 5 else 0.0


def reweight(C, logmix_hit, N, LV, block=16):
    """Balance-heuristic estimate, ESS fraction and PSIS k-hat for every column of LV (log source laws)."""
    mu, ess, kh = [], [], []
    for c in range(0, LV.shape[1], block):
        lw = np.asarray(C @ LV[:, c:c + block]) - logmix_hit[:, None]
        m = lw.max(0)
        w = np.exp(lw - m)
        mu.append(np.exp(m) * w.sum(0) / N)
        ess.append(w.sum(0) ** 2 / (w ** 2).sum(0) / N)
        kh.append([khat(w[:, j]) for j in range(w.shape[1])])
    return np.concatenate(mu), np.concatenate(ess), np.concatenate(kh)


def twist_tables(P, eta):
    """log of the product-form exponential twist q_k ~ p_k exp(eta * loss increment), concatenated over sources."""
    out = []
    for s in P.src:
        ell = sum(P.a_u * inc for _, inc in s["da"]) + sum(P.r_u * inc for _, inc in s["dr"])
        lg = s["logp"] + eta * ell
        out.append(lg - logsumexp(lg))
    return np.concatenate(out)


def twist_job(args):
    """Exponential twisting at an anchor: keep the hits' increments, the path count and the twist's log law."""
    th, T, seed, N = args
    t0 = time.process_time()
    rng = np.random.default_rng(seed)
    P = PortfolioProblem(ap_risk.model_of(ap_risk.CAL, th), "exceedance", threshold=T * 1e6)
    eta = _best_twist_sources(P)
    S = TiltSampler(P); S.set_theta(S.twist_theta(eta))
    acts = []
    for _ in range(N // 65536):
        o = S.sample(65536, rng, record=True)
        acts.append(o["acts"][:, o["f"] > 0].T.astype(np.int16))
    return {"acts": np.concatenate(acts), "N": N, "eta": eta, "logq": twist_tables(P, eta), "cpu": time.process_time() - t0}


def twist_ref_job(th):
    """What twisting from scratch costs at this point: per-path variance and seconds, and the fixed set-up time."""
    out = []
    for T in THRESH:
        t0 = time.perf_counter()
        P = PortfolioProblem(ap_risk.model_of(ap_risk.CAL, th), "exceedance", threshold=T * 1e6)
        x = tilt_estimate(P, np.random.default_rng(7), paths=65536, fit=False)
        out.append({"rho2": float(x["rho2"]), "tau": float(x["tau"]), "setup": max(0.0, time.perf_counter() - t0 - x["tau"] * x["paths_used"]), "estimate": float(x["estimate"])})
    return out


def design(inp, counts):
    """Fixed anchors (2016 posterior), then per year: posterior draws and a Sobol design. One flat list of points."""
    rng = np.random.default_rng(SEED)
    anc, _ = ap_risk.anchors(inp, rng)
    pts, index = [], {}
    for y in range(Y_CAL + 1, Y1 + 1):
        st = state(inp, counts, y)
        post = ap_risk.draws(st, rng, N_DRAW)
        A, B = ap_risk.draws(st, rng, N_SOB), ap_risk.draws(st, rng, N_SOB)
        sob = A + B + [dict(a, **{k: b[k] for k in ks}) for ks in GROUPS.values() for a, b in zip(A, B)]
        index[y] = (len(pts), len(pts) + N_DRAW, len(pts) + N_DRAW + len(sob))
        pts += post + sob
    return anc, pts, index


def run():
    from multiprocessing import Pool
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    inp = ap_risk.inputs()
    counts = json.loads((OUT / "twin_counts.json").read_text())
    anc, pts, index = design(inp, counts)
    chk = [index[y][0] + j for y in CHECK_YEARS for j in range(N_CHK)]
    friend = ap_risk.observed(inp)
    slim = {k: v for k, v in inp.items() if k != "model"}
    cpuA = {"train_sample": 0.0, "replay": 0.0, "reweight": 0.0}; cpuB = {"sample": 0.0, "reweight": 0.0}
    est = {"A": {}, "B": {}}; info = {}
    with Pool(PROCS, initializer=ap_risk._init, initargs=(slim,), maxtasksperchild=4) as pool:
        exact = pool.map(ap_risk.exact_job, [(pts[i],) for i in chk])
        tref = pool.map(twist_ref_job, [pts[i] for i in chk])
        je = np.array(pool.map(ap_risk.je_job, [pts[i] for y in index for i in range(index[y][0], index[y][1])] + [friend], chunksize=16))
        lv = pool.map(ap_risk.logvec_job, pts + [friend], chunksize=16)
        LV = np.stack([x[0] for x in lv], 1); cpu_laws = float(sum(x[1] for x in lv))
        sizes = [len(s["p"]) for s in PortfolioProblem(ap_risk.model_of(inp, anc[0]), "exceedance", threshold=THRESH[0] * 1e6).src]
        for T in THRESH:
            runs = pool.map(ap_risk.anchor_job, [(th, T, SEED + 1000 * T + j) for j, th in enumerate(anc)])
            cpuA["train_sample"] += sum(r["cpu"] for r in runs)
            acts = np.concatenate([r["acts"] for r in runs]); f = np.concatenate([r["f"] for r in runs])
            reps = pool.map(ap_risk.replay_job, [(th, T, r["theta"], acts) for th, r in zip(anc, runs)])
            cpuA["replay"] += sum(x[1] for x in reps)
            hit = np.flatnonzero(f > 0)
            logmix = logsumexp(np.stack([x[0] for x in reps], 1)[hit], axis=1) - math.log(len(anc))
            t0 = time.process_time()
            est["A"][T] = reweight(ap_risk.onehot(acts[hit], sizes), logmix, len(f), LV)
            cpuA["reweight"] += time.process_time() - t0
            tw = pool.map(twist_job, [(th, T, SEED + 5000 * T + j, TW_MULT * ap_risk.N_ANCHOR) for j, th in enumerate(anc)])
            cpuB["sample"] += sum(r["cpu"] for r in tw)
            t0 = time.process_time()
            C = ap_risk.onehot(np.concatenate([r["acts"] for r in tw]), sizes)
            logmix = logsumexp(np.asarray(C @ np.stack([r["logq"] for r in tw], 1)), axis=1) - math.log(len(anc))
            est["B"][T] = reweight(C, logmix, sum(r["N"] for r in tw), LV)
            cpuB["reweight"] += time.process_time() - t0
            info[T] = {"A_paths": int(len(f)), "A_hits": int(len(hit)), "B_paths": int(sum(r["N"] for r in tw)), "B_hits": int(C.shape[0]), "B_eta": [r["eta"] for r in tw]}
            print(f"T={T}: engines built, A hits {len(hit)}, B hits {C.shape[0]}", flush=True)
    ex = np.array([x[0] for x in exact])
    tot = {"A": sum(cpuA.values()) + cpu_laws, "B": sum(cpuB.values()) + cpu_laws}
    res = {"years": [Y_CAL + 1, Y1], "draws_per_year": N_DRAW, "sobol_rows": N_SOB, "check_years": list(CHECK_YEARS), "thresholds_musd": list(THRESH),
           "cpu_seconds": {"A": {**cpuA, "laws": cpu_laws, "total": tot["A"]}, "B": {**cpuB, "laws": cpu_laws, "total": tot["B"]}}, "paths": {str(T): v for T, v in info.items()},
           "accuracy": {}, "khat": {}, "yearly": {}, "decision": {}}
    post_cols = np.concatenate([np.arange(index[y][0], index[y][1]) for y in index])
    n_q, cpuC, ok1, ok3, ok4 = len(post_cols), 0.0, True, True, True
    cpuC_B, okB1, okB4 = 0.0, True, True
    for k, T in enumerate(THRESH):
        relA = np.abs(est["A"][T][0][chk] / ex[:, k] - 1); relB = np.abs(est["B"][T][0][chk] / ex[:, k] - 1)
        by_year = {str(y): float(np.median(relA[j * N_CHK:(j + 1) * N_CHK])) for j, y in enumerate(CHECK_YEARS)}
        rmsA, rmsB = float(np.sqrt(np.mean(relA ** 2))), float(np.sqrt(np.mean(relB ** 2)))
        rmsB_eq = rmsB * math.sqrt(tot["B"] / tot["A"])
        costC = float(np.median([r[k]["rho2"] / rmsA ** 2 * r[k]["tau"] + r[k]["setup"] for r in tref]))
        cpuC += n_q * costC
        kh = est["A"][T][2][post_cols]
        res["accuracy"][str(T)] = {"A_median_rel_err_by_year": by_year, "A_rms": rmsA, "A_max": float(relA.max()), "B_rms": rmsB, "B_max": float(relB.max()),
                                   "B_rms_at_A_cpu": rmsB_eq, "A_over_B_equal_cpu": rmsA / rmsB_eq,
                                   "twist_from_scratch": {"rho2_median": float(np.median([r[k]["rho2"] for r in tref])), "tau_median": float(np.median([r[k]["tau"] for r in tref])),
                                                          "setup_median": float(np.median([r[k]["setup"] for r in tref])), "seconds_per_query_at_A_error": costC}}
        by_year_B = {str(y): float(np.median(relB[j * N_CHK:(j + 1) * N_CHK])) for j, y in enumerate(CHECK_YEARS)}
        costC_B = float(np.median([r[k]["rho2"] / rmsB ** 2 * r[k]["tau"] + r[k]["setup"] for r in tref]))
        cpuC_B += n_q * costC_B
        res["accuracy"][str(T)].update(B_median_rel_err_by_year=by_year_B, twist_seconds_per_query_at_B_error=costC_B)
        okB1 &= all(v <= (0.02 if T == THRESH[0] else 0.05) for v in by_year_B.values())
        okB4 &= float(np.mean(est["B"][T][2][post_cols] < 0.7)) >= 0.90
        res["khat"][str(T)] = {"A_share_below_0.7": float(np.mean(kh < 0.7)), "A_median": float(np.median(kh)), "B_share_below_0.7": float(np.mean(est["B"][T][2][post_cols] < 0.7)),
                               "A_ess_median": float(np.median(est["A"][T][1][post_cols])), "B_ess_median": float(np.median(est["B"][T][1][post_cols]))}
        ok1 &= all(v <= (0.02 if T == THRESH[0] else 0.05) for v in by_year.values())
        ok3 &= rmsA <= 0.7 * rmsB_eq
        ok4 &= float(np.mean(kh < 0.7)) >= 0.90
    ok2 = tot["A"] <= cpuC / 5
    res["cpu_seconds"]["C_twist_from_scratch"] = cpuC; res["cpu_seconds"]["queries"] = int(n_q * len(THRESH)); res["cpu_seconds"]["A_over_C"] = tot["A"] / cpuC
    res["cpu_seconds"]["B_over_C"] = tot["B"] / cpuC
    okB2 = tot["B"] <= cpuC_B / 5
    res["cpu_seconds"]["C_at_B_error"] = cpuC_B; res["cpu_seconds"]["B_over_C_at_B_error"] = tot["B"] / cpuC_B
    E = "A" if ok1 and ok2 and ok3 and ok4 else "B"            # engine behind the yearly output
    res["engine_for_yearly_output"] = E
    rng2 = np.random.default_rng(SEED + 3)
    fr = {T: float(est[E][T][0][-1]) for T in THRESH}
    for y in index:
        a, b, c = index[y]; st = state(inp, counts, y)
        row = {"country_hits": float(st["K"].sum()), "shared": float(st["Kp"].sum())}
        dec = {}
        for T in THRESH:
            mu = est[E][T][0][a:b]
            row[f"exc_{T}"] = {"posterior_predictive": float(mu.mean()), "range95": [float(np.quantile(mu, 0.025)), float(np.quantile(mu, 0.975))], "P_above_friend_2016": float(np.mean(mu > fr[T]))}
            ys = np.log10(np.maximum(est[E][T][0][b:c], 1e-300)); k = N_SOB
            sob = voi.sobol(ys[:k], ys[k:2 * k], {g: ys[2 * k + m * k: 2 * k + (m + 1) * k] for m, g in enumerate(GROUPS)}, rng2)
            n_rec = y - Y0 + 1
            dec[f"exc_{T}"] = {"S": sob["S"], "T": sob["T"], "removed_by_10_more_years": {g: float(sob["S"][g] * 10 / (n_rec + 10)) for g in ("country_counts", "pair_counts")}}
        j0 = (y - Y_CAL - 1) * N_DRAW
        jp = je[j0:j0 + N_DRAW]
        row["joint_exhaustion_KG_TJ"] = {"posterior_predictive": float(jp.mean()), "range95": [float(np.quantile(jp, 0.025)), float(np.quantile(jp, 0.975))], "friend_2016": float(je[-1])}
        res["yearly"][str(y)] = row; res["decision"][str(y)] = dec
    res["friend_2016"] = {"exact_refs": inp["refs"]["refs"], "engine_at_observed": fr}
    res["checks_twisting_twin"] = {"accuracy_no_drift": bool(okB1), "cpu_fifth_of_twisting": bool(okB2), "khat": bool(okB4)}
    res["verdict_twisting_twin"] = "strong" if okB1 and okB2 and okB4 else "not strong"
    res["checks"] = {"accuracy_no_drift": bool(ok1), "cpu_fifth_of_twisting": bool(ok2), "rdpo_beats_twist_anchors": bool(ok3), "khat": bool(ok4)}
    res["verdict"] = "strong" if ok1 and ok2 and ok3 and ok4 else ("twin works without RDPO" if ok1 and ok2 and ok4 else "not strong")
    save("twin.json", res)
    for T in THRESH:
        a = res["accuracy"][str(T)]; kq = res["khat"][str(T)]
        print(f"T={T}: A median rel err by year {a['A_median_rel_err_by_year']} | RMS A {a['A_rms']:.4f} (max {a['A_max']:.4f}), B {a['B_rms']:.4f} -> at A's CPU {a['B_rms_at_A_cpu']:.4f}, A/B {a['A_over_B_equal_cpu']:.2f}"
              f" | twisting from scratch: rho2 {a['twist_from_scratch']['rho2_median']:.1f}, {a['twist_from_scratch']['seconds_per_query_at_A_error']:.2f}s per query at A's error"
              f" | k-hat<0.7: A {kq['A_share_below_0.7']:.2f} B {kq['B_share_below_0.7']:.2f}; ESS A {kq['A_ess_median']:.3f} B {kq['B_ess_median']:.3f}")
    c = res["cpu_seconds"]
    print(f"CPU: A {c['A']['total']:.0f}s { {k: round(v) for k, v in c['A'].items()} } | B {c['B']['total']:.0f}s | C {c['C_twist_from_scratch']:.0f}s for {c['queries']} queries -> A/C {c['A_over_C']:.3f}, B/C {c['B_over_C']:.3f}")
    for y, r in res["yearly"].items():
        d = res["decision"][y]
        print(f"{y}: hits {r['country_hits']:.0f}, shared {r['shared']:.0f} | " + " | ".join(f">{T}m {r[f'exc_{T}']['posterior_predictive']:.3g} [{r[f'exc_{T}']['range95'][0]:.3g}, {r[f'exc_{T}']['range95'][1]:.3g}]" for T in THRESH)
              + f" | JE {r['joint_exhaustion_KG_TJ']['posterior_predictive']:.3g} | S(450) " + ", ".join(f"{g} {v:.2f}" for g, v in d[f'exc_{THRESH[0]}']['S'].items()))
    print("friend 2016:", {k: f"{v:.4g}" for k, v in inp["refs"]["refs"].items() if "exceed" in k or "joint" in k}, f"| engine {E} at observed", fr)
    print("B median rel err by year:", {str(T): res["accuracy"][str(T)]["B_median_rel_err_by_year"] for T in THRESH}, f"| B/C at B's error {res['cpu_seconds']['B_over_C_at_B_error']:.3f}")
    print("checks:", res["checks"], "| verdict:", res["verdict"])
    print("twisting twin:", res["checks_twisting_twin"], "| verdict:", res["verdict_twisting_twin"])
    return res


def selftest():
    inp = ap_risk.inputs()
    ev, polys = friend_events()
    y = yearly(ev, polys, inp["order"], inp["edges"])
    assert np.array_equal(y["K"].sum(0), inp["K"]) and np.array_equal(y["Kp"].sum(0), inp["Kp"]) and not y["new_pairs"]
    counts = {"K_yearly": y["K"].tolist(), "Kp_yearly": y["Kp"].tolist()}
    st = state(inp, counts, Y_CAL)                              # assimilating nothing leaves the friend's state
    assert np.array_equal(st["K"], inp["K"]) and np.array_equal(st["Kp"], inp["Kp"])
    core_checks(PortfolioProblem(inp["model"], "exceedance", threshold=THRESH[0] * 1e6))

    print(f"selftest ok: yearly counts on the friend's catalogue sum to his calibration ({int(inp['K'].sum())} country hits, {int(inp['Kp'].sum())} shared, {y['main_shocks']} main shocks); an empty update keeps his state; twist log law = sampler weights; one anchor = plain IS; k-hat recovers Pareto shapes")


def core_checks(P):
    """The tabulated twist law reproduces the sampler's own weights; one anchor at the target is plain importance
    sampling; k-hat recovers known Pareto shapes."""
    eta = _best_twist_sources(P)
    S = TiltSampler(P); S.set_theta(S.twist_theta(eta))
    o = S.sample(20000, np.random.default_rng(1), record=True)
    sizes = [len(q["p"]) for q in P.src]
    C = ap_risk.onehot(o["acts"].T.astype(np.int64), sizes)
    lp = np.concatenate([q["logp"] for q in P.src])
    logp, lq = np.asarray(C @ lp).ravel(), np.asarray(C @ twist_tables(P, eta)).ravel()
    assert np.allclose(logp - lq, np.log(o["w"]), atol=1e-8)
    hit = np.flatnonzero(o["f"] > 0)
    mu, _, _ = reweight(C[hit], lq[hit], len(logp), lp[:, None])
    assert abs(mu[0] / np.mean(o["w"] * o["f"]) - 1) < 1e-9
    assert abs(khat(np.random.default_rng(0).pareto(2.0, 200000)) - 0.5) < 0.1 and khat(np.random.default_rng(0).pareto(1.0, 200000)) > 0.8


# ------------------------------------------------------------------------------------------- backtest
def predictive(k, n):
    """Next-year count when the annual rate has the Gamma(k + 1/2, n years) posterior: negative binomial."""
    from scipy.stats import nbinom
    return nbinom(k + 0.5, n / (n + 1.0))


def forecast_scores(k0, n0, Y, update=True):
    """One-step-ahead forecasts of the yearly counts Y (years x series), starting from k0 events in n0 years.
    Returns log scores, the expected coverage of the randomised 90% interval (randomised PIT, Czado, Gneiting & Held
    2009: the standard for counts) and the raw coverage of the integer interval [q05, q95]."""
    logs, cov, raw, z2 = [], [], [], []
    k, n = np.asarray(k0, float).copy(), float(n0)
    for t in range(Y.shape[0]):
        d = predictive(k, n)
        obs = Y[t]
        logs.append(d.logpmf(obs))
        lo, hi = d.cdf(obs - 1), d.cdf(obs)
        cov.append(np.clip(np.minimum(hi, 0.95) - np.maximum(lo, 0.05), 0, None) / np.maximum(hi - lo, 1e-300))
        raw.append((d.ppf(0.05) <= obs) & (obs <= d.ppf(0.95)))
        z2.append((obs - d.mean()) ** 2 / d.var())
        if update:
            k, n = k + obs, n + 1
    f = lambda a: np.concatenate([np.ravel(x) for x in a])
    return {"log_score": float(f(logs).sum()), "coverage_randomised": float(f(cov).mean()), "coverage_raw": float(f(raw).mean()),
            "dispersion": float(f(z2).mean()), "n": int(f(logs).size)}


def backtest():
    """Does the twin predict next year's counts? Rules (set before running): (1) the randomised 90% interval covers
    85-95% of the 2017-2025 forecasts, for country counts and for pair counts; (2) the yearly-updated twin's mean log
    score is not worse than the twin frozen in 2016 (difference per forecast >= -0.005 nats)."""
    c = json.loads((OUT / "twin_counts.json").read_text())
    K, Kp = np.asarray(c["K_yearly"]), np.asarray(c["Kp_yearly"])
    r0 = Y_CAL - Y0 + 1
    fk = np.array([x["friend"] for x in c["check_1965_2016"]["countries"]]); fp = np.array([x["friend"] for x in c["check_1965_2016"]["pairs"]])
    res = {"forecast_years": [Y_CAL + 1, Y1], "series": {}}
    ok1, ok2 = True, True
    for name, k0, Y in (("country_counts", fk, K[r0:]), ("pair_counts", fp, Kp[r0:])):
        up, fz = forecast_scores(k0, r0, Y, True), forecast_scores(k0, r0, Y, False)
        diff = (up["log_score"] - fz["log_score"]) / up["n"]
        res["series"][name] = {"updated": up, "frozen_2016": fz, "log_score_gain_per_forecast": diff}
        ok1 &= 0.85 <= up["coverage_randomised"] <= 0.95
        ok2 &= diff >= -0.005
    # supplementary, more forecasts: ComCat alone, trained on 1965-1999, forecasting 2000-2025
    cut = 2000 - Y0
    res["supplement_comcat_2000_2025"] = {"country_counts": forecast_scores(K[:cut].sum(0), cut, K[cut:]), "pair_counts": forecast_scores(Kp[:cut].sum(0), cut, Kp[cut:])}
    res["checks"] = {"coverage_85_95": bool(ok1), "updating_not_worse": bool(ok2)}
    res["verdict"] = "strong" if ok1 and ok2 else "not strong"
    save("twin_backtest.json", res)
    for name, r in res["series"].items():
        u = r["updated"]
        print(f"{name}: {u['n']} forecasts 2017-2025 | randomised 90% coverage {u['coverage_randomised']:.3f} (raw integer interval {u['coverage_raw']:.3f}) | dispersion {u['dispersion']:.2f}"
              f" | log score updated {u['log_score']:.1f} vs frozen {r['frozen_2016']['log_score']:.1f} (gain {r['log_score_gain_per_forecast']:+.4f} per forecast)")
    for name, u in res["supplement_comcat_2000_2025"].items():
        print(f"supplement {name}: {u['n']} forecasts 2000-2025 | randomised coverage {u['coverage_randomised']:.3f} (raw {u['coverage_raw']:.3f}) | dispersion {u['dispersion']:.2f}")
    print("checks:", res["checks"], "| verdict:", res["verdict"])
    return res


def selftest_fast():
    """The estimator checks on a small synthetic pool (no calibration, no downloads)."""
    from src.tensor.synthetic import SyntheticModel, topology
    core_checks(PortfolioProblem(SyntheticModel(4, topology("ring", 4)), "exceedance"))
    # backtest machinery: forecasts of data simulated from the model itself are calibrated
    rng = np.random.default_rng(5)
    lam = rng.gamma(3.0, 1.0, 200)
    Y = rng.poisson(lam, (60, 200)).astype(float)
    r = forecast_scores(Y[:30].sum(0), 30, Y[30:])
    assert abs(r["coverage_randomised"] - 0.90) < 0.02 and abs(r["dispersion"] - 1.0) < 0.1, r
    print("selftest ok (fast): twist log law = sampler weights; one anchor = plain IS; k-hat recovers Pareto shapes; forecasts of simulated Poisson counts are calibrated")


if __name__ == "__main__":
    selftest()
    if not (OUT / "twin_counts.json").exists():
        phase0()
    backtest()
    run()
