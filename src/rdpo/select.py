"""Work-normalised selector: choose the estimator that is predicted to be most accurate *at the wall-clock budget*.

At equal path budgets the learned sampler wins; at equal time what matters is the work-normalised variance
ρ²·τ (ρ² = per-path relative variance of the weighted payoff, τ = seconds per path) plus whatever training still has
to be paid. For a budget S the predicted relative variance of a candidate is

        err²  =  ρ² / n_eff,        n_eff = (S − pilots − training) / τ.

Candidates and how they are piloted (every pilot second is charged to S):
  twist        pilot batch of the fast product-form sampler at θ = (η*, 0): gives ρ²_tw and τ_tw.
  tilt         one source-tilt fit on a recorded batch, then a fresh test batch from the fitted measure; the fit is kept
               only if its estimated ρ² beats the twist's by a margin (and its cost is charged).
  rdpo (cold)  pilot rollout of the warm-started policy (τ_rdpo measured; ρ² of the *untrained* policy). The effect of
               training is not observable in a pilot, so it is predicted as ρ²_init / g with g = 10 (the median
               twist-to-RDPO variance ratio of results/rdpo_v2_scale.json is 17, the minimum 7), and the training cost is
               4·τ_rdpo per path (profile of fast.py: a full fit costs ≈ 3.7× a rollout of the same number of paths).
  rdpo (transfer) same pilot with the supplied θ; its ρ² is observed directly (no prediction factor) and training is
               free (amortised), only the estimation paths are paid.
A candidate that cannot win even if its training were free and ρ² were g_max = 30 times smaller than the twist's is
not piloted (prior τ_rdpo = 0.5 µs · K per path).

Unbiasedness. All pilot samples are DISCARDED. The chosen estimator then runs on the remaining time with a measure that
is fixed before its samples are drawn (θ taken from the pilot fit is a function of pilot data only), or with
`rdpo_estimate`, which is unbiased by construction. The only data-dependent quantity in the final stage is the number of
paths, which depends on timing and on the (independent) pilot, not on the final samples.
"""
import time

import numpy as np

from .problem import PortfolioProblem
from .fast import FastPolicy
from .tilt import TiltSampler, tilt_estimate, _stats, _ci, _pack
from .train import rdpo_estimate, v2_config, _warm_start, _best_twist_sources

G_COLD, G_MAX, TRAIN_FACTOR = 10.0, 30.0, 4.0


def _rdpo_cfg(P, theta0=None):
    from .experiments_v2 import final_config
    from dataclasses import replace
    cfg = final_config(P.target)
    return replace(cfg, theta0=theta0) if theta0 is not None else cfg


def _rho2(g):
    m = g.mean()
    return float(g.var(ddof=1) / m ** 2) if m > 0 and len(g) > 1 else float("inf")


def select_estimate(prob: PortfolioProblem, seconds, rng, theta_transfer=None, b_pilot=4096, rdpo_calibration=None):
    """Return the estimate of the chosen candidate within `seconds` of wall-clock (everything charged).
    rdpo_calibration = (t0, c): fixed overhead and seconds per path of a full RDPO fit (from a calibration run); when
    missing, c = TRAIN_FACTOR·τ_rdpo (profile-based)."""
    t0 = time.perf_counter()
    P = prob
    left = lambda: seconds - (time.perf_counter() - t0)
    info = {"pilots": {}}
    S = TiltSampler(P, "ell+ind")
    thresholded = P.target in ("exceedance", "stop_loss")
    eta0 = _best_twist_sources(P) if thresholded else 0.0
    th_tw = S.twist_theta(eta0)
    S.set_theta(th_tw)
    # ---- pilot 1: twist ---------------------------------------------------------------------------
    ts = time.perf_counter(); o = S.sample(b_pilot, rng); tau_tw = (time.perf_counter() - ts) / b_pilot
    rho_tw = _rho2(o["w"] * o["f"])
    if not np.isfinite(rho_tw) or rho_tw <= 0:               # twist pilot saw nothing: use a bigger pilot once
        o = S.sample(4 * b_pilot, rng); rho_tw = _rho2(o["w"] * o["f"])
    pred = {"twist": rho_tw * tau_tw / max(left() - 0.0, 1e-9)}
    info["pilots"]["twist"] = {"rho2": rho_tw, "tau": tau_tw}
    # ---- pilot 2: one source-tilt fit + test batch -------------------------------------------------------
    th_fit = None
    if thresholded and left() > 0.5:
        ts = time.perf_counter()
        rec = S.sample(b_pilot * 2, rng, True)
        bnd = np.zeros((S.K, S.m)); bnd[:, 0] = max(abs(eta0), 0.05) + 0.02; bnd[:, 1:] = 1.5
        th_new, fi = S.fit_chi2(rec["acts"], rec["f"], rec["w"], th_tw, 1e-3, bnd, maxiter=30)
        if fi.get("ok"):
            S.set_theta(th_tw + 0.8 * (th_new - th_tw))
            o2 = S.sample(b_pilot * 2, rng); rho_t = _rho2(o2["w"] * o2["f"])
            cost = time.perf_counter() - ts
            info["pilots"]["tilt"] = {"rho2": rho_t, "cost_s": cost}
            n_eff_t = max(left(), 1e-9) / tau_tw
            if np.isfinite(rho_t) and rho_t < 0.9 * rho_tw:
                pred["tilt"] = rho_t / n_eff_t
                th_fit = S.theta.copy()
            S.set_theta(th_tw)
    # ---- pilots 3/4: RDPO (cold / transferred) -----------------------------------------------------------
    best_twist_err = min(pred.values()) if pred else np.inf
    tau_prior = 0.5e-6 * P.K
    for name, th_in in (("rdpo_transfer", theta_transfer), ("rdpo_cold", None)):
        if name == "rdpo_transfer" and theta_transfer is None:
            continue
        free_train = name == "rdpo_transfer"
        if free_train:   # best case: ρ² can at best be G_MAX below the twist's
            best_case = (rho_tw / G_MAX) * tau_prior / max(left(), 1e-9)
        else:
            best_case = (rho_tw / G_MAX) * (TRAIN_FACTOR * tau_prior) / max(left(), 1e-9)
        if best_case > 0.8 * best_twist_err:
            info["pilots"][name] = {"skipped": "cannot win (prior)"}
            continue
        cfg = _rdpo_cfg(P, th_in)
        ts = time.perf_counter()
        pol = FastPolicy(P, cfg.alpha, cfg.edge_features, cfg.relevance, value=cfg.value, stages=cfg.stages)
        _warm_start(pol)
        if th_in is not None and len(th_in) == pol.d:
            pol.theta = np.array(th_in, float)
        tb = time.perf_counter()
        r = pol.rollout(2048, rng)
        tau_r = (time.perf_counter() - tb) / 2048
        rho_r = _rho2(r["w"] * r["f"])
        cost = time.perf_counter() - ts
        info["pilots"][name] = {"rho2_init": rho_r, "tau": tau_r, "cost_s": cost}
        rem = left() - 0.05
        if rem <= 0:
            continue
        if free_train:
            n_eff = rem / tau_r
            pred[name] = rho_r / n_eff
        else:
            c = TRAIN_FACTOR * tau_r if rdpo_calibration is None else rdpo_calibration[1]
            pred[name] = (rho_r / G_COLD) / (rem / c)
    choice = min(pred, key=pred.get)
    info["pred_err2"] = pred; info["choice"] = choice
    # ---- final stage: pilots discarded, remaining time spent with the chosen candidate ----------------------
    rem = left() * 0.97
    if choice in ("twist", "tilt"):
        S.set_theta(th_tw if choice == "twist" else th_fit)
        batches = []
        while True:
            rem = (left() * 0.9)
            b_fin = int(np.clip(0.01 * seconds / max(tau_tw, 1e-9), 4096, max(4096, 4_000_000 // S.K)))
            n = int(min(b_fin, rem / max(tau_tw, 1e-9)))
            if n < 256:
                break
            ob = S.sample(n, rng)
            batches.append(_pack(ob["w"] * ob["f"]))
        if not batches:
            o = S.sample(1024, rng); batches = [_pack(o["w"] * o["f"])]
        est, se, dof = _stats(batches)
        out = {"estimate": est, "se": se, "ci": _ci(est, se, dof), "paths_used": int(sum(b[0] for b in batches))}
    else:
        c_rate = rdpo_calibration[1] if rdpo_calibration else TRAIN_FACTOR * info["pilots"][choice]["tau"]
        t_fix = rdpo_calibration[0] if rdpo_calibration else 0.0
        B = int(max(1024, (rem - t_fix) / c_rate))
        cfg = _rdpo_cfg(P, theta_transfer if choice == "rdpo_transfer" else None)
        x = rdpo_estimate(P, B, rng, cfg)
        out = {"estimate": x["estimate"], "se": x["se"], "ci": x["ci"], "paths_used": int(x["paths_used"])}
    out.update({"seconds": time.perf_counter() - t0, "choice": choice, "info": info})
    return out
