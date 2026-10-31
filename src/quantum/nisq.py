"""Near-term (NISQ) feasibility: gate counts on hardware, noisy samplers, noisy low-depth QAE.

(1) nisq_counts: CNOT counts / depth of the structured A operator and Grover iterate, and a
    hardware-modality mapping (all-to-all ion/neutral-atom vs 2D superconducting with SWAPs).
(2) Noisy QCBM proposal sampler (8 qubits): global depolarising noise per entangling layer
    plus independent readout bit flips. Importance weights need q(y); if the *ideal* q is used
    while samples come from the noisy q', the estimator is biased. Mitigation: estimate q̂
    from calibration shots of the *sampled* (post-readout) law, smoothed, and use q̂ in the weights.
    (Readout-error inversion is the wrong mitigation here: it targets the pre-readout law.)
(3) Noisy maximum-likelihood amplitude estimation (MLAE, Suzuki et al. 2020) with a
    depolarising decay model P_k = ½ − ½ f^{2k+1} cos((4k+2)θ), and noise-aware MLE that fits
    f jointly with θ (Tanaka et al. 2021): error mitigation via the likelihood.
"""
import math

import numpy as np
from scipy.optimize import minimize_scalar, minimize

from . import structured_loading as SL


# ---------------------------------------------------------------------------
# (1) hardware-level counts
# ---------------------------------------------------------------------------
def nisq_counts(model):
    A = SL.a_operator_resources(model)
    cx_tof = 6                                     # standard Toffoli = 6 CNOT (+ 9 single-qubit)
    cnot_A = A["cnot_prep"] + cx_tof * A["toffoli"]
    # Grover iterate Q = A S0 A† Sχ ; reflections ~ multi-controlled Z on all qubits: ~2(Q-2) Toffolis
    refl_tof = 2 * (A["logical_qubits"] - 2)
    cnot_Q = 2 * cnot_A + 2 * cx_tof * refl_tof
    depth_A = 6 * A["toffoli_depth_est"] + 62     # deepest state-prep (6-qubit Möttönen) + arithmetic
    depth_Q = 2 * depth_A + 2 * 6 * int(2 * math.log2(A["logical_qubits"]) + 1)
    # superconducting 2D grid: each non-local CNOT needs ~sqrt(Q) SWAPs (3 CNOT each) in the worst case
    swap_factor_grid = 1 + 3 * math.sqrt(A["logical_qubits"]) * 0.5   # assume half of CNOTs non-local
    return {**A, "cnot_A": cnot_A, "cnot_Q": cnot_Q, "depth_A_cnot_layers": depth_A,
            "depth_Q_cnot_layers": depth_Q,
            "modality": {
                "trapped_ion_all_to_all": {"cnot_Q": cnot_Q, "note": "native all-to-all; dense dependence graphs map directly"},
                "neutral_atom_reconfigurable": {"cnot_Q": cnot_Q, "note": "atom shuttling gives effective all-to-all; parallel Rydberg gates suit source-parallel prep"},
                "superconducting_2d": {"cnot_Q": int(cnot_Q * swap_factor_grid),
                                        "note": "SWAP overhead; chain/ring graphs map to a line, complete graphs do not"}}}


# ---------------------------------------------------------------------------
# (2) noisy QCBM sampler and IS bias
# ---------------------------------------------------------------------------
def noisy_qcbm_probs(qcbm, p2=1e-3, p_ro=0.01):
    """Global depolarising per CZ-ring layer (8 CZ -> ε = 1-(1-p2)^8 per layer) + readout flips."""
    from .qcbm import NQ, DIM
    P = qcbm.probs_full()
    eps = 1 - (1 - p2) ** (NQ * qcbm.L)
    P = (1 - eps) * P + eps / DIM
    for q in range(NQ):                      # independent readout bit flips on each qubit
        st = P.reshape(2 ** q, 2, -1)
        a0, a1 = st[:, 0, :].copy(), st[:, 1, :].copy()
        st[:, 0, :] = (1 - p_ro) * a0 + p_ro * a1
        st[:, 1, :] = p_ro * a0 + (1 - p_ro) * a1
    return P


def readout_inverse(P, p_ro):
    from .qcbm import NQ
    M = np.linalg.inv(np.array([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    P = P.copy()
    for q in range(NQ):
        st = P.reshape(2 ** q, 2, -1)
        a0, a1 = st[:, 0, :].copy(), st[:, 1, :].copy()
        st[:, 0, :] = M[0, 0] * a0 + M[0, 1] * a1
        st[:, 1, :] = M[1, 0] * a0 + M[1, 1] * a1
    return np.maximum(P, 0)


def is_bias_study(prob, f, qcbm, alpha, p2, p_ro, calib_shots, rng):
    """Exact bias/variance of iid IS when samples come from the noisy QCBM.
    Returns relative bias with (i) ideal q in the weights, (ii) calibrated q̂ (shots + readout inversion)."""
    from .qcbm import BIT, iid_is_variance
    p = prob.p
    q_ideal, _ = qcbm.model_probs()
    Pn = noisy_qcbm_probs(qcbm, p2, p_ro)
    v = Pn[BIT]; q_noisy = v / v.sum()                       # post-selected noisy proposal
    mix = lambda q: (1 - alpha) * q + alpha * p
    mu = float(__import__("src.model.term", fromlist=["term_distribution"]).term_distribution(p) @ f)

    def exact_mean(q_true, q_used):
        # E_{q_true^3}[ Π p/q_used · f ] via capped convolution with r = q_true * p / q_used
        from src.model.term import ADD, YEARS
        r = q_true * p / q_used
        d = np.zeros(100); d[0] = 1.0
        for _ in range(YEARS):
            nxt = np.zeros(100); np.add.at(nxt, ADD, d[:, None] * r[None, :]); d = nxt
        return float(d @ f)

    qi, qn = mix(q_ideal), mix(q_noisy)
    bias_ideal = exact_mean(qn, qi) / mu - 1
    # Calibrate the law that is ACTUALLY sampled (post-readout, post-selected), with add-1/2
    # smoothing. Do NOT invert readout errors: the IS samples themselves carry readout noise,
    # so inverting would make q_used != q_sampled and re-introduce bias.
    counts = rng.multinomial(calib_shots, Pn / Pn.sum())
    vh = counts[BIT] + 0.5; q_hat = mix(vh / vh.sum())
    bias_cal = exact_mean(qn, q_hat) / mu - 1
    var_cal, _ = iid_is_variance(p, f, qn)                   # variance scale (approx., using q_noisy)
    return {"p2": p2, "p_ro": p_ro, "calib_shots": calib_shots,
            "tv_distance_noisy_vs_ideal": float(0.5 * np.abs(q_noisy - q_ideal).sum()),
            "rel_bias_ideal_weights": bias_ideal, "rel_bias_calibrated_weights": bias_cal,
            "rel_std_per_path_noisy": float(math.sqrt(max(var_cal, 0)) / mu)}


# ---------------------------------------------------------------------------
# (3) noisy MLAE with noise-aware likelihood
# ---------------------------------------------------------------------------
def mlae_noisy(a, f_iter, ks, shots, rng, fit_noise=True, f_known=None):
    """Measure P_k = ½ − ½ f^{2k+1} cos((4k+2)θ) for each k; estimate a = sin²θ by MLE.
    fit_noise: jointly fit f (noise-aware MLE, mitigation). Else assume f = f_known (1 = ignore noise)."""
    th = math.asin(math.sqrt(a))
    ks = np.asarray(ks)
    Pk = 0.5 - 0.5 * f_iter ** (2 * ks + 1) * np.cos((4 * ks + 2) * th)
    h = rng.binomial(shots, Pk)

    def nll(theta, ff):
        P = np.clip(0.5 - 0.5 * ff ** (2 * ks + 1) * np.cos((4 * ks + 2) * theta), 1e-12, 1 - 1e-12)
        return -np.sum(h * np.log(P) + (shots - h) * np.log(1 - P))

    # coarse grid over θ then refine (the likelihood is multimodal for large k)
    grid = np.linspace(1e-6, math.pi / 2 - 1e-6, 20000)
    if fit_noise:
        best = (np.inf, None)
        for ff in np.linspace(0.05, 1.0, 40):
            vals = [nll(t, ff) for t in grid[::20]]
            i = int(np.argmin(vals))
            if vals[i] < best[0]:
                best = (vals[i], (grid[::20][i], ff))
        r = minimize(lambda z: nll(z[0], z[1]), best[1], bounds=[(1e-7, math.pi / 2), (0.01, 1.0)], method="L-BFGS-B")
        t_hat = r.x[0]
    else:
        ff = 1.0 if f_known is None else f_known
        vals = np.array([nll(t, ff) for t in grid])
        i = int(np.argmin(vals))
        r = minimize_scalar(lambda t: nll(t, ff), bounds=(grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]), method="bounded")
        t_hat = r.x
    queries = int(np.sum(shots * (2 * ks + 1)))
    return math.sin(t_hat) ** 2, queries


def mlae_study(a, cnot_Q, p2_list, shots=100, reps=40, seed=0):
    """Relative RMSE of noisy MLAE vs queries, for several two-qubit error rates.
    Schedule k = 0,1,2,4,…, truncated where the circuit fidelity f^{2k+1} drops below 0.05."""
    rng = np.random.default_rng(seed)
    out = []
    for p2 in p2_list:
        f_iter = (1 - p2) ** cnot_Q
        kmax = 0
        while f_iter ** (2 * (2 * kmax if kmax else 1) + 1) > 0.05 and kmax < 2 ** 12:
            kmax = 2 * kmax if kmax else 1
        ks = [0] + [2 ** j for j in range(int(math.log2(kmax)) + 1)] if kmax >= 1 else [0]
        for mode in ["noise_aware", "naive"]:
            est, q = [], 0
            for _ in range(reps):
                e, q = mlae_noisy(a, f_iter, ks, shots, rng, fit_noise=(mode == "noise_aware"), f_known=1.0)
                est.append(e)
            est = np.array(est)
            mc_rel = math.sqrt((1 - a) / (a * q))
            out.append({"p2": p2, "f_per_Q": f_iter, "k_schedule": ks, "mode": mode, "queries": q,
                        "rel_rmse": float(np.sqrt(np.mean((est - a) ** 2)) / a),
                        "rel_bias": float((est.mean() - a) / a), "mc_rel_rmse_same_queries": mc_rel})
    return out
