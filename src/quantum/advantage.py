"""Quantum stage, part 2: falsifiable advantage criterion and noise/mitigation analysis.

(1) Bitwise exponential oracle. The twisted payoff g = 1{L>T} e^{−η(U−T)} factorises over the bits of
    D = U − T >= 0:  e^{−ηD} = Π_b (e^{−η 2^b})^{d_b}. Instead of a lookup table over all U values,
    one ancilla per bit is rotated by a controlled RY with sin² = e^{−η 2^b}; the good subspace is
    "flag and all bit-ancillas = 1". Cost: build U (2n adders of width w_U), subtract T, w_U
    controlled rotations; no QROM.
(2) Falsifiable wall-clock criterion. For a target relative error ε:
        T_classical(n) = ρ²(n)/ε² · t_path(n)       (measured RDPO: ρ² = relRMSE²·B, t_path incl. training)
        T_quantum(n)   = calls(n) · runtime-per-call (surface-code model of resources.py)
    The required hardware speed-up s*(n) = T_quantum / T_classical is the falsifiable quantity:
    Q-RDPO beats classical RDPO in wall-clock iff the fault-tolerant machine is s* times faster
    than the stated baseline assumptions. Also: the logical two-qubit error rate p_L needed for the
    deepest IQAE circuit to succeed with probability 1/2.
(3) Mitigation under miscalibration. Low-depth MLAE with depolarising decay f and a coherent
    control offset δ per Grover iterate:  P_k = ½ − ½ f^{2k+1} cos((4k+2)θ + 2kδ).
    Estimators: raw (assume f = 1, δ = 0); noise-aware "mitigated" MLE (fit f, assume δ = 0);
    guarded (mitigated, but a likelihood-ratio goodness-of-fit test across depths rejects the
    quantum estimate and falls back to the classical RDPO estimate).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from src.tensor.synthetic import SyntheticModel, topology
from . import structured_loading as SL
from . import twisted as TW
from .nisq import nisq_counts
from .iqae import iqae
from .resources import rotation_T, surface_code_footprint

R = Path("results")


def bitwise_oracle_counts(model, T_units):
    tw_eta, _ = TW.best_twist(model, T_units)
    umax = int(sum(ell.max() for _, ell in TW.source_losses(model)))
    wU = math.ceil(math.log2(umax + 1))
    A = SL.a_operator_resources(model)
    C = nisq_counts(model)
    extra_tof = 2 * (2 * model.n * wU + wU)          # build U, subtract T (compute + uncompute)
    tof = A["toffoli"] + extra_tof
    rot = A["rotations"] + wU
    qubits = A["logical_qubits"] + 2 * wU
    cnot_A = C["cnot_A"] + 6 * extra_tof + 2 * wU
    cnot_Q = 2 * cnot_A + (C["cnot_Q"] - 2 * C["cnot_A"])
    depth_A = C["depth_A_cnot_layers"] + 6 * (2 * wU) + 2 * wU
    return {"eta": tw_eta, "U_max": umax, "w_U": wU, "toffoli_A": tof, "rotations_A": rot, "logical_qubits": qubits,
            "cnot_A": cnot_A, "cnot_Q": cnot_Q, "depth_A": depth_A,
            "lookup_toffoli_avoided": int(2 * (umax + wU))}


def iqae_stats(a, eps_rel, reps=20, seed=0):
    class Counter:
        def __init__(self, s): self.g = np.random.default_rng(s); self.n = 0
        def binomial(self, n, p): self.n += n; return self.g.binomial(n, p)
    calls, kmax, shots = [], [], []
    for r in range(reps):
        c = Counter(seed + r)
        _, cl, km = iqae(a, eps_rel * a, rng=c)
        calls.append(cl); kmax.append(km); shots.append(c.n)
    return float(np.mean(calls)), float(np.mean(kmax)), float(np.mean(shots))


def advantage_map(eps_rel=0.1, e2_rows=None):
    """e2_rows: rows shaped like results/rdpo_e2.json (default: the file); the quantum side is untouched."""
    if e2_rows is None:
        e2_rows = json.loads((R / "rdpo_e2.json").read_text())["rows"]
    e2 = {(r["topology"], r["n"]): r for r in e2_rows if r["target"] == "exceedance"}
    tw = {(r["topology"], r["n"]): r for r in json.loads((R / "twisted.json").read_text())["rows"]}
    rows = []
    for key in sorted(e2, key=lambda k: (k[0] != "ring", k[1])):
        kind, n = key
        model = SyntheticModel(n, topology(kind, n))
        r2, t = e2[key], tw[key]
        B = r2["budget"]
        rho2 = r2["rdpo"]["rel_rmse"] ** 2 * B
        t_path = r2["seconds_per_fit"] / B
        Tc = rho2 / eps_rel ** 2 * t_path
        oc = bitwise_oracle_counts(model, 16 * n)
        calls, kmax, shots = iqae_stats(t["a_twisted"], eps_rel)
        tof_total = calls * (oc["toffoli_A"] + oc["logical_qubits"])
        rot_total = calls * oc["rotations_A"]
        Tcount = 4 * tof_total + rot_total * rotation_T(1e-3 / max(rot_total, 1))
        sc = surface_code_footprint(oc["logical_qubits"], Tcount)
        Tq = sc["runtime_s"]
        pL_req = math.log(2) / ((2 * kmax + 1) * oc["cnot_Q"] / 2 + oc["cnot_A"])
        rows.append({"topology": kind, "n": n, "mu": t["mu"], "a_twisted": t["a_twisted"], "eps_rel": eps_rel,
                     "classical_rho2": rho2, "classical_paths": rho2 / eps_rel ** 2, "classical_seconds": Tc,
                     "quantum_calls": calls, "quantum_kmax": kmax, "quantum_shots": shots, "quantum_T_count": Tcount,
                     "quantum_seconds": Tq, "required_speedup": Tq / Tc, "pL_required": pL_req,
                     "physical_qubits": sc["physical_qubits"], "code_distance": sc["code_distance"], **oc})
        print(f"{kind:8s} n={n:2d} classical {Tc:9.3g}s  quantum {Tq:9.3g}s  s*={Tq / Tc:9.3g}  pL_req={pL_req:.1e}  "
              f"kmax={kmax:.0f} qubits={oc['logical_qubits']}", file=sys.stderr, flush=True)
    return rows


# ------------------------------------------------------------------------------------------------
def _nll(theta, f, delta, ks, h, shots):
    P = np.clip(0.5 - 0.5 * f ** (2 * ks + 1) * np.cos((4 * ks + 2) * theta + 2 * ks * delta), 1e-12, 1 - 1e-12)
    return -np.sum(h * np.log(P) + (shots - h) * np.log(1 - P))


def _fit(ks, h, shots, fit_f):
    """MLE of θ (and f if fit_f) by a vectorised grid search followed by local refinement."""
    grid = np.linspace(1e-6, math.pi / 2 - 1e-6, 4000)
    fs = np.linspace(0.05, 1.0, 30) if fit_f else np.array([1.0])
    K = ks[None, None, :]
    P = 0.5 - 0.5 * fs[None, :, None] ** (2 * K + 1) * np.cos((4 * K + 2) * grid[:, None, None])
    P = np.clip(P, 1e-12, 1 - 1e-12)
    nll = -(h * np.log(P) + (shots - h) * np.log(1 - P)).sum(-1)          # (θ, f)
    i, j = np.unravel_index(np.argmin(nll), nll.shape)
    if fit_f:
        r = minimize(lambda z: _nll(z[0], z[1], 0.0, ks, h, shots), [grid[i], fs[j]],
                     bounds=[(1e-7, math.pi / 2), (0.01, 1.0)], method="L-BFGS-B")
        return r.x[0], r.x[1], r.fun
    r = minimize(lambda z: _nll(z[0], 1.0, 0.0, ks, h, shots), [grid[i]], bounds=[(1e-7, math.pi / 2)], method="L-BFGS-B")
    return r.x[0], 1.0, r.fun


def mitigation_study(n_inst=250, shots=100, seed=11, cnot_Q=6484, classical_rel_sd=0.005, pilot_rel_sd=0.05):
    """Random instances: amplitude a ~ logU[1e-3, 0.1], two-qubit error p2 ~ logU[1e-7, 1e-5] (f = (1-p2)^cnot_Q),
    control offset δ ~ U[-δmax, δmax] with δmax ∈ {0, 0.003, 0.01} rad per Grover iterate."""
    rng = np.random.default_rng(seed)
    out = []
    for dmax in (0.0, 0.003, 0.01):
        for i in range(n_inst):
            a = 10 ** rng.uniform(-3, -1)
            p2 = 10 ** rng.uniform(-7, -5)
            f = (1 - p2) ** cnot_Q
            delta = rng.uniform(-dmax, dmax)
            th = math.asin(math.sqrt(a))
            ks = [0]
            k = 1
            while f ** (2 * k + 1) > 0.1 and k <= 64:
                ks.append(k); k *= 2
            ks = np.array(ks)
            P = 0.5 - 0.5 * f ** (2 * ks + 1) * np.cos((4 * ks + 2) * th + 2 * ks * delta)
            h = rng.binomial(shots, P)
            t_raw, _, _ = _fit(ks, h, shots, False)
            t_mit, f_hat, nll_mit = _fit(ks, h, shots, True)
            # goodness of fit: G-statistic against the saturated model (each depth its own probability)
            phat = np.clip(h / shots, 1e-12, 1 - 1e-12)
            nll_sat = -np.sum(h * np.log(phat) + (shots - h) * np.log(1 - phat))
            G = 2 * (nll_mit - nll_sat)
            dof = max(len(ks) - 2, 1)
            pval = float(chi2.sf(G, dof)) if len(ks) > 2 else 1.0
            a_raw, a_mit = math.sin(t_raw) ** 2, math.sin(t_mit) ** 2
            classical = a * (1 + classical_rel_sd * rng.standard_normal())     # full RDPO estimate (fallback)
            pilot = a * (1 + pilot_rel_sd * rng.standard_normal())             # cheap RDPO pilot (detection only)
            a_guard = a_mit if pval >= 0.01 else classical
            disagree = abs(a_mit - pilot) > 3 * pilot_rel_sd * pilot
            a_guard2 = classical if (pval < 0.01 or disagree) else a_mit
            out.append({"delta_max": dmax, "a": a, "p2": p2, "delta": delta, "depths": int(len(ks)),
                        "err_raw": abs(a_raw - a) / a, "err_mit": abs(a_mit - a) / a, "err_guard": abs(a_guard - a) / a,
                        "err_guard2": abs(a_guard2 - a) / a, "gof_p": pval, "fallback": bool(pval < 0.01),
                        "fallback2": bool(pval < 0.01 or disagree)})
        sub = [r for r in out if r["delta_max"] == dmax]
        worse = np.mean([r["err_mit"] > r["err_raw"] for r in sub])
        print(f"δmax={dmax}: mitigation worse than raw in {100 * worse:.0f}%; p99 err raw {np.percentile([r['err_raw'] for r in sub], 99):.2f} "
              f"mit {np.percentile([r['err_mit'] for r in sub], 99):.2f} guard {np.percentile([r['err_guard'] for r in sub], 99):.2f}; "
              f"fallback {100 * np.mean([r['fallback'] for r in sub]):.0f}%", file=sys.stderr, flush=True)
    return out


def summarise_mitigation(rows):
    s = {}
    for dmax in sorted({r["delta_max"] for r in rows}):
        sub = [r for r in rows if r["delta_max"] == dmax]
        d = {}
        for k in ("raw", "mit", "guard", "guard2"):
            e = np.array([r["err_" + k] for r in sub])
            d[k] = {"median": float(np.median(e)), "p95": float(np.percentile(e, 95)), "p99": float(np.percentile(e, 99)),
                    "max": float(e.max()), "frac_above_50pct": float(np.mean(e > 0.5))}
        d["mitigation_worse_than_raw"] = float(np.mean([r["err_mit"] > r["err_raw"] for r in sub]))
        d["guard_worse_than_raw"] = float(np.mean([r["err_guard"] > r["err_raw"] for r in sub]))
        d["fallback_rate"] = float(np.mean([r["fallback"] for r in sub]))
        d["fallback2_rate"] = float(np.mean([r["fallback2"] for r in sub]))
        d["guard2_worse_than_raw"] = float(np.mean([r["err_guard2"] > r["err_raw"] for r in sub]))
        s[str(dmax)] = d
    return s


def run():
    out = R / "quantum_advantage.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    if "advantage" not in res:
        res["advantage"] = advantage_map()
        out.write_text(json.dumps(res, indent=2, default=float))
    if "mitigation_instances" not in res:
        mit = mitigation_study()
        res["mitigation_summary"] = summarise_mitigation(mit)
        res["mitigation_instances"] = mit
        out.write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    run()


def scaling_laws(adv=None, out_file=None):
    """Resource scaling of the Q-RDPO circuits and the quantum/classical crossover ('phase diagram').

    1. Least-squares fits of logical qubits, Toffolis per A, CNOTs per Grover iterate and A-depth against the
       portfolio size n and the number of dependence edges |E| (8 compiled settings, ring and complete graphs).
    2. Crossover law. For relative error ε, classical RDPO needs N_C = ρ²/ε² paths of cost τ_C each; amplitude
       estimation on the loaded measure needs N_Q ≈ c_Q / (ε √a) calls of A (IQAE) of cost τ_A each. Hence
           s*(ε, ρ²) = T_Q / T_C = (c_Q τ_A / τ_C) · ε / (ρ² √a),
       and quantum wins in wall-clock iff the hardware speed-up over the stated fault-tolerant baseline
       exceeds s*. c_Q, τ_A, τ_C and a are measured per setting from results/quantum_advantage.json (ε = 10%).
    """
    if adv is None:
        adv = json.loads((R / "quantum_advantage.json").read_text())["advantage"]
    rows = []
    for r in adv:
        n = r["n"]
        E = (1 if n == 2 else n) if r["topology"] == "ring" else n * (n - 1) // 2
        eps = r["eps_rel"]
        tau_C = r["classical_seconds"] / r["classical_paths"]
        tau_A = r["quantum_seconds"] / r["quantum_calls"]
        cQ = r["quantum_calls"] * eps * math.sqrt(r["a_twisted"])
        rows.append({"topology": r["topology"], "n": n, "E": E, "rho2": r["classical_rho2"], "a": r["a_twisted"],
                     "tau_C": tau_C, "tau_A": tau_A, "c_Q": cQ, "overhead": tau_A / tau_C, "s_star_10pct": r["required_speedup"],
                     **{k: r[k] for k in ("logical_qubits", "toffoli_A", "rotations_A", "cnot_Q", "depth_A")}})
    X = np.array([[1.0, r["n"], r["E"]] for r in rows])
    fits = {}
    for k in ("logical_qubits", "toffoli_A", "rotations_A", "cnot_Q", "depth_A"):
        y = np.array([r[k] for r in rows], float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        fits[k] = {"const": float(coef[0]), "per_country": float(coef[1]), "per_edge": float(coef[2]),
                   "max_rel_err": float(np.max(np.abs(pred - y) / y))}
    # crossover: rho^2 at which s* = 1 for a target error eps, per setting
    for r in rows:
        r["rho2_breakeven_at_1pct"] = r["c_Q"] * r["overhead"] * 0.01 / math.sqrt(r["a"])
        r["rho2_breakeven_at_10pct"] = r["c_Q"] * r["overhead"] * 0.10 / math.sqrt(r["a"])
    out = {"fits": fits, "rows": rows}
    if out_file is not False:
        (R / (out_file or "quantum_scaling.json")).write_text(json.dumps(out, indent=2))
    return out
