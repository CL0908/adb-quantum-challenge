"""Costed, simulated plan for a small real-hardware amplitude-estimation run of the two-country ADB benchmark.

Entry point: ``python -m src.quantum.hardware_plan``  (writes results/hardware_plan.json and
figures/paper/hardware_plan.svg).  qiskit / qiskit-aer / qiskit-ibm-runtime are imported inside functions only;
the exact-law, tilt and statistics helpers work without them.

What is planned
---------------
The joint term law of the two ADB countries is a 10 x 10 table over the terminal states s_i = 2 a_i + r_i
(a in 0..4 EQ quarters, r in 0..1 rain trigger), i.e. 100 states = 7 qubits (padded to 128).  Both targets
are functions of the pair (L1, L2) only:
    exceedance          f = 1{L1 + L2 > 80m}   mu = 1.2e-2
    joint exhaustion    f = 1{L1 = L2 = full}  mu = 3.2e-5

Encodings (all exact: a = Pr[good] on A|0> and mu is recovered from a to ~1e-15)
  direct   amplitude-encode sqrt(p) (7 qubits), a = mu.  Good set is relabelled so that the oracle is a few
           multi-controlled Z (one MCZ for exhaustion; <= popcount(|G|) cubes for exceedance).
  tilt     Proposition 6: load the exponentially tilted law q_eta ~ p e^{eta L} (L = L1 + L2) and write the
           bounded weight g = e^{-eta (L - L_min)} on an objective qubit; a = mu e^{eta L_min - Lambda(eta)},
           mu = a e^{Lambda(eta) - eta L_min}.  Exhaustion has a single good state (the maximum of L), so the
           weight is 1 and no objective qubit is needed (7 qubits); exceedance needs the objective qubit (8).
  coarse   (reported, not recommended) 2-class-per-country coarse-graining needs only 2 qubits, but its
           amplitudes are the target probabilities themselves, so it is circular for a 2-country model.

Noisy emulation: the transpiled circuits (cz/rz/sx/x, heavy-hex patch of a fake IBM Heron backend) are run with
Aer's density-matrix method under the backend noise model (gate errors + thermal relaxation, no idle noise),
readout errors are applied afterwards from the backend's calibration, outcomes are sampled with multinomial
shot noise.  The estimator is maximum-likelihood amplitude estimation with a noise-aware likelihood
(global-depolarising fidelity F_k = exp(-lambda n2q_k) fitted jointly, known readout confusion) and the trust
guard of src/quantum/advantage.py (G-test goodness of fit across depths, agreement with a classical RDPO
pilot within 3 sigma).
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2

OUT = Path("results/hardware_plan.json")
FIG = Path("figures/paper/hardware_plan.svg")
SEEDS_TRANSPILE = (11, 12, 13, 14, 15)
KS = (0, 1, 2, 3, 4)
SHOT_GRID = (1_000, 3_000, 10_000, 30_000, 100_000)
PILOT_REL_SD = 0.05            # cheap classical RDPO pilot (detection only), as in advantage.mitigation_study
CLASSICAL_REL_SD = 0.005       # full RDPO estimate used as fallback when the guard fires


# ---------------------------------------------------------------------------------------------------
# 1. exact joint law and targets
# ---------------------------------------------------------------------------------------------------
def joint_law():
    """Exact 10 x 10 terminal-state law of the base ADB model and the per-state losses (units of 2.5m)."""
    from src.rdpo.problem import adb_model, PortfolioProblem
    from src.tensor.synthetic import dense_term_distribution
    model = adb_model("adb-base")
    prob = PortfolioProblem(model, "exceedance", threshold=80e6)
    P = dense_term_distribution(model).reshape(10, 10)
    a, r = np.arange(10) // 2, np.arange(10) % 2
    L = prob.site_loss(a, r)
    return P, L, int(prob.T), int(prob.F)


def target_sets(P, L, T, F):
    L1, L2 = L[:, None] + 0 * L[None, :], L[None, :] + 0 * L[:, None]
    Lt = L1 + L2
    sets = {"exceedance": Lt > T, "joint_exhaustion": (L1 == F) & (L2 == F)}
    return Lt, sets


def tilt_scan(P, Lt, G, etas):
    """a(eta) = mu e^{eta L_min - Lambda(eta)} for the indicator target on G (weights g = e^{-eta(L-L_min)})."""
    mu = float(P[G].sum())
    Lmin = float(Lt[G].min())
    out = []
    for e in etas:
        Lam = math.log(float((P * np.exp(e * (Lt - Lt.max()))).sum())) + e * Lt.max()
        out.append(mu * math.exp(e * Lmin - Lam))
    return np.array(out), mu, Lmin


# ---------------------------------------------------------------------------------------------------
# 2. encodings
# ---------------------------------------------------------------------------------------------------
class Encoding:
    """A|0> = amp (length 2^nq).  `good` marks the basis states counted as 1 on measurement; `cubes` is the
    oracle S_chi as a union of subcubes {(qubit, bit)...}.  mu_from_a maps the encoded amplitude back to mu."""

    def __init__(self, name, target, nq, amp, good, cubes, a, mu, mu_from_a, eta=0.0, note=""):
        self.name, self.target, self.nq, self.amp, self.good, self.cubes = name, target, nq, amp, good, cubes
        self.a, self.mu, self.mu_from_a, self.eta, self.note = a, mu, mu_from_a, eta, note
        self.theta = math.asin(math.sqrt(a))

    def label(self):
        return f"{self.target}/{self.name}"


def _suffix_cubes(nq, m):
    """Basis states [2^nq - m, 2^nq) as aligned subcubes (one per set bit of m): list of {qubit: bit}."""
    cubes, start = [], 2 ** nq
    for b in range(nq, -1, -1):
        if m >> b & 1:
            start -= 2 ** b
            cubes.append({q: (start >> q) & 1 for q in range(b, nq)})
    return cubes


def _perm_amplitudes(vals, G_flat, nq=7):
    """Place the 100 states in a 2^nq register: bad states at the bottom, good states as a suffix."""
    n = len(vals)
    good_ids = np.flatnonzero(G_flat)
    bad_ids = np.flatnonzero(~G_flat)
    idx = np.empty(n, int)
    idx[bad_ids] = np.arange(len(bad_ids))
    idx[good_ids] = 2 ** nq - len(good_ids) + np.arange(len(good_ids))
    v = np.zeros(2 ** nq)
    v[idx] = vals
    return v, idx


def build_encodings(P, Lt, sets, eta_by_target=None):
    """Return {(target, name): Encoding} for direct and tilted encodings."""
    out = {}
    eta_by_target = eta_by_target or {}
    for tname, G in sets.items():
        Gf, pf, Lf = G.ravel(), P.ravel(), Lt.ravel().astype(float)
        mu = float(pf[Gf].sum())
        Lmin = float(Lf[Gf].min())
        # --- direct --------------------------------------------------------------------------
        v, idx = _perm_amplitudes(np.sqrt(pf), Gf)
        good = np.zeros(128, bool)
        good[idx[Gf]] = True
        out[(tname, "direct")] = Encoding(
            "direct", tname, 7, v, good, _suffix_cubes(7, int(Gf.sum())), mu, mu, lambda a: a,
            note="amplitude encoding of sqrt(p) on 100 states, a = mu")
        # --- tilted (Proposition 6) ----------------------------------------------------------
        etas = np.linspace(0, 3.0, 3001)
        aa, _, _ = tilt_scan(P, Lt, G, etas)
        eta = float(eta_by_target.get(tname, etas[int(np.argmax(aa))]))
        Lam = math.log(float((pf * np.exp(eta * (Lf - Lf.max()))).sum())) + eta * Lf.max()
        q = pf * np.exp(eta * Lf - Lam)
        w = np.where(Gf, np.exp(-eta * (Lf - Lmin)), 0.0)          # bounded weight in (0, 1] on G
        mu_from_a = (lambda a, Lam=Lam, eta=eta, Lmin=Lmin: a * math.exp(Lam - eta * Lmin))
        if int(Gf.sum()) == 1:                                       # single good state, weight 1: no objective qubit
            v, idx = _perm_amplitudes(np.sqrt(q), Gf)
            good = np.zeros(128, bool)
            good[idx[Gf]] = True
            out[(tname, "tilt")] = Encoding(
                "tilt", tname, 7, v, good, _suffix_cubes(7, 1), float(q[Gf].sum()), mu, mu_from_a, eta,
                note="tilted law, good state = unique maximum of L (weight 1, no objective qubit)")
        else:                                                        # objective qubit = qubit 7
            v7, idx = _perm_amplitudes(np.ones(len(pf)), Gf)         # only for the index map
            amp = np.zeros(256)
            amp[idx] = np.sqrt(q * (1 - w))
            amp[128 + idx] = np.sqrt(q * w)
            good = np.zeros(256, bool)
            good[128:] = True
            a = float((q * w).sum())
            out[(tname, "tilt")] = Encoding(
                "tilt", tname, 8, amp, good, [{7: 1}], a, mu, mu_from_a, eta,
                note="tilted law + bounded weight on objective qubit, a = mu e^{eta Lmin - Lambda}")
    return out


def coarse_encoding_note(P, L, F):
    """The 2-qubit coarse-graining (country exhausted / not) is exact for joint exhaustion but its amplitudes are the
    target probabilities themselves; recorded only to show why it is not used."""
    E = (L == F)
    p = np.array([[P[~E][:, ~E].sum(), P[~E][:, E].sum()], [P[E][:, ~E].sum(), P[E][:, E].sum()]])
    return {"qubits": 2, "class_probabilities": p.tolist(), "mu_read_off": float(p[1, 1]),
            "verdict": "circular: the loaded amplitude is the answer; a hardware run would test nothing about the model"}


# ---------------------------------------------------------------------------------------------------
# 3. circuits (qiskit imported lazily)
# ---------------------------------------------------------------------------------------------------
def qiskit_available():
    try:
        import qiskit, qiskit_aer, qiskit_ibm_runtime  # noqa: F401
        return True
    except Exception:
        return False


def _mcz(qc, qubits):
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.mcp(math.pi, list(qubits[:-1]), qubits[-1])


def _phase_flip_cubes(qc, cubes):
    """S_chi: phase -1 on every basis state in the union of (disjoint) subcubes."""
    for cube in cubes:
        qs = sorted(cube)
        zeros = [q for q in qs if cube[q] == 0]
        for q in zeros:
            qc.x(q)
        _mcz(qc, qs)
        for q in zeros:
            qc.x(q)


def _s0(qc, nq):
    qc.x(range(nq))
    _mcz(qc, list(range(nq)))
    qc.x(range(nq))


def build_A(enc):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation
    qc = QuantumCircuit(enc.nq, name="A")
    qc.append(StatePreparation(enc.amp / np.linalg.norm(enc.amp)), range(enc.nq))
    return qc


def build_circuit(enc, k, A=None, measure=False):
    """A Q^k with Q = A S_0 A^dagger S_chi (global sign dropped), applied to |0>."""
    from qiskit import QuantumCircuit
    A = A if A is not None else build_A(enc)
    qc = QuantumCircuit(enc.nq)
    qc.compose(A, inplace=True)
    for _ in range(k):
        _phase_flip_cubes(qc, enc.cubes)
        qc.compose(A.inverse(), inplace=True)
        _s0(qc, enc.nq)
        qc.compose(A, inplace=True)
    return qc


def statevector_check(enc, ks=KS):
    """Exact checks (noise free): good probability after A Q^k equals sin^2((2k+1) theta); mu recovered from a."""
    from qiskit.quantum_info import Statevector
    A = build_A(enc)
    res = []
    for k in ks:
        sv = Statevector(build_circuit(enc, k, A))
        p = np.abs(sv.data) ** 2
        pg = float(p[enc.good].sum())
        res.append({"k": k, "p_good": pg, "p_ideal": math.sin((2 * k + 1) * enc.theta) ** 2,
                    "abs_err": abs(pg - math.sin((2 * k + 1) * enc.theta) ** 2)})
    a_sv = res[0]["p_good"]
    return {"a_statevector": a_sv, "mu_recovered": enc.mu_from_a(a_sv), "mu_exact": enc.mu,
            "mu_abs_err": abs(enc.mu_from_a(a_sv) - enc.mu), "per_k": res}


# ---------------------------------------------------------------------------------------------------
# 4. backend model, transpilation
# ---------------------------------------------------------------------------------------------------
def get_backend():
    """Fake IBM Heron (156q, cz-native, heavy-hex) if installable; else None (generic fallback is used)."""
    try:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        return FakeFez()
    except Exception:
        return None


def pick_patch(backend, size=10):
    """Connected `size`-qubit patch of the device grown greedily from low-error cz edges (qubit labels = physical)."""
    tgt = backend.target
    err = {}
    for qs, props in tgt["cz"].items():
        if props is not None and props.error is not None:
            err[tuple(sorted(qs))] = props.error
    adj = {}
    for (u, v), e in err.items():
        adj.setdefault(u, {})[v] = e
        adj.setdefault(v, {})[u] = e
    best = None
    for start in sorted(adj, key=lambda q: min(adj[q].values()))[:25]:
        region = [start]
        while len(region) < size:
            cand = {}
            for u in region:
                for v, e in adj[u].items():
                    if v not in region:
                        cand[v] = min(cand.get(v, 9), e)
            if not cand:
                break
            region.append(min(cand, key=cand.get))
        edges = [(u, v) for (u, v) in err if u in region and v in region]
        score = sum(err[e] for e in edges) / max(len(edges), 1)
        if len(region) == size and (best is None or score < best[0]):
            best = (score, region, edges)
    return best[1], best[2]


def transpile_patch(qc, patch, edges, seed):
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    idx = {q: i for i, q in enumerate(patch)}
    cm = CouplingMap([(idx[u], idx[v]) for u, v in edges] + [(idx[v], idx[u]) for u, v in edges])
    return transpile(qc, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"], optimization_level=3,
                     seed_transpiler=seed)


def two_q_stats(tc):
    ops = tc.count_ops()
    return {"n2q": int(ops.get("cz", 0)), "n1q": int(sum(v for k, v in ops.items() if k != "cz")),
            "depth": int(tc.depth()), "depth_2q": int(tc.depth(lambda i: i.operation.num_qubits == 2))}


def resource_table(enc, backend, patch, edges, ks=KS, seeds=SEEDS_TRANSPILE):
    """2Q count and depth of A Q^k on (i) the heavy-hex patch used for emulation and (ii) the whole device."""
    from qiskit import transpile
    A = build_A(enc)
    rows, best = [], {}
    for k in ks:
        qc = build_circuit(enc, k, A)
        s_patch = [transpile_patch(qc, patch, edges, s) for s in seeds]
        st = [two_q_stats(t) for t in s_patch]
        ib = int(np.argmin([x["n2q"] for x in st]))
        best[k] = (s_patch[ib], seeds[ib])
        full = [two_q_stats(transpile(qc, backend=backend, optimization_level=3, seed_transpiler=s)) for s in seeds]
        rows.append({"k": k, "logical_ops": int(sum(qc.count_ops().values())),
                     "patch_n2q_median": float(np.median([x["n2q"] for x in st])),
                     "patch_n2q_min": min(x["n2q"] for x in st), "patch_n2q_max": max(x["n2q"] for x in st),
                     "patch_depth_median": float(np.median([x["depth"] for x in st])),
                     "patch_depth2q_median": float(np.median([x["depth_2q"] for x in st])),
                     "device_n2q_median": float(np.median([x["n2q"] for x in full])),
                     "device_n2q_min": min(x["n2q"] for x in full),
                     "device_depth_median": float(np.median([x["depth"] for x in full])),
                     "best_seed": seeds[ib]})
    return rows, best


# ---------------------------------------------------------------------------------------------------
# 5. noisy density-matrix probabilities
# ---------------------------------------------------------------------------------------------------
def _embed(tc, patch, nphys):
    """Put a circuit on patch-local qubits 0..len-1 onto physical qubits `patch` of an nphys-qubit register."""
    from qiskit import QuantumCircuit
    big = QuantumCircuit(nphys)
    big.compose(tc, qubits=list(patch), inplace=True)
    return big


def noisy_distribution(tc, enc, patch, nphys, noise_model):
    """Exact output distribution over the enc.nq logical qubits for transpiled circuit `tc` (patch-local indices)
    under `noise_model`, via Aer density-matrix simulation. Layout/permutation is taken from tc.layout."""
    from qiskit_aer import AerSimulator
    big = _embed(tc, patch, nphys)
    lay = tc.layout
    final = lay.final_index_layout(filter_ancillas=True) if lay is not None else list(range(enc.nq))
    phys = [patch[i] for i in final[:enc.nq]]
    big.save_probabilities(phys, label="p")
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    res = sim.run(big, shots=1).result()
    return np.asarray(res.data(0)["p"], float), phys


def readout_matrix(errs):
    """errs: list of (p(1|0), p(0|1)) per logical qubit (little endian). Returns tensor-product confusion C[meas, true]."""
    C = np.ones((1, 1))
    for e01, e10 in errs:                       # qubit 0 is least significant -> kron on the left
        C = np.kron(np.array([[1 - e01, e10], [e01, 1 - e10]]), C)
    return C


def ideal_noise_free_probs(tc_logical_qc):
    from qiskit.quantum_info import Statevector
    return np.abs(Statevector(tc_logical_qc).data) ** 2


def depol_noise_model(p2, p1, nphys):
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    nm = NoiseModel(basis_gates=["cz", "rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
    return nm


_CTX = {}


def _dm_worker(job):
    lab, k, kind, p2 = job
    enc, tc = _CTX["encs"][lab], _CTX["best"][lab][k][0]
    if kind == "backend":
        nm = _CTX["nm_backend"]
    else:
        nm = depol_noise_model(p2, p2 / 15, _CTX["nphys"])
    p, phys = noisy_distribution(tc, enc, _CTX["patch"], _CTX["nphys"], nm)
    return (lab, k, kind, p2), p, phys


def run_dm_jobs(jobs, workers=3):
    """Density-matrix runs in forked workers (context shared through _CTX). Returns {job: (probs, phys)}."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        out = pool.map(_dm_worker, jobs, chunksize=1)
    return {key: (p, phys) for key, p, phys in out}


# ---------------------------------------------------------------------------------------------------
# 6. noise-aware MLAE with trust guard
# ---------------------------------------------------------------------------------------------------
class Model:
    """P_k(theta, lambda) = F_k [ s_k^2 r_g + (1 - s_k^2) r_b ] + (1 - F_k) c_u,  s_k = sin((2k+1) theta),
    F_k = exp(-lambda n2q_k).  r_g / r_b: probability that the normalised good / bad component is *read* as good
    (from the model's conditional laws and the assumed readout confusion), c_u: same for the maximally mixed state."""

    def __init__(self, enc, ks, n2q, Cfit):
        self.ks, self.n2q = np.asarray(ks), np.asarray(n2q, float)
        gv = enc.amp ** 2
        pgood = gv[enc.good].sum()
        g = np.where(enc.good, gv, 0) / pgood
        b = np.where(~enc.good, gv, 0) / (1 - pgood)
        sel = enc.good.astype(float)
        self.r_g = float(sel @ (Cfit @ g))
        self.r_b = float(sel @ (Cfit @ b))
        self.c_u = float(sel @ (Cfit @ np.full(len(gv), 1 / len(gv))))
        self.nq = enc.nq

    def prob(self, theta, lam):
        """theta: (T,), lam: (L,) -> (T, L, K)."""
        th = np.asarray(theta)[:, None, None]
        s2 = np.sin((2 * self.ks[None, None, :] + 1) * th) ** 2
        F = np.exp(-np.asarray(lam)[None, :, None] * self.n2q[None, None, :])
        return np.clip(F * (s2 * self.r_g + (1 - s2) * self.r_b) + (1 - F) * self.c_u, 1e-12, 1 - 1e-12)


_TH_GRID = np.concatenate([[1e-7], np.geomspace(1e-5, math.pi / 2 - 1e-5, 6000)])


def _nll(P, h, n):
    return -(h * np.log(P) + (n - h) * np.log1p(-P)).sum(-1)


def fit_mlae(model, h, n, lam_max=None, fit_lam=True, lam_fixed=0.0):
    """Grid search + zoom for (theta, lambda). Returns theta_hat, lambda_hat, nll_min."""
    lam_max = lam_max if lam_max is not None else 2.0 / max(model.n2q.max(), 1)
    lams = np.linspace(0, lam_max, 41) if fit_lam else np.array([lam_fixed])
    nll = _nll(model.prob(_TH_GRID, lams), h, n)
    i, j = np.unravel_index(np.argmin(nll), nll.shape)
    th0, l0 = _TH_GRID[i], lams[j]
    th2 = np.linspace(th0 * 0.97, min(th0 * 1.03, math.pi / 2), 300) if th0 > 1e-6 else _TH_GRID[:50]
    l2 = np.linspace(max(l0 - lam_max / 40, 0), l0 + lam_max / 40, 41) if fit_lam else np.array([lam_fixed])
    nll2 = _nll(model.prob(th2, l2), h, n)
    i, j = np.unravel_index(np.argmin(nll2), nll2.shape)
    return float(th2[i]), float(l2[j]), float(nll2[i, j])


def run_estimates(Ptrue, ks, n2q, enc, Cfit, shots_total, reps, seed, mode="aware"):
    """Simulate `reps` experiments: shots split equally over depths ks, estimate a, mu, run the trust guard.
    mode: 'aware' (fit lambda), 'naive' (lambda = 0, ignore noise)."""
    rng = np.random.default_rng(seed)
    ks = list(ks)
    model = Model(enc, ks, n2q, Cfit)
    n = np.full(len(ks), shots_total // len(ks))
    rows = []
    for _ in range(reps):
        h = rng.binomial(n, Ptrue)
        th, lam, nll_m = fit_mlae(model, h, n, fit_lam=(mode == "aware"))
        a_hat = math.sin(th) ** 2
        mu_hat = enc.mu_from_a(a_hat)
        phat = np.clip(h / n, 1e-12, 1 - 1e-12)
        nll_sat = -np.sum(h * np.log(phat) + (n - h) * np.log1p(-phat))
        G = max(2 * (nll_m - nll_sat), 0.0)
        dof = len(ks) - (2 if mode == "aware" else 1)
        pval = float(chi2.sf(G, dof)) if dof >= 1 else 1.0
        pilot = enc.mu * (1 + PILOT_REL_SD * rng.standard_normal())
        agree = abs(mu_hat - pilot) <= 3 * PILOT_REL_SD * pilot
        accept = (pval >= 0.01) and agree
        classical = enc.mu * (1 + CLASSICAL_REL_SD * rng.standard_normal())
        rows.append((mu_hat, pval, agree, accept, classical, lam))
    mu_hat = np.array([r[0] for r in rows])
    acc = np.array([r[3] for r in rows])
    guard = np.where(acc, mu_hat, [r[4] for r in rows])
    rel = lambda x: (x - enc.mu) / enc.mu
    out = {"shots": int(shots_total), "ks": ks, "mode": mode,
           "rel_rmse": float(np.sqrt(np.mean(rel(mu_hat) ** 2))), "rel_bias": float(np.mean(rel(mu_hat))),
           "rel_median_abs": float(np.median(np.abs(rel(mu_hat)))),
           "frac_within_20pct": float(np.mean(np.abs(rel(mu_hat)) < 0.2)),
           "gof_reject_rate": float(np.mean([r[1] < 0.01 for r in rows])),
           "pilot_disagree_rate": float(np.mean([not r[2] for r in rows])),
           "guard_acceptance": float(acc.mean()),
           "rel_rmse_accepted": float(np.sqrt(np.mean(rel(mu_hat[acc]) ** 2))) if acc.any() else None,
           "rel_bias_accepted": float(np.mean(rel(mu_hat[acc]))) if acc.any() else None,
           "rel_rmse_guarded_with_fallback": float(np.sqrt(np.mean(rel(guard) ** 2))),
           "lambda_hat_median": float(np.median([r[5] for r in rows]))}
    return out


def ideal_sampling_rel_err(mu, N):
    """Relative std of a plain Bernoulli estimate of mu from N noiseless shots of the loaded law."""
    return math.sqrt((1 - mu) / (mu * N))


def noiseless_fisher_rel_err(enc, ks, N):
    """Cramer-Rao relative std of mu from N shots split over ks, ideal circuits (the best any k-schedule can do)."""
    th = enc.theta
    n = N / len(ks)
    I = 0.0
    for k in ks:                       # P = sin^2((2k+1) th), dP/dth = (2k+1) sin(2(2k+1)th); I = n (dP)^2/(P(1-P))
        P = math.sin((2 * k + 1) * th) ** 2
        dP = (2 * k + 1) * math.sin(2 * (2 * k + 1) * th)
        I += n * dP ** 2 / max(P * (1 - P), 1e-300)
    sd_th = 1 / math.sqrt(I)
    return 2 * sd_th / math.tan(th)      # a = sin^2 th: da/a = 2 cot(th) dth ; mu proportional to a


# ---------------------------------------------------------------------------------------------------
# 7. budget
# ---------------------------------------------------------------------------------------------------
RATE_HZ = {"low": 1000.0, "mid": 3000.0, "high": 5000.0}     # shots/s incl. rep delay and readout (Heron, Sampler)
OPEN_PLAN_MIN_PER_MONTH = 10.0
JOB_OVERHEAD_S = 15.0                                        # per submitted job (compilation, loading), conservative


def qpu_minutes(n_circuits, shots_each, rate_hz):
    return (n_circuits * shots_each / rate_hz + n_circuits * 0.0) / 60.0


def budget(plan_circuits, shots_each, jobs):
    """plan_circuits: number of distinct circuits; shots_each: shots per circuit; jobs: submitted jobs."""
    out = {}
    for name, r in RATE_HZ.items():
        sec = plan_circuits * shots_each / r
        out[name] = {"shot_rate_hz": r, "qpu_seconds": sec, "qpu_minutes": sec / 60,
                     "job_overhead_s_if_billed": jobs * JOB_OVERHEAD_S}
    return out


# ---------------------------------------------------------------------------------------------------
# 8. orchestration
# ---------------------------------------------------------------------------------------------------
def _cal_readout(backend, patch, nq_logical_phys):
    """Per-qubit (p(1|0), p(0|1)) for the listed physical qubits from backend calibration (prob_meas1_prep0/1)."""
    errs = []
    props = backend.properties() if hasattr(backend, "properties") else None
    for q in nq_logical_phys:
        if props is not None:
            try:
                errs.append((float(props.qubit_property(q, "prob_meas1_prep0")[0]),
                             float(props.qubit_property(q, "prob_meas0_prep1")[0])))
                continue
            except Exception:
                pass
        e = backend.target["measure"][(q,)].error
        errs.append((float(e), float(e)))
    return errs


def run(quick=False, reps=None):
    t_start = time.time()
    if not qiskit_available():
        raise SystemExit("qiskit, qiskit-aer and qiskit-ibm-runtime are required for the hardware plan")
    import qiskit, qiskit_aer, qiskit_ibm_runtime
    from qiskit_aer.noise import NoiseModel
    reps = reps or (60 if quick else 300)
    P, L, T, F = joint_law()
    Lt, sets = target_sets(P, L, T, F)
    encs = build_encodings(P, Lt, sets, {"joint_exhaustion": 1.0})
    backend = get_backend()
    patch, edges = pick_patch(backend, 10)
    nphys = backend.num_qubits
    nm_backend = NoiseModel.from_backend(backend, readout_error=False)
    meta = {"backend_model": type(backend).__name__ + " (fake_provider snapshot; Heron-class, cz/rz/sx/x, heavy-hex)",
            "device_qubits": nphys, "patch_physical_qubits": patch, "patch_edges": [list(e) for e in edges],
            "noise": "NoiseModel.from_backend(gate errors + thermal relaxation, no idle noise) + calibrated readout "
                     "applied to the exact density-matrix distribution; multinomial shot noise",
            "transpile": {"optimization_level": 3, "basis": ["cz", "rz", "sx", "x"], "seeds": list(SEEDS_TRANSPILE)},
            "estimator": {"reps": reps, "shot_grid": list(SHOT_GRID), "pilot_rel_sd": PILOT_REL_SD,
                          "classical_rel_sd": CLASSICAL_REL_SD, "gof_alpha": 0.01, "pilot_sigma": 3},
            "versions": {"qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
                         "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__, "numpy": np.__version__},
            "model": {"exact_mu": {t: float(P[G].sum()) for t, G in sets.items()}, "unit_usd": 2.5e6,
                      "threshold_units": T, "full_limit_units": F}}
    res = {"meta": meta, "encodings": {}, "resources": {}, "noisy": {}, "scan": {}, "budget": {}}
    res["coarse_encoding"] = coarse_encoding_note(P, L, F)

    best_circ, dists, Cs = {}, {}, {}
    for key, enc in encs.items():
        lab = enc.label()
        chk = statevector_check(enc)
        etas = np.linspace(0, 3, 3001)
        res["encodings"][lab] = {"qubits": enc.nq, "a": enc.a, "mu": enc.mu, "eta": enc.eta,
                                 "gain_a_over_mu": enc.a / enc.mu, "note": enc.note,
                                 "oracle_cubes": [{str(q): b for q, b in c.items()} for c in enc.cubes],
                                 "statevector": chk}
        rows, best = resource_table(enc, backend, patch, edges)
        res["resources"][lab] = rows
        best_circ[lab] = best
        print(f"[{time.time() - t_start:6.0f}s] {lab}: a={enc.a:.4g} qubits={enc.nq} "
              f"n2q(k=0..4)={[r['patch_n2q_min'] for r in rows]}", file=sys.stderr, flush=True)

    # --- noisy density-matrix distributions per (encoding, k) under the backend noise -------------------
    labs = {enc.label(): enc for enc in encs.values()}
    _CTX.update(encs=labs, best=best_circ, patch=patch, nphys=nphys, nm_backend=nm_backend)
    jobs = [(lab, k, "backend", None) for lab in labs for k in KS if not (quick and k > 2)]
    ts = time.time()
    dm = run_dm_jobs(jobs)
    print(f"[{time.time() - t_start:6.0f}s] backend DM runs: {len(jobs)} jobs in {time.time() - ts:.0f}s", file=sys.stderr, flush=True)
    for lab in labs:
        dists[lab] = {k: dm[(lab, k, "backend", None)][0] for k in KS if (lab, k, "backend", None) in dm}
        Cs[lab] = readout_matrix(_cal_readout(backend, patch, dm[(lab, 0, "backend", None)][1]))
    res["dm_seconds_backend"] = time.time() - ts

    def measured_good(enc, lab, k, Ctrue):
        p = dists[lab][k]
        return float(enc.good.astype(float) @ (Ctrue @ p))

    rng = np.random.default_rng(2026)
    for key, enc in encs.items():
        lab = enc.label()
        Ctrue = Cs[lab]
        # fit model uses a readout matrix that differs by 10% (relative) per-qubit calibration error
        nq = enc.nq
        errs_true = []
        # reconstruct per-qubit errors from Ctrue marginals is awkward; draw the fit matrix by rescaling Ctrue's factors
        Cfit = _perturbed_readout(Ctrue, enc.nq, rel=0.10, rng=np.random.default_rng(5))
        kmax_avail = max(k for k in dists[lab])
        n2q_all = [best_circ[lab][k][0].count_ops().get("cz", 0) for k in KS if k in dists[lab]]
        ks_all = [k for k in KS if k in dists[lab]]
        Pk = [measured_good(enc, lab, k, Ctrue) for k in ks_all]
        ideal_k = [math.sin((2 * k + 1) * enc.theta) ** 2 for k in ks_all]
        entry = {"ks_available": ks_all, "n2q_best_seed": [int(x) for x in n2q_all], "p_measured_good": Pk,
                 "p_ideal_good": ideal_k,
                 "fidelity_vs_ideal_total_variation": [float(0.5 * np.abs(dists[lab][k] - np.abs(_ideal_vec(enc, k)) ** 2).sum())
                                                      for k in ks_all],
                 "sampling_ideal_rel_err": {str(N): ideal_sampling_rel_err(enc.mu, N) for N in SHOT_GRID},
                 "noiseless_qae_cr_bound_rel_err": {str(N): noiseless_fisher_rel_err(enc, ks_all, N) for N in SHOT_GRID},
                 "schedules": {}}
        # depth schedules: k=0 only (plain sampling with noise-aware correction), 0-1, 0-2, ..., all available
        scheds = {f"k<={m}": [k for k in ks_all if k <= m] for m in ks_all}
        for sname, sk in scheds.items():
            if len(sk) < 1:
                continue
            ii = [ks_all.index(k) for k in sk]
            out = []
            for N in SHOT_GRID:
                for mode in (["aware"] if len(sk) > 1 else ["naive"]):
                    r = run_estimates(np.array([Pk[i] for i in ii]), sk, [n2q_all[i] for i in ii], enc, Cfit, N,
                                      reps, seed=int(rng.integers(1e9)), mode=mode)
                    out.append(r)
            entry["schedules"][sname] = out
        res["noisy"][lab] = entry
        print(f"[{time.time() - t_start:6.0f}s] estimates {lab} done", file=sys.stderr, flush=True)

    # --- 2Q error scan (global-depolarising DM with the transpiled circuits of the best encoding) ---------
    res["model_check"] = model_check(encs, best_circ, patch, nphys)
    res["scan"] = error_scan(encs, best_circ, nphys, reps, quick, t_start)

    # --- budget -----------------------------------------------------------------------------------------
    res["budget"] = make_budget(res)
    res["emulator_fallback"] = {"wall_seconds_this_run": time.time() - t_start,
                                "note": "all numbers here come from a CPU-only emulator (Aer density matrix on a 10-qubit "
                                        "patch, ~minutes); the fallback for the milestone is this same pipeline, "
                                        "with the hardware run replaced by sampling from the noisy emulator"}
    res["meta"]["wall_seconds"] = time.time() - t_start
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=float))
    try:
        make_figure(res)
    except Exception as e:                                           # figure is optional
        print("figure skipped:", e, file=sys.stderr)
    return res


def _ideal_vec(enc, k):
    from qiskit.quantum_info import Statevector
    return Statevector(build_circuit(enc, k)).data


def _perturbed_readout(C, nq, rel, rng):
    """Readout confusion assumed in the fit: each single-qubit error rate off by a random relative amount."""
    # recover single-qubit factors from the tensor product via marginalising: C[meas, true] is a Kronecker product
    # kron(C_{nq-1}, ..., C_0); rebuild with perturbed error rates.
    errs = []
    N = 2 ** nq
    for q in range(nq):
        idx0 = [i for i in range(N) if not (i >> q) & 1]
        # p(1|0) for qubit q = sum over meas with bit q=1 given true all-zero outside q... use true state with bit q=0, others 0
        e01 = sum(C[m, 0] for m in range(N) if (m >> q) & 1)
        e10 = sum(C[m, 1 << q] for m in range(N) if not (m >> q) & 1)
        errs.append((e01 * (1 + rel * rng.uniform(-1, 1)), e10 * (1 + rel * rng.uniform(-1, 1))))
    return readout_matrix(errs)


def mixture_distribution(enc, k, n2q, n1q, p2):
    """Global-depolarising approximation of the noisy output: F * ideal + (1 - F) * uniform with
    F = (1 - p2)^n2q (1 - p2/15)^n1q (n1q = sx/x pulses; rz is virtual)."""
    F = (1 - p2) ** n2q * (1 - p2 / 15) ** n1q
    ideal = np.abs(_ideal_vec(enc, k)) ** 2
    return F * ideal + (1 - F) / len(ideal)


def model_check(encs, best_circ, patch, nphys, p2=3e-3, ks=(0, 1)):
    """Density-matrix (depolarising noise on the transpiled circuits) versus the global mixture used in the scan."""
    out = {}
    nm = depol_noise_model(p2, p2 / 15, nphys)
    for enc in encs.values():
        lab = enc.label()
        rows = []
        for k in ks:
            tc, _ = best_circ[lab][k]
            ops = tc.count_ops()
            n2q, n1q = ops.get("cz", 0), ops.get("sx", 0) + ops.get("x", 0)
            p_dm, _ = noisy_distribution(tc, enc, patch, nphys, nm)
            p_mx = mixture_distribution(enc, k, n2q, n1q, p2)
            g = enc.good.astype(float)
            rows.append({"k": k, "n2q": n2q, "p_good_dm": float(g @ p_dm), "p_good_mixture": float(g @ p_mx),
                         "tv": float(0.5 * np.abs(p_dm - p_mx).sum())})
        out[lab] = rows
    return {"p2": p2, "rows": out}


def error_scan(encs, best_circ, nphys, reps, quick, t_start):
    """Achievable error vs 2Q error rate with density-matrix truth (depolarising 2Q p2, 1Q p2/15, no relaxation) on the
    transpiled circuits; readout error 1e-2 and 1e-3 (fit uses a +-10% wrong calibration). 3e4 shots per target."""
    labs = {enc.label(): enc for enc in encs.values()}
    plan = {"tilt": [1e-3, 3e-4, 1e-4, 1e-5], "direct": [1e-4, 1e-5]} if not quick else {"tilt": [1e-4], "direct": []}
    jobs = []
    keep = {}
    for lab, enc in labs.items():
        for p2 in plan[enc.name]:
            ks_ok = []
            for k in KS:
                ops = best_circ[lab][k][0].count_ops()
                n2q, n1q = ops.get("cz", 0), ops.get("sx", 0) + ops.get("x", 0)
                if (1 - p2) ** n2q * (1 - p2 / 15) ** n1q < 0.02 and k > 0:
                    break
                ks_ok.append(k)
                jobs.append((lab, k, "depol", p2))
            keep[(lab, p2)] = ks_ok
    ts = time.time()
    dm = run_dm_jobs(jobs)
    print(f"[{time.time() - t_start:6.0f}s] scan DM runs: {len(jobs)} jobs in {time.time() - ts:.0f}s", file=sys.stderr, flush=True)
    out = {}
    for (lab, p2), ks_ok in keep.items():
        enc = labs[lab]
        for ro in (1e-2, 1e-3):
            C = readout_matrix([(ro, ro)] * enc.nq)
            Cfit = _perturbed_readout(C, enc.nq, 0.10, np.random.default_rng(7))
            Pk = [float(enc.good.astype(float) @ (C @ dm[(lab, k, "depol", p2)][0])) for k in ks_ok]
            n2q = [int(best_circ[lab][k][0].count_ops().get("cz", 0)) for k in ks_ok]
            r = run_estimates(np.array(Pk), ks_ok, n2q, enc, Cfit, 30_000, max(reps // 2, 30), seed=3,
                              mode="aware" if len(ks_ok) > 1 else "naive")
            out.setdefault(lab, []).append(
                {"p2": p2, "readout_error": ro, "ks": ks_ok, "n2q": n2q, "shots": 30_000,
                 "p_measured_good": Pk, "p_ideal_good": [math.sin((2 * k + 1) * enc.theta) ** 2 for k in ks_ok],
                 "rel_rmse": r["rel_rmse"], "rel_bias": r["rel_bias"], "guard_acceptance": r["guard_acceptance"],
                 "rel_rmse_accepted": r["rel_rmse_accepted"], "rel_rmse_guarded_with_fallback": r["rel_rmse_guarded_with_fallback"],
                 "ideal_sampling_rel_err": ideal_sampling_rel_err(enc.mu, 30_000)})
    return out


def make_budget(res):
    """QPU-time bill for the recommended milestone plus the Deep Development calendar check."""
    plan = {}
    # recommended: both targets, best encoding each, depths k=0..kmax_useful, N total shots per target
    for tname, N in (("exceedance", 30_000), ("joint_exhaustion", 100_000)):
        labs = [l for l in res["noisy"] if l.startswith(tname)]
        ncirc = 0
        for l in labs:
            ncirc += len(res["noisy"][l]["ks_available"])
        plan[tname] = {"circuits_per_encoding": ncirc // max(len(labs), 1), "encodings": labs, "shots_per_target_per_encoding": N}
    circuits = sum(p["circuits_per_encoding"] * len(p["encodings"]) for p in plan.values())
    # per-circuit shots: split N over depths; plus readout calibration (10 circuits x 4096) and 3 repeat sessions
    shots_total_single = sum(p["shots_per_target_per_encoding"] * len(p["encodings"]) for p in plan.values())
    calib_shots = 10 * 4096
    repeats = 3
    total_shots = (shots_total_single + calib_shots) * repeats
    jobs = repeats * (len(plan) * 2 + 1)
    tab = {name: {"shot_rate_hz": r, "qpu_minutes": total_shots / r / 60} for name, r in RATE_HZ.items()}
    window_days = (np.datetime64("2027-01-22") - np.datetime64("2026-11-30")).astype(int) + 1
    return {"assumptions": {
                "shot_rate_hz": RATE_HZ,
                "basis": "Heron-class: ~250 us default rep delay + ~2 us circuit+readout (k<=1) => ~3.9 kHz upper bound; "
                         "1 kHz allows for dynamical decoupling, twirling and job overhead",
                "open_plan_min_per_month": OPEN_PLAN_MIN_PER_MONTH,
                "open_plan_note": "verify the current plan terms on the IBM Quantum Platform before booking; terms change",
                "deep_development_window": ["2026-11-30", "2027-01-22"], "window_days": int(window_days),
                "calendar_months_touched": 3, "conservative_monthly_allotments": 2},
            "recommended_run": {"plan": plan, "total_circuits_per_session": circuits, "sessions": repeats,
                                "total_shots": total_shots, "jobs": jobs, "calibration_shots_per_session": calib_shots},
            "qpu_minutes_total": tab,
            "fits_open_plan_one_month": {n: v["qpu_minutes"] <= OPEN_PLAN_MIN_PER_MONTH for n, v in tab.items()},
            "fits_window_2_allotments": {n: v["qpu_minutes"] <= 2 * OPEN_PLAN_MIN_PER_MONTH for n, v in tab.items()},
            "shots_available_open_plan_month": {n: int(OPEN_PLAN_MIN_PER_MONTH * 60 * r) for n, r in RATE_HZ.items()}}


# ---------------------------------------------------------------------------------------------------
# 9. figure
# ---------------------------------------------------------------------------------------------------
def make_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.viz import style as S
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), gridspec_kw={"wspace": 0.38})
    cols = {"direct": S.BLUE, "tilt": S.TERRA}
    ax = axes[0]
    for lab, rows in res["resources"].items():
        t, e = lab.split("/")
        ax.plot([r["k"] for r in rows], [r["patch_n2q_min"] for r in rows], "o-" if t == "exceedance" else "s--",
                color=cols[e], label=f"{'Pr(L>80m)' if t == 'exceedance' else 'joint exh.'}, {e}")
    ax.set_yscale("log"); ax.set_xlabel("Grover power k"); ax.set_ylabel("cz gates in A·Q$^k$ (heavy-hex patch)")
    ax.set_title("(a) circuit cost"); ax.legend(fontsize=6.5, frameon=False)
    for ax, t, ttl in [(axes[1], "exceedance", "(b) Pr(L>80m), 3e4 shots"), (axes[2], "joint_exhaustion", "(c) joint exhaustion, 3e4 shots")]:
        for e in ("direct", "tilt"):
            rows = res["scan"].get(f"{t}/{e}", [])
            rows = [r for r in rows if r["readout_error"] == 1e-2]
            if rows:
                ax.plot([r["p2"] for r in rows], [r["rel_rmse"] for r in rows], "o-", color=cols[e], label=f"{e} (noisy MLAE)")
        if rows:
            ax.axhline(rows[0]["ideal_sampling_rel_err"], color="grey", ls=":", label="ideal sampling of loaded law")
        ax.axhline(0.1, color="k", lw=0.6, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_xlabel("two-qubit error rate $p_2$"); ax.set_ylabel("relative RMSE of $\\mu$"); ax.set_title(ttl)
        ax.legend(fontsize=6.5, frameon=False)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, bbox_inches="tight")
    plt.close(fig)


def stage1_marginal(reps=200, seed=21):
    """Stage 1 of the milestone: one-country marginal (10 states, 4 qubits), event 'country 1 exhausted'.
    Same pipeline (transpile on the heavy-hex patch, backend-noise density matrix, noise-aware MLAE + guard)."""
    from qiskit_aer.noise import NoiseModel
    P, L, T, F = joint_law()
    p1 = P.sum(1)
    G = np.zeros(10, bool)
    G[9] = True                                   # state (a=4, r=1) = full limit
    v, idx = _perm_amplitudes(np.sqrt(p1), G, nq=4)
    v = np.concatenate([v, np.zeros(0)])
    good = np.zeros(16, bool)
    good[idx[G]] = True
    mu = float(p1[G].sum())
    enc = Encoding("direct", "marginal_exhaustion", 4, v, good, _suffix_cubes(4, 1), mu, mu, lambda a: a,
                   note="one-country marginal, single good state")
    backend = get_backend()
    patch, edges = pick_patch(backend, 8)
    nphys = backend.num_qubits
    nm = NoiseModel.from_backend(backend, readout_error=False)
    rows, best = resource_table(enc, backend, patch, edges, ks=KS)
    labs = {enc.label(): enc}
    _CTX.update(encs=labs, best={enc.label(): best}, patch=patch, nphys=nphys, nm_backend=nm)
    dm = run_dm_jobs([(enc.label(), k, "backend", None) for k in KS])
    C = readout_matrix(_cal_readout(backend, patch, dm[(enc.label(), 0, "backend", None)][1]))
    Cfit = _perturbed_readout(C, 4, 0.10, np.random.default_rng(5))
    Pk = [float(enc.good.astype(float) @ (C @ dm[(enc.label(), k, "backend", None)][0])) for k in KS]
    n2q = [int(best[k][0].count_ops().get("cz", 0)) for k in KS]
    out = {"mu": mu, "qubits": 4, "sv_check": statevector_check(enc), "resources": rows, "p_measured_good": Pk,
           "p_ideal_good": [math.sin((2 * k + 1) * enc.theta) ** 2 for k in KS], "schedules": {}}
    for m in KS:
        ks = [k for k in KS if k <= m]
        out["schedules"][f"k<={m}"] = [run_estimates(np.array(Pk[:len(ks)]), ks, n2q[:len(ks)], enc, Cfit, N, reps, seed,
                                                        mode="aware" if len(ks) > 1 else "naive") for N in SHOT_GRID]
    out["ideal_sampling_rel_err"] = {str(N): ideal_sampling_rel_err(mu, N) for N in SHOT_GRID}
    res = json.loads(OUT.read_text())
    res["stage1_marginal"] = out
    OUT.write_text(json.dumps(res, indent=2, default=float))
    return out


if __name__ == "__main__":
    if "--stage1" in sys.argv:
        stage1_marginal()
    else:
        run(quick="--quick" in sys.argv)
