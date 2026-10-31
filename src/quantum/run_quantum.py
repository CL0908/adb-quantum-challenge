"""Phase 3 driver: QCBM vs classical iid proposals, IS -> IQAE cascade, resource estimates."""
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.baselines.common import make_problem
from src.baselines.estimators import is_proposal
from src.model.scenarios import SCENARIOS, TARGETS
from src.learned.cross_entropy import fit as ce_fit
from .qcbm import QCBM, optimal_iid_proposal, factorized_fit, loglinear_fit, kl, iid_is_variance
from .iqae import iqae, amplitude_for_proposal
from . import resources as R

SEED = 20261102
ALPHA_MIX = 0.05
QCBM_LAYERS = 4
EPS_REL = 0.01
IQAE_REPS = 50
EXHAUST = "term_both_bonds_exhausted_probability"
CASES = [(s, EXHAUST) for s in SCENARIOS] + [("adb-base", t) for t in TARGETS if t != EXHAUST]


def iid_q(q):
    return lambda t, s: np.broadcast_to(q, (len(s), 100))


def run(out=Path("results/quantum.json")):
    res = {"seed": SEED, "alpha_mix": ALPHA_MIX, "qcbm_layers": QCBM_LAYERS, "eps_rel": EPS_REL,
           "proposals": {}, "iqae": {}, "resources": {}}
    for ci, (name, t) in enumerate(CASES):
        prob, sc = make_problem(name), SCENARIOS[name]
        p, f = prob.p, prob.f[t]
        t0 = time.perf_counter()
        teacher = optimal_iid_proposal(p, f)
        fam = {"mc": (p, 0), "pdf_is": (is_proposal(prob), 0), "teacher_iid": (teacher, 100)}
        qf, kf = factorized_fit(teacher); fam["factorized_iid"] = (qf, kf)
        ql, kl_ = loglinear_fit(teacher, p); fam["loglinear_iid"] = (ql, kl_)
        qc = QCBM(QCBM_LAYERS, seed=SEED + ci)
        _, leak = qc.fit_kl(teacher, restarts=2, seed=SEED + ci)
        qq, _ = qc.model_probs(); fam["qcbm_iid"] = (qq, qc.n_params)
        rows = {}
        for k, (q, npar) in fam.items():
            qm = q if k in ("mc", "pdf_is") else (1 - ALPHA_MIX) * q + ALPHA_MIX * p
            v, mu = iid_is_variance(p, f, qm)
            rows[k] = {"n_params": npar, "kl_to_teacher": kl(teacher, q) if k not in ("mc", "pdf_is") else None,
                       "rel_std_per_path": float(np.sqrt(max(v, 0)) / mu)}
        rows["qcbm_iid"]["leak"] = leak
        res["proposals"][f"{name}|{t}"] = rows
        print(f"{name:17s} {t[5:40]:36s} " + " ".join(f"{k}={r['rel_std_per_path']:.3g}" for k, r in rows.items())
              + f" leak={leak:.3f} {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True)

        # IS -> QAE cascade on the base scenario (exact amplitudes by 10^6-path enumeration)
        if name == "adb-base":
            doob, info = ce_fit(prob, t, sc, kind="pairwise", seed=SEED + ci)
            props = {
                "none (load p)": lambda tt, s: np.broadcast_to(p, (len(s), 100)),
                "pdf_is": iid_q(is_proposal(prob)),
                "loglinear_iid": iid_q((1 - ALPHA_MIX) * ql + ALPHA_MIX * p),
                "qcbm_iid": iid_q((1 - ALPHA_MIX) * qq + ALPHA_MIX * p),
                "learned_doob_state_dependent": doob.q,
            }
            amp = {k: amplitude_for_proposal(p, f, fn) for k, fn in props.items()}
            rng = np.random.default_rng(SEED + 99 + ci)
            for k, d in amp.items():
                eps = EPS_REL * d["a"]
                runs = [iqae(min(d["a"], 1 - 1e-12), eps, rng=rng) for _ in range(IQAE_REPS)]
                est = np.array([r[0] for r in runs]) * d["M"]
                d["iqae_A_calls_mean"] = float(np.mean([r[1] for r in runs]))
                d["iqae_kmax"] = int(max(r[2] for r in runs))
                d["iqae_rel_rmse"] = float(np.sqrt(np.mean((est - d["mu"]) ** 2)) / d["mu"])
                # classical IS paths for the same relative RMSE (exact variance)
                d["classical_paths_for_same_rel_rmse"] = float((d["rel_std_per_path"] / max(d["iqae_rel_rmse"], 1e-300)) ** 2)
            res["iqae"][t] = {"ce_training_paths": info["training_paths"], "proposals": amp}
            print("   IQAE:", {k: (f"a={d['a']:.3g}", f"calls={d['iqae_A_calls_mean']:.3g}") for k, d in amp.items()},
                  file=sys.stderr, flush=True)

    # Resource estimates for the rarest target (both exhausted), 1% relative error
    amp = res["iqae"][EXHAUST]["proposals"]
    for k, loading in [("none (load p)", "exact_iid"), ("pdf_is", "exact_iid"), ("loglinear_iid", "exact_iid"),
                       ("qcbm_iid", "qcbm"), ("learned_doob_state_dependent", "doob_state_dependent")]:
        c = R.total_cost(amp[k]["iqae_A_calls_mean"], loading, qcbm_layers=QCBM_LAYERS)
        c["surface_code"] = R.surface_code_footprint(c["logical_qubits"], c["T_count_total"])
        c["amplitude"] = amp[k]["a"]
        res["resources"][k] = c
    res["resources"]["nisq_qcbm_sampler_4096_paths"] = R.nisq_sampling_cost(
        4096, QCBM_LAYERS, res["proposals"][f"adb-base|{EXHAUST}"]["qcbm_iid"]["leak"])
    out.write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    run()
