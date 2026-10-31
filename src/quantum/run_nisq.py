"""Driver: hardware-level counts, noisy MLAE (structured A), noisy QCBM sampler bias + mitigation."""
import json
import sys
from pathlib import Path

import numpy as np

from src.tensor.synthetic import SyntheticModel, topology
from src.baselines.common import make_problem
from src.quantum.qcbm import QCBM, optimal_iid_proposal
from . import nisq

OUT = Path("results/nisq.json")


def run():
    res = {"counts": [], "mlae": {}, "qcbm_bias": []}
    for kind, n in [("chain", 2), ("ring", 8), ("ring", 28), ("complete", 6), ("complete", 12)]:
        c = nisq.nisq_counts(SyntheticModel(n, topology(kind, n)))
        res["counts"].append({"topology": kind, "n": n, **{k: c[k] for k in
                              ["logical_qubits", "toffoli", "rotations", "cnot_A", "cnot_Q", "depth_A_cnot_layers", "depth_Q_cnot_layers"]},
                              "modality": c["modality"]})
        print("counts", kind, n, c["cnot_Q"], file=sys.stderr, flush=True)
    c2 = res["counts"][0]["cnot_Q"]
    for name, a in [("exceedance_adb", 0.012046720907534033), ("both_exhausted_adb", 3.163304630917912e-05)]:
        res["mlae"][name] = nisq.mlae_study(a, c2, [1e-3, 1e-4, 1e-5, 1e-6], shots=100, reps=40, seed=1)
        print("mlae", name, file=sys.stderr, flush=True)
    prob = make_problem("adb-base"); t = "term_portfolio_loss_exceedance_probability"; f = prob.f[t]
    qc = QCBM(4, seed=1); qc.fit_kl(optimal_iid_proposal(prob.p, f), restarts=2, seed=1)
    for p2 in [1e-3, 5e-3, 1e-2]:
        for pro in [0.01, 0.03]:
            for cs in [10 ** 4, 10 ** 5, 10 ** 6]:
                rs = [nisq.is_bias_study(prob, f, qc, 0.05, p2, pro, cs, np.random.default_rng(s)) for s in range(10)]
                b = np.array([r["rel_bias_calibrated_weights"] for r in rs])
                res["qcbm_bias"].append({"p2": p2, "p_readout": pro, "calib_shots": cs,
                                         "tv_noisy_vs_ideal": rs[0]["tv_distance_noisy_vs_ideal"],
                                         "rel_bias_ideal_weights": rs[0]["rel_bias_ideal_weights"],
                                         "rel_bias_calibrated_mean": float(b.mean()), "rel_bias_calibrated_rms": float(np.sqrt(np.mean(b ** 2)))})
    OUT.write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    run()
