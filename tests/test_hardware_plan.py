import math

import numpy as np
import pytest

from src.quantum import hardware_plan as H


@pytest.fixture(scope="module")
def encs():
    P, L, T, F = H.joint_law()
    Lt, sets = H.target_sets(P, L, T, F)
    return P, sets, H.build_encodings(P, Lt, sets, {"joint_exhaustion": 1.0})


def test_exact_targets(encs):
    P, sets, _ = encs
    assert abs(P.sum() - 1) < 1e-12
    assert abs(P[sets["exceedance"]].sum() - 0.012046720907534038) < 1e-12
    assert abs(P[sets["joint_exhaustion"]].sum() - 3.163304630917903e-05) < 1e-12


def test_suffix_cubes_cover_exactly():
    for m in (1, 5, 45, 99):
        covered = set()
        for c in H._suffix_cubes(7, m):
            for x in range(128):
                if all(((x >> q) & 1) == b for q, b in c.items()):
                    assert x not in covered
                    covered.add(x)
        assert covered == set(range(128 - m, 128))


def test_encoding_amplitudes_exact_without_qiskit(encs):
    """Normalisation and mu recovery from the encoded amplitude, pure numpy."""
    for enc in encs[2].values():
        assert abs(np.sum(enc.amp ** 2) - 1) < 1e-12
        a = float(np.sum(enc.amp[enc.good] ** 2))
        assert abs(a - enc.a) < 1e-12
        assert abs(enc.mu_from_a(a) - enc.mu) < 1e-12 * max(1.0, enc.mu / enc.mu)
    # tilting increases the encoded amplitude
    assert encs[2][("exceedance", "tilt")].a > 5 * encs[2][("exceedance", "direct")].a
    assert encs[2][("joint_exhaustion", "tilt")].a > 1e3 * encs[2][("joint_exhaustion", "direct")].a


def test_mixture_and_readout_are_stochastic():
    C = H.readout_matrix([(0.01, 0.02)] * 3)
    assert np.allclose(C.sum(0), 1) and (C >= 0).all()


def test_budget_fits_open_plan():
    res = {"noisy": {"exceedance/direct": {"ks_available": [0, 1]}, "exceedance/tilt": {"ks_available": [0, 1]},
                     "joint_exhaustion/direct": {"ks_available": [0, 1]}, "joint_exhaustion/tilt": {"ks_available": [0, 1]}}}
    b = H.make_budget(res)
    assert b["qpu_minutes_total"]["low"]["qpu_minutes"] < H.OPEN_PLAN_MIN_PER_MONTH * 2


def test_statevector_reproduces_exact_mu(encs):
    pytest.importorskip("qiskit")
    for enc in encs[2].values():
        chk = H.statevector_check(enc, ks=(0, 1, 2))
        assert chk["mu_abs_err"] < 1e-12, (enc.label(), chk["mu_abs_err"])
        for r in chk["per_k"]:
            assert r["abs_err"] < 1e-9


def test_noise_aware_mle_recovers_amplitude_without_noise(encs):
    enc = encs[2][("exceedance", "tilt")]
    ks = [0, 1, 2]
    n2q = [100, 300, 500]
    Cid = np.eye(2 ** enc.nq)
    P = [math.sin((2 * k + 1) * enc.theta) ** 2 for k in ks]
    r = H.run_estimates(np.array(P), ks, n2q, enc, Cid, 30_000, 20, seed=1)
    assert abs(r["rel_bias"]) < 0.05 and r["rel_rmse"] < 0.1
