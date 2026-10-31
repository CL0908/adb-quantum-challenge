"""Correctness tests for Phases 1-5 building blocks."""
import numpy as np
import pytest
from dataclasses import replace
from scipy.linalg import expm

from src.baselines.common import make_problem
from src.baselines import estimators as E
from src.model.scenarios import SCENARIOS, TARGETS
from src.model.annual import eq_generator, rain_annual, annual_distribution
from src.model.term import term_distribution

PROB = make_problem("adb-base")
EXH = "term_both_bonds_exhausted_probability"


def test_term_equals_three_year_ctmc():
    """Capped addition commutes => 3-year convolution == expm(3G) (rain at rate 3mu)."""
    for sc in SCENARIOS.values():
        G, _ = eq_generator(sc)
        d = np.einsum("ab,rs->abrs", expm(3 * G)[0].reshape(5, 5),
                      rain_annual(replace(sc, rain_mult=3 * sc.rain_mult))).ravel()
        np.testing.assert_allclose(d, term_distribution(annual_distribution(sc)), atol=1e-15)


@pytest.mark.parametrize("t", TARGETS)
def test_zero_variance_is_exact(t):
    rng = np.random.default_rng(0)
    est = E.zero_variance_single(PROB, t, 500, rng)
    assert est == pytest.approx(PROB.exact[t], rel=1e-10)


@pytest.mark.parametrize("est", [E.mc, E.sobol, E.stratified, E.importance])
def test_baselines_unbiased_on_expected_loss(est):
    rng = np.random.default_rng(1)
    x = np.array([est(PROB, 4096, rng)["term_expected_principal_loss_usd"] for _ in range(60)])
    mu = PROB.exact["term_expected_principal_loss_usd"]
    assert abs(x.mean() - mu) < 4 * x.std(ddof=1) / np.sqrt(len(x))


def test_learned_doob_unbiased_and_beats_mc():
    from src.learned.cross_entropy import fit
    prop, _ = fit(PROB, EXH, SCENARIOS["adb-base"], kind="pairwise", seed=3)
    rng = np.random.default_rng(4)
    x = np.array([prop.estimate(PROB.f[EXH], 2048, rng) for _ in range(40)])
    mu = PROB.exact[EXH]
    assert abs(x.mean() - mu) < 4 * x.std(ddof=1) / np.sqrt(len(x))
    assert x.std() / mu < 0.1  # plain MC at N=2048 has rel. std ~ 3.9


def test_iid_variance_formula_matches_sampling():
    from src.quantum.qcbm import iid_is_variance
    q = E.is_proposal(PROB)
    f = PROB.f["term_portfolio_loss_exceedance_probability"]
    v, mu = iid_is_variance(PROB.p, f, q)
    rng = np.random.default_rng(5)
    x = np.array([E.importance(PROB, 2000, rng, q)["term_portfolio_loss_exceedance_probability"] for _ in range(300)])
    assert x.var(ddof=1) == pytest.approx(v / 2000, rel=0.25)


def test_qcbm_parameter_shift_gradient():
    from src.quantum.qcbm import QCBM
    m = QCBM(2)
    th = np.random.default_rng(0).normal(0, 1, m.n_params)
    G = m.grad_full(th)
    h = 1e-6
    for k in [0, 5, 13]:
        e = np.zeros_like(th); e[k] = h
        fd = (m.probs_full(th + e) - m.probs_full(th - e)) / (2 * h)
        np.testing.assert_allclose(G[k], fd, atol=1e-7)


def test_iqae_accuracy():
    from src.quantum.iqae import iqae
    rng = np.random.default_rng(6)
    a = 0.05
    est = np.array([iqae(a, 1e-3, rng=rng)[0] for _ in range(40)])
    assert np.mean(np.abs(est - a) <= 1e-3) >= 0.9


def test_mps_reproduces_adb_table5():
    from src.tensor.synthetic import SyntheticModel, topology
    from src.tensor.mps import build_mps, targets_from_mps
    mdl = SyntheticModel(2, topology("chain", 2))
    psi, _ = build_mps(mdl, tol=1e-12)
    r = targets_from_mps(mdl, psi, 80e6)
    ex = PROB.exact
    assert r["expected_loss"] == pytest.approx(ex[TARGETS[0]], rel=1e-10)
    assert r["prob_at_least_two_pay"] == pytest.approx(ex[TARGETS[1]], rel=1e-10)
    assert r["exceedance_prob"] == pytest.approx(ex[TARGETS[2]], rel=1e-10)
    assert r["stop_loss"] == pytest.approx(ex[TARGETS[3]], rel=1e-10)
    assert r["all_exhausted_prob"] == pytest.approx(ex[TARGETS[4]], rel=1e-10)


@pytest.mark.parametrize("kind", ["chain", "ring", "complete"])
def test_tilted_mps_matches_dense(kind):
    from src.tensor.synthetic import SyntheticModel, topology, dense_term_distribution
    from src.tensor.mps import build_mps, targets_from_mps
    from src.tensor.scaling import dense_targets
    mdl = SyntheticModel(4, topology(kind, 4))
    d = dense_targets(mdl, dense_term_distribution(mdl), 160e6)
    psi, _ = build_mps(mdl, tol=1e-12, tilt=True)
    m = targets_from_mps(mdl, psi, 160e6)
    for k in d:
        assert m[k] == pytest.approx(d[k], rel=1e-9), k


def test_structured_loading_exact_adb():
    """Poisson-source circuit A reproduces ADB targets exactly (cap makes increments finite)."""
    from src.tensor.synthetic import SyntheticModel, topology, dense_term_distribution
    from src.quantum.structured_loading import simulate_A, payoff_library
    m = SyntheticModel(2, topology("chain", 2))
    lib, _ = payoff_library(m, 80e6)
    r = simulate_A(m, lib["all_exhausted"])
    assert r["norm"] == pytest.approx(1.0, abs=1e-12)
    assert np.abs(r["joint"] - dense_term_distribution(m)).max() < 1e-14
    assert r["a"] == pytest.approx(PROB.exact[EXH], rel=1e-10)
    assert simulate_A(m, lib["exceedance"])["a"] == pytest.approx(
        PROB.exact["term_portfolio_loss_exceedance_probability"], rel=1e-10)


def test_structured_loading_n3_matches_dense():
    from src.tensor.synthetic import SyntheticModel, topology, dense_term_distribution
    from src.quantum.structured_loading import simulate_A, payoff_library
    m = SyntheticModel(3, topology("chain", 3))
    lib, _ = payoff_library(m, 120e6)
    r = simulate_A(m, lib["at_least_two_pay"])
    assert np.abs(r["joint"] - dense_term_distribution(m)).max() < 1e-14


def test_vqc_encoding_is_informative():
    """Re-uploading with 1/L-scaled angles keeps a and r linearly recoverable at θ=0."""
    from src.hybrid.vqc_policy import quantum_features
    from src.model.annual import STATES
    for L in (1, 2, 3):
        F = quantum_features(np.zeros(4 * L), L)
        X = np.c_[F, np.ones(100)]
        for col in STATES.T:
            c = np.linalg.lstsq(X, col, rcond=None)[0]
            r2 = 1 - np.sum((X @ c - col) ** 2) / np.sum((col - col.mean()) ** 2)
            assert r2 > 0.85


def test_renyi2_refinement_rescues_failed_ce_fit():
    from src.learned.cross_entropy import fit
    from src.learned.renyi import renyi2_refine
    prop, _ = fit(PROB, EXH, SCENARIOS["adb-base"], kind="pairwise", seed=20261142)  # known CE failure (learned.json: adb-base, pairwise)
    from src.learned.renyi import exact_rel_std
    before = exact_rel_std(prop, PROB.f[EXH])        # exact: heavy tails are not missed
    prop, _ = renyi2_refine(prop, PROB.f[EXH], seed=1)
    after = exact_rel_std(prop, PROB.f[EXH])
    assert after < before / 3
