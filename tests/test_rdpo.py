"""Tests for the final RDPO package (src/rdpo)."""
import numpy as np
import pytest
from scipy.optimize import check_grad

from src.baselines.common import make_problem
from src.rdpo.problem import PortfolioProblem, adb_model, ADB_SCENARIOS, TARGETS
from src.rdpo.policy import RDPOPolicy
from src.rdpo.train import rdpo_estimate, RDPOConfig
from src.tensor.synthetic import SyntheticModel, topology

KEY = {"expected_loss": "term_expected_principal_loss_usd", "joint_attachment": "term_joint_attachment_probability",
       "exceedance": "term_portfolio_loss_exceedance_probability", "stop_loss": "term_portfolio_stop_loss_usd",
       "joint_exhaustion": "term_both_bonds_exhausted_probability"}


@pytest.mark.parametrize("sc", list(ADB_SCENARIOS))
def test_source_form_reproduces_all_reference_values(sc):
    ex = make_problem(sc).exact
    for t in TARGETS:
        assert PortfolioProblem(adb_model(sc), t, threshold=80e6).exact_dense() == pytest.approx(ex[KEY[t]], rel=1e-12)


@pytest.mark.parametrize("target", ["joint_exhaustion", "exceedance"])
def test_policy_gradients(target):
    P = PortfolioProblem(SyntheticModel(4, topology("ring", 4)), target)
    pol = RDPOPolicy(P); pol.theta = np.random.default_rng(1).normal(0, .3, pol.d)
    out = pol.rollout(30, np.random.default_rng(0), record=True); c = np.ones(30) / 30
    v = pol.theta.copy()
    f = lambda x: (setattr(pol, "theta", x), pol.replay(out["acts"], c, "ce")[0])[1]
    g = lambda x: (setattr(pol, "theta", x), pol.replay(out["acts"], c, "ce")[1])[1]
    assert check_grad(f, g, v) < 1e-5


def test_rollout_is_unbiased_for_any_policy():
    P = PortfolioProblem(adb_model("adb-base"), "joint_attachment", threshold=80e6)
    mu = P.exact_dense()
    pol = RDPOPolicy(P); pol.theta = np.random.default_rng(3).normal(0, 1.0, pol.d)
    x = np.array([np.mean((o := pol.rollout(4000, np.random.default_rng(s)))["w"] * o["f"]) for s in range(30)])
    assert abs(x.mean() - mu) < 4 * x.std(ddof=1) / np.sqrt(len(x))


def test_rdpo_estimate_accuracy_and_interval():
    P = PortfolioProblem(adb_model("adb-base"), "joint_exhaustion", threshold=80e6)
    mu = P.exact_dense()
    res = [rdpo_estimate(P, 16384, np.random.default_rng(s)) for s in range(6)]
    e = np.array([r["estimate"] for r in res])
    assert np.sqrt(np.mean((e - mu) ** 2)) / mu < 0.05          # MC at this budget: ~1.4
    assert all(r["paths_used"] == 16384 for r in res)           # training is inside the budget
    assert np.mean([r["ci"][0] <= mu <= r["ci"][1] for r in res]) >= 0.5


# ---------------- RDPO v2 (roadmap WP1 / WP3) ----------------
def test_fast_engine_identical_to_v1():
    from src.rdpo.policy import RDPOPolicy
    from src.rdpo.fast import FastPolicy
    from src.tensor.synthetic import SyntheticModel, topology
    cases = [(adb_model("adb-base"), t, 80e6) for t in TARGETS] + [(SyntheticModel(6, topology("complete", 6)), "exceedance", None),
                                                                  (SyntheticModel(8, topology("ring", 8)), "joint_exhaustion", None)]
    for model, t, thr in cases:
        P = PortfolioProblem(model, t, threshold=thr)
        a, b = RDPOPolicy(P), FastPolicy(P)
        th = np.random.default_rng(1).normal(size=a.d) * 0.5
        a.theta, b.theta = th.copy(), th.copy()
        U = np.random.default_rng(2).random((2000, P.K))
        ra, rb = a.rollout(2000, None, record=True, U=U), b.rollout(2000, None, record=True, U=U)
        assert (ra["acts"] == rb["acts"]).all() and np.allclose(ra["w"], rb["w"], rtol=1e-12)
        c = np.random.default_rng(3).random(2000)
        assert np.allclose(a.replay(ra["acts"], c, "mix"), b.replay(rb["acts"], c, "mix"))
        assert np.allclose(a.replay(ra["acts"], c, "ce")[1], b.replay(rb["acts"], c, "ce")[1])


def test_spline_policy_is_valid_importance_sampler():
    from src.rdpo.fast import FastPolicy
    P = PortfolioProblem(adb_model("adb-base"), "expected_loss", threshold=80e6)
    pol = FastPolicy(P, value="spline")
    pol.theta = np.random.default_rng(0).normal(size=pol.d)
    out = pol.rollout(200000, np.random.default_rng(1))
    g = out["w"] * out["f"]
    assert abs(g.mean() / P.exact_dense() - 1) < 4 * g.std() / np.sqrt(len(g)) / P.exact_dense() + 1e-3


def test_v2_unbiased_and_t_interval():
    from src.rdpo.train import v2_config
    P = PortfolioProblem(adb_model("adb-base"), "joint_exhaustion", threshold=80e6)
    mu = P.exact_dense()
    xs = [rdpo_estimate(P, 4096, np.random.default_rng(100 + r), v2_config()) for r in range(30)]
    est = np.array([x["estimate"] for x in xs])
    assert abs(est.mean() / mu - 1) < 4 * est.std() / np.sqrt(len(est)) / mu
    assert all(np.isfinite(x["dof"]) or x["dof"] == float("inf") for x in xs)
    assert all(x["ci"][1] - x["ci"][0] >= 2 * 1.96 * x["se"] - 1e-15 for x in xs)


def test_exact_pair_reduction_matches_dense_and_mps():
    for sc in ADB_SCENARIOS:
        for t in ("joint_attachment", "joint_exhaustion"):
            P = PortfolioProblem(adb_model(sc), t, threshold=80e6)
            assert abs(P.exact_pair() / P.exact_dense() - 1) < 1e-12
    P = PortfolioProblem(SyntheticModel(6, topology("complete", 6)), "joint_exhaustion")
    assert abs(P.exact_pair() / P.exact_dense() - 1) < 1e-12


def test_stage_dependent_policy_is_unbiased_sampler():
    from src.rdpo.fast import FastPolicy
    P = PortfolioProblem(adb_model("adb-base"), "exceedance", threshold=80e6)
    pol = FastPolicy(P, stages=4)
    pol.theta = np.random.default_rng(0).normal(size=pol.d) * 0.5
    out = pol.rollout(200000, np.random.default_rng(1))
    g = out["w"] * out["f"]; mu = P.exact_dense()
    assert abs(g.mean() - mu) < 4 * g.std() / np.sqrt(len(g))


def test_realdata_check_runs_on_cached_catalogue():
    from pathlib import Path
    if not Path("data/real/usgs_m55_central_asia_1965_2016.csv").exists():
        pytest.skip("real-data cache not present")
    from src.realdata.eq_check import analyse
    out, ev, dist, polys = analyse()
    r = out["radii_km"]["50"]["5.5"]
    assert r["KG"]["events"] > 10 and r["TJ"]["events"] > 10
    assert 0.3 < r["KG"]["dispersion_index"] < 2.0
    rows = out["implied_scenarios"]["rows"]
    assert rows["brief"]["m"] == 1.0
    assert abs(rows["brief"]["joint_exhaustion"] / 3.163304630917903e-05 - 1) < 1e-10


def test_projected_doob_is_zero_variance_and_exact():
    from src.rdpo.projected import ProjectedDoob
    for model, thr in [(adb_model("adb-common-shock"), 80e6), (SyntheticModel(8, topology("ring", 8)), None)]:
        P = PortfolioProblem(model, "joint_exhaustion", threshold=thr)
        d = ProjectedDoob(P)
        assert abs(d.mu / P.exact_pair() - 1) < 1e-12
        o = d.rollout(500, np.random.default_rng(0))
        g = o["w"] * o["f"]
        assert np.all(o["f"] == 1) and np.max(np.abs(g / d.mu - 1)) < 1e-10
