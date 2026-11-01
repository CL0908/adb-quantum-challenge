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
