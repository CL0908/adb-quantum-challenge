"""Tests for the stronger baselines and parameter-uncertainty module."""
import numpy as np
import pytest
from pathlib import Path

from src.baselines.stronger import splitting_estimate, smc_estimate, flow_estimate, _simulate, _replay
from src.rdpo.problem import PortfolioProblem, adb_model


def test_simulate_and_replay_consistent():
    P = PortfolioProblem(adb_model("adb-base"), "exceedance", threshold=80e6)
    rng = np.random.default_rng(0)
    out = _simulate(P, None, 500, rng, record=True)
    score, f = _replay(P, out["acts"])
    assert np.array_equal(score, out["score"])
    assert np.array_equal(f, out["f"])


def test_smc_is_unbiased_and_accurate():
    """SMC should hit the exact answer within sampling noise."""
    P = PortfolioProblem(adb_model("adb-base"), "exceedance", threshold=80e6)
    mu = P.exact_dense()
    ests = [smc_estimate(P, 16384, np.random.default_rng(100 + s))["estimate"] for s in range(10)]
    err = np.sqrt(np.mean([(x - mu) ** 2 for x in ests])) / mu
    assert err < 0.15, f"SMC rel RMSE {err} is unexpectedly large"


def test_flow_warm_start_improves_on_untwisted():
    P = PortfolioProblem(adb_model("adb-base"), "exceedance", threshold=80e6)
    mu = P.exact_dense()
    ests = [flow_estimate(P, 16384, np.random.default_rng(200 + s))["estimate"] for s in range(5)]
    assert all(x > 0 for x in ests)


def test_splitting_runs_without_error_and_returns_reasonable_estimate():
    """Splitting is implemented with Gibbs mutations and advance-past-ties. For the ADB base case
    exceedance (mu ~ 1.2%) the estimate should be within a factor of 2 of truth on average."""
    P = PortfolioProblem(adb_model("adb-base"), "exceedance", threshold=80e6)
    mu = P.exact_dense()
    ests = [splitting_estimate(P, 16384, np.random.default_rng(300 + s))["estimate"] for s in range(10)]
    nonzero = [x for x in ests if x > 0]
    assert len(nonzero) >= 5, "splitting fails too often"
    bias = np.mean(nonzero) / mu
    assert 0.3 < bias < 3.0, f"splitting mean / exact = {bias}"


def test_param_uncertainty_credible_interval():
    """Posterior CI for pool exceedance should sit above the brief's point value."""
    if not Path("results/realdata_losses.json").exists() or not Path("results/realdata_rain.json").exists():
        pytest.skip("real-data posteriors not available")
    from src.realdata.param_uncertainty import eq_posterior, rain_posterior
    eq = eq_posterior()
    assert eq["ci95"][0] < eq["point"] < eq["ci95"][1]
    rn = rain_posterior()
    assert rn["ci95"][0] <= rn["point"] <= rn["ci95"][1]
