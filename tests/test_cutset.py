"""Cutset conditioning: exact contraction, full enumeration, unbiased sampling, rigorous enumeration bounds."""
import numpy as np
import pytest

from src.rdpo.problem import PortfolioProblem
from src.tensor.cutset import CutsetSolver, split_edges
from src.tensor.junction import dense_topology, junction_solve
from src.tensor.synthetic import SyntheticModel, topology


def _complete(n):
    m = SyntheticModel(n, topology("complete", n))
    return m, split_edges(m.edges, lambda i, j: j - i <= 1)


def test_empty_cutset_equals_junction():
    m = SyntheticModel(8, dense_topology("band2", 8))
    P = PortfolioProblem(m, "exceedance")
    ref = junction_solve(P)
    S = CutsetSolver(P, [])
    ge, gs = S.g(np.zeros(8, int))
    assert ge == pytest.approx(ref["exceedance"], rel=1e-12)
    assert gs == pytest.approx(ref["stop_loss"], rel=1e-12)


def test_full_enumeration_equals_dense():
    m, (R, D) = _complete(4)
    P = PortfolioProblem(m, "exceedance")
    S = CutsetSolver(P, D)
    e = S.enumerate_best_first(max_configs=10 ** 7, check_every=10 ** 7)
    assert e["mass_enumerated"] == pytest.approx(1.0, abs=1e-12)
    assert e["exceedance_lower"] == pytest.approx(P.exact_dense(), rel=1e-10)
    assert e["stop_loss_lower"] == pytest.approx(PortfolioProblem(m, "stop_loss").exact_dense(), rel=1e-10)


def test_sampler_is_unbiased_complete5():
    m, (R, D) = _complete(5)
    P = PortfolioProblem(m, "exceedance")
    S = CutsetSolver(P, D)
    ex = P.exact_dense()
    o = S.sampled(np.random.default_rng(3), n_samples=20000)
    for t, ref in (("exceedance", ex), ("stop_loss", PortfolioProblem(m, "stop_loss").exact_dense())):
        z = (o[t]["estimate"] - ref) / o[t]["se"]
        assert abs(z) < 4.0
    # importance weights integrate to one
    _, w = S.sample_configs(200000, np.random.default_rng(1), 0.4, 0.1)
    assert w.mean() == pytest.approx(1.0, abs=0.02)


def test_truncated_enumeration_bounds_contain_exact():
    m = SyntheticModel(6, dense_topology("band4", 6))
    R, D = split_edges(m.edges, lambda i, j: j - i <= 1)
    for t in ("exceedance", "stop_loss"):
        P = PortfolioProblem(m, t)
        ref = P.exact_dense()
        S = CutsetSolver(P, D)
        e = S.enumerate_best_first(max_configs=1500, check_every=500)
        lo = e["exceedance_lower"] if t == "exceedance" else e["stop_loss_lower"]
        rem = e["remainder_bound_exceedance"] if t == "exceedance" else e["remainder_bound_stop_loss"]
        assert lo <= ref * (1 + 1e-12) and ref <= lo + rem
