"""Exact junction/frontier solver: agreement with dense enumeration (n <= 6) and the tilted MPS."""
import numpy as np
import pytest

from src.rdpo.problem import PortfolioProblem
from src.tensor.synthetic import SyntheticModel
from src.tensor.junction import (junction_solve, pair_law, dense_topology, band_edges, ladder_edges,
                                 treewidth_minfill, frontier_width, best_order)
from src.tensor.mps import build_mps, targets_from_mps


@pytest.mark.parametrize("kind,n", [("band2", 5), ("band3", 6), ("ladder-xdiag", 6), ("ladder", 6),
                                    ("ring", 6), ("chain", 5), ("complete", 5), ("star", 5)])
def test_matches_dense_enumeration(kind, n):
    m = SyntheticModel(n, dense_topology(kind, n))
    for target in ("exceedance", "stop_loss"):
        P = PortfolioProblem(m, target)
        r = junction_solve(P)
        assert abs(r["total_mass"] - 1) < 1e-12
        assert r[target] == pytest.approx(P.exact_dense(), rel=1e-11)


def test_other_threshold_and_order_independence():
    m = SyntheticModel(6, band_edges(6, 2))
    P = PortfolioProblem(m, "exceedance", threshold=100e6)
    base = junction_solve(P)
    assert base["exceedance"] == pytest.approx(P.exact_dense(), rel=1e-11)
    rev = junction_solve(P, order=[5, 3, 1, 0, 2, 4])
    assert rev["exceedance"] == pytest.approx(base["exceedance"], rel=1e-11)
    assert rev["stop_loss"] == pytest.approx(base["stop_loss"], rel=1e-11)


@pytest.mark.parametrize("kind,n", [("ring", 12), ("chain", 14)])
def test_matches_tilted_mps_in_the_tail(kind, n):
    m = SyntheticModel(n, dense_topology(kind, n))
    P = PortfolioProblem(m, "exceedance")
    r = junction_solve(P)
    psi, _ = build_mps(m, 1e-12, True)
    g = targets_from_mps(m, psi, P.T * 2.5e6)
    assert r["exceedance"] < 1e-6
    assert r["exceedance"] == pytest.approx(g["exceedance_prob"], rel=1e-9)
    assert r["stop_loss"] == pytest.approx(g["stop_loss"], rel=1e-9)


def test_pair_law_matches_dense_and_exact_pair():
    m = SyntheticModel(6, band_edges(6, 2))
    P = PortfolioProblem(m, "joint_exhaustion", pair=(1, 3))
    d = pair_law(P)
    assert d.sum() == pytest.approx(1.0, rel=1e-12)
    assert d[9, 9] == pytest.approx(P.exact_dense(), rel=1e-11)
    assert d[9, 9] == pytest.approx(P.exact_pair(), rel=1e-11)
    assert pair_law(P, (3, 1))[9, 9] == pytest.approx(d[9, 9], rel=1e-12)
    # full pair law against the dense tensor marginal
    from src.tensor.synthetic import dense_term_distribution
    D = dense_term_distribution(m)
    marg = D.sum(axis=(0, 2, 4, 5))
    np.testing.assert_allclose(d, marg, rtol=1e-10, atol=1e-18)


def test_widths():
    for k in (2, 3):
        e = band_edges(10, k)
        assert treewidth_minfill(10, e) == k
        assert frontier_width(10, e, best_order(10, e)) == k
    assert treewidth_minfill(8, ladder_edges(8, 2)) == 3
