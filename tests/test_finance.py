"""Tests for the finance analyses (src/finance)."""
import numpy as np
import pytest

from src.finance import analysis as A
from src.model.scenarios import REFERENCE


@pytest.fixture(scope="module")
def P():
    return A.term_law()


def test_anchor_expected_loss():
    a = A.anchor()["computed"]
    assert a["KG"]["annual_EL_fraction_of_eq_sublimit"] == pytest.approx(0.02297, abs=5e-6)
    assert a["TJ"]["annual_EL_fraction_of_eq_sublimit"] == pytest.approx(0.02011, abs=5e-6)
    r = A.anchor()["ratios_computed_over_cited"]
    assert 2.5 < r["KG"]["risk_margin_over_cited_EL"] < 2.7 and 2.9 < r["TJ"]["risk_margin_over_cited_EL"] < 3.1


def test_brief_values_match_paper(P):
    m = A.metrics(P)
    ref = REFERENCE["adb-base"]
    assert m["expected_loss"] == pytest.approx(ref[0], rel=1e-3)
    assert m["prob_exceed"] == pytest.approx(ref[2], rel=1e-2)
    assert m["shortfall"] == pytest.approx(ref[3], rel=1e-2)
    assert m["joint_exhaustion"] == pytest.approx(ref[4], rel=0.05)


def test_delivered_fraction_brief(P):
    assert abs(A.metrics(P)["delivered_fraction"] - 0.972) < 0.005
    d = [A.metrics(P, x)["delivered_fraction"] for x in (80e6, 100e6, 120e6)]
    assert d[0] < d[1] < d[2] <= 1


def test_lp_bounds_contain_independent_and_points(P):
    m_d, s_d, _ = A.data_setting()
    pts = {"brief": P, "data": A.term_law(m_d, s_d)}
    lp = A.coupling_bounds(P, pts)
    for name, t in lp["targets"].items():
        lo, hi = t["full_coupling"]["min"], t["full_coupling"]["max"]
        eps = 1e-9 * max(1.0, abs(hi))
        assert lo - eps <= t["independent_countries"] <= hi + eps
        for p in t["points"].values():
            assert lo - eps <= p["value"] <= hi + eps
        for k in ("eq_coupled_rain_independent", "rain_coupled_eq_independent"):
            assert lo - eps <= t[k]["min"] <= t[k]["max"] <= hi + eps
    # marginals identical across settings, so the same LP applies to all
    assert np.allclose(pts["data"].sum(1), P.sum(1)) and np.allclose(pts["data"].sum(0), P.sum(0))


def test_lp_extreme_known():
    t = A.coupling_bounds(A.term_law(), {"brief": A.term_law()})["targets"]
    assert t["joint_exhaustion"]["full_coupling"]["max"] == pytest.approx(
        min(A.term_law().sum(1)[9], A.term_law().sum(0)[9]), rel=1e-6)
