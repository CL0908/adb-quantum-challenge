"""Phase 0 acceptance: checkpoints, Table 5 to published precision, independent cross-checks."""
import itertools
import numpy as np
import pytest

from src.model.annual import annual_distribution, eq_annual, rain_annual, STATES, eq_generator
from src.model.poisson_fit import TAILS, compound_poisson_rates
from src.model.scenarios import SCENARIOS, REFERENCE, TARGETS
from src.model.term import term_distribution, ADD
from src.model.targets import exact_targets, target_functions

BASE = SCENARIOS["adb-base"]


def test_rates_table2():
    lam, L = compound_poisson_rates(TAILS["KG"])
    np.testing.assert_allclose(np.r_[lam, L], [0.016147, 0.005498, 0.006215, 0.011743, 0.039604], atol=6e-7)
    lam, L = compound_poisson_rates(TAILS["TJ"])
    np.testing.assert_allclose(np.r_[lam, L], [0.022711, 0.004020, 0.002915, 0.010395, 0.040041], atol=6e-7)


def test_checkpoints():
    _, c = eq_generator(BASE)
    assert round(c, 7) == 0.0029667
    assert round(eq_annual(BASE)[0, 0] * 100, 2) == 92.62
    R = rain_annual(BASE)
    assert round(R[0, 0] * 100, 3) == 96.526
    assert round(R[1, 0] * 100, 3) == 1.474 and round(R[0, 1] * 100, 3) == 1.474
    assert round(R[1, 1] * 100, 3) == 0.526
    assert round(annual_distribution(BASE)[0] * 100, 2) == 89.40


# (target index, scale, decimals) matching how Table 5 prints values
FMT = [(0, 1e-6, 2), (1, 100, 3), (2, 100, 3), (3, 1, 0), (4, 100, 4)]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_table5(name):
    sc = SCENARIOS[name]
    vals = exact_targets(sc, term_distribution(annual_distribution(sc)))
    for i, scale, dec in FMT:
        got = round(vals[TARGETS[i]] * scale, dec)
        want = round(REFERENCE[name][i] * scale, dec)
        assert got == pytest.approx(want, abs=10 ** -dec / 2), (name, TARGETS[i], got, want)


@pytest.mark.parametrize("name", ["adb-base", "adb-common-shock"])
def test_bruteforce_paths(name):
    """Enumerate all 100^3 paths explicitly (independent of the step() convolution)."""
    sc = SCENARIOS[name]
    p = annual_distribution(sc)
    s1 = ADD[0][:, None, None] * np.ones((1, 100, 100), dtype=int)
    s2 = ADD[s1, np.arange(100)[None, :, None]]
    s3 = ADD[s2, np.arange(100)[None, None, :]]
    w = p[:, None, None] * p[None, :, None] * p[None, None, :]
    d = np.bincount(s3.ravel(), weights=w.ravel(), minlength=100)
    np.testing.assert_allclose(d, term_distribution(p), rtol=1e-12, atol=1e-18)


def test_event_simulation_matches_expm():
    """Simulate the raw Poisson event process (no generator / expm) and compare."""
    rng = np.random.default_rng(12345)
    lam1, L1 = compound_poisson_rates(TAILS["KG"])
    lam2, L2 = compound_poisson_rates(TAILS["TJ"])
    _, c = eq_generator(BASE)
    N = 2_000_000
    a = np.zeros(N, int); b = np.zeros(N, int)
    for rate, hit1, hit2 in [(L1 - c, 1, 0), (L2 - c, 0, 1), (c, 1, 1)]:
        n = rng.poisson(rate, N)
        for k in range(1, n.max() + 1):
            m = n >= k
            if hit1:
                a[m] += rng.choice(4, m.sum(), p=lam1 / L1) + 1
            if hit2:
                b[m] += rng.choice(4, m.sum(), p=lam2 / L2) + 1
    a = np.minimum(a, 4); b = np.minimum(b, 4)
    emp = np.bincount(a * 5 + b, minlength=25) / N
    ex = eq_annual(BASE).ravel()
    se = np.sqrt(ex * (1 - ex) / N)
    assert np.all(np.abs(emp - ex) < 5 * se + 1e-7)
