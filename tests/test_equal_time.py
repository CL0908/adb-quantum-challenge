import numpy as np

from src.rdpo import experiments_equal_time as eq


def test_fit_and_equal_time_budget():
    a, b = eq.fit_linear(1000, 0.0019, 101000, 0.0919)       # t = 0.001 + 9e-7 N
    assert abs(a - 0.001) < 1e-9 and abs(b - 9e-7) < 1e-12
    n, capped = eq.n_eq_from(0.1, a, b, mult=8)
    assert n % 8 == 0 and not capped and abs(a + b * n - 0.1) < 1e-5
    n, capped = eq.n_eq_from(1e6, a, b)
    assert n == eq.CAP and capped


def test_block_summary_counts():
    rows = [{"target": "x", "rmse_paths": {"rdpo": 0.1, "mc": 0.5, "smc": 0.05},
             "rmse_time": {"rdpo": 0.1, "mc": 0.05, "smc": 0.3}},
            {"target": "x", "rmse_paths": {"rdpo": 0.1, "mc": 0.2, "smc": 0.3},
             "rmse_time": {"rdpo": 0.1, "mc": 0.2, "smc": 0.3}}]
    s = eq.block_summary(rows, ["mc"])["x"]
    assert s["paths"]["rdpo_wins_vs_brief_methods"] == 2 and s["paths"]["rdpo_wins_vs_all_baselines"] == 1
    assert s["time"]["rdpo_wins_vs_brief_methods"] == 1 and s["time"]["rdpo_wins_vs_all_baselines"] == 1
    assert s["time"]["most_accurate_method_counts"] == {"mc": 1, "rdpo": 1}


def test_rel_rmse_and_cells():
    assert abs(eq.rel_rmse([1.1, 0.9], 1.0) - 0.1) < 1e-12
    assert len(eq.CELLS_A) == 25 and len(eq.CELLS_B) == 9
