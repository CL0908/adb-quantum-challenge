"""Shared helpers: path simulation, problem bundle, RMSE harness.

A "query" = one sampled three-year path (three annual draws). All estimators
receive the same path budget N.
"""
from dataclasses import dataclass, field

import numpy as np

from src.model.annual import annual_distribution
from src.model.scenarios import SCENARIOS, TARGETS
from src.model.term import ADD, term_distribution, YEARS
from src.model.targets import target_functions, exact_targets


@dataclass
class Problem:
    name: str
    p: np.ndarray                 # annual distribution (100,)
    f: dict                       # target -> f(term state) (100,)
    exact: dict
    cdf: np.ndarray = field(init=False)

    def __post_init__(self):
        self.cdf = np.cumsum(self.p)
        self.cdf[-1] = 1.0


def make_problem(name):
    sc = SCENARIOS[name]
    p = annual_distribution(sc)
    return Problem(name, p, target_functions(sc), exact_targets(sc, term_distribution(p)))


def inv_cdf(cdf, u):
    return np.minimum(np.searchsorted(cdf, u, side="right"), len(cdf) - 1)


def terminal_states(Y):
    """Y: (N, T) annual state indices -> capped term state indices (N,)."""
    s = np.zeros(Y.shape[0], dtype=np.int64)
    for t in range(Y.shape[1]):
        s = ADD[s, Y[:, t]]
    return s


def rmse_study(estimator, prob, budgets, reps, seed, targets=TARGETS):
    """Returns {target: {N: {"rmse":..,"rel_rmse":..,"bias":..}}}.

    estimator(prob, N, rng) -> dict target -> estimate (all targets share paths).
    """
    rng = np.random.default_rng(seed)
    out = {t: {} for t in targets}
    for N in budgets:
        ests = {t: [] for t in targets}
        for _ in range(reps):
            e = estimator(prob, N, rng)
            for t in targets:
                ests[t].append(e[t])
        for t in targets:
            x = np.array(ests[t]); mu = prob.exact[t]
            rmse = float(np.sqrt(np.mean((x - mu) ** 2)))
            out[t][int(N)] = {"rmse": rmse, "rel_rmse": rmse / abs(mu), "bias": float(x.mean() - mu),
                              "frac_zero": float(np.mean(x == 0))}
    return out
