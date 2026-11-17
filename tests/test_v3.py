"""Tests for the equal-wall-clock additions: source-tilt sampler, work-normalised selector, engine speed edits."""
import numpy as np
import pytest

from src.rdpo.problem import PortfolioProblem
from src.rdpo.fast import FastPolicy
from src.rdpo.tilt import TiltSampler, tilt_estimate
from src.rdpo.select import select_estimate
from src.tensor.synthetic import SyntheticModel, topology


def small(target="exceedance", n=3):
    return PortfolioProblem(SyntheticModel(n, topology("ring", n)), target)


def _z(est, mu):
    est = np.asarray(est)
    return (est.mean() - mu) / (est.std(ddof=1) / np.sqrt(len(est)))


@pytest.mark.parametrize("target", ["exceedance", "stop_loss"])
@pytest.mark.parametrize("fit", [False, True])
def test_tilt_unbiased_on_small_exact_case(target, fit):
    P = small(target)
    mu = P.exact_dense()
    est = [tilt_estimate(P, np.random.default_rng(s), paths=60000, fit=fit)["estimate"] for s in range(14)]
    assert abs(_z(est, mu)) < 4.0
    assert abs(np.mean(est) / mu - 1) < 0.05


def test_tilt_weights_are_normalised_for_any_theta():
    P = small()
    S = TiltSampler(P)
    S.set_theta(np.random.default_rng(0).normal(0, 0.15, (S.K, S.m)))
    o = S.sample(200000, np.random.default_rng(1))
    assert abs(o["w"].mean() - 1) < 0.05            # E_q[p/q] = 1 (same support)


def test_tilt_chi2_fit_reduces_the_sample_second_moment():
    P = small(); S = TiltSampler(P)
    from src.rdpo.train import _best_twist_sources
    th0 = S.twist_theta(_best_twist_sources(P)); S.set_theta(th0)
    o = S.sample(20000, np.random.default_rng(2), True)
    pos = o["f"] > 0

    def second_moment(th):
        S.set_theta(th)
        k = np.arange(S.K)[:, None]
        logw_new = S.lr[k, o["acts"][:, pos].astype(int)].sum(0)
        return float(np.sum(o["f"][pos] ** 2 * o["w"][pos] * np.exp(logw_new)))

    th, info = S.fit_chi2(o["acts"], o["f"], o["w"], th0, kappa=1e-3, bound=None, maxiter=50)
    assert info["ok"]
    assert second_moment(th) <= second_moment(th0) * (1 + 1e-9)


def test_tilt_interval_and_budgets():
    P = small()
    x = tilt_estimate(P, np.random.default_rng(0), seconds=0.4)
    assert x["seconds"] < 0.4 * 1.5 and x["ci"][0] <= x["estimate"] <= x["ci"][1] and x["paths_used"] > 1000
    a = TiltSampler(P).amplitude_bound(0.01)
    assert 0 < a["a_lower"] <= 0.01


def test_selector_is_unbiased_and_charges_time():
    P = small()
    mu = P.exact_dense()
    outs = [select_estimate(P, 0.5, np.random.default_rng(s)) for s in range(12)]
    est = [o["estimate"] for o in outs]
    assert abs(_z(est, mu)) < 4.0
    assert all(o["seconds"] < 0.5 * 1.5 for o in outs)
    assert all(o["choice"] in ("twist", "tilt", "rdpo_cold", "rdpo_transfer") for o in outs)


def test_compile_row_unique_matches_lexicographic_reference():
    """The integer-encoded unique in FastPolicy.compile must group paths exactly like the original row-unique."""
    P = small("exceedance", 4)
    pol = FastPolicy(P, stages=4)
    rng = np.random.default_rng(0)
    pol.theta = rng.normal(0, 0.1, pol.d)
    acts = pol.rollout(1500, rng, record=True)["acts"]
    comp = pol.compile(acts)
    # reference: replay with the original np.unique(axis=0) grouping
    N = acts.shape[0]
    S, Lt, Li, Lj = pol._init_state(N)
    idx = np.arange(N); k_out = 0
    for k in range(P.K):
        v = acts[:, k]; plan = pol.plan[k]
        if not plan["frozen"]:
            ends = sorted({i for e in plan["edges"] for i in P.edges[e]} | set(plan["sites"]))
            key = np.stack([S[:, i] for i in ends] + [Lt], 1)
            _, rep, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
            assert np.array_equal(inv.reshape(-1), comp[k_out]["inv"])
            k_out += 1
        newS = {i: pol.next[k][i][S[:, i]] for i in plan["sites"]}
        dL = sum(pol.loss_tab[newS[i]][idx, v] - pol.loss_tab[S[:, i]] for i in plan["sites"])
        for i in newS:
            S[:, i] = newS[i][idx, v]
        Lt = Lt + dL


def test_compiled_objective_equals_replay():
    P = small("stop_loss", 4)
    pol = FastPolicy(P, stages=4)
    rng = np.random.default_rng(3)
    pol.theta = rng.normal(0, 0.2, pol.d)
    acts = pol.rollout(2000, rng, record=True)["acts"]
    c = np.ones(2000) / 2000
    comp = pol.compile(acts); Cs = pol.aggregate(comp, c)
    v1, g1 = pol.ce_from(comp, Cs); v2, g2 = pol.replay(acts, c, "ce")
    assert np.allclose(g1, g2, atol=1e-12) and abs(v1 - v2) < 1e-9
    assert np.allclose(pol.mix_from(comp, Cs), pol.replay(acts, c, "mix"), atol=1e-12)
