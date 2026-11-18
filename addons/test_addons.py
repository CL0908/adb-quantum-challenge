"""Minimal checks for the add-ons: run with  ../.venv/Scripts/python.exe -m pytest -q test_addons.py"""
import numpy as np

import ae
import climate_cc
import rates_unc
import rotations
import variant


def test_rotations_reproduce_friend_and_reduce():
    rotations.selftest()


def test_chebae_odd_is_cheaper_and_covers():
    a, er, g = 0.01, 0.1, np.random.default_rng(0)
    res = np.array([ae.chebae(a, er * a, 0.05, 100, g, odd_only=True) for _ in range(60)], float)
    assert np.mean(abs(res[:, 0] - a) <= er * a) >= 0.9
    assert res[:, 1].mean() * er * np.sqrt(a) < 40          # friend's IQAE is ~60-140 here


def test_variant_reproduces_friend_annual_model():
    variant.selftest()


def test_rates_unc_variant_equals_friend_targets():
    rates_unc.selftest()


def test_climate_cc_zero_warming_equals_friend():
    climate_cc.selftest()


def test_sensitivity_semigroup_derivative_and_exact():
    import sensitivity
    sensitivity.selftest()


def test_voi_sobol_preposterior_and_brief_point():
    import voi
    voi.selftest()


def test_model_risk_reproduces_friend_carec():
    import model_risk
    model_risk.selftest()


def test_nested_uq_draws_and_brief_point():
    import nested_uq
    nested_uq.selftest()




def test_twin_estimator_checks():
    import twin
    twin.selftest_fast()


def test_spread_band_curves():
    import spread_band
    spread_band.selftest()


def test_event_response_fit_recovers_simulated_model():
    import event_response
    event_response.selftest()


def test_live_mark_matches_friend_term_and_markov():
    import live_mark
    live_mark.selftest()


def test_desk_default_state_and_attribution():
    import desk
    desk.selftest()


def test_crisis_chain_gradient_and_second_moment():
    import crisis
    crisis.selftest()


def test_learner_one_world_and_belief_direction():
    import learner
    learner.selftest()


def test_twinstar_alarm_levels_and_gamma_tilt():
    import twinstar
    twinstar.selftest()
