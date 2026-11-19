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


def test_ceiling_split_rates_and_twist_family():
    import ceiling
    ceiling.selftest()


def test_rdpo_gap_enumeration_matches_chain_and_sampler():
    import rdpo_gap
    rdpo_gap.selftest()


def test_rdpo_engine_same_paths_as_friend_rollout():
    import rdpo_engine
    rdpo_engine.selftest()


def test_dreamwake_untrained_policy_and_scores():
    import dreamwake
    dreamwake.selftest()


def test_rdpo_hybrid_unbiased_for_any_fixed_subset():
    import rdpo_hybrid
    rdpo_hybrid.selftest()


def test_small_world_teacher_unbiased_and_bounds_exact():
    import small_world
    small_world.selftest()


def test_small_world_loop_training_runs_on_taught_policy():
    import small_world_loop
    small_world_loop.selftest()


def test_llm_gate_number_check_without_calling_a_model():
    import llm_gate
    llm_gate.selftest()


def test_multi_country_teacher_matches_exact_three_country_ring():
    import multi_country
    multi_country.selftest()


def test_cap_modules_own_answer_exact_on_two_countries():
    import cap_modules
    cap_modules.selftest()


def test_policy_library_borrowed_modules_unbiased():
    import policy_library
    policy_library.selftest()


def test_dense_pools_modules_exact_on_complete_graph_of_four():
    import dense_pools
    dense_pools.selftest()
