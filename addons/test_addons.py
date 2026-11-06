"""Minimal checks for the add-ons: run with  ../.venv/Scripts/python.exe -m pytest -q test_addons.py"""
import numpy as np

import ae
import joint_k
import rain_lag
import rates_unc
import risk_measures
import variant
import measure
import psis
import rotations
import router


def test_rotations_reproduce_friend_and_reduce():
    rotations.selftest()


def test_router_exact_branches():
    router.selftest()


def test_measure_sampler_matches_exact_twisted_A():
    measure.selftest()


def test_psis_recovers_tail_shape():
    psis.selftest()


def test_chebae_odd_is_cheaper_and_covers():
    a, er, g = 0.01, 0.1, np.random.default_rng(0)
    res = np.array([ae.chebae(a, er * a, 0.05, 100, g, odd_only=True) for _ in range(60)], float)
    assert np.mean(abs(res[:, 0] - a) <= er * a) >= 0.9
    assert res[:, 1].mean() * er * np.sqrt(a) < 40          # friend's IQAE is ~60-140 here


def test_variant_reproduces_friend_annual_model():
    variant.selftest()


def test_rain_lag_reproduces_friend_shared_counts():
    rain_lag.selftest()


def test_joint_k_pairwise_fit_on_pairwise_data():
    joint_k.selftest()


def test_rates_unc_variant_equals_friend_targets():
    rates_unc.selftest()


def test_risk_measures_mps_equals_dense_and_euler_adds_up():
    risk_measures.selftest()
