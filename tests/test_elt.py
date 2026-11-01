"""Tests for the ELT/YLT converter (src/elt)."""
import csv

import numpy as np

from src.elt.convert import Event, ELTModel, read_elt, read_ylt
from src.rdpo.problem import PortfolioProblem
from src.rdpo.train import rdpo_estimate
from src.rdpo.experiments_v2 import final_config

EVENTS = [Event(0.03, {0: (1, 0)}), Event(0.01, {0: (4, 0)}), Event(0.02, {1: (2, 0)}),
          Event(0.005, {0: (2, 0), 1: (1, 0), 2: (3, 0)}), Event(0.02, {2: (0, 1)}), Event(0.004, {1: (0, 1), 2: (0, 1)}),
          Event(0.02, {0: (0, 1)}), Event(0.015, {1: (4, 1)})]


def _bruteforce(events, n, N=400000, seed=0, years=3):
    rng = np.random.default_rng(seed)
    a = np.zeros((N, n), int); r = np.zeros((N, n), int)
    for e in events:
        k = rng.poisson(e.rate * years, N)
        for c, (da, dr) in e.payout.items():
            a[:, c] = np.minimum(a[:, c] + k * da, 4); r[:, c] = np.minimum(r[:, c] + (k > 0) * dr, 1)
    return a * 17.5 + r * 10.0


def test_term_law_matches_bruteforce_simulation():
    m = ELTModel(3, EVENTS)
    L = _bruteforce(EVENTS, 3)
    for thr in (40, 60, 90):
        mu = PortfolioProblem(m, "exceedance", threshold=thr * 1e6).exact_dense()
        emp = (L.sum(1) > thr).mean()
        assert abs(mu - emp) < 5 * np.sqrt(emp * (1 - emp) / len(L)) + 1e-6
    el = PortfolioProblem(m, "expected_loss").exact_dense() / 1e6
    assert abs(el - L.sum(1).mean()) < 5 * L.sum(1).std() / np.sqrt(len(L))
    ja = PortfolioProblem(m, "joint_attachment")
    assert abs(ja.exact_dense() - ja.exact_pair()) < 1e-12


def test_footprints_and_multicountry_sources():
    m = ELTModel(3, EVENTS)
    assert (0, 1, 2) in m.footprints and m.footprints[(0, 1, 2)]["events"] == 1
    assert all(abs(s["p"].sum() - 1) < 1e-12 for s in m.custom_sources)


def test_ylt_roundtrip_and_rdpo_runs(tmp_path):
    elt = tmp_path / "elt.csv"
    with open(elt, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["event_id", "rate", "country", "loss_usd", "rain"])
        for i, e in enumerate(EVENTS):
            for c, (a, r) in e.payout.items():
                w.writerow([f"E{i}", e.rate, "ABC"[c], a * 17.5e6, r])
    ev = read_elt(elt, list("ABC"))
    assert len(ev) == len(EVENTS)
    m = ELTModel(3, ev)
    mu = PortfolioProblem(m, "exceedance", threshold=60e6).exact_dense()
    # YLT simulated from the ELT recovers the rates
    rng = np.random.default_rng(1); ny = 200000
    ylt = tmp_path / "ylt.csv"
    with open(ylt, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["year", "event_id", "country", "eq_steps", "rain"])
        for i, e in enumerate(EVENTS):
            for y in np.flatnonzero(rng.poisson(e.rate, ny)):
                for c, (a, r) in e.payout.items():
                    w.writerow([y, f"E{i}", "ABC"[c], a, r])
    m2 = ELTModel(3, read_ylt(ylt, list("ABC"), ny))
    mu2 = PortfolioProblem(m2, "exceedance", threshold=60e6).exact_dense()
    assert abs(mu2 / mu - 1) < 0.05
    P = PortfolioProblem(m, "exceedance", threshold=60e6)
    x = rdpo_estimate(P, 4096, np.random.default_rng(0), final_config("exceedance"))
    assert abs(x["estimate"] / mu - 1) < 0.1
