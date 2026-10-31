"""Annual joint distribution: 25-state EQ CTMC (expm) x 4-outcome rain -> 100 states.

State indexing: s = ((a*5 + b)*2 + r1)*2 + r2, a,b in 0..4, r in {0,1}.
"""
import numpy as np
from scipy.linalg import expm

from .poisson_fit import TAILS, P_TJ_GIVEN_KG, P_KG_GIVEN_TJ, compound_poisson_rates
from .scenarios import Scenario

N_EQ = 5
N_STATES = 100
STATES = np.array([(a, b, r1, r2) for a in range(5) for b in range(5)
                   for r1 in range(2) for r2 in range(2)])


def encode(a, b, r1, r2):
    return ((a * 5 + b) * 2 + r1) * 2 + r2


def eq_generator(sc: Scenario):
    lam1, L1 = compound_poisson_rates(TAILS["KG"])
    lam2, L2 = compound_poisson_rates(TAILS["TJ"])
    w1, w2 = lam1 / L1, lam2 / L2
    c = sc.m * 0.5 * (P_TJ_GIVEN_KG * L1 + P_KG_GIVEN_TJ * L2)
    G = np.zeros((25, 25))
    for a in range(5):
        for b in range(5):
            i = a * 5 + b
            for k in range(1, 5):
                G[i, min(a + k, 4) * 5 + b] += (L1 - c) * w1[k - 1]
                G[i, a * 5 + min(b + k, 4)] += (L2 - c) * w2[k - 1]
                for j in range(1, 5):
                    G[i, min(a + k, 4) * 5 + min(b + j, 4)] += c * w1[k - 1] * w2[j - 1]
            G[i, i] -= G[i].sum()  # diagonal (self-loops at cap cancel correctly)
    return G, c


def eq_annual(sc: Scenario):
    G, _ = eq_generator(sc)
    return expm(G)[0].reshape(5, 5)  # P[a, b]


def rain_annual(sc: Scenario):
    q = 0.02
    mu = -np.log(1 - q) * sc.rain_mult
    mu1 = mu2 = mu
    cr = sc.rain_shared_share * min(mu1, mu2)
    P00 = np.exp(-(mu1 + mu2 - cr))
    P10 = np.exp(-mu2) - P00
    P01 = np.exp(-mu1) - P00
    P11 = 1 - P00 - P10 - P01
    return np.array([[P00, P01], [P10, P11]])  # R[r1, r2]


def annual_distribution(sc: Scenario):
    E = eq_annual(sc)
    R = rain_annual(sc)
    return np.einsum("ab,rs->abrs", E, R).reshape(-1)
