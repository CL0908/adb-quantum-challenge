"""Tail probabilities -> compound Poisson rates (brief eq. 2-3)."""
import numpy as np

# Annual EQ payout exceedance probabilities: >=25%, >=50%, >=75%, 100%
TAILS = {
    "KG": np.array([0.03883, 0.02331, 0.01790, 0.01184]),
    "TJ": np.array([0.03925, 0.01743, 0.01332, 0.01043]),
}
P_TJ_GIVEN_KG = 0.075
P_KG_GIVEN_TJ = 0.074


def compound_poisson_rates(t):
    """Return (lam[1..4] as array of length 4, Lambda)."""
    t1, t2, t3, t4 = t
    p0, p1, p2, p3 = 1 - t1, t1 - t2, t2 - t3, t3 - t4
    Lam = -np.log(p0)
    l1 = p1 / p0
    l2 = p2 / p0 - l1 ** 2 / 2
    l3 = p3 / p0 - l1 * l2 - l1 ** 3 / 6
    l4 = Lam - l1 - l2 - l3
    return np.array([l1, l2, l3, l4]), Lam
