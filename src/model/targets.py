"""Loss mapping and the five focus targets as functions of the (capped) term state."""
import numpy as np

from .annual import STATES
from .scenarios import Scenario, TARGETS


def losses(sc: Scenario, states=STATES):
    a, b, r1, r2 = states.T
    L1 = sc.eq_limit * a / 4 + sc.rain_limit * r1
    L2 = sc.eq_limit * b / 4 + sc.rain_limit * r2
    return L1, L2


def target_functions(sc: Scenario, states=STATES):
    """Dict target -> f(state) vector over the 100 term states."""
    L1, L2 = losses(sc, states)
    L = L1 + L2
    full = sc.eq_limit + sc.rain_limit
    return {
        TARGETS[0]: L,
        TARGETS[1]: ((L1 > 0) & (L2 > 0)).astype(float),
        TARGETS[2]: (L > sc.threshold).astype(float),
        TARGETS[3]: np.maximum(L - sc.threshold, 0.0),
        TARGETS[4]: ((np.isclose(L1, full)) & (np.isclose(L2, full))).astype(float),
    }


def exact_targets(sc: Scenario, term_dist):
    return {k: float(term_dist @ f) for k, f in target_functions(sc).items()}
