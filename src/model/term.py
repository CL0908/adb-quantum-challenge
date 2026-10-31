"""Three-year capped convolution of annual states."""
import numpy as np

from .annual import STATES, N_STATES, encode

YEARS = 3


def _build_add_table():
    T = np.empty((N_STATES, N_STATES), dtype=np.int64)
    for s, (a, b, r1, r2) in enumerate(STATES):
        for y, (da, db, d1, d2) in enumerate(STATES):
            T[s, y] = encode(min(a + da, 4), min(b + db, 4), min(r1 + d1, 1), min(r2 + d2, 1))
    return T


ADD = _build_add_table()  # ADD[s, y] = s (+) y (capped)


def step(dist, p):
    """One year of convolution: dist over cumulative states -> next dist."""
    out = np.zeros(N_STATES)
    np.add.at(out, ADD, dist[:, None] * p[None, :])
    return out


def term_distribution(p, years=YEARS):
    dist = np.zeros(N_STATES)
    dist[0] = 1.0
    for _ in range(years):
        dist = step(dist, p)
    return dist
