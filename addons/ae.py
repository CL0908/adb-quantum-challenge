# Drafted by the QContagion session (D:/quantun_contagion/reports/ADB_addon/ae.py), reviewed and integrated here.
"""Exact (noise-free) emulators of two amplitude-estimation algorithms, drop-in for ADB's iqae().

Interface, counting and measurement model copied from src/quantum/iqae.py:iqae (CL0908/adb-quantum-challenge):
    f(a, eps, alpha=0.05, shots=..., rng=None) -> (estimate, A-operator calls, kmax)
  * a     probability of the good state, a = sin^2(theta_a)        (callers pass eps = eps_rel * a)
  * eps   absolute half-width target ON a
  * a sample at Grover power k returns 1 w.p. sin^2((2k+1) theta_a); one shot costs 2k+1 calls of A
  * kmax  largest Grover power used (degree-d Chebyshev sample counted as k = floor(d/2))

miqae  — Fukuzawa, Ho, Irani, Zion, "Modified Iterative Quantum Amplitude Estimation is Asymptotically
         Optimal", arXiv:2208.14612 (ALENEX 2023), Algorithm 1 + Algorithm 2 (FindNextK) + Table 1.
chebae — Rall & Fuller, "Amplitude Estimation from Quantum Signal Processing", arXiv:2207.08628
         (Quantum 7, 937, 2023), Section 3 "ChebAE" + find_next_cheb; cross-checked against the
         reference notebook github.com/qiskit-community/ChebAE/blob/main/chebae.ipynb.

ADAPTATIONS (both papers state their eps on a different quantity than ADB's iqae does):
  A1 MIQAE's eps is a precision on theta (loop `while theta_u - theta_l > 2 eps`). Here the loop stops on the
     probability interval, `sin^2 theta_u - sin^2 theta_l <= 2 eps`, exactly like ADB's iqae. Kmax = pi/(4 eps)
     is still a valid upper bound on every K used (while running, theta-width >= a-width > 2 eps), so the
     paper's alpha_i allocation keeps sum_i alpha_i <= alpha.
  A2 ChebAE's eps is on the AMPLITUDE x = sqrt(a). Here the loop stops on x_max^2 - x_min^2 <= 2 eps.
     The number of confidence intervals is bounded by T = ceil(log_r(pi / (2 eps))) + 1 (degree d <= pi/(2 dtheta),
     dtheta >= x-width > eps while running); the paper's T = ceil(log_r(1/(2 eps))) is in amplitude units.
     The 'late' test (paper eq. 70) compares against nu * eps_x with eps_x = eps / (x_min + x_max), the amplitude
     half-width that corresponds to eps on a for the current interval.
  A3 Query cost of a Chebyshev degree-d sample is d calls of A (odd d = 2k+1 is Q^k A, the same 2k+1 as ADB).
     The reference notebook counts floor(d/2) reflections and leaves out the final A; ADB's convention is used.
     Even d needs a QSP circuit (phases on Z_psi), not a plain Grover power; pass odd_only=True to restrict
     ChebAE to Grover-power circuits (paper: even degrees "do not improve performance much").

NOT REPRODUCIBLE / UNCERTAIN (flagged, not guessed):
  U1 MIQAE: the paper's rigorous version uses Chernoff-Hoeffding intervals (Alg. 1, lines 15-17); its Beta
     (Clopper-Pearson) variant is reported numerically but "not amenable to a rigorous analysis". ci="beta"
     uses CP at the same alpha_i — a choice the paper itself says may not be optimal.
  U2 MIQAE: the paper reports Nshots=100 in its headline plots and Nshots=1 as a large improvement (Fig. 5).
     Default here is shots=1. Its exact numerical constant is only shown graphically (no fitted formula).
  U3 MIQAE: the paper notes its own code "further handles numerical floating errors" at quadrant boundaries
     without saying how; here boundaries use a 1e-12 tolerance.
  U4 ChebAE: eps_p_max in the 'late' test is "the largest possible error ... guaranteed by Clopper-Pearson with
     Nshots flips and confidence 1-delta/T"; computed here as max over heads of the CP half-width (the
     notebook's exact routine was not read line-by-line).
  U6 MIQAE (paper) REPLACES [theta_l, theta_u] by the new round interval; here it is intersected with the old
     one (as ADB's iqae and ChebAE step 4(f) do). Both are valid CIs under the union bound; intersection can only
     narrow. Repeated looks inside a round (shots=1) are the papers' own design; coverage is checked empirically.
  U5 ChebAE's empirical claim (<QPi> ~ 1.71/eps ln(2.08 ln 1/eps), r=2, Nshots=100, nu=8) is for a=0.5 and eps
     on the amplitude; it is not a proof, and no constant is claimed for small a.
"""
from __future__ import annotations

import numpy as np
from scipy.special import betaincinv

HALF_PI = np.pi / 2
_TOL = 1e-12


def _cp(n1: int, n: int, alpha: float) -> tuple[float, float]:
    """Two-sided Clopper-Pearson interval at level 1-alpha."""
    lo = 0.0 if n1 == 0 else float(betaincinv(n1, n - n1 + 1, alpha / 2))
    hi = 1.0 if n1 == n else float(betaincinv(n1 + 1, n - n1, 1 - alpha / 2))
    return lo, hi


# ----------------------------------------------------------------------------------------------- MIQAE
def _find_next_k(k: int, th_l: float, th_u: float) -> int:
    """Algorithm 2 (FindNextK): largest odd K in [3K_i, (pi/2)/(th_u-th_l)] with K*th_l, K*th_u in one quadrant."""
    K_i = 2 * k + 1
    K = int(np.floor(HALF_PI / (th_u - th_l)))
    if K % 2 == 0:
        K -= 1
    while K >= 3 * K_i:
        if np.floor(K * th_l / HALF_PI + _TOL) == np.ceil(K * th_u / HALF_PI - _TOL) - 1:
            return (K - 1) // 2
        K -= 2
    return k


def miqae(a, eps, alpha=0.05, shots=1, rng=None, ci="chernoff", max_iter=10_000_000):
    """Modified IQAE (arXiv:2208.14612, Alg. 1). shots = N_shots per inner iteration (paper Fig. 5: 1 is best).

    Per round i: alpha_i = (2 alpha / 3) K_i / K_max, K_max = pi/(4 eps);
    N_i^max = 2 / (sin^2(pi/21) sin^2(8pi/21)) * ln(2/alpha_i);  Chernoff half-width sqrt(ln(2/alpha_i)/(2N)).
    Returns (estimate, A calls, kmax)."""
    rng = rng or np.random.default_rng()
    th = np.arcsin(np.sqrt(a))
    th_l, th_u = 0.0, HALF_PI
    K_max = np.pi / (4 * eps)
    nmax_c = 2.0 / (np.sin(np.pi / 21) ** 2 * np.sin(8 * np.pi / 21) ** 2)
    k, calls, kmax, it = 0, 0, 0, 0
    done = lambda: np.sin(th_u) ** 2 - np.sin(th_l) ** 2 <= 2 * eps  # noqa: E731  (adaptation A1)
    while not done():
        K = 2 * k + 1
        a_i = (2 * alpha / 3) * K / K_max
        n_max = int(np.ceil(nmax_c * np.log(2 / a_i)))
        R = int(np.floor(K * th_l / HALF_PI + _TOL))
        p1 = np.sin(K * th) ** 2
        N = n1 = 0
        k_old = k
        while k == k_old:
            m = min(shots, n_max - N)
            n1 += rng.binomial(m, p1)
            N += m
            calls += m * K
            kmax = max(kmax, k)
            ah = n1 / N
            if ci == "chernoff":
                e = np.sqrt(np.log(2 / a_i) / (2 * N))
                lo, hi = max(0.0, ah - e), min(1.0, ah + e)
            else:
                lo, hi = _cp(n1, N, a_i)
            if R % 2 == 0:  # Table 1
                g_lo, g_hi = np.arcsin(np.sqrt(lo)), np.arcsin(np.sqrt(hi))
            else:
                g_lo, g_hi = HALF_PI - np.arcsin(np.sqrt(hi)), HALF_PI - np.arcsin(np.sqrt(lo))
            th_l = max(th_l, (R * HALF_PI + g_lo) / K)
            th_u = min(th_u, (R * HALF_PI + g_hi) / K)
            if done():
                break
            k = _find_next_k(k, th_l, th_u)
            if N >= n_max and k == k_old:  # Lemma 3.2/3.3 say this cannot happen (with prob >= 1-alpha_i)
                n_max += int(np.ceil(nmax_c * np.log(2 / a_i)))
            it += 1
            if it > max_iter:
                raise RuntimeError("miqae: max_iter exceeded")
    est = 0.5 * (np.sin(th_l) ** 2 + np.sin(th_u) ** 2)
    return float(est), int(calls), int(kmax)


# ----------------------------------------------------------------------------------------------- ChebAE
def _cheb2(d: int, x: float) -> float:
    """|T_d(x)|^2 = cos^2(d arccos x)."""
    return float(np.cos(d * np.arccos(np.clip(x, -1.0, 1.0))) ** 2)


def _find_next_cheb(x_lo: float, x_hi: float, min_deg: int, odd: bool) -> int | None:
    """Highest d >= min_deg with cos^2(d theta) monotone on [arccos x_hi, arccos x_lo]; None if none."""
    t_lo, t_hi = np.arccos(x_hi), np.arccos(x_lo)
    if t_hi - t_lo <= 0:
        return None
    d = int(np.floor(HALF_PI / (t_hi - t_lo)))
    step = 1
    if odd:
        d -= (d + 1) % 2
        step = 2
    while d >= max(min_deg, 1):
        if np.floor(d * t_lo / HALF_PI + _TOL) == np.floor(d * t_hi / HALF_PI - _TOL):
            return d
        d -= step
    return None


def _invert(d: int, x_lo: float, x_hi: float, p_lo: float, p_hi: float) -> tuple[float, float]:
    """Map a CI on cos^2(d theta) back to x = cos theta, on an interval where it is monotone (paper Fig. 5)."""
    t_lo = np.arccos(x_hi)
    m = int(np.floor(d * t_lo / HALF_PI + _TOL))  # quadrant of d*theta
    if m % 2 == 0:   # cos^2 decreasing in gamma on even quadrants
        g = lambda p: np.arccos(np.sqrt(p))  # noqa: E731
    else:
        g = lambda p: np.arcsin(np.sqrt(p))  # noqa: E731
    xs = [np.cos((m * HALF_PI + g(p)) / d) for p in (p_lo, p_hi)]
    return float(min(xs)), float(max(xs))


def chebae(a, eps, alpha=0.05, shots=100, rng=None, r=2, nu=8, odd_only=False, max_iter=10_000_000):
    """ChebAE (arXiv:2207.08628, Sec. 3). Paper's best hyperparameters: r=2, N_shots=100, nu=8 (eq. 69).
    Returns (estimate, A calls, kmax) with estimate = midpoint of the final interval on a."""
    rng = rng or np.random.default_rng()
    x_true = np.sqrt(a)
    T = int(np.ceil(np.log(np.pi / (2 * eps)) / np.log(r))) + 1        # adaptation A2
    a_T = alpha / T
    # U4: largest CP half-width over all head counts with `shots` flips at level 1 - a_T
    eps_p_max = max((lambda lo_hi: (lo_hi[1] - lo_hi[0]) / 2)(_cp(h, shots, a_T)) for h in range(shots + 1))
    x_lo, x_hi, d = 0.0, 1.0, 1
    n1 = n = 0
    calls = kmax = it = 0
    while x_hi ** 2 - x_lo ** 2 > 2 * eps:                              # adaptation A2
        d_new = _find_next_cheb(x_lo, x_hi, r * d, odd_only)
        if d_new is not None and d_new >= r * d:
            d, n1, n = d_new, 0, 0
        gap = abs(_cheb2(d, x_hi) - _cheb2(d, x_lo))
        eps_x = eps / (x_lo + x_hi)
        late = gap > 0 and eps_p_max * (x_hi - x_lo) / gap <= nu * eps_x   # eq. (70)
        m = 1 if late else shots
        n1 += rng.binomial(m, _cheb2(d, x_true))
        n += m
        calls += m * d                                                   # adaptation A3
        kmax = max(kmax, d // 2)
        p_lo, p_hi = _cp(n1, n, a_T)
        lo, hi = _invert(d, x_lo, x_hi, p_lo, p_hi)
        x_lo, x_hi = max(x_lo, lo), min(x_hi, hi)
        if x_lo > x_hi:  # numerical guard, same spirit as ADB iqae
            x_lo, x_hi = min(x_lo, x_hi), max(x_lo, x_hi)
        it += 1
        if it > max_iter:
            raise RuntimeError("chebae: max_iter exceeded")
    return float(0.5 * (x_lo ** 2 + x_hi ** 2)), int(calls), int(kmax)


# ----------------------------------------------------------------------------------------------- self-check
def _table(reps=200, seed=0):
    import sys
    import time
    sys.dont_write_bytecode = True  # never write __pycache__ into the friend's tree
    import _src  # noqa: F401  (puts the friend repo on sys.path, read-only)
    from src.quantum.iqae import iqae as adb_iqae

    algos = {"iqae (ADB, shots=100)": lambda a, e, g: adb_iqae(a, e, 0.05, 100, g),
             "miqae chernoff shots=1": lambda a, e, g: miqae(a, e, 0.05, 1, g, "chernoff"),
             "miqae beta shots=1": lambda a, e, g: miqae(a, e, 0.05, 1, g, "beta"),
             "chebae r=2 N=100 nu=8": lambda a, e, g: chebae(a, e, 0.05, 100, g),
             "chebae odd_only": lambda a, e, g: chebae(a, e, 0.05, 100, g, odd_only=True)}
    print(f"reps={reps}, alpha=0.05; c = mean(calls)*eps_rel*sqrt(a); cover = P(|est-a|<=eps), need >= 0.95")
    print(f"{'algorithm':24s} {'a':>7s} {'eps_rel':>7s} {'c':>8s} {'cover':>6s} {'kmax':>8s} {'k0':>4s} {'s/run':>7s}")
    rows = []
    for name, f in algos.items():
        for a in (1e-6, 1e-4, 1e-2, 0.3):
            for er in (0.1, 0.01):
                g = np.random.default_rng(seed)
                eps = er * a
                t0 = time.time()
                out = np.array([f(a, eps, g) for _ in range(reps)], dtype=float)
                dt = (time.time() - t0) / reps
                c = out[:, 1].mean() * er * np.sqrt(a)
                cover = np.mean(np.abs(out[:, 0] - a) <= eps)
                k0 = int(np.sum(out[:, 2] == 0))
                rows.append((name, a, er, c, cover, out[:, 2].mean(), k0, dt))
                print(f"{name:24s} {a:7.0e} {er:7.2f} {c:8.1f} {cover:6.3f} {out[:, 2].mean():8.1f} {k0:4d} {dt:7.3f}",
                      flush=True)
    return rows


if __name__ == "__main__":
    import sys
    # unit checks first: inversion round-trip and FindNextK invariant
    for x_lo, x_hi in ((0.30, 0.31), (0.001, 0.0011), (0.9, 0.95), (0.5, 0.5001)):
        for odd in (False, True):
            dd = _find_next_cheb(x_lo, x_hi, 1, odd)
            assert dd is not None and (not odd or dd % 2 == 1)
            x0 = 0.5 * (x_lo + x_hi)
            p = _cheb2(dd, x0)
            lo, hi = _invert(dd, x_lo, x_hi, p, p)
            assert abs(lo - x0) < 1e-9 and abs(hi - x0) < 1e-9, (dd, lo, hi)
    k = _find_next_k(0, 0.10, 0.11)
    K = 2 * k + 1
    assert k > 0 and np.floor(K * 0.10 / HALF_PI) == np.ceil(K * 0.11 / HALF_PI) - 1
    print("unit checks ok")
    _table(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
