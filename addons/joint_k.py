"""Quality 3: do multi-country rain extremes exceed what pairwise dependence implies? (QContagion cds_highdim TEST 1,
rebuilt for this question.) Additive; friend's src/ read-only.

The friend's n-country model (SyntheticModel / DataModel, used for the CAREC and Asia-Pacific pools and the scaling
argument) has pairwise shared sources only. If storms that hit 3+ countries at once are more common than any
pairwise structure can produce, k-of-N tails are understated.
Data: the friend's daily GHCN country index (rain_check.country_index), 8 countries with long records (PK has ~20
years and is left out). Days are grouped in 3-day blocks (a block counts for a country when >= 2 of its days are
valid; only blocks valid for all 8 countries are used). A country is "extreme" in a block if any valid day exceeds its
daily quantile u (levels below). Per season (DJF/MAM/JJA/SON, to keep seasonality from posing as dependence):
  M0  independence (product of the marginals)
  M2  pairwise maximum-entropy model (Ising on 2^8 states, exact): reproduces every country's extreme rate and every
      pair's co-extreme rate, nothing higher
Statistic: number of blocks with >= 3 (>= 4) countries extreme, observed vs expected under M0 and M2; p-value from a
parametric bootstrap of M2 (fit held fixed).
"""
from datetime import date
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from _src import save
from rain_lag import country_data

COUNTRIES = ["AZ", "GE", "KG", "KZ", "MN", "TJ", "TM", "UZ"]
LEVELS = (0.99, 0.995, 0.998)          # daily quantile of each country's valid days
D0, NBOOT, SEED = date(1951, 1, 1), 4000, 20261107
SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}


def blocks(ci, u):
    """Binary matrix (blocks x countries) and the season of each block."""
    K, nd = len(COUNTRIES), (date(2023, 12, 31) - D0).days + 1
    val = np.full((nd, K), np.nan)
    for k, c in enumerate(COUNTRIES):
        ys = set(ci[c]["years"])
        for key, v in ci[c]["idx"].items():
            if int(key[:4]) in ys:
                val[(date(int(key[:4]), int(key[4:6]), int(key[6:])) - D0).days, k] = v
    thr = np.array([np.nanquantile(val[:, k], u) for k in range(K)])
    nb = nd // 3
    V = val[:nb * 3].reshape(nb, 3, K)
    nvalid = np.sum(~np.isnan(V), axis=1)
    ok = np.all(nvalid >= 2, axis=1)
    X = (np.nan_to_num(V, nan=-1) > thr).any(axis=1)[ok].astype(int)
    season = np.array([SEASON[date.fromordinal(D0.toordinal() + 3 * b).month] for b in np.flatnonzero(ok)])
    return X, season, thr


def ising_fit(X, lam=1e-2):
    """Exact pairwise maximum-entropy fit on 2^K states (convex dual, small L2 ridge for empty pairs)."""
    K = X.shape[1]
    S = np.array([[(s >> k) & 1 for k in range(K)] for s in range(2 ** K)], float)
    pairs = list(combinations(range(K), 2))
    F = np.hstack([S, np.stack([S[:, i] * S[:, j] for i, j in pairs], 1)])
    mu = np.r_[X.mean(0), [np.mean(X[:, i] * X[:, j]) for i, j in pairs]]
    n = len(X)

    def f(th):
        lp = F @ th
        lz = logsumexp(lp)
        p = np.exp(lp - lz)
        return lz - th @ mu + 0.5 * lam / n * th @ th, F.T @ p - mu + lam / n * th

    th = minimize(f, np.zeros(F.shape[1]), jac=True, method="L-BFGS-B").x
    lp = F @ th
    return np.exp(lp - logsumexp(lp)), S


def run():
    ci = country_data()
    rng = np.random.default_rng(SEED)
    out = {"countries": COUNTRIES, "block_days": 3, "levels": {}}
    for u in LEVELS:
        X, season, thr = blocks(ci, u)
        res = {"blocks": int(len(X)), "per_season": {}}
        tot = {k: {"obs": 0, "m0": 0.0, "m2": 0.0} for k in (3, 4)}
        boot = {3: np.zeros(NBOOT), 4: np.zeros(NBOOT)}
        for s in range(4):
            Xs = X[season == s]
            if len(Xs) == 0:
                continue
            p2, S = ising_fit(Xs)
            p0 = np.prod(np.where(S == 1, Xs.mean(0), 1 - Xs.mean(0)), axis=1)
            cnt = S.sum(1)
            draws = rng.multinomial(len(Xs), p2, size=NBOOT)
            row = {"blocks": int(len(Xs))}
            for k in (3, 4):
                obs = int((Xs.sum(1) >= k).sum())
                e0, e2 = len(Xs) * p0[cnt >= k].sum(), len(Xs) * p2[cnt >= k].sum()
                row[f"ge{k}"] = {"observed": obs, "independence": float(e0), "pairwise": float(e2)}
                tot[k]["obs"] += obs; tot[k]["m0"] += e0; tot[k]["m2"] += e2
                boot[k] += draws[:, cnt >= k].sum(1)
            # sanity: the fit reproduces the pairwise co-extreme rates
            row["max_pair_fit_error"] = float(max(abs(len(Xs) * (p2 * S[:, i] * S[:, j]).sum() - (Xs[:, i] * Xs[:, j]).sum())
                                                 for i, j in combinations(range(len(COUNTRIES)), 2)))
            res["per_season"][str(s)] = row
        for k in (3, 4):
            t = tot[k]
            t["ratio_obs_to_pairwise"] = t["obs"] / t["m2"] if t["m2"] > 0 else None
            t["ratio_obs_to_independence"] = t["obs"] / t["m0"] if t["m0"] > 0 else None
            t["p_value_vs_pairwise"] = float((1 + (boot[k] >= t["obs"]).sum()) / (1 + NBOOT))
            t["pairwise_q95"] = float(np.quantile(boot[k], 0.95))
            res[f"ge{k}"] = t
        out["levels"][str(u)] = res
        print(f"u={u}: {len(X)} blocks  " + "  ".join(
            f">={k}: obs {res[f'ge{k}']['obs']} vs pairwise {res[f'ge{k}']['m2']:.1f} (x{res[f'ge{k}']['ratio_obs_to_pairwise']:.2f}, "
            f"p={res[f'ge{k}']['p_value_vs_pairwise']:.3f}) vs indep {res[f'ge{k}']['m0']:.1f}" for k in (3, 4)), flush=True)
    save("quality_joint_k.json", out)
    return out


def selftest():
    """On data drawn from a known Ising model, the >=3 count must sit inside the model's bootstrap band."""
    rng = np.random.default_rng(1)
    K = 5
    S = np.array([[(s >> k) & 1 for k in range(K)] for s in range(2 ** K)], float)
    lp = -3 * S.sum(1) + 1.2 * (S[:, 0] * S[:, 1] + S[:, 1] * S[:, 2] + S[:, 3] * S[:, 4])
    p = np.exp(lp - logsumexp(lp))
    X = S[rng.choice(len(S), 20000, p=p)].astype(int)
    p2, S2 = ising_fit(X)
    e = len(X) * p2[S2.sum(1) >= 3].sum()
    obs = (X.sum(1) >= 3).sum()
    assert abs(obs - e) < 4 * np.sqrt(e) + 1, (obs, e)
    print(f"selftest ok: on true-pairwise data, observed >=3 = {obs} vs fitted pairwise {e:.1f}")


if __name__ == "__main__":
    import sys
    selftest()
    if "--selftest" not in sys.argv:
        run()
