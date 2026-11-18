"""E3: sensitivities of a tail probability to every source's rate, from the same RDPO paths as the estimate.
Additive; the friend's rdpo_estimate is called unchanged and his trained policy is only rolled out again.

Each source k is a compound Poisson process with a capped-increment term law p_k (Proposition 1). Scaling its
rate by s gives p_k(s) = row 0 of exp(s·log P_k), where P_k is the source's "add then cap" term transition built
from p_k itself (the capped chain is a semigroup in time, and time and rate enter only through their product).
So dp_k/ds at s = 1 is (log P_k · P_k)[0] exactly, and the score S_k(v) = dp_k(v) / p_k(v) needs no model rebuild.
Score-function identity: ∂μ/∂s_k = E_q[f·w·S_k(v_k)] — one batch of N paths from the trained policy gives μ and all
K derivatives together. Reported as elasticities (s/μ)·∂μ/∂s_k: "a 1% higher rate of source k raises μ by E_k %".
Exact references: μ is multilinear in the source laws, so ∂μ/∂s_k = μ with p_k replaced by dp_k (dense enumeration
from sources for n <= 6; the friend's exact_pair for pair targets in any portfolio size).
Pre-set rule: "strong" if, for sources with |E_k| >= 0.05, the median rel RMSE of the RDPO elasticities <= 20%.
"""
import copy
import time

import numpy as np
from scipy.linalg import expm, logm

from _src import save, seed as _seed
from src.rdpo.experiments_v2 import final_config
from src.rdpo.fast import FastPolicy
from src.rdpo.problem import PortfolioProblem, adb_model
from src.rdpo.train import rdpo_estimate
from src.tensor.synthetic import SyntheticModel, topology

B, N, REPS = 16384, 16384, 20
BEDS = {
    "adb_joint_exhaustion": lambda: PortfolioProblem(adb_model("adb-base"), "joint_exhaustion", threshold=80e6),
    "adb_exceedance": lambda: PortfolioProblem(adb_model("adb-base"), "exceedance", threshold=80e6),
    "complete4_exceedance": lambda: PortfolioProblem(SyntheticModel(4, topology("complete", 4)), "exceedance"),
    "ring28_joint_exhaustion": lambda: PortfolioProblem(SyntheticModel(28, topology("ring", 28)), "joint_exhaustion"),
}
RULE_MIN_E, RULE_MAX_RMSE = 0.05, 0.20


def source_derivative(src):
    """dp/ds at s = 1 for one source (same index as src['p']), via the capped-chain semigroup."""
    coords = [(inc, 4) for _, inc in src["da"]] + [(inc, 1) for _, inc in src["dr"]]
    dims = tuple(c + 1 for _, c in coords)
    S = int(np.prod(dims))
    states = np.array(np.unravel_index(np.arange(S), dims)).T
    inc = np.stack([i for i, _ in coords], 1)                    # (V, d)
    caps = np.array([c for _, c in coords])
    Pm = np.zeros((S, S))
    for v, pv in enumerate(src["p"]):
        nxt = np.ravel_multi_index(tuple(np.minimum(states + inc[v], caps).T), dims)
        np.add.at(Pm, (np.arange(S), nxt), pv)
    L = np.real(logm(Pm))
    d_states = (L @ Pm)[0]
    code = np.ravel_multi_index(tuple(np.minimum(inc, caps).T), dims)
    return d_states[code], L, Pm, code


def exact_value(P, src):
    Q = copy.copy(P); Q.src = src
    if P.target in ("joint_attachment", "joint_exhaustion"):
        return Q.exact_pair()
    n = P.n
    D = Q._dense_from_sources()
    idx = np.arange(10 ** n)
    digits = np.stack([(idx // 10 ** (n - 1 - i)) % 10 for i in range(n)], 1)
    Ls = P.site_loss(digits // 2, digits % 2)
    i, j = P.pair
    return float(D @ P.payoff_from(Ls.sum(1), Ls[:, i], Ls[:, j]))


def exact_elasticities(P, dps):
    mu = exact_value(P, P.src)
    out = []
    for k, dp in enumerate(dps):
        if not any(P.pair[0] in P.src[k]["sites"] or P.pair[1] in P.src[k]["sites"] for _ in [0]) and \
                P.target in ("joint_attachment", "joint_exhaustion"):
            out.append(0.0)                                      # exact_pair: sources off the pair do not matter
            continue
        src = list(P.src); s = dict(src[k]); s["p"] = dp; src[k] = s
        out.append(exact_value(P, src) / mu)
    return mu, np.array(out)


def label(s):
    return f"{s['kind']}:{'-'.join(map(str, s['sites']))}"


def job(args):
    bed, rep = args
    P = BEDS[bed]()
    cfg = final_config(P.target)
    seed = _seed(8_100_000) + 10_000 * list(BEDS).index(bed) + rep
    t0 = time.perf_counter()
    r = rdpo_estimate(P, B, np.random.default_rng(seed), cfg)
    pol = FastPolicy(P, cfg.alpha, cfg.edge_features, cfg.relevance, value=cfg.value, stages=cfg.stages)
    pol.theta = np.array(r["theta"])
    out = pol.rollout(N, np.random.default_rng(seed + 1), record=True)
    g = out["f"] * out["w"]
    S = np.stack([np.where(P.src[k]["p"] > 0, SCORES[bed][k], 0.0)[out["acts"][:, k]] for k in range(P.K)], 1)
    mu = g.mean()
    return {"rep": rep, "mu_rdpo": float(r["estimate"]), "mu_batch": float(mu),
            "elasticity": ((g[:, None] * S).mean(0) / mu).tolist() if mu > 0 else [float("nan")] * P.K,
            "sec": time.perf_counter() - t0}


SCORES = {}


def _scores(bed):
    P = BEDS[bed]()
    dps = [source_derivative(s)[0] for s in P.src]
    return P, dps, [dp / np.maximum(s["p"], 1e-300) for dp, s in zip(dps, P.src)]


for _b in BEDS:                                                  # module-level so pool workers have them
    SCORES[_b] = _scores(_b)[2]


def run(procs=14):
    from multiprocessing import Pool
    res = {}
    for bed in BEDS:
        P, dps, _ = _scores(bed)
        mu, ex = exact_elasticities(P, dps)
        with Pool(procs) as pool:
            rows = list(pool.imap_unordered(job, [(bed, r) for r in range(REPS)]))
        E = np.array([r["elasticity"] for r in rows])
        rmse = np.sqrt(np.nanmean((E - ex[None, :]) ** 2, 0))
        big = np.abs(ex) >= RULE_MIN_E
        rel = rmse[big] / np.abs(ex[big])
        res[bed] = {"mu_exact": mu, "K": P.K, "sources": [label(s) for s in P.src], "elasticity_exact": ex.tolist(),
                    "elasticity_rdpo_mean": np.nanmean(E, 0).tolist(), "elasticity_rmse": rmse.tolist(),
                    "n_big": int(big.sum()), "median_rel_rmse_big": float(np.median(rel)) if big.any() else None,
                    "max_rel_rmse_big": float(rel.max()) if big.any() else None,
                    "mu_rel_rmse_batch": float(np.sqrt(np.mean([(r["mu_batch"] / mu - 1) ** 2 for r in rows]))),
                    "median_sec": float(np.median([r["sec"] for r in rows])), "reps": len(rows),
                    "paths": {"rdpo_fit": B, "sensitivity_batch": N, "finite_difference_would_need_fits": 2 * P.K}}
        top = np.argsort(-np.abs(ex))[:6]
        print(f"{bed}: mu {mu:.3g}, K={P.K}, big sources {big.sum()}, median rel RMSE {res[bed]['median_rel_rmse_big']:.3f} "
              f"(max {res[bed]['max_rel_rmse_big']:.3f})", flush=True)
        for k in top:
            print(f"   {label(P.src[k]):14s} exact {ex[k]:+.3f}  rdpo {np.nanmean(E[:, k]):+.3f} ± {rmse[k]:.3f}", flush=True)
        save("sensitivity.json", res)
    meds = [v["median_rel_rmse_big"] for v in res.values() if v["median_rel_rmse_big"] is not None]
    verdict = "strong" if all(m <= RULE_MAX_RMSE for m in meds) else "not strong"
    out = {"beds": res, "rule": f"median rel RMSE <= {RULE_MAX_RMSE} for |E| >= {RULE_MIN_E} on every bed", "verdict": verdict}
    save("sensitivity.json", out)
    print("verdict:", verdict)
    return out


def selftest():
    P = BEDS["adb_joint_exhaustion"]()
    for s in P.src:
        dp, L, Pm, code = source_derivative(s)
        assert np.allclose(expm(L), Pm, atol=1e-10)
        h = 1e-5                                                 # dp matches a finite difference of the semigroup
        fd = (expm((1 + h) * L)[0] - expm((1 - h) * L)[0])[code] / (2 * h)
        assert np.allclose(dp, fd, atol=1e-8), s["kind"]
        assert abs(dp.sum()) < 1e-10                             # a derivative of a probability law sums to 0
        if s["kind"] == "rain1":                                 # Bernoulli p0 = exp(-λT): d log p0 / ds = log p0
            assert abs(dp[0] / s["p"][0] - np.log(s["p"][0])) < 1e-10
    mu, ex = exact_elasticities(P, [source_derivative(s)[0] for s in P.src])
    assert abs(mu / P.exact_dense() - 1) < 1e-12
    k = 0; h = 1e-4                                              # exact elasticity matches a rebuilt finite difference
    _, L, _, code = source_derivative(P.src[k])
    up, dn = list(P.src), list(P.src)
    up[k] = dict(P.src[k], p=expm((1 + h) * L)[0][code]); dn[k] = dict(P.src[k], p=expm((1 - h) * L)[0][code])
    fd = (exact_value(P, up) - exact_value(P, dn)) / (2 * h) / mu
    assert abs(fd - ex[k]) < 1e-6 * max(1, abs(ex[k]))
    print("selftest ok: semigroup derivative = finite difference; Bernoulli case closed form; exact elasticity = FD")


if __name__ == "__main__":
    selftest()
    run()
