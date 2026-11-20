"""The small world with saturation modules against the main author's exact solver, and the cost per query. Additive, read-only.

Until here the large pools were checked against the twist (another estimator). On graphs of small width the main author's
junction solver (src/tensor/junction.py) gives the exceedance probability exactly, with machine-level relative accuracy
in the far tail. Pools: rings of 10, 20, 28 countries at US$25m and US$40m per country, and the ap_risk pool at its
two thresholds (US$450m and US$850m).
The brief's criterion is accuracy per query (per path). Paths needed for a 1% relative standard error = relative variance / 1e-4.
Rules (written knowing the US$40m cells against the twist; the exact values, the US$25m cells and the US$850m cell are new):
 E1 the estimate with modules is within 3 standard errors of the exact value in every pool (3 seeds, 200,000 paths each);
 E2 the small world's own answer with modules is within 5% of the exact value in every pool;
 E3 the twist reference used in the earlier modules is within 3 standard errors of the exact value in every pool;
 E4 paths for a 1% error: with modules at least 10x fewer than the twist in every pool.
"""
import numpy as np

import cap_modules as C
import multi_country as M
import small_world as W
import small_world_loop as L
from _src import save, seed
from src.rdpo.problem import PortfolioProblem
from src.tensor.junction import junction_solve


def pools():
    import ap_risk
    out = {f"ring {n} @ {t / 1e6:.0f}m": M.pool(n, t) for n in (10, 20, 28) for t in (25e6, 40e6)}
    model = ap_risk.inputs()["model"]
    for t in ap_risk.THRESH:
        out[f"ap_risk 28 @ {t}m total"] = PortfolioProblem(model, "exceedance", threshold=t * 1e6)
    return out


def cell(P, n=200_000):
    ex = junction_solve(P)
    p = ex["exceedance"]
    p_tw, se_tw, rho2_tw, _ = L.twist_ref(P, 2_000_000, np.random.default_rng(seed(1)))
    pol0, a0, _ = W.teacher(P)
    polm, am, tb, _ = C.build(P, 1)
    s0, sm = L.stat(pol0, p, 0.0, n), L.stat(polm, p, 0.0, n)
    rho2 = {"plain": (1 - p) / p, "twist": rho2_tw * (p_tw / p) ** 2, "uncapped": s0["rho2_worst"], "modules": sm["rho2_worst"]}
    return {"sources": P.K, "p_exact": p, "exact_seconds": ex["seconds"], "exact_width": ex["width"], "twist_z": float((p_tw - p) / se_tw),
            "uncapped_z_max": s0["z_max"], "modules_z_max": sm["z_max"], "uncapped_answer_over_exact": a0 / p, "modules_answer_over_exact": am / p,
            "cpu_build_modules": tb, "relative_variance": rho2, "paths_for_1pct": {k: v / 1e-4 for k, v in rho2.items()}}


def run():
    res = {k: cell(P) for k, P in pools().items()}
    res["checks"] = {"E1_modules_unbiased_vs_exact": bool(max(r["modules_z_max"] for r in res.values()) <= 3),
                     "E2_own_answer_within_5pct": {k: bool(abs(r["modules_answer_over_exact"] - 1) <= 0.05) for k, r in res.items()},
                     "E3_twist_reference_ok": bool(max(abs(r["twist_z"]) for r in res.values()) <= 3),
                     "E4_modules_10x_fewer_paths_than_twist": {k: bool(r["paths_for_1pct"]["twist"] >= 10 * r["paths_for_1pct"]["modules"]) for k, r in res.items()}}
    save("exact_check.json", res)
    for k, r in res.items():
        if k == "checks":
            continue
        q = r["paths_for_1pct"]
        print(f"{k:24s} exact {r['p_exact']:.6g} ({r['exact_seconds']:.1f} s) | max |z|: twist {abs(r['twist_z']):.1f}, modules {r['modules_z_max']:.1f} | own answer / exact: uncapped {r['uncapped_answer_over_exact']:.2f}, "
              f"modules {r['modules_answer_over_exact']:.3f} | paths for 1%: plain {q['plain']:.2g}, twist {q['twist']:.2g}, uncapped {q['uncapped']:.2g}, modules {q['modules']:.2g} (x{q['twist'] / q['modules']:.0f} fewer than twist)")
    print("checks:", res["checks"])
    return res


def selftest():
    """The junction solver agrees with brute-force enumeration on a 3-country ring, and the modules' answer with both."""
    from src.tensor.synthetic import dense_term_distribution
    P = M.pool(3, 25e6)
    dist = dense_term_distribution(P.model)
    codes = np.indices(dist.shape).reshape(3, -1)
    brute = float(dist.reshape(-1)[(P.a_u * (codes // 2) + P.r_u * (codes % 2)).sum(0) > P.T].sum())
    ex = junction_solve(P)["exceedance"]
    am = C.build(P, 1)[1]
    assert abs(ex / brute - 1) < 1e-10 and abs(am / ex - 1) < 0.02, (brute, ex, am)
    print(f"selftest ok: 3-country ring, enumeration {brute:.8g}, junction solver {ex:.8g}, small world with modules {am:.8g}")


if __name__ == "__main__":
    selftest()
    run()
