"""The small world / RDPO / twin loop on many countries. Additive, read-only.

small_world.py and small_world_loop.py hold on the two-country model and on one 28-country pool (ap_risk). Here: the
the main author's synthetic ring pools of 5, 10, 20 and 28 countries, two thresholds per pool (US$25m and US$40m per country;
the second is the main author's default and is far rarer), and the twin's worlds on 10 and 28 countries. No exact values
exist at these sizes; the reference is the twist with 2,000,000 paths.
Rules (written after a one-seed probe of the US$40m rings, which already showed the last two columns of the table
getting worse with size; nothing was tuned afterwards):
 M1 every estimate is within 3 standard errors of the twist's;
 M2 small-world teacher, equal-CPU efficiency against the twist > 1 for every size and threshold (building charged);
 M3 the small world's own answer is within a factor 2 of the truth for every size and threshold;
 M4 worlds (10 and 28 countries, US$40m): rebuilt relative variance <= 2x the base world's in every world, and not
    rebuilding is >= 1.5x worse in the worst world;
 M5 28 countries, US$40m: the main author's training on top of the small world lowers the relative variance by >= 20%.
"""
import numpy as np

import dreamwake as D
import small_world as W
import small_world_loop as L
from _src import save, seed
from src.rdpo.problem import PortfolioProblem
from src.tensor.synthetic import SyntheticModel, topology

SIZES = (5, 10, 20, 28)
PER_COUNTRY = (25e6, 40e6)


def pool(n, per_country, **world):
    return PortfolioProblem(SyntheticModel(n, topology("ring", n), **{**D.BASE, **world}), "exceedance", threshold=per_country * n)


def cell(P, n_tw=2_000_000, n=200_000, s=1):
    p, p_se, rho2_tw, c_tw = L.twist_ref(P, n_tw, np.random.default_rng(seed(s)))
    pol, answer, t_build = W.teacher(P)
    st = L.stat(pol, p, p_se, n)
    return {"sources": P.K, "p_twist": p, "p_twist_se": p_se, "rho2_twist": rho2_tw, "rho2_teacher": st["rho2_worst"], "z_max": st["z_max"],
            "cost_ratio": st["cost_per_path"] / c_tw, "cpu_build": t_build, "small_world_answer_over_truth": answer / p,
            "equal_cpu_gain": rho2_tw * c_tw / (st["rho2_worst"] * (st["cost_per_path"] + t_build / n))}, (p, p_se, rho2_tw, c_tw)


def worlds(n, M=8, paths=100_000):
    rng = np.random.default_rng(seed(20261201 + n))
    P0 = pool(n, 40e6)
    base = [np.log(np.maximum(v, 1e-300)) for v in W.value_tables(P0)[0]]
    rows = []
    for w in [dict(D.BASE)] + D.worlds(rng, M):
        P = pool(n, 40e6, **w)
        p, p_se, _, _ = L.twist_ref(P, 1_000_000, rng)
        pol, _, _ = W.teacher(P)
        fresh = L.stat(pol, p, p_se, paths, seeds=(61,))
        psi_t, chi_t, _ = pol._tables()
        pol._tables = lambda: (psi_t, chi_t, base)
        stale = L.stat(pol, p, p_se, paths, seeds=(61,))
        rows.append({**w, "p_twist": p, "z": fresh["z_max"], "rho2_rebuilt": fresh["rho2_worst"], "rho2_not_rebuilt": stale["rho2_worst"]})
    reb, old = np.array([r["rho2_rebuilt"] for r in rows]), np.array([r["rho2_not_rebuilt"] for r in rows])
    return {"worlds": rows, "base_rho2": float(reb[0]), "worst_rho2_rebuilt": float(reb.max()), "worst_ratio_rebuilt_over_base": float((reb / reb[0]).max()),
            "worst_ratio_not_rebuilt_over_rebuilt": float((old / reb).max()), "z_max": max(r["z"] for r in rows)}


def run():
    scale, ref28 = {}, None
    for n in SIZES:
        for t in PER_COUNTRY:
            scale[f"{n}@{t / 1e6:.0f}m"], ref = cell(pool(n, t))
            if (n, t) == (28, 40e6):
                ref28 = ref
    tr = L.with_training(pool(28, 40e6), *ref28, n=200_000)
    wl = {n: worlds(n) for n in (10, 28)}
    res = {"scale": scale, "training_28": tr, "worlds": wl}
    res["checks"] = {"M1_unbiased": bool(max([c["z_max"] for c in scale.values()] + [w["z_max"] for w in wl.values()] + [tr["trained"]["z_max"]]) <= 3),
                     "M2_equal_cpu_gt_1_everywhere": bool(min(c["equal_cpu_gain"] for c in scale.values()) > 1),
                     "M3_own_answer_within_2x": {k: bool(0.5 <= c["small_world_answer_over_truth"] <= 2) for k, c in scale.items()},
                     "M4_worlds": {n: bool(w["worst_ratio_rebuilt_over_base"] <= 2 and w["worst_ratio_not_rebuilt_over_rebuilt"] >= 1.5) for n, w in wl.items()},
                     "M5_training_lowers_variance_20pct": bool(tr["variance_reduction_by_training"] >= 0.2)}
    save("multi_country.json", res)
    for k, c in scale.items():
        print(f"{k:>7s} ({c['sources']:3d} sources): p {c['p_twist']:.3g} | relative variance twist {c['rho2_twist']:.3g}, teacher {c['rho2_teacher']:.3g} | cost x{c['cost_ratio']:.1f} | "
              f"equal-CPU x{c['equal_cpu_gain']:.2f} | small world alone {c['small_world_answer_over_truth']:.2f}x the truth | max |z| {c['z_max']:.1f}")
    print(f"training on top, 28@40m: relative variance {tr['untrained']['rho2_worst']:.3g} -> {tr['trained']['rho2_worst']:.3g} ({tr['variance_reduction_by_training']:+.0%}), {tr['cpu_training']:.0f} s; "
          f"equal-CPU vs twist x{tr['equal_cpu_gain_trained']:.2f}, x{tr['equal_cpu_gain_trained_training_charged']:.2f} with training charged")
    for n, w in wl.items():
        print(f"worlds, {n} countries: base {w['base_rho2']:.3g}, rebuilt worst x{w['worst_ratio_rebuilt_over_base']:.2f} the base, not rebuilt worst x{w['worst_ratio_not_rebuilt_over_rebuilt']:.1f} the rebuilt, max |z| {w['z_max']:.1f}")
    print("checks:", res["checks"])
    return res


def selftest():
    """A 3-country ring is small enough to enumerate: the teacher is unbiased there and the uncapped answer is an upper bound."""
    import rdpo_engine as E
    from src.tensor.synthetic import dense_term_distribution
    P = pool(3, 25e6)
    dist = dense_term_distribution(P.model)
    codes = np.indices(dist.shape).reshape(3, -1)
    loss = (P.a_u * (codes // 2) + P.r_u * (codes % 2)).sum(0)
    exact = float(dist.reshape(-1)[loss > P.T].sum())
    pol, answer, _ = W.teacher(P)
    o = E.rollout(pol, 300_000, np.random.default_rng(4))
    g = o["w"] * o["f"]
    assert answer >= exact * (1 - 1e-9) and abs(g.mean() - exact) < 4 * g.std() / np.sqrt(len(g)), (answer, g.mean(), exact)
    print(f"selftest ok: 3-country ring, exact {exact:.5g}, teacher {g.mean():.5g}, uncapped answer {answer:.5g} (an upper bound)")


if __name__ == "__main__":
    selftest()
    run()
