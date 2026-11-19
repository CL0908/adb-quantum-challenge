"""Two open links of the small world / RDPO / twin loop, tested. Additive, read-only.

small_world.py: the uncapped pool's exact table, used untrained as the progress table of the main author's FastPolicy,
beats the twist at equal CPU on the 28-country pool. Two questions were left open.
A. Does training still add anything? Here the policy is "small world's table + the main author's learned features", and
   the main author's own training (rdpo_estimate, unchanged) fits only the learned part, starting from zero.
   A1 the trained estimate is within 3 standard errors of the reference;
   A2 training lowers the relative variance by >= 20% against the untrained small world (two-country and pool, each);
   A3 on the pool, equal-CPU efficiency against the twist stays > 1 with the training time charged.
B. When the twin changes the world, is rebuilding the small world enough? Worlds: the twin's three scenario inputs
   (dreamwake.py), two-country model, exact probabilities by enumeration.
   B1 every world's estimate is within 3 standard errors of its exact value;
   B2 the relative variance in every world is <= 2x the base world's;
   B3 keeping the base world's table instead of rebuilding is worse in the worst world by >= 1.5x.
How the base table reaches the main author's training without touching src/: a subclass adds it in _tables (rollout) and
in compile (as a per-state offset of log p), and rdpo_estimate is pointed at that subclass for the duration of the call.
Limit: B is on the two-country model only; A uses one training budget (65,536 paths).
"""
import dataclasses
import json
import time

import numpy as np

import dreamwake as D
import rdpo_engine as E
import small_world as W
import src.rdpo.fast as fast
from _src import OUT, save, seed
from src.rdpo.experiments_v2 import final_config
from src.rdpo.problem import PortfolioProblem, adb_model
from src.rdpo.tilt import TiltSampler
from src.rdpo.train import _best_twist_sources, rdpo_estimate

FastPolicy = fast.FastPolicy


class _Rows:
    """Stands in for a (V,) log p row where the main author's ce_from / mix_from write logp[None, :]: returns per-state rows."""

    def __init__(self, a):
        self.a = a

    def __getitem__(self, _):
        return self.a


class Taught(FastPolicy):
    """FastPolicy whose progress table is the small world's (one per source) plus the learned one."""
    base_tables = None                                                        # a subclass may carry other tables (cap_modules.py)

    def __init__(self, prob, alpha=0.05, edge_features=True, relevance=True, value="linear", n_knots=10, stages=1):
        super().__init__(prob, alpha, edge_features, relevance, value=value, n_knots=n_knots, stages=prob.K)
        self.base = [np.log(np.maximum(v, 1e-22)) for v in (self.base_tables or W.value_tables(prob)[0])]   # floor: keeps the main author's exp() away from 0/0
        self._off = None

    def _tables(self):
        psi_t, chi_t, h_t = super()._tables()
        return psi_t, chi_t, [b + h for b, h in zip(self.base, h_t)]

    def delta_codes(self, k, S, Lt, Li, Lj):
        out = super().delta_codes(k, S, Lt, Li, Lj)
        if self._off is not None and not self.plan[k]["frozen"]:
            self._off.append(self.base[k][out[2]] - self.base[k][Lt][:, None])
        return out

    def compile(self, acts):
        self._off = []
        out = super().compile(acts)
        off, self._off = self._off, None
        for st, o in zip(out, off):
            st["logp"] = _Rows(st["logp"][None, :] + np.where(st["p"][None, :] > 0, o, 0.0))   # impossible outcomes keep the main author's floor (no 0/0)
        return out


def train(P, b_fit, rng, tables=None, theta0=None):
    """tables: other base tables than the uncapped ones; theta0: start the learned part from a neighbour's (the main author's transfer hook)."""
    cfg = dataclasses.replace(final_config(P.target), warm_start=False, select=True, theta0=theta0)
    Taught = type("Taught", (globals()["Taught"],), {"base_tables": tables})
    fast.FastPolicy = Taught                                                  # rdpo_estimate imports the name at call time
    try:
        t0 = time.process_time()
        r = rdpo_estimate(P, b_fit, rng, cfg)
        t = time.process_time() - t0
    finally:
        fast.FastPolicy = FastPolicy
    pol = Taught(P, cfg.alpha, cfg.edge_features, cfg.relevance)
    pol.theta = np.array(r["theta"], float)
    return pol, t


def stat(pol, p, p_se, n, seeds=(41, 42, 43)):
    rs = []
    for s in seeds:
        t, o = E.cpu(lambda: E.rollout(pol, n, np.random.default_rng(s)), reps=1)
        g = o["w"] * o["f"]
        rs.append((float(g.mean()), float(g.std() / np.sqrt(n)), float(g.var() / p ** 2), t / n))
    return {"z_max": float(max(abs(r[0] - p) / np.hypot(r[1], p_se) for r in rs)), "rho2_worst": max(r[2] for r in rs),
            "rho2_seeds": [r[2] for r in rs], "cost_per_path": float(np.median([r[3] for r in rs]))}


def with_training(P, p, p_se, rho2_tw, c_tw, n, b_fit=65_536, s=20261129):
    cfg = final_config(P.target)
    plain = stat(Taught(P, cfg.alpha, cfg.edge_features, cfg.relevance), p, p_se, n)
    pol, t_train = train(P, b_fit, np.random.default_rng(seed(s)))
    tr = stat(pol, p, p_se, n)
    return {"untrained": plain, "trained": tr, "cpu_training": t_train, "learned_norm": float(np.linalg.norm(pol.theta)),
            "variance_reduction_by_training": 1 - tr["rho2_worst"] / plain["rho2_worst"], "rho2_twist": rho2_tw,
            "equal_cpu_gain_untrained": rho2_tw * c_tw / (plain["rho2_worst"] * plain["cost_per_path"]),
            "equal_cpu_gain_trained": rho2_tw * c_tw / (tr["rho2_worst"] * tr["cost_per_path"]),
            "equal_cpu_gain_trained_training_charged": rho2_tw * c_tw / (tr["rho2_worst"] * (tr["cost_per_path"] + t_train / n))}


def twist_ref(P, n, rng):
    S = TiltSampler(P); S.set_theta(S.twist_theta(_best_twist_sources(P)))
    t, o = E.cpu(lambda: S.sample(n, rng), reps=1)
    g = o["w"] * o["f"]
    return float(g.mean()), float(g.std() / np.sqrt(n)), float(g.var() / g.mean() ** 2), t / n


def test_a(n=300_000):
    import ap_risk
    P2 = PortfolioProblem(adb_model(), "exceedance")
    ref = json.loads((OUT / "rdpo_gap.json").read_text())["exceedance"]
    _, _, _, c2 = twist_ref(P2, 1_000_000, np.random.default_rng(1))
    two = with_training(P2, ref["p"], 0.0, ref["rho2_twist_exact"], c2, n)
    Pb = PortfolioProblem(ap_risk.inputs()["model"], "exceedance", threshold=ap_risk.THRESH[0] * 1e6)
    p, p_se, rho2_tw, c_tw = twist_ref(Pb, 2_000_000, np.random.default_rng(seed(20261128)))
    return {"two_country": two, "pool_28": with_training(Pb, p, p_se, rho2_tw, c_tw, n)}


def test_b(M=20, n=200_000):
    rng = np.random.default_rng(seed(20261130))
    P0 = D.problem("exceedance")
    base = [np.log(np.maximum(v, 1e-300)) for v in W.value_tables(P0)[0]]
    rows = []
    for w in [dict(D.BASE)] + D.worlds(rng, M):
        P = D.problem("exceedance", **w)
        p = D.exact(P)[2]
        pol, answer, _ = W.teacher(P)
        fresh = stat(pol, p, 0.0, n, seeds=(51,))
        psi_t, chi_t, _ = pol._tables()
        pol._tables = lambda: (psi_t, chi_t, base)                             # the base world's table, not rebuilt
        stale = stat(pol, p, 0.0, n, seeds=(51,))
        rows.append({**w, "p_exact": p, "small_world_answer_over_truth": answer / p, "z": fresh["z_max"], "rho2_rebuilt": fresh["rho2_worst"],
                     "rho2_not_rebuilt": stale["rho2_worst"], "z_not_rebuilt": stale["z_max"]})
    return rows


def run():
    a, b = test_a(), test_b()
    reb, old = np.array([r["rho2_rebuilt"] for r in b]), np.array([r["rho2_not_rebuilt"] for r in b])
    res = {"A": a, "B": {"worlds": b, "base_rho2": float(reb[0]), "worst_rho2_rebuilt": float(reb.max()), "worst_rho2_not_rebuilt": float(old.max()),
                         "worst_ratio_not_rebuilt_over_rebuilt": float((old / reb).max()), "p_range": [min(r["p_exact"] for r in b), max(r["p_exact"] for r in b)]}}
    res["checks"] = {"A1_unbiased": bool(max(a[k]["trained"]["z_max"] for k in a) <= 3),
                     "A2_training_lowers_variance_20pct": {k: bool(a[k]["variance_reduction_by_training"] >= 0.2) for k in a},
                     "A3_pool_equal_cpu_with_training_gt_1": bool(a["pool_28"]["equal_cpu_gain_trained_training_charged"] > 1),
                     "B1_unbiased_every_world": bool(max(r["z"] for r in b) <= 3),
                     "B2_variance_le_2x_base": bool(reb.max() <= 2 * reb[0]),
                     "B3_not_rebuilding_worse_1.5x": bool((old / reb).max() >= 1.5)}
    save("small_world_loop.json", res)
    for k, r in a.items():
        print(f"A {k}: relative variance untrained {r['untrained']['rho2_worst']:.3g} -> trained {r['trained']['rho2_worst']:.3g} ({r['variance_reduction_by_training']:+.0%} reduction, "
              f"{r['cpu_training']:.1f} s of training, max |z| {r['trained']['z_max']:.1f}) | equal-CPU vs twist: untrained x{r['equal_cpu_gain_untrained']:.2f}, "
              f"trained x{r['equal_cpu_gain_trained']:.2f}, x{r['equal_cpu_gain_trained_training_charged']:.2f} with training charged")
    B = res["B"]
    print(f"B {len(b)} worlds, p from {B['p_range'][0]:.3g} to {B['p_range'][1]:.3g}: relative variance rebuilt base {B['base_rho2']:.3g}, worst {B['worst_rho2_rebuilt']:.3g}; "
          f"not rebuilt worst {B['worst_rho2_not_rebuilt']:.3g} (worst ratio x{B['worst_ratio_not_rebuilt_over_rebuilt']:.1f}); max |z| {max(r['z'] for r in b):.1f}")
    print("checks:", res["checks"])
    return res


def selftest():
    """With nothing learned the taught policy is the small world's teacher; the main author's training runs on it and stays unbiased."""
    P = PortfolioProblem(adb_model(), "exceedance")
    cfg = final_config("exceedance")
    a = E.rollout(Taught(P, cfg.alpha, cfg.edge_features, cfg.relevance), 5000, np.random.default_rng(1), record=True)
    b = E.rollout(W.teacher(P)[0], 5000, np.random.default_rng(1), record=True)
    assert (a["acts"] == b["acts"]).all() and np.allclose(a["w"], b["w"])
    pol, _ = train(P, 8192, np.random.default_rng(2))
    assert fast.FastPolicy is FastPolicy                                       # the main author's class is back in place
    assert E.same_paths(pol, 3000)["paths_equal"] == 1.0
    exact = D.exact(P)[2]
    o = E.rollout(pol, 300_000, np.random.default_rng(3))
    g = o["w"] * o["f"]
    assert abs(g.mean() - exact) < 4 * g.std() / np.sqrt(len(g)), (g.mean(), exact)
    assert np.isfinite(pol.theta).all() and np.linalg.norm(pol.theta) > 0          # the training did move the learned part
    print(f"selftest ok: untrained taught policy = teacher; trained one unbiased (|theta| = {np.linalg.norm(pol.theta):.2f}); the main author's class restored")


if __name__ == "__main__":
    selftest()
    run()
