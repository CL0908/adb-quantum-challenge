"""A library of saturation modules and learned corrections, so a new world borrows instead of rebuilding. Additive, read-only.

cap_modules.py builds one module per risk source from a pilot run (about 3 s on 28 countries); small_world_loop.py
trains the main author's features on top (about 30 s). When the twin moves to another world, what must be redone?
Always, and exactly: the backward pass with the new world's own probabilities (the source laws change with the world).
Borrowed from the nearest library entry: the modules rho_k(state | L) and the learned correction theta.
A new world: nearest entry (distance weighted by how strongly each input moves the small world's own answer) ->
backward pass with the borrowed modules -> a short check run; if its relative variance is within TOL of the entry's own,
use it; otherwise build fresh modules by a pilot and add the world to the library.
28-country ring at US$40m each, 30 worlds drawn uniformly from the twin's ranges (far apart: the hard case).
Rules (written after a probe on 5 other worlds, which showed borrowed modules within 6% of fresh ones):
 L1 every estimate is within 3 standard errors of the twist's (1,000,000 paths per world);
 L2 with the library, the relative variance in every world is <= 1.5x that of fresh modules in the same world;
 L3 the library builds fresh modules in at most a third of the worlds;
 L4 the correction trained once in the base world, borrowed in 5 other worlds, gives a relative variance <= 1.5x that
    of training again in that world (so one training serves the stream);
 L5 total preparation CPU with the library is <= half of building fresh modules in every world.
The small world's own answer is reported for both; it is not used by the estimate.
"""
import time

import numpy as np

import cap_modules as C
import dreamwake as D
import multi_country as M
import rdpo_engine as E
import small_world as W
import small_world_loop as L
from _src import save, seed

TOL, CHECK = 1.5, 10_000


def rel_var(pol, n, s):
    o = E.rollout(pol, n, np.random.default_rng(s))
    g = o["w"] * o["f"]
    return float(g.var() / g.mean() ** 2)


class Library:
    def __init__(self, make):
        self.make, self.entries, self.fresh_builds = make, [], 0             # make(world) -> PortfolioProblem

    def gradient(self, world):
        """d log(own answer of the uncapped small world) / d input: which inputs matter, at no sampling cost."""
        g = {}
        for n in D.INPUTS:
            lo, hi = D.RANGE[n]
            a, b = max(lo, world[n] - D.EPS[n]), min(hi, world[n] + D.EPS[n])
            fa, fb = (np.log(W.value_tables(self.make({**world, n: x}))[1]) for x in (a, b))
            g[n] = (fb - fa) / (b - a)
        return g

    def add(self, world, P=None):
        P = P or self.make(world)
        pol, answer, _, rho = C.build(P, 1, s=len(self.entries))
        e = {"world": dict(world), "rho": rho, "theta": None, "grad": self.gradient(world), "rho2_check": rel_var(pol, CHECK, 99)}
        self.entries.append(e)
        self.fresh_builds += 1
        return pol, answer, e

    def nearest(self, world):
        return min(self.entries, key=lambda e: sum(abs(e["grad"][n] * (world[n] - e["world"][n])) for n in D.INPUTS))

    def policy(self, world):
        """Returns policy, own answer, what happened ('borrowed' or 'fresh'), preparation CPU seconds."""
        t0 = time.process_time()
        P = self.make(world)
        e = self.nearest(world)
        pol, answer, _ = W.teacher(P, C.value_tables(W.teacher(P)[0], e["rho"]))
        how = "borrowed"
        if rel_var(pol, CHECK, 98) > TOL * e["rho2_check"]:
            pol, answer, _ = self.add(world, P)
            how = "fresh"
        return pol, answer, how, time.process_time() - t0


def stream(M_worlds=30, n=100_000):
    make = lambda w: M.pool(28, 40e6, **w)
    rng = np.random.default_rng(seed(20261203))
    lib = Library(make)
    lib.add(dict(D.BASE))
    rows = []
    for w in D.worlds(rng, M_worlds):
        P = make(w)
        p, p_se, rho2_tw, _ = L.twist_ref(P, 1_000_000, rng)
        pol_f, a_f, t_f, _ = C.build(P, 1, s=1000 + len(rows))
        pol_l, a_l, how, t_l = lib.policy(w)
        sf, sl = L.stat(pol_f, p, p_se, n, seeds=(71,)), L.stat(pol_l, p, p_se, n, seeds=(71,))
        rows.append({**w, "p_twist": p, "rho2_twist": rho2_tw, "rho2_fresh": sf["rho2_worst"], "rho2_library": sl["rho2_worst"], "z_fresh": sf["z_max"],
                     "z_library": sl["z_max"], "how": how, "cpu_fresh": t_f, "cpu_library": t_l, "answer_fresh_over_truth": a_f / p, "answer_library_over_truth": a_l / p})
    return rows, lib


def one_training(lib, k=5, n=100_000, b_fit=65_536):
    """The correction trained once at the base world, borrowed in k other worlds, against training again in each."""
    make = lib.make
    rng = np.random.default_rng(seed(20261204))
    P0 = make(dict(D.BASE))
    base = lib.entries[0]
    tabs0 = C.value_tables(W.teacher(P0)[0], base["rho"])[0]
    pol0, t0 = L.train(P0, b_fit, rng, tables=tabs0)
    base["theta"] = pol0.theta.copy()
    rows = []
    for w in D.worlds(rng, k):
        P = make(w)
        p, p_se, _, _ = L.twist_ref(P, 1_000_000, rng)
        tabs = C.value_tables(W.teacher(P)[0], base["rho"])[0]
        cfg = L.final_config(P.target)
        borrowed = type("T", (L.Taught,), {"base_tables": tabs})(P, cfg.alpha, cfg.edge_features, cfg.relevance)
        plain = borrowed.__class__(P, cfg.alpha, cfg.edge_features, cfg.relevance)                 # same tables, nothing learned
        borrowed.theta = base["theta"].copy()
        again, t_again = L.train(P, b_fit, rng, tables=tabs)
        s = {k_: L.stat(pol, p, p_se, n, seeds=(72,)) for k_, pol in (("no_correction", plain), ("borrowed", borrowed), ("trained_again", again))}
        rows.append({**w, "p_twist": p, **{f"rho2_{k_}": v["rho2_worst"] for k_, v in s.items()}, "z_max": max(v["z_max"] for v in s.values()), "cpu_training_again": t_again})
    return {"cpu_training_base": t0, "worlds": rows}


def run():
    rows, lib = stream()
    tr = one_training(lib)
    ratio = np.array([r["rho2_library"] / r["rho2_fresh"] for r in rows])
    tw = np.array([r["rho2_borrowed"] / r["rho2_trained_again"] for r in tr["worlds"]])
    res = {"worlds": rows, "entries": len(lib.entries), "fresh_builds_in_stream": sum(r["how"] == "fresh" for r in rows),
           "worst_ratio_library_over_fresh": float(ratio.max()), "median_ratio_library_over_fresh": float(np.median(ratio)),
           "cpu_fresh_total": sum(r["cpu_fresh"] for r in rows), "cpu_library_total": sum(r["cpu_library"] for r in rows),
           "answer_library_over_truth_range": [min(r["answer_library_over_truth"] for r in rows), max(r["answer_library_over_truth"] for r in rows)],
           "answer_fresh_over_truth_range": [min(r["answer_fresh_over_truth"] for r in rows), max(r["answer_fresh_over_truth"] for r in rows)],
           "one_training": tr, "worst_ratio_borrowed_over_trained_again": float(tw.max())}
    res["checks"] = {"L1_unbiased": bool(max([max(r["z_fresh"], r["z_library"]) for r in rows] + [r["z_max"] for r in tr["worlds"]]) <= 3),
                     "L2_library_within_1.5x_of_fresh": bool(ratio.max() <= 1.5),
                     "L3_fresh_builds_le_third": bool(res["fresh_builds_in_stream"] <= len(rows) / 3),
                     "L4_one_training_serves": bool(tw.max() <= 1.5),
                     "L5_preparation_cpu_halved": bool(res["cpu_library_total"] <= 0.5 * res["cpu_fresh_total"])}
    save("policy_library.json", res)
    print(f"{len(rows)} worlds, 28-country ring, p from {min(r['p_twist'] for r in rows):.2g} to {max(r['p_twist'] for r in rows):.2g}: library builds fresh in {res['fresh_builds_in_stream']} "
          f"({res['entries']} entries) | relative variance library / fresh: median {res['median_ratio_library_over_fresh']:.2f}, worst {res['worst_ratio_library_over_fresh']:.2f} | "
          f"preparation CPU {res['cpu_library_total']:.0f} s vs {res['cpu_fresh_total']:.0f} s | own answer / truth: library {res['answer_library_over_truth_range'][0]:.2f}-{res['answer_library_over_truth_range'][1]:.2f}, "
          f"fresh {res['answer_fresh_over_truth_range'][0]:.2f}-{res['answer_fresh_over_truth_range'][1]:.2f}")
    for r in tr["worlds"]:
        print(f"  one training: no correction {r['rho2_no_correction']:.3g}, borrowed {r['rho2_borrowed']:.3g}, trained again {r['rho2_trained_again']:.3g} ({r['cpu_training_again']:.0f} s)")
    print("checks:", res["checks"])
    return res


def selftest():
    """Two-country model, exact probabilities: a borrowed-module policy in another world stays unbiased, and a forced miss builds a new entry."""
    make = lambda w: D.problem("exceedance", **w)
    lib = Library(make)
    lib.add(dict(D.BASE))
    w = {"m": 1.6, "rain_mult": 1.5, "rain_shared": 0.1}
    pol, _, how, _ = lib.policy(w)
    exact = D.exact(make(w))[2]
    o = E.rollout(pol, 300_000, np.random.default_rng(8))
    g = o["w"] * o["f"]
    assert how == "borrowed" and abs(g.mean() - exact) < 4 * g.std() / np.sqrt(len(g)), (how, g.mean(), exact)
    lib.entries[0]["rho2_check"] = 0.0                                         # nothing can pass the check now
    assert lib.policy(w)[2] == "fresh" and len(lib.entries) == 2
    print("selftest ok: borrowed modules unbiased in another world; a failed check adds a library entry")


if __name__ == "__main__":
    selftest()
    run()
