"""Saturation modules: what the uncapped small world is missing in a large pool. Additive, read-only.

multi_country.py: the uncapped small world guides RDPO at every size, but it gets worse with size (28 countries at
US$40m each: relative variance 10.2, its own answer 45x too high). Reason: a large pool crosses its threshold only
when many countries are near their payout cap, and a capped country adds nothing more. The uncapped world cannot see that.
One small module per risk source, answering one question: "when the pool's running loss is L, in what state are the
one or two countries I touch?" A module is a table rho_k(state | L), at most 100 states. All modules speak about the
same coordinate, the running loss, so they combine in one backward pass:
    V_k(L) = sum_state rho_k(state | L) * sum_v p_k(v) * V_{k+1}(L + capped increment of outcome v in that state).
With exact modules this is the exact projection of the pool onto (source, running loss). The modules are estimated
from a pilot run of the current teacher (importance-weighted counts; a loss never visited borrows the nearest), so this small
world is no longer sampling-free; the pilot is charged as build time. Storage and work grow linearly with the sources.
Pre-set rules (rings of 10, 20, 28 countries at US$40m each, and the ap_risk pool; reference: the twist, 2,000,000 paths):
 C1 every estimate is within 3 standard errors of the twist's;
 C2 28-country ring: relative variance <= half the uncapped teacher's;
 C3 the small world's own answer is within a factor 2 of the truth at every size;
 C4 28-country ring: build <= 3 s and equal-CPU efficiency against the twist above the uncapped teacher's (build charged).
"""
import time

import numpy as np

import multi_country as M
import rdpo_engine as E
import small_world as W
import small_world_loop as L
from _src import save, seed
from src.rdpo.problem import PortfolioProblem

# pilot paths; half-width (loss units) of the pooling over L. 0 after a probe on the two-country model (pooling 4 units biased the
# own answer by 0.7%, none is exact) and on the 10-country ring (no gain); the 28-country results were not looked at for this choice.
PILOT, SMOOTH = 50_000, 0


def sites_of(pol, k):
    return list(pol.next[k].keys())


def increments(pol, k):
    """D[state, v]: capped loss increment of outcome v when the touched countries are in `state` (codes, base 10)."""
    sites = sites_of(pol, k)
    V = len(pol.P.src[k]["p"])
    codes = np.indices((10,) * len(sites)).reshape(len(sites), -1)             # (sites, 10^sites)
    return sum(pol.loss_tab[pol.next[k][i][c]] - pol.loss_tab[c][:, None] for i, c in zip(sites, codes)) * np.ones((1, V), dtype=np.int64)


def modules(pol, n, rng):
    """rho_k(state | L) under the model's own law, from a weighted pilot of `pol`: list of (Lmax + 1, 10^sites) tables."""
    P = pol.P
    H = [np.zeros((P.Lmax + 1, 10 ** len(sites_of(pol, k)))) for k in range(P.K)]

    def trace(k, S, Lt, logw):
        c = np.zeros(len(Lt), dtype=np.int64)
        for i in sites_of(pol, k):
            c = c * 10 + S[:, i]
        np.add.at(H[k], (Lt, c), np.exp(logw))

    E.rollout(pol, n, rng, trace=trace)
    ker = np.ones(2 * SMOOTH + 1)
    out = []
    for h in H:
        h = np.apply_along_axis(np.convolve, 0, h, ker, "same")               # pool neighbouring losses
        tot = h.sum(1)
        seen = np.flatnonzero(tot > 0)
        near = seen[np.abs(np.arange(len(tot))[:, None] - seen[None, :]).argmin(1)]   # a loss never visited borrows the nearest visited one
        out.append(h[near] / tot[near][:, None])
    return out


def value_tables(pol, rho):
    """Same contract as small_world.value_tables, with the modules in the backward pass."""
    P = pol.P
    Lg = np.arange(P.Lmax + 1)
    V = P.payoff_from(Lg, 0 * Lg, 0 * Lg).astype(float)
    out = [None] * P.K
    for k in reversed(range(P.K)):
        out[k] = V
        D, p = increments(pol, k), P.src[k]["p"]
        A = (p[None, None, :] * V[np.minimum(Lg[:, None, None] + D[None, :, :], P.Lmax)]).sum(2)   # (L, state)
        V = (rho[k] * A).sum(1)
    return out, float(V[0])


def build(P, rounds=1, n=PILOT, s=0):
    """Uncapped teacher -> pilot -> modules -> capped tables (-> pilot of the new teacher -> ...). Returns policy, own answer, CPU seconds, modules."""
    t0 = time.process_time()
    pol, answer, _ = W.teacher(P)
    rng = np.random.default_rng(seed(20261202 + s))
    rho = None
    for _ in range(rounds):
        rho = modules(pol, n, rng)
        tables = value_tables(pol, rho)
        pol, answer, _ = W.teacher(P, tables)
    return pol, answer, time.process_time() - t0, rho


def cell(P, rounds=(1, 2), n_tw=2_000_000, n=200_000):
    p, p_se, rho2_tw, c_tw = L.twist_ref(P, n_tw, np.random.default_rng(seed(1)))
    pol0, a0, t0 = W.teacher(P)
    s0 = L.stat(pol0, p, p_se, n)
    eq = lambda st, tb: rho2_tw * c_tw / (st["rho2_worst"] * (st["cost_per_path"] + tb / n))
    out = {"sources": P.K, "p_twist": p, "p_twist_se": p_se, "rho2_twist": rho2_tw,
           "uncapped": {"rho2": s0["rho2_worst"], "z_max": s0["z_max"], "answer_over_truth": a0 / p, "cpu_build": t0, "equal_cpu_gain": eq(s0, t0)}}
    for r in rounds:
        pol, a, tb, _ = build(P, r)
        st = L.stat(pol, p, p_se, n)
        out[f"modules_{r}"] = {"rho2": st["rho2_worst"], "z_max": st["z_max"], "answer_over_truth": a / p, "cpu_build": tb,
                               "cost_ratio": st["cost_per_path"] / c_tw, "equal_cpu_gain": eq(st, tb)}
    return out


def run():
    import ap_risk
    pools = {f"ring {n}": M.pool(n, 40e6) for n in (10, 20, 28)}
    pools["ap_risk 28"] = PortfolioProblem(ap_risk.inputs()["model"], "exceedance", threshold=ap_risk.THRESH[0] * 1e6)
    res = {k: cell(P) for k, P in pools.items()}
    best = {k: min((r[m] for m in ("modules_1", "modules_2")), key=lambda x: x["rho2"]) for k, r in res.items()}
    big = res["ring 28"]
    res["checks"] = {"C1_unbiased": bool(max(r[m]["z_max"] for r in res.values() for m in ("uncapped", "modules_1", "modules_2")) <= 3),
                     "C2_ring28_variance_halved": bool(best["ring 28"]["rho2"] <= 0.5 * big["uncapped"]["rho2"]),
                     "C3_own_answer_within_2x": {k: bool(0.5 <= b["answer_over_truth"] <= 2) for k, b in best.items()},
                     "C4_ring28_build_and_equal_cpu": bool(best["ring 28"]["cpu_build"] <= 3 and best["ring 28"]["equal_cpu_gain"] > big["uncapped"]["equal_cpu_gain"])}
    save("cap_modules.json", res)
    for k, r in res.items():
        if k == "checks":
            continue
        print(f"{k} ({r['sources']} sources, p {r['p_twist']:.3g}, twist relative variance {r['rho2_twist']:.3g})")
        for m in ("uncapped", "modules_1", "modules_2"):
            x = r[m]
            print(f"   {m:10s}: relative variance {x['rho2']:.3g} | own answer {x['answer_over_truth']:.2f}x the truth | build {x['cpu_build']:.2f} s | equal-CPU x{x['equal_cpu_gain']:.2f} | max |z| {x['z_max']:.1f}")
    print("checks:", res["checks"])
    return res


def selftest():
    """Two-country model, exact by enumeration: with modules the small world's own answer moves to the truth and the policy stays unbiased."""
    import dreamwake as D
    P = D.problem("exceedance")
    exact = D.exact(P)[2]
    pol0, a0, _ = W.teacher(P)
    pol, a, _, rho = build(P, 1, n=100_000)
    assert all(np.allclose(r.sum(1), 1) for r in rho)
    o = E.rollout(pol, 300_000, np.random.default_rng(6))
    g = o["w"] * o["f"]
    assert abs(g.mean() - exact) < 4 * g.std() / np.sqrt(len(g)), (g.mean(), exact)
    assert abs(a / exact - 1) < abs(a0 / exact - 1) and abs(a / exact - 1) < 0.01, (a0, a, exact)
    print(f"selftest ok: exact {exact:.6f}; own answer uncapped {a0:.6f}, with modules {a:.6f}; policy unbiased")


if __name__ == "__main__":
    selftest()
    run()
