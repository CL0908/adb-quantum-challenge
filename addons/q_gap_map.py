"""Where is the gap smallest? The same accounts on larger rings and on dense pools. Additive, read-only.

q_link.py and q_link_gates.py were run on the main author's rings up to 28 countries. Two things could move the gap:
size (the cap-aware loader went from a loss at 8 countries to a gain of 1.8 at 28) and density (on dense pools the
small world is weaker, dense_pools.py, while the quantum circuit still grows linearly with the number of sources).
Pools: rings of 28, 40 and 56 countries at US$40m per country (his setting, T = 16 n); the complete graphs of 6 and 10
countries at US$40m; the circulants of 28 countries with degree 8 and 12 at US$30m (the pools of dense_pools.py).
For each pool, at eps = 1%:
  classical   the faster of the uncapped small world and the small world with modules, 100,000 paths, build charged
              (quantum_side.classical); the reference probability is exact on rings (junction tree) and on the complete
              graph of 6 (enumeration), otherwise the twist with 2,000,000 paths from dense_pools.json
  quantum     his oracle counts for that pool (src/quantum/advantage.bitwise_oracle_counts), his twisted amplitude,
              ChebAE calls simulated, mixed-fallback rotations with QROM levels (the re-pricing already in the paper);
              and the cap-aware loader of q_link_gates.py with tree preparation, single tilt chosen from k/128,
              k = 8..16, on a 200,000-path pilot and measured on 1,000,000 fresh paths
Pre-set rules:
 R1 on some pool the best counted s* (twisting or loader) is at least 10x below that of the ring of 28
 R2 on rings the gain of the loader over twisting keeps growing with size and exceeds 2 at 56 countries
Limits: classical times are single timings on this machine; the reference on three dense pools is itself a sampling
estimate; the gate formulas of the loader assume two touched countries per source, which holds for every pool here.
"""
import json

import numpy as np

import dense_pools as DP
import multi_country as M
import q_link_gates as G
import quantum_side as QS
from _src import OUT, save, seed
from q_link import cheb_calls
from src.quantum import twisted as TW
from src.quantum.advantage import bitwise_oracle_counts
from src.tensor.junction import junction_solve

EPS, TILTS, N_PILOT, N_FINAL = 0.01, range(8, 17), 200_000, 1_000_000


def pools():
    out = {f"ring {n}": M.pool(n, 40e6) for n in (28, 40, 56)}
    out.update({name: DP.pool(name) for name in DP.POOLS})
    return out


def reference(name, P):
    if name.startswith("ring"):
        return junction_solve(P)["exceedance"], "exact"
    r = json.loads((OUT / "dense_pools.json").read_text(encoding="utf-8"))[name]
    return (r["p_exact"], "exact") if "p_exact" in r else (r["p_twist"], "twist, 2,000,000 paths")


def his_row(P, mu):
    """The fields of his advantage rows that the pricing needs, for any pool."""
    T = int(round(P.T))
    return {"n": P.n, "mu": mu, "T_units": T, "a_twisted": TW.twisted_amplitude(P.model, mu, T)["a_twisted"], **bitwise_oracle_counts(P.model, T)}


def cell(name, P):
    mu, ref = reference(name, P)
    h = his_row(P, mu)
    cl = QS.classical(P, mu)
    t = {k: v["cpu_build"] + v["rho2"] / EPS ** 2 * v["t_path"] for k, v in cl.items()}
    best = min(t, key=t.get)
    rng = np.random.default_rng(seed(4100 + P.K))
    pilot = []
    for k in TILTS:
        st = G.stats(*G.sample(G.design(P.model, [k / 128], capaware=True, T=h["T_units"]), N_PILOT, rng), mu)
        if abs(st["z"]) < 4 and st["amplitude"] > 0:
            pilot.append((st["amplitude"], k, st["cap"]))
    amp, k, cap = max(pilot)
    st = G.stats(*G.sample(G.design(P.model, [k / 128], capaware=True, T=h["T_units"]), N_FINAL, np.random.default_rng(seed(5100 + P.K))), mu, cap)
    c0, c1 = cheb_calls(h["a_twisted"], EPS)["calls"], cheb_calls(st["amplitude"], EPS)["calls"]
    q0, q1 = G.cost(h, c0), G.cost(h, c1, capaware=True)
    return {"countries": P.n, "edges": len(P.edges), "sources": P.K, "mu": mu, "reference": ref, "a_twisted": h["a_twisted"],
            "classical": {"kind": best, "seconds": t[best], "rho2": cl[best]["rho2"], "rho2_uncapped": cl["uncapped"]["rho2"], "rho2_modules": cl["modules"]["rho2"]},
            "loader": {"tilt": k / 128, "amplitude": st["amplitude"], "pilot_amplitude": amp, "bias": st["bias"], "estimate_over_mu": st["estimate_over_mu"], "z": st["z"], "rho2": st["rho2"]},
            "twisting": {"calls": c0, "T_per_call": q0["T_per_call"], "quantum_seconds": q0["runtime_s"], "physical_qubits": q0["physical_qubits"], "s_star": q0["runtime_s"] / t[best]},
            "loader_tree": {"calls": c1, "T_per_call": q1["T_per_call"], "quantum_seconds": q1["runtime_s"], "physical_qubits": q1["physical_qubits"], "s_star": q1["runtime_s"] / t[best]},
            "loader_gain": q0["runtime_s"] / q1["runtime_s"], "best_s_star": min(q0["runtime_s"], q1["runtime_s"]) / t[best]}


def run():
    res = {"eps_rel": EPS, "pools": {}}
    for name, P in pools().items():
        res["pools"][name] = cell(name, P)
        r = res["pools"][name]
        print(f"{name} ({r['sources']} sources, mu {r['mu']:.3g} {r['reference']}): classical {r['classical']['kind']} {r['classical']['seconds']:.3g} s "
              f"(relative variance uncapped {r['classical']['rho2_uncapped']:.3g}, modules {r['classical']['rho2_modules']:.3g})", flush=True)
        print(f"   amplitude twisted {r['a_twisted']:.3g} -> loader {r['loader']['amplitude']:.3g} (tilt {r['loader']['tilt'] * 128:.0f}/128, bias {r['loader']['bias']:.2%}, z {r['loader']['z']:+.1f}) | "
              f"s* twisting {r['twisting']['s_star']:.3g} ({r['twisting']['calls']:.3g} calls x {r['twisting']['T_per_call'] / 1e3:.0f}k T), loader {r['loader_tree']['s_star']:.3g} "
              f"({r['loader_tree']['calls']:.3g} x {r['loader_tree']['T_per_call'] / 1e3:.0f}k T), gain x{r['loader_gain']:.2f}", flush=True)
    p = res["pools"]
    base = p["ring 28"]["best_s_star"]
    for r in p.values():
        r["gap_vs_ring_28"] = base / r["best_s_star"]
    g = [p[f"ring {n}"]["loader_gain"] for n in (28, 40, 56)]
    res["checks"] = {"R1_some_pool_10x_below_ring_28": bool(max(r["gap_vs_ring_28"] for r in p.values()) >= 10),
                     "R2_loader_gain_grows_and_passes_2_at_56": bool(g[0] < g[1] < g[2] and g[2] >= 2)}
    save("q_gap_map.json", res)
    print("best s* per pool, and how far below the ring of 28: " + " | ".join(f"{k} {r['best_s_star']:.3g} (x{r['gap_vs_ring_28']:.2f})" for k, r in p.items()))
    print("checks:", res["checks"])
    return res


def selftest():
    """For the ring of 28 the row built here is the main author's own row, and the pricing of it is the one in q_link_gates.py."""
    P = M.pool(28, 40e6)
    his = {r["n"]: r for r in json.loads(QS.reference().read_text(encoding="utf-8"))["advantage"] if r["topology"] == "ring"}[28]
    h = his_row(P, junction_solve(P)["exceedance"])
    assert all(h[k] == his[k] for k in ("toffoli_A", "rotations_A", "logical_qubits", "w_U")) and abs(h["a_twisted"] / his["a_twisted"] - 1) < 1e-6, (h, his)
    assert abs(G.cost(h, 8000.0)["T_count"] / G.cost(his, 8000.0)["T_count"] - 1) < 1e-12
    print(f"selftest ok: ring 28 row rebuilt (amplitude {h['a_twisted']:.6g}, {h['toffoli_A']} Toffolis, {h['rotations_A']} rotations)")


if __name__ == "__main__":
    selftest()
    run()
