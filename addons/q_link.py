"""The gap ledger: how far the small world can pull the quantum side back, counted before any new circuit is designed.

quantum_side.py showed that the small world, taken as the classical baseline, raises the main author's required hardware
speed-up s* (16x at 28 countries, 25x at eps = 1%). The small world can also be handed to the quantum side: its table
tells a state loader where to put the probability, which raises the amplitude and cuts the calls. This file stacks every
lever we have on the main author's own rows (quantum_advantage_v3.json), ring pools at US$40m per country:
  declared     his model: Ross-Selinger rotations, IQAE calls (scaled 1/eps), twisted amplitude
  repriced     + cheaper rotations (rotations.py, mixed fallback + QROM levels) and ChebAE calls (ae.py), both already
                 in the paper; ChebAE calls are simulated here at every amplitude and precision, not extrapolated
  u_loader     + a loader that samples the uncapped world given U > T. Its weights are constant on the event, so the
                 amplitude is exactly mu / mu_uncapped (quantum_side.py)
  table        + a loader that follows the uncapped table on the capped progress (the small world as it runs
                 classically). Any bounded-weight encoding of a sampler with relative variance rho2 has amplitude
                 a <= 1 / (1 + rho2), because E_q[w^2 1_A] <= w_max mu. The bound is used, so this row is a ceiling
  modules      + the same with saturation modules (cap_modules.py), a <= 1 / (1 + rho2_modules). Also a ceiling
The three loader rows keep the per-call gate count of the twisted loader. That is the optimistic assumption this file
exists to bound: a state-dependent loader costs more per call, and q_link_gates.py has to count it. Each row is priced
against two classical times: his RDPO (his file) and the best small world (this machine, quantum_side.json).
Separately, as a hardware assumption and not a gain of ours: his runtime model has 4 T factories and runtime scales as
1 / factories until the reaction-time floor (T-count x 10 microseconds, not in his model). The curve is reported for
4, 8, 16, 32 factories with the physical qubits each costs.
Pre-set rules, at 28 countries and eps = 1%, mixed-fallback pricing, 4 factories:
 G1 the u_loader row against the small world is no worse than repriced twisting against his RDPO
    (the link cancels what the small world cost the quantum side)
 G2 the same for the modules ceiling
 G3 the u_loader row against the small world is below the s* his file declares at the same precision
Limits: classical times are single timings (his on his machine, ours on this one, build times at the 0.016 s timer
resolution); the loader rows are ceilings, not circuits; the deeper circuits of ChebAE and of small eps need better
logical qubits (kmax is recorded, s* does not charge for it); nothing here changes his criterion.
"""
import json
import math

import numpy as np

import ae
from _src import OUT, save, seed
from quantum_side import reference
from rotations import price
from src.quantum.resources import surface_code_footprint

SIZES, EPS, REPS = (8, 16, 20, 28), (0.1, 0.01, 0.001), 200
PRICING, FACTORIES, REACTION_S, FACTORY_QUBITS = "qrom_mixed", (4, 8, 16, 32), 1e-5, 6000
LOADERS = ("u_loader", "table", "modules")


def cheb_calls(a, eps, reps=REPS):
    g = np.random.default_rng(seed(7))
    res = np.array([ae.chebae(a, eps * a, 0.05, 100, g, odd_only=True) for _ in range(reps)], float)
    return {"calls": float(res[:, 1].mean()), "coverage": float(np.mean(abs(res[:, 0] - a) <= eps * a)), "kmax": float(res[:, 2].max())}


def amplitudes(h, r):
    sw = r["small_world"]
    return {"twisted": h["a_twisted"], "u_loader": r["a_ceiling_U_loader"], "table": 1 / (1 + sw["uncapped"]["rho2"]),
            "modules": 1 / (1 + sw["modules"]["rho2"])}


def classical(h, r, eps):
    sw = {k: v["cpu_build"] + v["rho2"] / eps ** 2 * v["t_path"] for k, v in r["small_world"].items()}
    return {"his_rdpo": h["classical_rho2"] / eps ** 2 * r["his_t_path"], "small_world": min(sw.values()), "small_world_kind": min(sw, key=sw.get)}


def factory_curve(p):
    """His footprint model with more factories, floored at the reaction time; logical qubits recovered from his own formula."""
    q = (p["physical_qubits"] - 4 * FACTORY_QUBITS) / (2 * p["code_distance"] ** 2)
    out = {}
    for f in FACTORIES:
        sc = surface_code_footprint(q, p["T_count"], factories=f)
        out[f] = {"runtime_s": max(sc["runtime_s"], p["T_count"] * REACTION_S), "physical_qubits": sc["physical_qubits"],
                  "at_reaction_floor": bool(sc["runtime_s"] < p["T_count"] * REACTION_S)}
    return out


def run():
    his = {r["n"]: r for r in json.loads(reference().read_text(encoding="utf-8"))["advantage"] if r["topology"] == "ring"}
    side = json.loads((OUT / "quantum_side.json").read_text(encoding="utf-8"))["rows"]
    res = {"his_reference": reference().name, "pricing": PRICING, "rows": {}}
    for n in SIZES:
        h, r = his[n], side[str(n)]
        amp = amplitudes(h, r)
        row = {"amplitude": amp, "by_eps": {}}
        for eps in EPS:
            tc = classical(h, r, eps)
            q = {"declared": {"calls": h["quantum_calls"] * h["eps_rel"] / eps, **price(h, "rs", h["quantum_calls"] * h["eps_rel"] / eps)}}
            for name, a in (("repriced", amp["twisted"]), *((k, amp[k]) for k in LOADERS)):
                c = cheb_calls(a, eps)
                q[name] = {**c, **price(h, PRICING, c["calls"])}
            lv = {k: {"calls": v["calls"], "coverage": v.get("coverage"), "kmax": v.get("kmax"), "quantum_seconds": v["runtime_s"], "T_count": v["T_count"],
                      "s_star_vs_his_rdpo": v["runtime_s"] / tc["his_rdpo"], "s_star_vs_small_world": v["runtime_s"] / tc["small_world"]} for k, v in q.items()}
            row["by_eps"][str(eps)] = {"classical_seconds": tc, "levels": lv,
                                       "factor": {"small_world_raises": tc["his_rdpo"] / tc["small_world"],
                                                  "repricing_lowers": q["declared"]["runtime_s"] / q["repriced"]["runtime_s"],
                                                  **{f"{k}_lowers": q["repriced"]["runtime_s"] / q[k]["runtime_s"] for k in LOADERS}},
                                       "factories": {k: factory_curve(q[k]) for k in ("repriced", "u_loader")}}
        res["rows"][n] = row
    b = res["rows"][28]["by_eps"]["0.01"]["levels"]
    res["checks"] = {"G1_u_loader_cancels_small_world": bool(b["u_loader"]["s_star_vs_small_world"] <= b["repriced"]["s_star_vs_his_rdpo"]),
                     "G2_modules_ceiling_cancels_small_world": bool(b["modules"]["s_star_vs_small_world"] <= b["repriced"]["s_star_vs_his_rdpo"]),
                     "G3_u_loader_below_his_declared": bool(b["u_loader"]["s_star_vs_small_world"] <= b["declared"]["s_star_vs_his_rdpo"])}
    save("q_link.json", res)
    for n, row in res["rows"].items():
        a = row["amplitude"]
        print(f"ring {n}: amplitude twisted {a['twisted']:.3g} | u_loader {a['u_loader']:.3g} | table <= {a['table']:.3g} | modules <= {a['modules']:.3g}")
        for eps, e in row["by_eps"].items():
            lv, f = e["levels"], e["factor"]
            print(f"   eps {eps}: s* declared {lv['declared']['s_star_vs_his_rdpo']:.3g} (his RDPO) | against the small world ({e['classical_seconds']['small_world_kind']}, x{f['small_world_raises']:.1f}): "
                  + " | ".join(f"{k} {lv[k]['s_star_vs_small_world']:.3g}" for k in ("declared", "repriced", *LOADERS)))
            print(f"      factors: repricing /{f['repricing_lowers']:.1f}, u_loader /{f['u_loader_lowers']:.1f}, table /{f['table_lowers']:.1f}, modules /{f['modules_lowers']:.1f}"
                  f" | twisting repriced against his RDPO {lv['repriced']['s_star_vs_his_rdpo']:.3g}")
    e = res["rows"][28]["by_eps"]["0.01"]
    for k, curve in e["factories"].items():
        print(f"ring 28, eps 0.01, {k}, factories: " + " | ".join(
            f"{f}: {c['runtime_s'] / e['classical_seconds']['small_world']:.3g} ({c['physical_qubits'] / 1e6:.2f}m qubits{', floor' if c['at_reaction_floor'] else ''})" for f, c in curve.items()))
    print("checks:", res["checks"])
    return res


def selftest():
    """His own runtime is reproduced; ChebAE calls follow c / (eps sqrt(a)) with one constant; the factory curve starts at his number."""
    his = {r["n"]: r for r in json.loads(reference().read_text(encoding="utf-8"))["advantage"] if r["topology"] == "ring"}
    h = his[28]
    p = price(h, "rs")
    assert abs(p["runtime_s"] / h["quantum_seconds"] - 1) < 1e-9 and p["code_distance"] == h["code_distance"]
    c = [cheb_calls(a, eps, 40)["calls"] * eps * math.sqrt(a) for a in (h["a_twisted"], 0.02) for eps in (0.1, 0.01)]
    assert 15 < min(c) and max(c) < 35, c
    f = factory_curve(p)
    assert abs(f[4]["runtime_s"] / h["quantum_seconds"] - 1) < 1e-9 and f[4]["physical_qubits"] == h["physical_qubits"] and f[8]["runtime_s"] < f[4]["runtime_s"]
    print(f"selftest ok: his runtime reproduced ({p['runtime_s']:.4g} s, d = {p['code_distance']}); ChebAE constant {min(c):.1f}-{max(c):.1f}; 4 factories give his {f[4]['physical_qubits']} qubits")


if __name__ == "__main__":
    selftest()
    run()
