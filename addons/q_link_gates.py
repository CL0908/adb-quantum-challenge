"""A loader that reads the state of the pool: measured amplitude and counted gates, to see what is left of the q_link.py ceilings.

q_link.py stacked ceilings under the assumption that a state-dependent loader costs the same per call as the twisted one.
Here concrete loaders are built on the main author's source picture (src/quantum/structured_loading.py: independent
sources, increments added into registers, caps applied at the end) and both sides of the trade are measured.
Why twisting loses amplitude. His weight is e^{Lambda - eta U} with U the uncapped loss, bounded on hits by its value at
U = T. A hit of the capped loss needs U well above T (the rest is wasted on saturated layers), so most hits carry a
weight far below the bound: at 28 countries 19% of twisted paths are hits but the amplitude is 0.00088.
Loaders. Sources are loaded country by country (a country's solo sources, then the edges it starts).
  progress   before each country the circuit reads a progress register and picks one of J + 1 tilts, raising the tilt
             when the path is behind: j = number of thresholds the gap T - progress exceeds, thresholds = kappa x the
             mean remaining loss under the mid tilts. Progress is the uncapped sum ("U") or the capped loss ("capped",
             the small-world reading). The tilt still acts on uncapped increments.
  cap-aware  each source is tilted on what it can still pay: q(x | c) = p(x) e^{eta e(x; c)} / Z(c), c the capped state
             of the countries it touches (5 states for a solo earthquake source, 25 for an edge, 2 and 4 for rain) and
             e the capped increment. The weight is e^{-eta L + sum log Z(c)}: it is tied to the capped loss L, which is
             what the event is about. This is the saturation-module idea of cap_modules.py in the source picture.
  both       cap-aware with progress-dependent tilts.
  J = 0 with his eta and no capped state is his loader.
Weights in the circuit. log w = sum of log Z (lookup-add in fixed point) - sum of tilt x increment (adders; tilts are
multiples of 1/128). Y = c - log w is formed in a register and the objective rotations act on its bits, as his act on
the bits of U. Weights above the cap c are clipped, which biases the answer low; c is the smallest cap whose clipped
mass is <= 0.2% of mu on a pilot sample, and the bias is measured again on a fresh sample. a = E_q[f min(w, c)] / c.
Gates of one A, by explicit component formulas in the style of resources.py (estimates, not compiled circuits), on top
of his Toffolis and reflection: comparators for the tilt index; the running capped loss; the capped state of the
touched countries; lookup-adds of log Z; the adders for tilt x increment and for Y; B_W objective rotations instead of
w_U. Source preparation in two ways:
  tree   his Moettoenen levels with the angles loaded by QROM over (tilt, state, prefix) and one phase-gradient
         addition per level (rotations.py, same error budget); no extra qubits beyond the two b-bit registers.
         A cap-aware edge is prepared as two 3-qubit trees (first country's step, then the second given the first)
  alias  coherent alias sampling (Babbush et al. 2018): QROM over (tilt, state, outcome), a mu-bit comparison and a
         swap, an edge in two stages; cheap in Toffolis but leaves log2(outcomes) + 2 mu garbage qubits per stage,
         mu from a separate 1e-3 budget
Runtime and physical qubits from his surface_code_footprint; ChebAE calls simulated (ae.py); mixed-fallback rotations.
A design is chosen per size and preparation on a pilot sample of 200,000 paths by estimated runtime at eps = 1%, then
measured on a fresh sample of 2 million paths. Designs whose pilot estimate is off (|z| > 4) are not eligible.
Pre-set rules:
 L1 exactness on the 2-country ring by enumerating all 5000 source outcomes: each loader's law sums to 1, its weighted
    hits give the exact mu to 1e-9, J = 0 gives his twisted amplitude to 1e-9, and sampling agrees within 4 standard errors
 L2 at 28 countries the measured amplitude of the best loader that reads capped state reaches the exact U-loader
    amplitude (mu / mu_uncapped, 0.022)
 L3 at 28 countries and eps = 1%, the best loader with tree preparation has s* <= half that of repriced twisting
 L4 the same with alias preparation
How the rules came about, said plainly: they were written when only the progress family existed. A 200,000-path trial
showed that family below twisting at 28 countries (amplitude 0.0007 against 0.0009) for the reason given above, and the
cap-aware family was added before the recorded run. L2 first read "capped-progress loader"; the thresholds are unchanged.
The first recorded run priced a cap-aware edge under tree preparation as one 6-qubit tree over 25 states (1575 angles);
it gave 0.69 for L3 at 28 countries (log and results kept as repro/logs/q_link_gates_run1.*). The alias row already
used two stages, so the tree row was given the same two-stage preparation (350 angles) and the run repeated; samples,
seeds and chosen designs are unaffected except where the cheaper tree changes which design is fastest.
Limits: the 3-country ring has 1.25e8 source outcomes and was not enumerated; component formulas, not circuits; the
alias rows buy Toffolis with qubits and their physical-qubit count is reported next to them; classical times as in
q_link.py; the tilt grid is coarse (steps of 1/128 to 1/32) and no design was optimised beyond it.
"""
import json
import math

import numpy as np

from _src import OUT, save, seed
from q_link import cheb_calls, classical
from quantum_side import reference
from rotations import LEVELS_EDGE, LEVELS_EQ, price, t_rot
from src.quantum import structured_loading as SL
from src.quantum import twisted as TW
from src.quantum.resources import B_W, surface_code_footprint
from src.tensor.synthetic import SyntheticModel, topology

AU, RU, SIZES, EPS = 7, 4, (8, 16, 20, 28), (0.1, 0.01, 0.001)
N_PILOT, N_FINAL, CHUNK, BIAS = 200_000, 2_000_000, 250_000, 0.002
step = lambda lo, hi, d: [k / d for k in range(round(lo * d), round(hi * d) + 1)]
GRID = ([{"family": "progress", "etas": step(0, J * e, round(1 / e)), "kappa": 1.0, "progress": p, "capaware": False} for e, J in ((1 / 32, 3), (1 / 16, 3), (1 / 16, 7)) for p in ("U", "capped")]
        + [{"family": "cap-aware", "etas": [k / 128], "kappa": 1.0, "progress": "capped", "capaware": True} for k in (10, 11, 12, 13, 14)]
        + [{"family": "both", "etas": e, "kappa": k, "progress": "capped", "capaware": True} for e in (step(1 / 16, 1 / 8, 32), step(1 / 32, 1 / 8, 32), step(0, 7 / 32, 32)) for k in (0.7, 0.85)])
N_STATES = {"eq1": 5, "eq2": 25, "rain1": 2, "rain2": 4}


def increments(s, ell, capaware):
    """e[c, x]: what outcome x adds to the loss the tilt acts on, given capped state c of the touched countries."""
    if not capaware:
        return ell[None, :].astype(float)
    x = np.arange(len(s.law))
    if s.kind == "eq1":
        return np.array([AU * (np.minimum(c + x, 4) - c) for c in range(5)], float)
    if s.kind == "eq2":
        return np.array([AU * (np.minimum(ci + x // 5, 4) - ci + np.minimum(cj + x % 5, 4) - cj) for ci in range(5) for cj in range(5)], float)
    if s.kind == "rain1":
        return np.array([RU * (np.minimum(c + x, 1) - c) for c in range(2)], float)
    return np.array([RU * (np.minimum(ci + x, 1) - ci + np.minimum(cj + x, 1) - cj) for ci in range(2) for cj in range(2)], float)


def state(s, S, R, capaware):
    if not capaware:
        return np.zeros(len(S), np.int64)
    if s.kind == "eq1":
        return np.minimum(S[:, s.sites[0]], 4).astype(np.int64)
    if s.kind == "eq2":
        return (5 * np.minimum(S[:, s.sites[0]], 4) + np.minimum(S[:, s.sites[1]], 4)).astype(np.int64)
    if s.kind == "rain1":
        return np.minimum(R[:, s.sites[0]], 1).astype(np.int64)
    return (2 * np.minimum(R[:, s.sites[0]], 1) + np.minimum(R[:, s.sites[1]], 1)).astype(np.int64)


def design(model, etas, kappa=1.0, progress="U", capaware=False, T=None, **_):
    """Per country block: for each source the laws q[j, c, x], log-normalisers lz[j, c], increments e[c, x]; and the gap thresholds."""
    n, etas = model.n, np.asarray(etas, float)
    blocks = []
    for i in range(n):
        blk = []
        for s, ell in TW.source_losses(model):
            if s.sites[0] == i:
                e = increments(s, ell, capaware)
                q = s.law[None, None, :] * np.exp(etas[:, None, None] * e[None])
                Z = q.sum(2)
                blk.append({"s": s, "ell": ell, "e": e, "q": q / Z[..., None], "cdf": np.cumsum(q / Z[..., None], 2)[..., :-1], "lz": np.log(Z)})
        blocks.append(blk)
    mean = lambda t, eta: float((t["e"][0] * t["s"].law * np.exp(eta * t["e"][0])).sum() / (t["s"].law * np.exp(eta * t["e"][0])).sum())   # from an empty state
    mids = (etas[1:] + etas[:-1]) / 2
    thr = np.array([[kappa * sum(mean(t, m) for b in blocks[i:] for t in b) for m in mids] for i in range(n)]).reshape(n, len(mids))
    return {"n": n, "T": 16 * n if T is None else T, "etas": etas, "thr": thr, "blocks": blocks, "progress": progress, "capaware": capaware}


def loss(S, R):
    return AU * np.minimum(S, 4).sum(1) + RU * np.minimum(R, 1).sum(1)


def add(S, R, s, x):
    if s.kind == "eq1":
        S[:, s.sites[0]] += x
    elif s.kind == "eq2":
        S[:, s.sites[0]] += x // 5; S[:, s.sites[1]] += x % 5
    else:
        for c in s.sites:
            R[:, c] += x


def tilt_index(d, i, U, S, R):
    return ((d["T"] - (U if d["progress"] == "U" else loss(S, R)))[:, None] > d["thr"][i][None, :]).sum(1)


def sample(d, N, rng):
    hit, logw = [], []
    for m in [CHUNK] * (N // CHUNK) + ([N % CHUNK] if N % CHUNK else []):
        S, R, U, lw = np.zeros((m, d["n"]), np.int16), np.zeros((m, d["n"]), np.int16), np.zeros(m), np.zeros(m)
        for i, blk in enumerate(d["blocks"]):
            j = tilt_index(d, i, U, S, R)
            for t in blk:
                c = state(t["s"], S, R, d["capaware"])
                x = (rng.random(m)[:, None] > t["cdf"][j, c]).sum(1)
                lw += t["lz"][j, c] - d["etas"][j] * t["e"][c, x]; U += t["ell"][x]
                add(S, R, t["s"], x.astype(np.int16))
        hit.append(loss(S, R) > d["T"]); logw.append(lw)
    return np.concatenate(hit), np.concatenate(logw)


def enumerate_all(d):
    """Every source outcome with the loader's probability (small rings only)."""
    n = d["n"]
    q, lw, U, S, R = np.ones(1), np.zeros(1), np.zeros(1), np.zeros((1, n), np.int64), np.zeros((1, n), np.int64)
    for i, blk in enumerate(d["blocks"]):
        j = tilt_index(d, i, U, S, R)
        for t in blk:
            Lk, c = len(t["s"].law), state(t["s"], S, R, d["capaware"])
            x = np.tile(np.arange(Lk), len(q))
            q = (q[:, None] * t["q"][j, c]).ravel()
            j, c = np.repeat(j, Lk), np.repeat(c, Lk)
            lw, U, S, R = np.repeat(lw, Lk) + t["lz"][j, c] - d["etas"][j] * t["e"][c, x], np.repeat(U, Lk) + t["ell"][x], np.repeat(S, Lk, 0), np.repeat(R, Lk, 0)
            add(S, R, t["s"], x)
    return q, loss(S, R) > d["T"], lw


def cap_for(r, N, budget=BIAS):
    """Smallest cap on w / mu whose clipped mass, sum max(r - c, 0) / N, stays within the budget."""
    r = np.sort(r)[::-1]
    clipped = (np.cumsum(r) - np.arange(1, len(r) + 1) * r) / N
    return float(r[max(np.searchsorted(clipped, budget, side="right") - 1, 0)])


def stats(hit, logw, mu, c=None, prob=None):
    """r = w / mu on hits; prob = exact probabilities (enumeration) or None (a sample of len(hit) paths)."""
    r = np.exp(logw[hit] - math.log(mu))
    p = np.full(len(r), 1 / len(hit)) if prob is None else prob[hit]
    c = cap_for(r, len(hit)) if c is None else c
    est, m2 = float(p @ r), float(p @ r ** 2)
    return {"cap": c, "amplitude": float(p @ np.minimum(r, c)) / c if c > 0 else 0.0, "bias": float(p @ np.maximum(r - c, 0)), "estimate_over_mu": est,
            "rho2": m2 - est ** 2, "hit_rate": float(p.sum()), "z": None if prob is not None else (est - 1) / math.sqrt(max(m2 - est ** 2, 1e-300) / len(hit))}


def cost(h, calls, J=0, progress="U", prep="tree", capaware=False):
    """T-count, runtime and qubits of the whole run; J = 0 with tree preparation and no capped state is rotations.price(h, 'qrom_mixed', calls)."""
    n, wU = h["n"], h["w_U"]
    E = (h["rotations_A"] - 8 * n - wU) // 64
    groups = [(n, LEVELS_EQ, 3, "eq1"), (E, LEVELS_EDGE, 6, "eq2"), (n, [1], 1, "rain1"), (E, [1], 1, "rain2")]
    mult = {k: (J + 1) * (v if capaware else 1) for k, v in N_STATES.items()}       # lookup entries per outcome or angle
    jb, wL = math.ceil(math.log2(J + 1)), math.ceil(math.log2(32 * n + 1))
    tof0, qubits, rot_obj = h["toffoli_A"] + h["logical_qubits"], h["logical_qubits"], wU
    if J:
        tof0 += n * 2 * J * (wU if progress == "U" else wL)                          # which tilt: J comparators with constants per country
        tof0 += n * 2 * 2 * (4 + 2 + 3 + 2 * wL) if progress == "capped" else 0       # running capped loss of the two touched countries
        tof0 += n * 2 * (4 * 7 + jb * (wU + jb))                                      # block increment, then j x increment
        qubits += n * jb + wU + jb
    if capaware:
        tof0 += 2 * (3 * n + 6 * E + n + 2 * E)                                       # capped state of the touched countries
        tof0 += sum(cnt * 2 * (mult[k] - 1 + B_W) for cnt, _, _, k in groups)         # lookup-add of log Z per source
        tof0 += 2 * 2 * wL                                                            # eta x L, eta a multiple of 1/128 with two set bits
    elif J:
        tof0 += n * 2 * (J + B_W)                                                     # lookup-add of log Z per country
    if J or capaware:
        tof0, qubits, rot_obj = tof0 + 2 * B_W, qubits + 2 * B_W, B_W                 # Y = c - log w; rotate its bits
    delta = 1e-3 / max(calls * h["rotations_A"], 1)
    for _ in range(3):
        b = math.ceil(math.log2(2 * math.pi / delta))
        syn, pg, tof, extra = rot_obj, 0, tof0, 0
        if prep == "alias":
            mu_bits = math.ceil(math.log2(calls * (2 * n + 2 * E) / 1e-3))
            for cnt, _, qb, k in groups:
                if k == "eq2" and capaware:                                           # first country's step, then the second given the first
                    tof += cnt * (2 * (J + 1) * 25 * 8 - 2 + 2 * (mu_bits + 3)); extra += cnt * 2 * (3 + 2 * mu_bits)
                else:
                    tof += cnt * (mult[k] * 2 ** qb - 1 + mu_bits + qb); extra += cnt * (qb + 2 * mu_bits)
        else:
            extra = 2 * b
            for cnt, levels, _, k in groups:
                if k == "eq2" and capaware:                                           # two 3-qubit trees: first country's step, then the second given the first
                    cnt, levels = 2 * cnt, LEVELS_EQ
                for L in levels:
                    qrom = 2 * (mult[k] * L - 1) + b
                    if mult[k] > 1 or (L > 1 and 4 * qrom < L * t_rot("mixed", delta)):
                        tof += cnt * qrom; pg += cnt
                    else:
                        syn += cnt * L
        delta = 1e-3 / max(calls * (syn + pg), 1)
    T = 4 * calls * tof + (calls * syn + (b if prep == "tree" else 0)) * t_rot("mixed", delta)
    sc = surface_code_footprint(qubits + extra, T)
    return {"T_count": T, "T_per_call": T / calls, "toffoli_per_call": tof, "logical_qubits": qubits + extra, "runtime_s": sc["runtime_s"],
            "physical_qubits": sc["physical_qubits"], "code_distance": sc["code_distance"]}


def model_of(n):
    return SyntheticModel(n, topology("ring", n))


def knobs(c):
    return {"J": len(c["etas"]) - 1, "progress": c["progress"], "capaware": c["capaware"]}


def run():
    his = {r["n"]: r for r in json.loads(reference().read_text(encoding="utf-8"))["advantage"] if r["topology"] == "ring"}
    side = json.loads((OUT / "quantum_side.json").read_text(encoding="utf-8"))["rows"]
    res = {"his_reference": reference().name, "bias_budget": BIAS, "exact_ring_2": exact_ring_2(), "rows": {}}
    for n in SIZES:
        h, model, rng = his[n], model_of(n), np.random.default_rng(seed(3900 + n))
        pilot = []
        for g in GRID:
            st = stats(*sample(design(model, **g), N_PILOT, rng), h["mu"])
            ok = abs(st["z"]) < 4 and st["amplitude"] > 0
            calls = 23.5 / (0.01 * math.sqrt(max(st["amplitude"], 1e-12)))
            pilot.append({**g, "amplitude": st["amplitude"], "cap": st["cap"], "z": st["z"], "rho2": st["rho2"], "eligible": bool(ok),
                          **{f"runtime_{p}": cost(h, calls, prep=p, **knobs(g))["runtime_s"] for p in ("tree", "alias")}})
        el = [c for c in pilot if c["eligible"]]
        chosen = {"tree": min(el, key=lambda c: c["runtime_tree"]), "alias": min(el, key=lambda c: c["runtime_alias"]),
                  "amplitude": max((c for c in el if c["capaware"] or c["progress"] == "capped"), key=lambda c: c["amplitude"])}
        row = {"a_twisted": h["a_twisted"], "a_u_loader": side[str(n)]["a_ceiling_U_loader"], "pilot": pilot, "loaders": {}, "by_eps": {},
               "best_pilot_amplitude_by_family": {f: max(c["amplitude"] for c in el if c["family"] == f) for f in ("progress", "cap-aware", "both") if any(c["family"] == f for c in el)}}
        for name, c in chosen.items():
            st = stats(*sample(design(model, **c), N_FINAL, np.random.default_rng(seed(4900 + n))), h["mu"], c["cap"])
            row["loaders"][name] = {**{k: c[k] for k in ("family", "etas", "kappa", "progress", "capaware")}, "pilot_amplitude": c["amplitude"], **st}
        for eps in EPS:
            tc = classical(h, side[str(n)], eps)["small_world"]
            c0 = cheb_calls(h["a_twisted"], eps)["calls"]
            lv = {"twisting_tree": {"calls": c0, **cost(h, c0)}, "twisting_alias": {"calls": c0, **cost(h, c0, prep="alias")}}
            for prep in ("tree", "alias"):
                L = row["loaders"][prep]
                cl = cheb_calls(L["amplitude"], eps)["calls"]
                lv[f"loader_{prep}"] = {"calls": cl, **cost(h, cl, prep=prep, **knobs(L))}
            for v in lv.values():
                v["s_star_vs_small_world"] = v["runtime_s"] / tc
            row["by_eps"][str(eps)] = {"classical_seconds_small_world": tc, "levels": lv,
                                       "factor_vs_twisting_tree": {k: lv["twisting_tree"]["runtime_s"] / v["runtime_s"] for k, v in lv.items()}}
        res["rows"][n] = row
    big = res["rows"][28]
    f = big["by_eps"]["0.01"]["factor_vs_twisting_tree"]
    res["checks"] = {"L1_exact_on_ring_2": res["exact_ring_2"]["ok"],
                     "L2_amplitude_reaches_u_loader_at_28": bool(big["loaders"]["amplitude"]["amplitude"] >= big["a_u_loader"]),
                     "L3_tree_halves_s_star_at_28": bool(f["loader_tree"] >= 2), "L4_alias_halves_s_star_at_28": bool(f["loader_alias"] >= 2)}
    save("q_link_gates.json", res)
    report(res)
    return res


def report(res):
    for n, row in res["rows"].items():
        print(f"ring {n}: amplitude twisted {row['a_twisted']:.3g}, exact U-loader {row['a_u_loader']:.3g} | best pilot amplitude by family: "
              + ", ".join(f"{k} {v:.3g}" for k, v in row["best_pilot_amplitude_by_family"].items()))
        for name, L in row["loaders"].items():
            print(f"   chosen for {name}: {L['family']}, tilts {[round(e, 4) for e in L['etas']]}, kappa {L['kappa']} | amplitude {L['amplitude']:.3g} (pilot {L['pilot_amplitude']:.3g}) | "
                  f"clipping bias {L['bias']:.2%} | estimate / mu {L['estimate_over_mu']:.4f} (z {L['z']:+.1f}) | relative variance {L['rho2']:.3g}")
        for eps, e in row["by_eps"].items():
            lv, f = e["levels"], e["factor_vs_twisting_tree"]
            print(f"   eps {eps}: " + " | ".join(f"{k} s* {v['s_star_vs_small_world']:.3g} (/{f[k]:.2f}; {v['calls']:.3g} calls x {v['T_per_call'] / 1e3:.0f}k T; {v['physical_qubits'] / 1e6:.1f}m qubits)" for k, v in lv.items()))
    print("exact on ring 2:", {k: v for k, v in res["exact_ring_2"].items() if k != "cases"})
    print("checks:", res["checks"])


def exact_ring_2():
    """L1: enumeration of the 2-country ring against the exact answer, his twisted amplitude and sampling."""
    model = model_of(2)
    mu = SL.simulate_A(model, SL.payoff_library(model, 16 * 2 * 2.5e6)[0]["exceedance"])["a"]
    eta, rate = TW.best_twist(model, 16 * 2)
    q, hit, lw = enumerate_all(design(model, [eta]))
    tw = stats(hit, lw, mu, c=math.exp(-rate) / mu, prob=q)                    # his bound: w <= e^{Lambda - eta T} on hits
    out = {"mu_exact": mu, "states": len(q), "twisted_amplitude_ratio": tw["amplitude"] / TW.twisted_amplitude(model, mu, 32)["a_twisted"], "cases": []}
    ok = abs(out["twisted_amplitude_ratio"] - 1) < 1e-9 and tw["bias"] < 1e-12
    for g in ({"etas": step(0, 3 / 16, 16), "progress": "U"}, {"etas": step(0, 3 / 16, 16), "progress": "capped"}, {"etas": [3 / 32], "capaware": True},
              {"etas": step(1 / 16, 1 / 8, 32), "kappa": 0.85, "progress": "capped", "capaware": True}):
        d = design(model, **g)
        q, hit, lw = enumerate_all(d)
        ex = stats(hit, lw, mu, c=0.5 * float(np.exp(lw[hit] - math.log(mu)).max()), prob=q)      # a cap that does clip
        mc = stats(*sample(d, 400_000, np.random.default_rng(seed(39))), mu, ex["cap"])
        case = {**g, "law_sum": float(q.sum()), "estimate_over_mu": ex["estimate_over_mu"], "amplitude": ex["amplitude"], "bias": ex["bias"],
                "sampled_amplitude": mc["amplitude"], "sampled_estimate_z": mc["z"]}
        ok &= (abs(case["law_sum"] - 1) < 1e-12 and abs(ex["estimate_over_mu"] - 1) < 1e-9 and abs(mc["z"]) < 4
               and abs(mc["amplitude"] - ex["amplitude"]) < 4 * math.sqrt(ex["amplitude"] / 400_000))
        out["cases"].append(case)
    out["ok"] = bool(ok)
    return out


def selftest():
    """The cost function with no conditioning is the existing re-pricing; the enumeration is exact on the 2-country ring."""
    h = {r["n"]: r for r in json.loads(reference().read_text(encoding="utf-8"))["advantage"] if r["topology"] == "ring"}[28]
    for calls in (h["quantum_calls"], 8000.0):
        a, b = cost(h, calls), price(h, "qrom_mixed", calls)
        assert abs(a["T_count"] / b["T_count"] - 1) < 1e-12 and a["physical_qubits"] == b["physical_qubits"], (a, b)
    t = [cost(h, 8000.0, **k)["T_per_call"] for k in ({}, {"J": 3}, {"J": 3, "progress": "capped"}, {"capaware": True}, {"J": 3, "progress": "capped", "capaware": True})]
    assert t == sorted(t), t
    ex = exact_ring_2()
    assert ex["ok"], ex
    print(f"selftest ok: cost(J = 0) equals rotations.price; ring 2 enumerated ({ex['states']} outcomes), mu {ex['mu_exact']:.4g}, "
          f"twisted amplitude ratio {ex['twisted_amplitude_ratio']:.12f}, {len(ex['cases'])} loaders exact")


if __name__ == "__main__":
    selftest()
    run()
