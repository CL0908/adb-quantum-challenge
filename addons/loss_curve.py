"""One pilot, the whole loss curve. Additive, read-only.

A saturation module (cap_modules.py) says in what state a source's countries are when the pool's running loss is L. Nothing
in it refers to a threshold. So one pilot, aimed at the highest threshold of interest, should serve every lower threshold
and other payoffs of the total loss:
  * a forward pass through the modules gives the small world's whole loss distribution at once (the exceedance curve);
  * a backward pass per threshold (no new pilot) gives the guide for an unbiased estimate at that threshold;
  * the same with the stop-loss payoff E[(L - T)+] gives a layer's expected loss.
Pools: the 28-country ring (US$15m to US$40m per country) and the ap_risk pool (US$250m to US$850m in total). Exact values:
the main author's junction solver. Pre-set rules:
 K1 the forward-pass curve is within 10% of the exact exceedance probability at every threshold of the grid;
 K2 at every lower threshold the borrowed-module estimate is within 3 standard errors of exact and its relative variance
    is <= 1.5x that of modules built by a fresh pilot for that threshold;
 K3 stop-loss at the middle and the highest threshold: estimate within 3 standard errors of exact, own answer within 10%;
 K4 preparing the whole sheet from one pilot takes <= half the CPU of one pilot per threshold.
"""
import time

import numpy as np

import cap_modules as C
import multi_country as M
import small_world as W
import small_world_loop as L
from _src import save
from src.rdpo.problem import PortfolioProblem
from src.tensor.junction import junction_solve


def forward(pol, rho):
    """The small world's law of the final total loss: pmf over 0..Lmax, one pass over the sources."""
    P = pol.P
    Lg = np.arange(P.Lmax + 1)
    pmf = np.zeros(P.Lmax + 1); pmf[0] = 1.0
    for k in range(P.K):
        D, p = C.increments(pol, k), P.src[k]["p"]
        m = pmf[:, None] * rho[k]                                              # mass at (L, state)
        new = np.zeros_like(pmf)
        for v in range(len(p)):
            new += np.bincount(np.minimum(Lg[:, None] + D[None, :, v], P.Lmax).ravel(), weights=(m * p[v]).ravel(), minlength=P.Lmax + 1)
        pmf = new
    return pmf


def borrowed(P, rho):
    t0 = time.process_time()
    pol, answer, _ = W.teacher(P, C.value_tables(W.teacher(P)[0], rho))
    return pol, answer, time.process_time() - t0


def sheet(model, thresholds, n=100_000):
    hi = max(thresholds)
    P_hi = PortfolioProblem(model, "exceedance", threshold=hi)
    pol_hi, _, t_pilot, rho = C.build(P_hi, 1)
    t0 = time.process_time()
    pmf = forward(pol_hi, rho)
    t_forward = time.process_time() - t0
    Lg = np.arange(len(pmf))
    rows, t_borrow, t_fresh = [], 0.0, 0.0
    for t in thresholds:
        P = PortfolioProblem(model, "exceedance", threshold=t)
        p = junction_solve(P)["exceedance"]
        pol_b, a_b, tb = borrowed(P, rho)
        pol_f, a_f, tf, _ = C.build(P, 1, s=7)
        sb, sf = L.stat(pol_b, p, 0.0, n, seeds=(81,)), L.stat(pol_f, p, 0.0, n, seeds=(81,))
        t_borrow += tb; t_fresh += tf
        rows.append({"threshold_musd": t / 1e6, "p_exact": p, "curve_over_exact": float(pmf[Lg > P.T].sum() / p), "borrowed_answer_over_exact": a_b / p,
                     "z_borrowed": sb["z_max"], "rho2_borrowed": sb["rho2_worst"], "rho2_fresh": sf["rho2_worst"], "cpu_borrowed": tb, "cpu_fresh": tf})
    sl = []
    for t in (sorted(thresholds)[len(thresholds) // 2], hi):
        P = PortfolioProblem(model, "stop_loss", threshold=t)
        exact = junction_solve(P)["stop_loss"]
        pol, a, _ = borrowed(P, rho)
        s = L.stat(pol, exact, 0.0, n, seeds=(82,))
        sl.append({"threshold_musd": t / 1e6, "stop_loss_exact_usd": exact, "own_answer_over_exact": a / exact, "z": s["z_max"], "rho2": s["rho2_worst"]})
    return {"thresholds": rows, "stop_loss": sl, "cpu_one_pilot_sheet": t_pilot + t_forward + t_borrow, "cpu_pilot_per_threshold": t_fresh, "cpu_forward_pass": t_forward}


def run():
    import ap_risk
    res = {"ring 28": sheet(M.pool(28, 40e6).model, [t * 28 for t in (15e6, 20e6, 25e6, 30e6, 35e6, 40e6)]),
           "ap_risk 28": sheet(ap_risk.inputs()["model"], [t * 1e6 for t in (250, 350, 450, 550, 650, 750, 850)])}
    res["checks"] = {"K1_curve_within_10pct": {k: bool(max(abs(r["curve_over_exact"] - 1) for r in s["thresholds"]) <= 0.10) for k, s in res.items()},
                     "K2_borrowed_unbiased_and_within_1.5x": {k: bool(max(r["z_borrowed"] for r in s["thresholds"]) <= 3 and max(r["rho2_borrowed"] / r["rho2_fresh"] for r in s["thresholds"]) <= 1.5) for k, s in res.items()},
                     "K3_stop_loss": {k: bool(max(r["z"] for r in s["stop_loss"]) <= 3 and max(abs(r["own_answer_over_exact"] - 1) for r in s["stop_loss"]) <= 0.10) for k, s in res.items()},
                     "K4_sheet_cpu_halved": {k: bool(s["cpu_one_pilot_sheet"] <= 0.5 * s["cpu_pilot_per_threshold"]) for k, s in res.items()}}
    save("loss_curve.json", res)
    for k, s in res.items():
        if k == "checks":
            continue
        print(f"{k}: one pilot + forward pass ({s['cpu_forward_pass']:.2f} s) + {len(s['thresholds'])} backward passes = {s['cpu_one_pilot_sheet']:.1f} s; a pilot per threshold = {s['cpu_pilot_per_threshold']:.1f} s")
        for r in s["thresholds"]:
            print(f"   US${r['threshold_musd']:.0f}m: exact {r['p_exact']:.4g} | curve {r['curve_over_exact']:.3f}x | relative variance borrowed {r['rho2_borrowed']:.3g}, fresh {r['rho2_fresh']:.3g} | |z| {r['z_borrowed']:.1f}")
        for r in s["stop_loss"]:
            print(f"   stop-loss above US${r['threshold_musd']:.0f}m: exact US${r['stop_loss_exact_usd']:.4g} | own answer {r['own_answer_over_exact']:.3f}x | relative variance {r['rho2']:.3g} | |z| {r['z']:.1f}")
    print("checks:", res["checks"])
    return res


def selftest():
    """Two-country model: the forward pass sums to one and reproduces the exact exceedance probability at a lower threshold than the pilot's."""
    import dreamwake as D
    P_hi = D.problem("exceedance")
    pol, _, _, rho = C.build(P_hi, 1, n=100_000)
    pmf = forward(pol, rho)
    P = PortfolioProblem(P_hi.model, "exceedance", threshold=0.6 * P_hi.T * 2.5e6)
    exact = D.exact(P)[2]
    got = float(pmf[np.arange(len(pmf)) > P.T].sum())
    assert abs(pmf.sum() - 1) < 1e-9 and abs(got / exact - 1) < 0.03, (pmf.sum(), got, exact)
    print(f"selftest ok: forward pass sums to 1; exceedance at a lower threshold {got:.5f} vs exact {exact:.5f}")


if __name__ == "__main__":
    selftest()
    run()
