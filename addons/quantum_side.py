"""What the small world does to the quantum stage, in the main author's own resource model. Additive, read-only.

The main author's Q-RDPO loads the exponentially twisted product law and runs amplitude estimation on a bounded payoff of
amplitude a_twisted = mu * e^{I(T)} (src/quantum/twisted.py). His criterion (src/quantum/advantage.py):
    T_classical = rho2 / eps^2 * t_path      (measured RDPO, training included)
    T_quantum   = calls * runtime per call   (surface-code model),   calls ~ C / (eps * sqrt(a))
    required hardware speed-up  s* = T_quantum / T_classical.
His own note in twisted.py: "the quadratic QAE gain is in 1/sqrt(a) vs 1/a". So a better classical sampler leaves less.
Three things are computed here on his ring pools at US$40m per country (his setting T = 16 n), nothing is re-derived:
 1. s* with the classical side replaced by the small world (uncapped, and with saturation modules), his T_quantum unchanged;
 2. queries: classical paths against quantum calls, and the relative error below which amplitude estimation needs fewer
    queries than classical sampling (his fitted IQAE constant C);
 3. a ceiling for any loader that conditions the source amplitudes on the uncapped loss register U only: the best such
    loader samples the uncapped world given U > T, whose amplitude is exactly mu / mu_uncapped (weights are constant on
    the event). mu_uncapped is the small world's own answer, so the ceiling costs nothing to compute.
Pre-set questions, answered yes or no:
 Q1 does the small world lower s* at 28 countries?  (expected: no, it raises it; the factor is the result)
 Q2 is there a relative error >= 1% at which amplitude estimation on the twisted law needs fewer queries than classical
    sampling with modules, at 28 countries?
 Q3 does the U-register ceiling cut the quantum calls by >= 10x at 28 countries?
Limits: T_quantum is his number at eps = 10% and is scaled by 1/eps for other errors (his rotation-synthesis precision
term changes slowly and is ignored); classical times are this machine's NumPy timings; no gate count is given for the
U-register loader, only the ceiling on what it could gain in calls.
"""
import json

import numpy as np

import cap_modules as C
import multi_country as M
import small_world as W
import small_world_loop as L
from _src import REPO, save
from src.tensor.junction import junction_solve

SIZES = (8, 16, 20, 28)


def his_rows():
    adv = json.loads((REPO / "results" / "quantum_advantage.json").read_text())["advantage"]
    tw = json.loads((REPO / "results" / "twisted.json").read_text())
    return {r["n"]: r for r in adv if r["topology"] == "ring"}, tw["iqae_C"]


def classical(P, p, n=100_000):
    out = {}
    for name, (pol, tb) in {"uncapped": W.teacher(P)[::2], "modules": (lambda b: (b[0], b[2]))(C.build(P, 1))}.items():
        s = L.stat(pol, p, 0.0, n, seeds=(91,))
        out[name] = {"rho2": s["rho2_worst"], "t_path": s["cost_per_path"], "cpu_build": tb, "z": s["z_max"]}
    return out


def run():
    his, iqae_c = his_rows()
    res = {"iqae_C": iqae_c, "rows": {}}
    for n in SIZES:
        P = M.pool(n, 40e6)
        h = his[n]
        mu = junction_solve(P)["exceedance"]
        mu_unc = W.value_tables(P)[1]
        cl = classical(P, mu)
        row = {"mu_exact": mu, "mu_his": h["mu"], "a_twisted": h["a_twisted"], "mu_uncapped": mu_unc, "a_ceiling_U_loader": mu / mu_unc,
               "his_classical_rho2": h["classical_rho2"], "his_t_path": h["classical_seconds"] / h["classical_paths"], "small_world": cl, "by_eps": {}}
        row["calls_reduction_ceiling"] = float(np.sqrt(row["a_ceiling_U_loader"] / h["a_twisted"]))
        # relative error below which C / (eps sqrt(a)) < rho2 / eps^2, i.e. eps < rho2 sqrt(a) / C
        row["crossover_eps"] = {"vs_his_rdpo": h["classical_rho2"] * np.sqrt(h["a_twisted"]) / iqae_c,
                                **{f"vs_{k}": v["rho2"] * np.sqrt(h["a_twisted"]) / iqae_c for k, v in cl.items()},
                                "ceiling_loader_vs_modules": cl["modules"]["rho2"] * np.sqrt(row["a_ceiling_U_loader"]) / iqae_c}
        for eps in (0.1, 0.01):
            tq = h["quantum_seconds"] * (h["eps_rel"] / eps)
            calls = iqae_c / (eps * np.sqrt(h["a_twisted"]))
            t_his = h["classical_rho2"] / eps ** 2 * row["his_t_path"]
            t_new = {k: v["cpu_build"] + v["rho2"] / eps ** 2 * v["t_path"] for k, v in cl.items()}
            best = min(t_new, key=t_new.get)
            row["by_eps"][str(eps)] = {"quantum_seconds": tq, "quantum_calls": calls, "classical_paths": {"his_rdpo": h["classical_rho2"] / eps ** 2, **{k: v["rho2"] / eps ** 2 for k, v in cl.items()}},
                                       "classical_seconds": {"his_rdpo": t_his, **t_new}, "best_small_world": best,
                                       "s_star_his": tq / t_his, "s_star_new": tq / t_new[best], "s_star_factor": t_his / t_new[best]}
        res["rows"][n] = row
    big = res["rows"][28]
    res["answers"] = {"Q1_small_world_lowers_s_star_at_28": bool(big["by_eps"]["0.1"]["s_star_factor"] < 1),
                      "Q2_quantum_fewer_queries_than_modules_at_eps_ge_1pct_at_28": bool(big["crossover_eps"]["vs_modules"] >= 0.01),
                      "Q3_U_loader_ceiling_cuts_calls_10x_at_28": bool(big["calls_reduction_ceiling"] >= 10)}
    save("quantum_side.json", res)
    for n, r in res["rows"].items():
        e1, e2 = r["by_eps"]["0.1"], r["by_eps"]["0.01"]
        print(f"ring {n}: mu {r['mu_exact']:.3g} | amplitude: twisted {r['a_twisted']:.3g}, ceiling of a U-register loader {r['a_ceiling_U_loader']:.3g} (at most x{r['calls_reduction_ceiling']:.1f} fewer calls)")
        print(f"   relative variance: his RDPO {r['his_classical_rho2']:.3g}, uncapped {r['small_world']['uncapped']['rho2']:.3g}, modules {r['small_world']['modules']['rho2']:.3g} | "
              f"quantum needs fewer queries only below eps = {r['crossover_eps']['vs_his_rdpo']:.2g} (his RDPO), {r['crossover_eps']['vs_modules']:.2g} (modules)")
        print(f"   s* at eps 10%: {e1['s_star_his']:.3g} -> {e1['s_star_new']:.3g} (x{e1['s_star_factor']:.0f}, {e1['best_small_world']}) | at eps 1%: {e2['s_star_his']:.3g} -> {e2['s_star_new']:.3g} (x{e2['s_star_factor']:.0f}, {e2['best_small_world']})")
    print("answers:", res["answers"])
    return res


def selftest():
    """His own row is reproduced from his own numbers: s* = T_quantum / T_classical, and the uncapped answer bounds the truth."""
    his, c = his_rows()
    h = his[28]
    assert abs(h["quantum_seconds"] / h["classical_seconds"] / h["required_speedup"] - 1) < 1e-9
    assert abs(h["classical_rho2"] / h["eps_rel"] ** 2 / h["classical_paths"] - 1) < 1e-9
    P = M.pool(28, 40e6)
    mu, mu_unc = junction_solve(P)["exceedance"], W.value_tables(P)[1]
    assert abs(mu / h["mu"] - 1) < 0.01 and mu_unc >= mu and h["a_twisted"] <= mu / mu_unc <= 1, (mu, h["mu"], mu_unc, h["a_twisted"])
    print(f"selftest ok: his s* reproduced; mu {mu:.4g} = his {h['mu']:.4g}; twisted amplitude {h['a_twisted']:.3g} <= ceiling {mu / mu_unc:.3g} <= 1")


if __name__ == "__main__":
    selftest()
    run()
