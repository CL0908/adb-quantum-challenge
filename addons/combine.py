"""Combine the add-on experiments into one table: what each change does to the friend's quantum/classical comparison.

Per advantage_map setting (ε_rel = 10%):
  s*_declared   friend's model (Ross–Selinger rotations, IQAE calls)               reproduced exactly by rotations.price
  s*_Q1c        + cheaper rotation synthesis, conservative: fallback (no randomisation) + QROM levels   rotations.py
  s*_Q1         + cheaper rotation synthesis, mixed fallback (re-sampled per gate instance) + QROM levels
  s*_Q2         + ChebAE (odd degrees): same-type swap for IQAE (both AE, same Grover-power circuits)   q2_ae.py
All of these are same-type component swaps or additions; the main pipeline (twisted loading -> AE on that measure;
RDPO CE -> Rényi-2 -> RQMC on the classical side) is not changed.
  s*_Q1cQ2, s*_Q1Q2   both (conservative / mixed)
  ring-28 also with the I25 pooled ρ² (E2 config and v2-final) instead of the 6-rep value.
"""
import json

from _src import HERE, OUT, load, save
from rotations import price


def run():
    adv = load("quantum_advantage.json")["advantage"]
    q2 = {(r["topology"], r["n"]): r for r in json.loads((OUT / "q2_ae.json").read_text())["rows"]}
    p = HERE.parent / "repro" / "i25" / "summary.json"
    i25 = {c["config"]: c["quantum"]["point"]["rho2"] for c in json.loads(p.read_text(encoding="utf-8"))["cells"]
           if c["n"] == 28} if p.exists() else {}
    rows = []
    for r in adv:
        key = (r["topology"], r["n"])
        cheb = q2[key]["chebae_odd"]["calls_mean"]
        s = {"declared": r["required_speedup"], "Q1c": price(r, "qrom_fallback")["s_star"],
             "Q1": price(r, "qrom_mixed")["s_star"], "Q2": price(r, "rs", cheb)["s_star"],
             "Q1cQ2": price(r, "qrom_fallback", cheb)["s_star"], "Q1Q2": price(r, "qrom_mixed", cheb)["s_star"]}
        assert abs(price(r, "rs")["s_star"] / s["declared"] - 1) < 1e-9
        row = {"topology": key[0], "n": key[1], "s_star": s, "factor_Q1Q2": s["declared"] / s["Q1Q2"],
               "factor_Q1cQ2": s["declared"] / s["Q1cQ2"],
               "quantum_seconds_Q1Q2": price(r, "qrom_mixed", cheb)["runtime_s"],
               "classical_seconds_eps10": r["classical_seconds"],
               "pL_required_iqae": r["pL_required"], "pL_required_chebae": q2[key]["chebae_odd"]["pL_required"]}
        if key == ("ring", 28):
            for cfg, rho2 in i25.items():
                row[f"s_star_Q1Q2_rho2_i25_{cfg}"] = s["Q1Q2"] * r["classical_rho2"] / rho2
                row[f"s_star_Q1cQ2_rho2_i25_{cfg}"] = s["Q1cQ2"] * r["classical_rho2"] / rho2
                row[f"s_star_declared_rho2_i25_{cfg}"] = s["declared"] * r["classical_rho2"] / rho2
        rows.append(row)
        print(f"{key[0]:8s} n={key[1]:2d}  s* {s['declared']:.3g} -> Q1c {s['Q1c']:.3g} / Q1 {s['Q1']:.3g}, Q2 {s['Q2']:.3g}, "
              f"both {s['Q1cQ2']:.3g}..{s['Q1Q2']:.3g} (/{row['factor_Q1cQ2']:.1f}..{row['factor_Q1Q2']:.1f})")
    save("combined.json", {"eps_rel": 0.1, "rows": rows})
    return rows


if __name__ == "__main__":
    run()
