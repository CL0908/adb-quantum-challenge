"""Driver: end-to-end Q-RDPO amplitudes with exponentially twisted Poisson-source loading."""
import json
import math
from pathlib import Path

from src.tensor.synthetic import SyntheticModel, topology
from src.scale.run_scale import exact_refs
from . import twisted as TW
from . import structured_loading as SL

OUT = Path("results/twisted.json")


def run(eps_rel=0.1):
    C = json.loads(Path("results/structured.json").read_text())["iqae_C"]
    ref = exact_refs()
    rows = []
    for (kind, n), r in sorted(ref.items()):
        if kind not in ("ring", "complete") or n < 2:
            continue
        m = SyntheticModel(n, topology(kind, n))
        mu = r["exceedance"]
        tw = TW.twisted_amplitude(m, mu, 16 * n)
        A = SL.a_operator_resources(m)
        rows.append({"topology": kind, "n": n, "mu": mu, **tw,
                     "qae_calls_plain": C / (eps_rel * math.sqrt(mu)),
                     "qae_calls_twisted": C / (eps_rel * math.sqrt(tw["a_twisted"])),
                     "classical_twisted_paths_upper": (1 / tw["a_twisted"] - 1) / eps_rel ** 2,
                     "mc_paths": (1 - mu) / mu / eps_rel ** 2,
                     "A_toffoli_plain": A["toffoli"], "A_toffoli_twisted": A["toffoli"] + 2 * tw["lookup_toffoli"],
                     "logical_qubits": A["logical_qubits"] + math.ceil(math.log2(tw["U_max"] + 1)) + 1})
    OUT.write_text(json.dumps({"eps_rel": eps_rel, "iqae_C": C, "rows": rows}, indent=2))
    return rows


if __name__ == "__main__":
    for r in run():
        print(r["topology"], r["n"], f"a={r['a_twisted']:.3g}", f"calls {r['qae_calls_plain']:.2e} -> {r['qae_calls_twisted']:.2e}",
              f"classical twisted <= {r['classical_twisted_paths_upper']:.2e}", f"Tof/A {r['A_toffoli_twisted']}")
