"""Market sizing for the commercialisation section (TAM / SAM / SOM).

Top-down context figures are third-party estimates (sources listed in SOURCES); the serviceable and
obtainable markets are bottom-up: number of institutions x annual contract value (ACV) per segment.
All assumptions are explicit here so the numbers in the proposal can be audited and changed.
Run: python -m src.business.market -> results/market_model.json
"""
import json
from pathlib import Path

SOURCES = {
    "cat_bonds": "Artemis.bm, outstanding catastrophe bond market at end-2025 (US$61.3bn)",
    "parametric": "Global Market Insights, parametric insurance market 2025 (US$19.4bn)",
    "gap": "Swiss Re Institute sigma 1/2025: 2024 economic nat-cat losses US$318bn, protection gap US$181bn",
    "software": "TrendX Insights / Market Intelo: catastrophe modelling & exposure software 2025 (US$2.1bn)",
}

CONTEXT_BN = {"Catastrophe bonds outstanding (2025)": 61.3, "Parametric insurance premiums (2025)": 19.4,
              "Uninsured nat-cat losses, protection gap (2024)": 181.0}

TAM_BN = 2.1   # catastrophe modelling & exposure-management software, global, 2025

# segment: (institutions addressable, ACV US$k, institutions won by year 5)
SEGMENTS = {
    "Development banks & sovereign risk pools": (15, 150, 4),
    "Reinsurers": (50, 250, 5),
    "ILS fund managers & cat-bond investors": (70, 100, 8),
    "Brokers & catastrophe-model vendors (licence)": (12, 300, 1),
    "Parametric insurers & MGAs": (100, 50, 10),
}


def model():
    rows = []
    for name, (n, acv, won) in SEGMENTS.items():
        rows.append({"segment": name, "institutions": n, "acv_k": acv, "sam_m": n * acv / 1e3,
                     "won_y5": won, "som_m": won * acv / 1e3})
    sam = sum(r["sam_m"] for r in rows)
    som = sum(r["som_m"] for r in rows)
    return {"tam_bn": TAM_BN, "sam_m": sam, "som_m": som, "som_share_of_sam": som / sam,
            "customers_y5": sum(r["won_y5"] for r in rows), "segments": rows, "context_bn": CONTEXT_BN,
            "sources": SOURCES}


if __name__ == "__main__":
    out = Path("results/market_model.json")
    m = model()
    out.write_text(json.dumps(m, indent=2))
    print(f"TAM {m['tam_bn']}bn  SAM {m['sam_m']:.1f}m  SOM {m['som_m']:.1f}m ({m['customers_y5']} customers)")
