"""Numbers behind the small data graphics of the public site (frontend/public/data/story.json).

Every figure is read from the committed results files; nothing is typed here. Called by export_web_data.main().
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
BRIEF_METHODS = ("mc", "sobol", "stratified", "importance")
E1_BUDGET = 16384
TARGETS = ["expected_loss", "joint_attachment", "exceedance", "stop_loss", "joint_exhaustion"]


def story() -> dict:
    ex = json.loads((R / "exact.json").read_text())
    cp = ex["checkpoints"]
    lam = {k: ex["rates"][k]["Lambda"] for k in ("KG", "TJ")}
    c = cp["shared_rate_c"]
    decomp = {
        "kg_solo": lam["KG"] - c, "shared": c, "tj_solo": lam["TJ"] - c,
        "rain_kg_or_tj_only": cp["rain_one_only"], "rain_both": cp["rain_both"],
        "no_payout": cp["annual_no_payout"],
    }

    tn = json.loads((R / "tensor.json").read_text())["B_scaling_n"]
    ring = [(r["n"], r["runtime_s"]) for r in tn if r["topology"] == "ring" and r["max_bond"] >= 58 or (r["topology"] == "ring" and r["n"] <= 3 and r["max_bond"] >= 9)]
    seen = {}
    for n, s in ring:
        seen[n] = s
    scaling = {"ring": sorted(seen.items()), "max_bond": max(r["max_bond"] for r in tn if r["topology"] == "ring")}

    eq = json.loads((R / "equal_time.json").read_text())
    heat = {"targets": TARGETS, "paths": {}, "time": {}}
    # equal paths: the paper's e1_heatmap / e1_values source (results/rdpo_e1.json, B = 16,384, RDPO vs best of four)
    for r in json.loads((R / "rdpo_e1.json").read_text())["rows"]:
        if r["budget"] == E1_BUDGET:
            heat["paths"].setdefault(r["scenario"], {})[r["target"]] = r["improvement_vs_best"]
    for cell in eq["A_cells"]:
        best = min(cell["rmse_time"][m] for m in BRIEF_METHODS)
        heat["time"].setdefault(cell["scenario"], {})[cell["target"]] = best / cell["rmse_time"]["rdpo"]

    qa = json.loads((R / "quantum_advantage_v3.json").read_text())["advantage"]
    quantum = [
        {"n": r["n"], "s_star": r["required_speedup"], "ci": r.get("s_star_ci95"), "classical_s": r["classical_seconds"],
         "quantum_s": r["quantum_seconds"]}
        for r in qa if r["topology"] == "ring"
    ]
    return {"decomp": decomp, "scaling": scaling, "heat": heat, "quantum": sorted(quantum, key=lambda r: r["n"])}


PDFS = {"proposal": "RDPO_Proposal_Short.pdf", "annex": "RDPO_Proposal.pdf", "deck": "RDPO_Presentation.pdf"}


def documents(web: Path) -> dict:
    """Page count and a first-page thumbnail (data/thumb_<key>.jpg) of each PDF, from the PDFs that are published."""
    from pypdf import PdfReader
    out = {}
    for key, name in PDFS.items():
        pdf = web / name
        if not pdf.exists():
            pdf = ROOT / "docs" / name
        pages = len(PdfReader(str(pdf)).pages)
        stem = web / "data" / f"thumb_{key}"
        subprocess.run(["pdftoppm", "-jpeg", "-jpegopt", "quality=82", "-f", "1", "-l", "1", "-scale-to-x", "520", "-scale-to-y", "-1",
                        "-singlefile", str(pdf), str(stem)], check=True)
        out[key] = {"pages": pages, "file": name, "thumb": f"data/thumb_{key}.jpg"}
    return out


def write(web: Path) -> None:
    (web / "data").mkdir(parents=True, exist_ok=True)
    d = story()
    d["documents"] = documents(web)
    (web / "data" / "story.json").write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    write(ROOT / "frontend" / "public")
