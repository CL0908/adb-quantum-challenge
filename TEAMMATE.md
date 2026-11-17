# For the new team member: what to do before 22 Nov 2026, 23:59 (GMT+8)

Read `HANDOFF.md` first (project state, code map, results, rules). This file is the short to-do list.

## 1. Setup (5 min)

```bash
git clone https://github.com/CL0908/adb-quantum-challenge && cd adb-quantum-challenge
pip install -r requirements.txt
python -m pytest -q tests/                      # all tests should pass (about 1-3 min)
(cd docs/paper && npm install)                  # once, for the PDF build (needs pandoc, pdfunite, Chromium/Playwright)
PYTHONPATH=. python -m src.viz.build_paper      # rebuilds both PDFs in ~20 s; must print "unfilled: []"
```

Edit only the Markdown sources `docs/paper/short.md` (main proposal, ~12 pages) and `docs/paper/proposal.md`
(technical annex, ~58 pages). Never type a result number by hand: numbers are `{{PLACEHOLDERS}}` filled from
`results/*.json` by `src/viz/build_paper.py`. Figures and tables are numbered by labels (`[fig:x]`, `[tab:x]`).

## 2. Add yourself to the team (required)

The text currently describes a one-person team. Change these places:

| File | Line (approx.) | What to change |
|---|---|---|
| `docs/paper/short.md` | §6 Team (~l. 379–393) | Add your paragraph; replace "(sole member)" and the "One-person team, mitigated" paragraph with who does what |
| `docs/paper/short.md` | §5 (~l. 348, 369) | "Three work packages, one person", "a one-person team" in the risk list; assign work packages to people |
| `docs/paper/proposal.md` | §9 (~l. 1313, 1339) | "one person" in the work-plan intro and the risk table |
| `docs/paper/proposal.md` | §11 Team (~l. 1370–1410) | "sole member", your profile bullets, "One-person team, mitigated" |
| `README.md` / cover | `src/viz/art.py` l. ~150 | Cover shows one author; add your name if you want it on the cover |

State only facts that are on your CV (the jury will read the CVs).

## 3. CVs and the upload package

Put CVs in `submission/` (gitignored, never commit them: they contain phone numbers and emails), then:

```bash
PYTHONPATH=. python -m src.viz.package_submission --cv submission/Tan_Chun_Loong_CV.pdf --cv submission/<You>_CV.pdf
```

This writes `submission/upload/`:
* `01_Technical_Proposal.pdf`: main proposal followed by the technical annex
* `02_Team_Profile_CVs.pdf`: the CVs, in the order given
* `README_upload.txt`: what goes where on the portal

The portal asks for (organisers' email, 15 Nov): submission form; Technical Proposal [PDF]; Lightweight
Prototype [GitHub URL] = https://github.com/CL0908/adb-quantum-challenge; Team Profile/CVs [PDF]; optional
presentation (not prepared). Commit and push before uploading so the repository matches the PDFs.

## 4. Open items, in priority order

1. **Team text** (section 2 above).
2. **Check three references** added in the last rewrite (`docs/paper/references.bib`): Kastoryano & Pancotti,
   arXiv:2203.02804; Benedetti et al. 2016; Herbert 2021. Confirm title, authors, venue. `errmit2026` still has
   no author list: add it or drop the citation.
3. **Length.** The main proposal is ~12 pages (target was 8–10); p.5 has a gap. Trim §3 or §5 if you can.
4. **Cutset conditioning (exploratory, unfinished).** `src/tensor/cutset.py` (+ `tests/test_cutset.py`) samples the
   long-range dependence links and integrates the rest exactly with the tensor-network solver. Validation and
   reference runs on band-6/band-8 graphs are in `results/raw/cutset_*.json*`; the equal wall-clock comparison
   against twisting was not finished, so the papers do not report it (the Phase 2 plan lists it as WP1 work).
   Both papers contain an invisible `<!-- CUTSET-RESULTS -->` marker where a result would go.
5. **Optional:** a 10-slide presentation for the portal (and later the 25 Jan pitch).
6. Weekly effort in the Phase 2 plan is 20 h/week; adjust per person if two people share the work.

## 5. Things not to change without reason

* The honest equal wall-clock results (§3 of the main proposal, §6.5 of the annex): the brief scores accuracy
  per query, and we say plainly that at equal CPU time cheap methods win most cells.
* The data caveats on the dependence finding (stress range, not a re-pricing).
* The AI-use statement (required by the brief).
