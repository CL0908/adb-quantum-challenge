# Final submission checklist for Ziyu

Deadline: 22 Nov 2026, 23:59 (GMT+8). Read `HANDOFF.md` for project state. The team text, cover and slides already
list both members (state of 18 Nov 2026).

## 1. Pull and rebuild

```bash
git clone https://github.com/CL0908/adb-quantum-challenge && cd adb-quantum-challenge   # or: git pull origin main
pip install -r requirements.txt
(cd docs/paper && npm install)                  # once; needs pandoc, pdfunite, Chromium/Playwright
python -m pytest -q
mkdir -p submission                             # put both CVs here yourself: they are NOT in git (phone numbers, emails)
#   submission/Tan_Chun_Loong_CV.pdf   submission/Xu_Zi_Yu_CV.pdf
PYTHONPATH=. python -m src.viz.build_paper      # must print "unfilled: []"
PYTHONPATH=. python -m src.viz.build_slides
PYTHONPATH=. python -m src.viz.package_submission --cv submission/Tan_Chun_Loong_CV.pdf --cv submission/Xu_Zi_Yu_CV.pdf
```

Never commit anything under `submission/`. Edit only `docs/paper/short.md` / `proposal.md`; numbers are
`{{PLACEHOLDERS}}` filled from `results/*.json`.

## 2. Portal mapping

| Portal field | File / value |
|---|---|
| Submission form | fill in online |
| Technical Proposal [PDF] | `submission/upload/01_Technical_Proposal.pdf` |
| Lightweight Prototype [URL] | https://github.com/CL0908/adb-quantum-challenge |
| Team Profile / CVs [PDF] | `submission/upload/02_Team_Profile_CVs.pdf` |
| Presentation (optional) | `submission/upload/03_Presentation.pdf` |

**The repository is currently PRIVATE.** The owner (Tan Chun Loong) must make it public before submitting:
GitHub, Settings, Danger Zone, Change visibility. Open the URL in a logged-out browser to check.

## 3. Confirm before submitting

a. Three references added in the last rewrite (`docs/paper/references.bib`): Kastoryano & Pancotti,
   arXiv:2203.02804; Benedetti et al. 2016; Herbert 2021 (title, authors, venue). `errmit2026` has no author list:
   add it or drop the citation.
b. The two 2026 preprints listed in `addons/NOTES_FOR_AUTHOR.md` and the Kliuchnikov et al. rotation-synthesis
   constants (Quantum 7, 1208): the add-on resource reductions rely on them. Check they exist and say what we cite.
c. The reductions are presented as conditional (the ChebAE constant is simulated; mixed-fallback synthesis needs
   per-shot randomisation). Keep that wording; do not turn it into an unconditional claim.
d. The contract's definition of "event" (aftershock clustering add-on, not in the papers) is an open choice; the
   text says so.
e. Final read of page 1 and §6 (Team) of the main proposal, and a check that the main proposal is at most 12 pages.

Commit and push any change before uploading so the repository matches the PDFs.

## 4. Do not change without reason

* The honest equal wall-clock results (§3 main proposal, §6.5 annex).
* The data caveats on the dependence finding.
* The AI-use statement (required by the brief).
