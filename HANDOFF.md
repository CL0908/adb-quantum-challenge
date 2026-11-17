# HANDOFF: state of the project (for a new Claude Code session)

**New team member: start with `TEAMMATE.md`** (to-do list before the 22 Nov deadline).

**Status 17 Nov 2026.** Main proposal rewritten around the brief's five Phase 1 deliverables (`docs/paper/short.md`, ~12 pages) and the annex aligned (`proposal.md`, ~58 pages). New since 10 Nov: equal wall-clock audit (`results/equal_time.json`: at equal CPU time RDPO keeps 5/25 benchmark cells and 4/9 pool cells, the joint-exhaustion ones), method v3 negative result (`results/rdpo_v3_equal_time.json`: a learned product-form tilt and an automatic selector only tie twisting), exploratory cutset conditioning (`src/tensor/cutset.py`, unfinished). Portal package: `python -m src.viz.package_submission --cv ...` (organisers want a PDF proposal, a GitHub URL and a CV PDF). The repository is public.


Repo: https://github.com/CL0908/adb-quantum-challenge (branch `main`). Everything below is in git except the
CV (see "Not in git"). To continue: clone, read this file and `README.md`, run `pytest -q tests/` (84 tests pass; hardware-plan tests skip without qiskit).

## 1. What this is
Phase 1 submission to the **ADB Quantum Challenge** (joint-loss modelling for the ADB Kyrgyz–Tajik disaster
bond). Author/sole team member: Tan Chun Loong (NTU; incoming Phortify fellow at ICFO, Barcelona, host group
of Prof. Antonio Acín; the "one of three worldwide" claim is the user's, not verifiable from the email).
Method name: **Rényi-Doob Policy Optimization (RDPO)**; quantum extension **Q-RDPO** (future feature, no QPU in Phase 1).
The user's working language is Chinese/English mixed; reply in the language they use.

## 2. Deliverables (all built from `docs/paper/*.md` by `python -m src.viz.build_paper`)
| File | What |
|---|---|
| `docs/RDPO_Proposal_Short.pdf` | 6-page main proposal for judges (`docs/paper/short.md`) |
| `docs/RDPO_Proposal.pdf` | full technical annex, ~59 pages (`docs/paper/proposal.md`) |
| `submission/RDPO_Proposal_with_CV_annex.pdf` | full + CV; **local only, gitignored** (CV has phone/email) |
Build needs pandoc, Node + Playwright/Chromium, `docs/paper/node_modules` (`npm install` in `docs/paper`),
pdfunite. Every number in the text is a `{{PLACEHOLDER}}` filled from `results/*.json` by `src/viz/build_paper.py`;
figures/tables are numbered by label (`[fig:x]`, `[tab:x]`). Never type numbers by hand.
Build with CV annex: `PYTHONPATH=. python -m src.viz.build_paper --cv <path to CV pdf>`.

## 3. Code map
- `src/model, src/baselines, src/learned, src/quantum, src/tensor, src/causal, src/scale, src/hybrid`: Phases 0–5 (exact model, baselines, tensor networks, quantum sims).
- `src/rdpo/`: final method. `problem.py` (source form, exact references, `exact_pair`), `policy.py` (v1), `fast.py` (Phase 2 table-driven engine, stage-dependent and spline soft values), `train.py` (budgeted KL→χ² training, unbiased estimator, t-intervals), `projected.py` (projected Doob sampler, zero variance for pair targets), `experiments*.py` (resumable, raw results in `results/raw/*.jsonl`).
- `src/baselines/stronger.py`: splitting, SMC-on-twist, diagonal flow IS (added last).
- `src/realdata/`: real-data checks. `eq_check.py` (USGS catalogue), `loss_backtest.py` (NOAA/EM-DAT loss years, bivariate Poisson), `rain_check.py` (GHCN-Daily stations), `param_uncertainty.py` (Jeffreys-Beta posteriors → credible intervals), `central_asia_pool.py` (5 countries), `carec_pool.py` (10), `asia_pacific_pool.py` (28 + 50-country ring).
- `src/finance/analysis.py`: bond anchor (brief vs issued notes' expected loss), pooled-limit decision, LP coupling bounds over all dependence → `results/finance.json`.
- `src/tensor/junction.py`: frontier variable elimination, exact for bounded-pathwidth (dense) graphs to n=28; `src/rdpo/experiments_dense.py` → `results/dense_exact.json`.
- `src/quantum/hardware_plan.py`: costed IBM Heron-class emulation of two-country amplitude estimation (qiskit imported inside functions) → `results/hardware_plan.json`.
- `src/elt/`: ELT/YLT converter (`convert.py`) and demo.
- `src/business/market.py`: TAM/SAM/SOM model. `src/viz/`: figures (art*.py = SVG illustrations/cats, paper_figs.py = data plots, build_paper.py = build).
- `api/estimate.py`: estimation API (Vercel Python function). `web/` is empty: the user will build the front end themselves.
- `data/real/`: cached public data (sources in file headers). `tests/`: 59 tests.

## 4. Key results (all reproducible from `results/`)
- **Benchmark** (25 ADB cells, 16,384 paths, training included): RDPO beats the best of the brief's four methods in 20/25; joint exhaustion ×6.3–×171 (median ×19). Coverage ~94%.
- **Phase 2 engine**: 6.5–9.9× faster rollouts, identical policy; 13/16 scale settings better than Phase 1; beats exponential twisting 16/16; pair targets exact in ≤3 ms (`exact_pair`); overnight transfer cuts small-budget error ×1.8–×84.
- **Stronger baselines** (splitting / SMC-twist / flow IS, same budget and seeds): RDPO wins on rare targets by ×1.4–×18. SMC-on-twist is the strongest. Splitting is weak on this tie-heavy discrete score (reported honestly).
- **Real data**: earthquake arrivals Poisson; shared-quake share from 120-yr loss records ≈48% (CI 12–79%) vs brief 7.5%; extreme-rain share ≈2% (0.6–7.7%) vs brief 25%. Posterior pool exceedance (>US$80m): median 3.76% (3.1× brief 1.2%), 95% CI [2.08%, 5.27%]; joint exhaustion ≈ unchanged (effects cancel).
- **WP2**: 10-country CAREC pool 1.25% error at μ=4.3e-9; 28-country Asia–Pacific 0.76%, 50-country ring 0.79% (65,536 paths); target was ≤10% at μ≤1e-8.
- **ELT/YLT**: rarest question 5.4% (RDPO) vs 132% read off a 10,000-year YLT.
- **Quantum**: propositions 6–7 (amplitude encoding, query complexity), crossover law s* = κ ε/ρ², Algorithm 2 (guarded Q-RDPO), resources linear in n and |E|, break-even needs ρ²≈6e4–1e6 at 1% error vs measured 0.13–117, hardware ≈8e4× faster needed at 28 countries. No advantage claimed.

- **Model risk / pooling** (`finance.json`): brief tiers reproduce issued EL 2.296%/2.011% (Artemis, verified); Pr(L>80m) over all couplings ∈ [0, 9.0%] (brief at 13% of range); shared US$80m limit keeps 97.2% of expected payouts (brief) vs 90.6% [86.7, 94.9] (posterior).
- **Dense exact**: 21 references to n=28 (band-2..4, ladder+diagonals); RDPO beats twisting 10/10 per path (×1.9–3.9).
- **Wall-clock caveat (important)**: at equal CPU time exponential twisting is at least as accurate as RDPO in 13/16 scaling and 10/10 dense settings. All headline comparisons are per path; the paper says so. WP1 criterion: close this gap.
- **Hardware plan**: on a Heron-class noise model best estimates are off 19–25%, guard falls back; tilted circuits reach a few % at p2=1e-4, sub-% at 1e-5; ~9e5 shots, 3–15 QPU min.

## 4b. Challenge facts (challenges.adb.org, fetched 10 Nov 2026)
Deadline 22 Nov 2026 23:59 GMT+8; PDF/PPT/video, English; no page limit found (Challenge Brief PDF on Google Drive not reachable). Criteria: Problem & Approach 25, Technical Rigor 20, Proof of Concept 20, Work Plan 15, Resource Estimation 10, Team 10 (no business criterion). Deep Development 30 Nov 2026–22 Jan 2027, pitch 25 Jan; USD 20k to one team. Rules require submission content to stay confidential until results: keep the repo private. Network: this repo's cloud environment allows challenges.adb.org and www.artemis.bm (WebFetch does not use it; use curl).

## 5. Caveats to keep stating (do not overclaim)
All real-data results use **public proxies** (epicentre distance, reported losses, station rain), not the bond's actual trigger definitions or payout history; indicative, not a re-pricing. No vendor ELT/YLT used (demo event set is built from public data). No quantum hardware run. Mitigation paper `errmit2026` has no author list in `references.bib`. USGS, ERA5, EM-DAT, Yahoo, Artemis hosts are blocked in the cloud sandbox; data came via GitHub / AWS open-data mirrors.

## 6. Open items (priority order)
1. Proposal reframed (Nov 10) around the six criteria: quantum-inspired tensor network first, RDPO for the dense regime, model risk + pooling, 8-week work plan, commercialisation in Appendix B. Remaining: user to decide repo access for the jury, drop or source the 'one of three worldwide' claim, upload the Challenge Brief PDF to check format limits; optional PPT/video.
2. Add the three external items only the user can supply: a named advisor/letter of support, a real vendor ELT/YLT test, optionally a tiny real-hardware amplitude-estimation run (IBM open plan).
3. Gridded rainfall (ERA5/GPCP mirrors reachable via AWS/GCS), ground-motion/exposure models instead of epicentre distance (WP2 remainder).
4. (Descoped from Deep Development) Web front end + MCP tool server on top of `api/estimate.py` (user will do this with Claude Code; design reference was an emotionx.ai-style minimal site, which was not reachable from the sandbox).
5. Cosmetics: scan the full PDF visually once more; the user asked for more illustrations/cats and a less dry style (done through section mascots and infographics).

## 7. Working rules from the user
- Commit messages end with the attribution lines the session specifies; no model identifiers in repo files; no PRs unless asked.
- Never commit the CV or anything under `submission/`.
- Long runs use resumable job files in `results/raw/`; background watchers whose pattern appears in their own command line never terminate (use `pgrep -f` carefully).
- Generated text must be data-driven (placeholders); an earlier version had false claims typed by hand.

## 8. Not in git
The CV PDF (privacy), the conversation transcript, and scratch files. A new session cannot see the old conversation; this file is the substitute.
