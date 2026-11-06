# RDPO / ADB Quantum Challenge — claims inventory (read-only audit, 2026-11-06)

Repo: `adb-quantum-challenge` (scratch clone). No repo code was executed. Stored JSON values were parsed with
independent read-only scripts (stdlib only, no repo imports) to recompute every aggregate the paper builder
(`src/viz/build_paper.py`) derives.

Conventions
* `R/` = `results/`; `raw/` = `results/raw/`. CLI prefix for every command: `PYTHONPATH=. python -m`.
* "Declared" = number as written in README / proposal (after placeholder fill) / short.md.
* Placeholder → JSON mapping follows `build_paper.py` (function names given as `fn:` in the Source column).
* Runtime: **log** = from `.log`; **json** = Σ(seconds_per_fit × reps) or Σ raw `sec`; **est** = my extrapolation.
  Pooled drivers (`run_jobs`, 4 processes) are given as CPU time / ≈ wall on 4 cores.
* Tier: T0 deterministic/exact (s–min); T1 stochastic < 30 min; T2 > 30 min.
* Match: ✅ stored value reproduces declared number; ⚠️ partial / rounding / labelling issue; ❌ mismatch; ∅ no backing file.

## Inventory

| ID | Claim | Declared value | Where declared | Source file → key path | Producer CLI (sub-exp) | Seeds / reps / budget | Runtime | Ext. deps / offline | Match | Tier |
|---|---|---|---|---|---|---|---|---|---|---|
| C01 | Posterior Pr(L>80m): median, factor, 95% CI, brief | 3.76% (3.12× of 1.20%), CI [2.08%, 5.27%] | proposal Headline #1 (L15–22), §6.6.5 (L780); short.md L15–19 (filled); README Real-world row "Parameter-uncertainty" | R/param_uncertainty.json → posterior_summary.exceedance.{median,ci95}; factor_vs_brief.exceedance.median_x; brief_point.exceedance (fn: pu_values) | `src.realdata.param_uncertainty` | N=2000 posterior draws, seed 4242; exact dense enumeration per draw | unknown (est 1–2 min) | needs R/realdata_losses.json + R/realdata_rain.json (offline) | ✅ (values); see logic note L8 on EQ posterior | T1 |
| C02 | Weather stations used | 48 | proposal Headline #1, §6.6 (RAIN_NST); short.md L90 | R/realdata_rain.json → Σ len(stations[c]) (fn: rain_values) | `src.realdata.rain_check` | deterministic | est < 2 min | NOAA GHCN S3 (urllib) only if cache missing; cached `data/real/ghcnd-inventory-prcp.txt`, `data/real/ghcn/*.csv.gz` (54 files) → offline. Overwrites `data/real/ghcn_country_index.csv.gz` | ✅ (PK 1, TJ 5 stations) | T0 |
| C03 | Headline shares: EQ "about 48%", rain "about 2%" | 48% / 2% | proposal Headline #1 (HL_EQ_SHARE, HL_RAIN_SHARE) | R/realdata_losses.json → tests.eq_noaa_1900.share_of_A; R/realdata_rain.json → pooled."0.2".share (fn: headline_values) | `src.realdata.loss_backtest`; `src.realdata.rain_check` | deterministic (MLE) | est < 1 min each | OWID git clone only if cache missing; cached `data/real/owid_*_central_asia.csv` → offline | ✅ | T0 |
| C04 | 28-country Asia–Pacific pool, rare exceedance | μ=9.5×10⁻⁹, RDPO 0.76% (paper "0.8%"), 21 s/estimate, 65,536 paths | README Real-world "WP2 at scale"; proposal Headline #2 + glance tile; short.md L20–22 | R/wp2_asia_pacific.json → rows[case=asia_pacific,target=exceedance,threshold_musd=850,method=rdpo].{mu,rel_rmse,seconds} (fn: headline_values, ap_values) | `src.realdata.asia_pacific_pool` | 24 reps, seed 6161+r (same seed across rdpo/mc/twist), B=65,536 | json: 5,228 CPU-s (1.45 CPU-h) ≈ 25 min wall + MPS ~1–2 min | urllib only if cache missing; cached `usgs_m55_asia_pacific_1965_2016.csv`, `asia_pacific_boundaries.geojson`; needs R/realdata_rain.json; raw/wp2_asia_pacific.jsonl 480/480 done (resume ⇒ no-op) | ✅ (short.md renders "9.5×10⁻⁹-probability-probability": template bug) | T2 |
| C05 | Full 25-cell stress sweep time | 5 seconds (one core) | proposal Headline #3; short.md L24, L171 | R/sweep_timing.json → seconds_total (=5.19) | `src.rdpo.timing` | seed 0+n, B=16,384, 1 estimate per cell | json: 5.2 s | none | ✅ (timing, machine-dependent) | T1 |
| C06 | Overnight transfer gain at small budget | ×1.8–×84 (×1.8–×83.8) | README Phase 2 "Overnight transfer"; proposal Headline #3, §6.5 | R/rdpo_v2_transfer.json → rows grouped by (case,target,budget): rel_rmse[transfer=False]/rel_rmse[transfer=True] (fn: v2_values V2_TRANSFER_GAIN) | `src.rdpo.experiments_v2 transfer` | 40 reps, seed 9000+r (dst) and +999 (source fit); B∈{2048,4096}; source fit 65,536 paths NOT charged | unknown (raw has no `sec`); est ~20 CPU-min | raw/transfer.jsonl 800/800 (resume ⇒ no-op) | ✅ (gains 1.81…83.75) | T1 |
| C07 | Pool exceedance with both dependences from data vs brief | 2.8× | proposal glance tile (RAIN_EXC_FACTOR), §6.6; README "Rain dependence" | R/realdata_rain.json → implied["both from data"].exceedance / implied.brief.exceedance (=2.755) | `src.realdata.rain_check` | deterministic | est < 2 min | as C02 | ✅ | T0 |
| C08 | Hardware speed-up needed at 28 countries | 8.2×10⁴ | proposal glance tile + §7.8 takeaway (Q_SSTAR_28); README Quantum; short.md "8×10⁴" | R/quantum_advantage.json → advantage[ring,28].required_speedup (=8.19e4) | `src.quantum.advantage` (advantage_map) | IQAE emulation 20 reps seed 0+r; classical side from R/rdpo_e2.json (ρ², seconds) | unknown (est minutes) | needs R/rdpo_e2.json, R/twisted.json; resumes (skips if keys exist) | ✅ (depends on measured E2 timings ⇒ approx only) | T1 |
| C09 | Benchmark wins vs best of 4 reference methods | 20/25 cells at 16,384 | README Phase 1; short.md L73–74; proposal §6.1 (TABLE_E1/E1_TEXT) | R/rdpo_e1.json → rows[budget=16384].improvement_vs_best > 1 (count) | `src.rdpo.experiments e1` | RDPO 40 reps, seed 1000+100000·si+1000·ti+r; baselines from R/baselines.json (200 reps, seed 20261031); B∈{4096,16384,65536} | json: 1,659 CPU-s (28 min, serial) | needs R/baselines.json (`src.baselines.run_baselines`, log ≈ 2.7 min); resumes from R/rdpo_e1.json (complete ⇒ no-op) | ✅ (20/25 at all 3 budgets; all 5 losses = expected loss; min winning margin ×3.0 at 16k) | T1 |
| C10 | Joint exhaustion improvement & error | ×6.3–×171, median ×19; median rel RMSE 1.4% | README Phase 1; proposal Abstract (E1_EXH_MED, E1_EXH_RANGE), glance tile; short.md L75–76 | R/rdpo_e1.json → rows[target=joint_exhaustion,budget=16384].{improvement_vs_best, rdpo.rel_rmse} | `src.rdpo.experiments e1` | as C09 | as C09 | as C09 | ✅ (6.26, 170.55, median 19.3; 1.39%) | T1 |
| C11 | Benchmark interval coverage | mean 95% (paper), "94%" (short.md), as low as 78% (README) | proposal §6.1 E1_TEXT; short.md L76; README "What we do not claim" | R/rdpo_e1.json → mean/min of rows[].rdpo.ci_coverage (0.947 / 0.775) | `src.rdpo.experiments e1` | as C09 | as C09 | as C09 | ⚠️ short.md "94%" (0.947 rounds to 95%) | T1 |
| C12 | Per-target median factors & budget dependence | exh ×19, stop-loss ×9.8, exceedance ×7.9, attachment ×3.4, EL ×0.53; exh ×3.5 at 4,096, ×72 at 65,536 | proposal §6.1 E1_TEXT | R/rdpo_e1.json → median improvement_vs_best per target/budget | `src.rdpo.experiments e1` | as C09 | as C09 | as C09 | ✅ (computed) | T1 |
| C13 | RDPO time per benchmark estimate | 0.4 s at B=16,384 | proposal §8.1 (E1_SEC_16K) | R/rdpo_e1.json → mean seconds_per_fit [budget=16384] (=0.377) | `src.rdpo.experiments e1` | as C09 | as C09 | — | ✅ (timing) | T1 |
| C14 | Scaling ring 2→28: target to 1.3×10⁻¹³, error 0.1%→4.2%; MC ~10¹⁵ paths | 0.1%–4.2% | README Phase 1; proposal Abstract (E2_RANGE), §6.2 | R/rdpo_e2.json → rows[topology=ring,target=exceedance].{mu, rdpo.rel_rmse, mc_theory_rel_rmse} (0.14%→4.23%; MC theory 10,743 at B=65,536 ⇒ 7.6×10¹⁴ paths for 10%) | `src.rdpo.experiments e2` | reps 20 (n≤4) / 10 (n≤16) / 6; seed 2000+1000n+100·len(rows)+r (**depends on resume order**); B=65,536 | json: 6,459 CPU-s RDPO (1.8 h serial) + twisting + MPS pair refs ⇒ ~2 h | needs R/tensor.json (`src.tensor.scaling`); resumes; log shows a mid-run "restart with target-relevance freezing" | ✅ values / not bit-reproducible | T2 |
| C15 | Phase 1 vs exponential twisting | 15/16 wins; exception ring-28 stop-loss 5.9% vs 3.5% | proposal §6.2 E2_TEXT; README "What we do not claim" | R/rdpo_e2.json → rows[].twisted.rel_rmse vs rdpo.rel_rmse | `src.rdpo.experiments e2` | twist: single rng seed 2007 shared across cases, same reps as RDPO | as C14 | as C14 | ✅ | T2 |
| C16 | Pair exhaustion in portfolio & coverage | 0.34%–3.4%; coverage mean 93%, min 80% | proposal §6.2 E2_TEXT | R/rdpo_e2.json → rows[target=joint_exhaustion].rdpo.{rel_rmse,ci_coverage} | `src.rdpo.experiments e2` | as C14 (6–20 reps) | as C14 | as C14 (these rows were re-run after a code change) | ✅ (mixed code versions in file) | T2 |
| C17 | RDPO time at 28 countries | 65 s (B=65,536) | proposal §8.1 (E2_SEC_28) | R/rdpo_e2.json → rows[ring,28,exceedance].seconds_per_fit (=65.4) | `src.rdpo.experiments e2` | as C14 | as C14 | — | ✅ (timing) | T2 |
| C18 | Ablation: indispensable components | CE ×81, twist warm start ×37 (ring 16), relevance freezing ×2.6 (ring 8 pair) | README Phase 1; proposal §6.3 (E3_TEXT) | R/rdpo_e3.json → rows[case,ablation].rel_rmse / rows[case,"full RDPO"].rel_rmse (80.5, 37.3, 2.61) | `src.rdpo.experiments e3` | 40 reps, seed 3000+r (common across ablations), B=16,384 | no timing in json/log; est ~55 min serial | needs R/tensor.json; resumes | ✅ ("no twist warm start" on pair targets is a no-op: identical rows) | T2 |
| C19 | Exact tilted MPS: bond dims constant (chain 9, star 10, ring 58); complete 58→135 (n=4→6); 28 countries in ~1 min | 9/10/58; 58→135; ~1 min | proposal §6.4 | R/tensor.json → B_scaling_n[topology,n,tol=1e-9].{max_bond,runtime_s} | `src.tensor.scaling` (Exp A/B/C) | deterministic | json: Σ runtime_s(B)=770 s + Exp A/C ⇒ est 15–20 min | none | ✅ | T0 |
| C20 | Untilted SVD truncation error at tol 10⁻¹² | "up to 2%" on rarest target | proposal §6.4 L505; REPORT §0 item 9 | R/tensor.json → A_correctness[tilted_gauge=false,tol=1e-12].max_rel_err (max 6.7×10⁻⁴, complete n=5) | `src.tensor.scaling` (Exp A) | deterministic | as C19 | none | ❌ JSON (and REPORT's own §5.2 table) give 0.067%; 2% not found | T0 |
| C21 | Table-driven engine rollout speed-up (n≥4) | 6.5–9.9×; "identical policy (tested)" | README Phase 2; proposal §6.5 (V2_SPEEDUP) | R/rdpo_v2_throughput.json → rows[n≥4].speedup (6.47–9.92) | `src.rdpo.experiments_v2 throughput` | θ ~ N(0,0.1²) seed 0, best of 3 timings, N=65,536 | est ~8–9 min (v1 engine ~1.3k paths/s at n=28) | none | ✅ (timing, machine-dependent) | T1 |
| C22 | Exact compressed training cost reduction | 50–500× cheaper per evaluation | proposal §6.5 L534 | none found | — | — | — | — | ∅ | n/a |
| C23 | Phase 2 vs Phase 1 at scale; vs twisting | 13/16 better (up to ×1.7, median ×1.18); beats twisting 16/16 | README Phase 2; proposal §6.5 (V2_GAIN, V2_TW_WINS); short.md L79 | R/rdpo_v2_scale.json → rows[variant=v1].rel_rmse / rows[variant=v2-final].rel_rmse (non-pair); rows[variant=twist] | `src.rdpo.experiments_v2 scale` | 24 reps, seed 7000+1000n+100·TARGETS.index(t)+r, identical across v1/v2/v2-final/twist; B=65,536 | json: Σ sec 15,805 (4.4 CPU-h) ≈ 1.1 h wall | needs R/tensor.json, R/rdpo_e2.json; raw/scale.jsonl 1920/1920 (resume ⇒ no-op) | ✅ | T2 |
| C24 | Wall-clock reduction Phase 1 study → Phase 2 | 2.3–11.9× | README Phase 2; short.md L80; proposal §6.5 (V2_TIME) | R/rdpo_v2_scale.json → rows[v1].v1_study_seconds / rows[v2-final].seconds_per_fit | `src.rdpo.experiments_v2 scale` | as C23 | as C23 | as C23 | ⚠️ compares serial E2 timings with 4-process-pool timings | T2 |
| C25 | Phase 2 interval coverage, portfolio-wide | 88–100%, mean 95% | README Phase 2; proposal §6.5 (V2_COV) | R/rdpo_v2_scale.json → rows[v2-final].ci_coverage (0.875–1.0, mean 0.953) | `src.rdpo.experiments_v2 scale` | as C23 | as C23 | as C23 | ✅ | T2 |
| C26 | Exact pair reduction speed & accuracy | ≤ 3 ms any size; agrees with MPS to 1.8×10⁻¹⁰ | README Phase 2; short.md L68–69; proposal §6.5 (V2_EXACT_MS/ERR) | R/rdpo_v2_scale.json → exact_pair.*.{ms, rel_err_vs_mps} (max 3.33 ms; 1.8e-10) | `src.rdpo.experiments_v2 scale` (aggregation step) | deterministic | ms (aggregation also rebuilds MPS refs, minutes) | needs R/tensor.json | ⚠️ max is 3.3 ms (rounded to "3 ms") | T0 |
| C27 | Phase 2 benchmark | keeps 20/25; expected loss "×0.6–0.8 instead of ×0.5" vs Sobol | README Phase 2; proposal §6.5 (V2_BENCH_TABLE) | R/rdpo_v2_bench.json → rows[budget].{improvement_vs_best, v1_improvement_vs_best} | `src.rdpo.experiments_v2 bench` | 40 reps, same seed formula as E1 (1000+…); B∈{4096,16384,65536}; baselines reused from E1 rows | raw has no `sec`; est 10–20 CPU-min | needs R/rdpo_e1.json; raw/bench.jsonl 3000/3000 | ⚠️ 20/25 ✅; EL medians v2 = 0.75/0.62/0.45 vs v1 0.72/0.53/0.33 (4k/16k/65k) — README range only fits 16k | T1 |
| C28 | Phase 2 benchmark per-cell coverage | 83%–100% (82.5%) | proposal §6.9 (V2_BENCH_COV) | R/rdpo_v2_bench.json → rows[].v2.ci_coverage | `src.rdpo.experiments_v2 bench` | as C27 | as C27 | as C27 | ✅ | T1 |
| C29 | Projected Doob sampler | rel spread ~10⁻¹⁵ (8.9×10⁻¹⁶); 20,000 paths in 130 ms (CAREC); cond. mean US$194m, p95 US$275m; P(>400m\|pair)=0.04%; triple exhaustion 2.1×10⁻⁸ | README Real-world; proposal §6.5 (PROJ_*) | R/projected_doob.json → ["CAREC pool (10 countries)"].{rel_sd_per_path, seconds, conditional_pool_loss_mean_musd, conditional_pool_loss_p95_musd, "P(pool > 400m \| pair exhausted)", triple_exhaustion_KG_TJ_UZ} | `src.rdpo.projected` | seed 7, 20,000 draws (μ exact; conditional stats are MC: P(>400m) = 8 hits) | json: 0.06 s + 0.13 s (+CAREC calibration) | uses cached CAREC data (offline) | ✅ | T0 |
| C30 | Negative results | spline soft values up to ×19 worse on stop-loss; value-of-training stop "slightly increased" error | proposal §6.5 "Negative results" | none found ("development probes") | — | — | — | — | ∅ | n/a |
| C31 | Stronger rare-event baselines | README: "wins on every rare target by ×1.4–×18" (same budget, same seeds); filled paper/short: "×5.4–×8" | README Real-world row; proposal §6.5.5 (STR_TABLE/STR_RANGE); short.md L85 | R/stronger_baselines.json → rows[case,target,method].rel_rmse; RDPO from R/rdpo_e1.json[budget=16384] and R/rdpo_e2.json[ring,16] (fn: stronger_values) | `src.baselines.stronger` | 20 reps, seed 12121+r; B=16,384 (ADB) / 65,536 (ring 16). RDPO comparator: different runs/seeds/reps (40 / 10) | json: ≈ 35 s of method time + ring-16 MPS (est 1–2 min) | none | ❌ README stale: JSON gives ×5.4–×7.8; "same seeds" false; table says "Phase 2" but uses Phase 1 numbers | T1 |
| C32 | USGS catalogue checks | 168 main shocks; dispersion 0.66–1.24 (every p ≥ 0.12); b = 0.92 ± 0.11; 25% trigger ≈ M6.9 (KG) / M7.0 (TJ); shared share 29–75% | README Real-world "USGS"; proposal §6.6 (RD_*); short.md L91 | R/realdata_eq.json → data.events_declustered; radii_km["50"][M][KG/TJ].{dispersion_index,poisson_p_value}; gutenberg_richter.{b,b_se,KG/TJ.implied_trigger_magnitude}; radii_km[*][*][*].shared_share | `src.realdata.eq_check` | deterministic | est seconds | urllib (plotly/datasets, datasets/geo-countries) only if cache missing; cached `usgs_m55_central_asia_1965_2016.csv`, `kg_tj_boundaries.geojson` | ⚠️ min p = 0.115 ("≥ 0.12" is rounding) | T0 |
| C33 | Catalogue-implied joint targets | joint exhaustion 2.1–3.6× brief; joint attachment 6.2%–8.8% | proposal §6.6 (RD_EXH_FACTOR, RD_ATT) | R/realdata_eq.json → implied_scenarios.rows.{ci_low,ci_high}.{joint_exhaustion,joint_attachment} / rows.brief | `src.realdata.eq_check` | deterministic (exact enumeration) | est seconds | as C32 | ✅ | T0 |
| C34 | Loss-record backtest | 120-yr record: shared share 48% (12–79%), independence rejected p = 0.005 | README Real-world; proposal §6.6 (LOSS_SHARE, LOSS_P); short.md L92–93 | R/realdata_losses.json → tests.eq_noaa_1900.{share_of_A, share_of_A_ci95, lr_test_independence_p} | `src.realdata.loss_backtest` | deterministic (MLE + profile likelihood) | est seconds | `git clone owid/owid-datasets` only if cache missing; cached `owid_noaa_events/deaths_central_asia.csv`, `owid_emdat_central_asia.csv` | ⚠️ values ✅; short.md calls these a "posterior median/CI" — actual Jeffreys posterior (param_uncertainty) is 55% [25%, 83%] | T0 |
| C35 | Implied joint exhaustion from loss data | ≈3× brief; 9.9×10⁻⁵ (4.0×10⁻⁵–1.5×10⁻⁴) | README Real-world "Implied bond risk"; proposal §6.6 (LOSS_EXH, LOSS_EXH_FACTOR) | R/realdata_losses.json → implied["loss data, point estimate"/"lower 95% bound"/"upper 95% bound"].joint_exhaustion / implied.brief | `src.realdata.loss_backtest` | deterministic | est seconds | as C34 | ✅ (3.1×) | T0 |
| C36 | Rain dependence from GHCN | shared share 2.2% (0.6–7.7%); both from data: exceedance 2.8×, exhaustion 0.85× ("about unchanged") | README Real-world "Rain"; proposal §6.6 (RAIN_SHARE, RAIN_EXC/EXH_FACTOR); short.md L93–94 | R/realdata_rain.json → pooled."0.2".{share,share_ci95}; implied["both from data"] / implied.brief | `src.realdata.rain_check` | deterministic | est < 2 min | as C02; needs R/realdata_losses.json | ⚠️ values ✅; short.md calls 2% [0.6–7.7%] a "posterior" (actual posterior 2.7% [0.46%, 6.9%]); short.md §7 "2.8× posterior median" conflates 2.8× point factor with 3.12× posterior median | T0 |
| C37 | Data-calibrated 5-country Central Asian pool | RDPO ×1.7–×15 over best of MC/twisting on rare questions; EL ×3.7; pair exhaustion 9.5% (72.5% coverage); b = 0.86, trigger M7.1 | README Real-world; proposal §6.6 (POOL_*) | R/realdata_pool.json → rows[target,threshold_musd,method].{rel_rmse,ci_coverage}; calibration.{b,m1} | `src.realdata.central_asia_pool` | 40 reps, seed 4242+r (same across methods), B=16,384 | json: Σ sec 374 CPU-s ≈ 2–3 min wall + exact dense (10⁵ states) | urllib only if cache missing; cached `usgs_m55_central_asia_wide_1965_2016.csv`, `central_asia5_boundaries.geojson`; raw/ca_pool.jsonl 600/600 | ✅ ("rare" set includes joint attachment μ=5.6%) | T1 |
| C38 | WP2 CAREC 10-country pool | Pr(>US$500m)=4.3×10⁻⁹; RDPO 1.25% @65,536 (8.3% @16,384); twisting 7.0%; MC sees nothing; bond 588; MPS 4.5 min; tol check 4×10⁻¹⁰; 10 edges; 731 events | README Real-world "WP2"; proposal §6.7 (CAREC_*) | R/wp2_carec.json → references["exceedance\|500"]; rows[exceedance,500,method,budget]; mps.{max_bond,mps_seconds}; references["check\|*"]; calibration.{edges,events} | `src.realdata.carec_pool` | 24 reps, seed 5151+r; B=65,536 (+16,384 for one test) | json: Σ sec 1,134 CPU-s ≈ 5 min wall + MPS ≈ 4.5 min (×2 tolerances) | urllib + `git clone owid-datasets` to /tmp only if cache missing; cached `usgs_m55_carec_1965_2016.csv`, `carec_boundaries.geojson`, `owid_emdat_carec.csv`; raw/wp2_carec.jsonl 456/456 | ✅ | T1 |
| C39 | WP2 at scale: 28 real countries + 50-country ring | AP 9.5×10⁻⁹ @0.76%; ring-50 4.1×10⁻⁹ @0.79%; twisting 2.3% / 1.8%; WP2 criterion met; bond 367 / 64; bandwidth 5; 7,424 events; 22 edges; 100× tighter tol agrees to 5.7×10⁻¹⁰ | README Real-world; short.md L96–97; proposal §6.7 (AP_*) | R/wp2_asia_pacific.json → rows[case,exceedance,T_rare,method]; cases.{asia_pacific,ring50}.{max_bond,mps_seconds,n,edges,tol_check}; calibration.{events,bandwidth} | `src.realdata.asia_pacific_pool` | as C04 | as C04 | as C04 | ✅ | T2 |
| C40 | ELT/YLT converter demo | 2,411 events → 19 sources (max footprint 4) in 0.10 s; rarest question (>US$320m, μ=2.3×10⁻⁴): RDPO 5.4% vs YLT 132% (MC 43%) | README Real-world "ELT/YLT"; short.md L101–102; proposal §6.8 (ELT_*) | R/elt_demo.json → event_set.events, n_sources, max_footprint, conversion_seconds; rows[exceedance,320,{rdpo,ylt,mc}].rel_rmse | `src.elt.demo` | 24 reps, seed 8080+r; RDPO/MC B=16,384; YLT 10,000 yrs (seed per rep) | json: Σ sec 109 CPU-s ≈ 1 min | cached catalogue/boundaries; needs R/realdata_rain.json; writes `data/elt/*.csv` + temp `_ylt_*.csv`; raw/elt_demo.jsonl 360/360 | ✅ | T1 |
| C41 | Posterior attachment / exhaustion | attachment 7.6% (1.73×) [4.9%, 10.2%]; exhaustion 3.1×10⁻⁵ (0.99×) [1.6×10⁻⁵, 5.6×10⁻⁵] | proposal §6.6.5 (PU_ATT_*, PU_EXH_*) | R/param_uncertainty.json → posterior_summary.{joint_attachment,joint_exhaustion}; factor_vs_brief | `src.realdata.param_uncertainty` | as C01 | as C01 | as C01 | ✅ | T1 |
| C42 | AE calls at 28 countries: untwisted → twisted vs MC | 2.8×10⁹ → 3.4×10⁴ vs 7.6×10¹⁴ MC paths | README Quantum "Loading"; proposal §7.4 (hard-coded text) | R/twisted.json → rows[ring,28].{qae_calls_plain, qae_calls_twisted, mc_paths} | `src.quantum.run_twisted` | deterministic given iqae_C from R/structured.json | est < 1 min | needs R/structured.json, R/tensor.json | ✅ (note quantum_advantage.json uses 4.7×10⁴ emulated calls for the same setting) | T0 |
| C43 | Bitwise weight oracle size | ~12 controlled rotations at 28 countries | proposal §7.4 | R/quantum_advantage.json → advantage[ring,28].{w_U=12, U_max=2688} | `src.quantum.advantage` | deterministic | — | — | ✅ | T0 |
| C44 | Resource laws (8 compiled settings) | logical qubits ≈ 38 + 13.2n + 8.2\|E\|; Toffoli/A ≈ −26 + 106n + 39\|E\|; depth(A) ≈ 378 + 0.3n + 14.7\|E\|; all within 7% | README Quantum; short.md L123–124; proposal §7.5 (QS_FITS, QS_FIT_ERR) | R/quantum_scaling.json → fits.*.{const,per_country,per_edge,max_rel_err} (max 7.12%) | `scaling_laws()` in src/quantum/advantage.py — **not reachable from CLI** (defined after the `__main__` block; no caller) | deterministic post-processing of R/quantum_advantage.json | seconds | — | ✅ | T0 |
| C45 | Phase diagram: measured ρ² vs break-even ρ² at 1% | ρ² 0.13–117 vs break-even ≈ 6×10⁴–10⁶ | README Quantum; proposal §7.5 (QS_RHO2, QS_BREAKEVEN_1) | R/quantum_scaling.json → rows[].{rho2, rho2_breakeven_at_1pct} (5.8×10⁴–9.6×10⁵) | as C44 | ρ² = rel_rmse²·B from E2 (6–20 reps ⇒ noisy) | seconds | — | ✅ | T1 |
| C46 | Falsifiable wall-clock criterion | s* 8.2×10⁴–4.4×10⁶; p_L ≈ 5.6×10⁻⁸ at n=28 (short.md "near 10⁻⁷"); 635 logical / 1.2M physical qubits; RDPO paths for 10%: 13–11,717 | README Quantum; short.md L126–128; proposal §7.6 (Q_ADV_TABLE/TEXT) | R/quantum_advantage.json → advantage[].{required_speedup, pL_required, logical_qubits, physical_qubits, classical_paths} | `src.quantum.advantage` | as C08 | as C08 | as C08 | ✅ | T1 |
| C47 | MC comparison overstates advantage | up to 10¹¹ | proposal §7.2; short.md L116 | derived: (1−μ)/μ ÷ ρ²_RDPO at ring 28 = 7.6×10¹²/117 ≈ 6.5×10¹⁰ (no stored field) | — | — | — | — | ⚠️ order-of-magnitude, no file | T0 |
| C48 | Error mitigation trust boundary | mitigation worse than none in 21–38%; GoF-guard fallback 2–4%; guard 2 p99 0.15–0.17 (≤0.17), max 0.20; fallback 10–22% | README Quantum; short.md L130–132; proposal §7.7 (MIT_TABLE/TEXT) | R/quantum_advantage.json → mitigation_summary[δ].{mitigation_worse_than_raw, fallback_rate, guard2.p99, guard2.max, fallback2_rate} | `src.quantum.advantage` (mitigation_study) | 250 instances × 3 δ levels, seed 11, 100 shots/depth | unknown (est minutes) | resumes | ✅ (classical fallback/pilot are synthetic, see logic L6) | T1 |
| C49 | Near-term QCBM sampler | noise biases weights 9–60%; 10⁵-shot calibration ≤ 0.6% | proposal §7.8 | R/nisq.json → qcbm_bias[].{rel_bias_ideal_weights (0.091–0.597), rel_bias_calibrated_rms[calib_shots=1e5] (≤0.0061)} | `src.quantum.run_nisq` | QCBM seed 1; 10 reps seed 0..9 per setting | unknown | none | ✅ ("8 qubits, depth 9, 32 entangling gates" not in any JSON) | T1 |
| C50 | Exact structured loading | joint law reproduced to 10⁻¹⁶ | proposal §7.4 | R/structured.json → verify[].max_abs_err_vs_dense (4.4×10⁻¹⁶) | `src.quantum.run_structured` | deterministic (IQAE calibration seed 0, 30 reps) | unknown | none | ✅ | T0 |
| C51 | Exact benchmark model time | 0.2 ms / scenario | proposal §8.1 table, §2.10 | none found (REPORT text only; R/exact.json has no timing) | (`src.model.exact`) | — | — | — | ∅ | n/a |
| C52 | Exact tilted TN, 28-country ring | 68 s | proposal §8.1 table | R/tensor.json → B_scaling_n[ring,28,tol=1e-9].runtime_s (=67.98) | `src.tensor.scaling` | deterministic | as C19 | none | ✅ (timing) | T0 |
| C53 | Full reproduction cost | ≈ 7 CPU-hours (≈ 3 h wall, 4 cores) | README Reproduce; proposal §8.1 table | none (hard-coded). Stored timings sum to ≈ 9–11 CPU-h (v2 scale alone 4.4 CPU-h; E2 ~2 h; AP 1.45 h) | all drivers | — | — | — | ⚠️ likely understated | T2 |
| C54 | Per-setting quantum resource table | logical/physical qubits, code distance, Toffoli/rotations/CNOT, runtime per AE run | proposal §8.2 (Q_RES_TABLE) | R/quantum_advantage.json → advantage[].{logical_qubits, toffoli_A, rotations_A, cnot_Q, lookup_toffoli_avoided, code_distance, physical_qubits, quantum_seconds} | `src.quantum.advantage` | as C08 | as C08 | as C08 | ✅ (auto-filled) | T1 |
| C55 | Test suite | 59 passing tests | README badge + layout; short.md L136 | tests/ (40 functions; 59 with parametrisation) | `python -m pytest -q tests/` | fixed seeds inside tests | unknown | `test_realdata_check_runs_on_cached_catalogue` and `test_param_uncertainty_credible_interval` skip if caches/results missing | ✅ count (REPORT.md §9 says 35: stale) | T1 |

Commercial (not audited, one line): TAM US$2.1bn, SAM US$30m, SOM/ARR US$3.5m by year 5, 28 customers —
`R/market_model.json` (`src/business/market.py`, hand-entered assumptions; regenerated by `build_paper`). Consistent with short.md §7.

## Tier counts

* T0: 18 (C02, C03, C07, C19, C20, C26, C29, C32–C36, C42–C44, C47, C50, C52)
* T1: 23 (C01, C05, C06, C08–C13, C21, C27, C28, C31, C37, C38, C40, C41, C45, C46, C48, C49, C54, C55)
* T2: 11 (C04, C14–C18, C23–C25, C39, C53)
* n/a, no producer: 3 (C22, C30, C51)

Total: 55 claims.

## Runtime budget (estimated)

| Driver | CPU | Wall (4 cores) | Basis |
|---|---|---|---|
| baselines.run_baselines | 3 min | 3 min | log |
| tensor.scaling | 15–20 min | same (serial) | json Σ runtime_s 770 s + Exp A/C |
| experiments e1 / e2 / e3 | 0.46 h / ~2 h / ~0.9 h | serial | json / json / est |
| experiments_v2 throughput / scale / bench / transfer | 0.15 h / 4.4 h / ~0.25 h / ~0.3 h | 0.15 / 1.1 / 0.07 / 0.08 h | est / json / est / est |
| realdata eq, loss, rain, param_uncertainty | < 0.1 h | < 0.1 h | est |
| central_asia_pool / carec_pool / asia_pacific_pool | 0.1 / 0.45 / 1.5 h | 0.05 / 0.25 / 0.45 h | json + MPS |
| elt.demo, rdpo.projected, rdpo.timing, baselines.stronger | < 0.1 h | < 0.1 h | json |
| quantum run_structured, run_twisted, run_nisq, advantage, scaling_laws | unknown (est ≤ 0.5 h) | ≤ 0.5 h | — |
| **Total** | **≈ 10–11 CPU-h** | **≈ 5–6 h if run sequentially; ≈ 3 h with parallel launches** | |

## Reproduction mechanics (important)

1. **Every main driver resumes and will silently no-op on the shipped results.** `experiments e1/e2/e3` skip cells already in
   `R/rdpo_e*.json`; all `run_jobs` drivers (v2 scale/bench/transfer, ca_pool, carec, asia_pacific, elt) skip job ids already in
   `raw/*.jsonl` (all are 100% complete); `quantum.advantage.run` skips if keys exist. Move `results/` aside (keep a copy for diffing).
2. **Dependency order:** `src.model.exact` → `baselines.run_baselines` → `experiments e1` → `experiments_v2 bench`, `baselines.stronger`;
   `tensor.scaling` → `experiments e2/e3`, `experiments_v2 scale`, `quantum.run_structured` → `quantum.run_twisted` →
   `quantum.advantage` (needs `rdpo_e2.json`) → `scaling_laws()` (manual call); `realdata.loss_backtest` → `realdata.rain_check` →
   `param_uncertainty`, `asia_pacific_pool`, `elt.demo`; `realdata.eq_check` → `central_asia_pool`; `carec_pool` → `rdpo.projected`.
3. **Offline:** all network paths (urllib to raw.githubusercontent.com / NOAA S3; `git clone owid/owid-datasets`) are guarded by
   `if not cache.exists()`, and every cache file is present under `data/real/` ⇒ fully offline. `carec_pool` would clone to `/tmp/owid_ds`
   (POSIX path) if its cache were missing. Side effects: `rain_check` rewrites `data/real/ghcn_country_index.csv.gz`; `elt.demo` rewrites
   `data/elt/*.csv` and writes temporary `_ylt_*.csv`; `build_paper` rewrites `R/market_model.json` and figures and needs pandoc, node +
   Playwright, pdfunite.
4. `quantum_scaling.json` has no CLI entry point (`scaling_laws()` sits after `if __name__ == "__main__"` in `src/quantum/advantage.py`).

## (a) Seeds / determinism

Seeds are explicit almost everywhere (`np.random.default_rng(seed + r)` per replication/job; RQMC uses `Sobol(seed=rng)`), with
numpy 2.5.3 / scipy 1.18.1 pinned. With identical library versions, per-job estimates should repeat bit-for-bit for E1, E3, v2,
real-data pools, ELT, stronger baselines, param-uncertainty and baselines. Exceptions:
* **E2 is not bit-reproducible:** seed = `2000 + 1000n + 100·len(rows) + r` depends on resume state, and `rdpo_e2.log` shows a mid-run
  "restart with target-relevance freezing" (pair rows re-run under new code). Only statistical agreement is possible for C14–C17, and
  this propagates to quantum ρ²/s* (C08, C45, C46), the v2 "Phase 1 study" columns (C24) and the ring-16 stronger-baseline ratio.
* Pooled drivers aggregate in `imap_unordered` completion order ⇒ last-digit float differences in RMSE/coverage only.
* All timing-derived numbers (C05, C13, C17, C21, C24, C29 ms, C38/C39 MPS seconds, C52, and s*/runtime in C08/C46/C54) are machine-dependent.
* Cross-version: L-BFGS-B, Sobol scrambling and BLAS differences can change training trajectories ⇒ statistical match only on other stacks.

## (b) Logic issues in method comparisons

* L1 (OK) Training is charged: `train.py` counts CE, χ²-PG, RQMC and final batches in `used`; total = B exactly (asserted in
  `test_rdpo_estimate_accuracy_and_interval`). Recycling is unbiased (eligibility fixed before each batch). The analytic twist warm start
  and relevance freezing use the model (no paths); the same analytic twist is also given to twisting/SMC/flow, so that is fair.
* L2 Transfer (C06): the 65,536-path "overnight" source fit is **not charged** to the 2,048/4,096 budget (by design, but the ×84 is vs cold start with zero prior compute).
* L3 (OK) No reference leakage into training: `rdpo_estimate` never sees μ. References: dense enumeration (n ≤ 6), exact pair reduction,
  tilted MPS (tol 1e-9/1e-12; complete-7 at 1e-6). WP2 thresholds are *chosen* from the exact curve (question selection only).
* L4 Seeds across methods: v2 scale/bench, WP2 pools and ELT use common seeds across methods (fair). E1 compares RDPO (40 reps, seed 1000…)
  with baselines from a separate run (200 reps, seed 20261031) — statistically fine (min margin ×3). **Stronger baselines claim "same seeds"
  but reuse Phase 1 RDPO numbers from E1/E2 (different seeds, 40/10 reps) under a "Phase 2" header.**
* L5 `smc_estimate` is a single exponential-twist IS batch (no tempering/resampling); the flow IS is claimed to "dominate SMC by
  construction" yet is worse in 4/4 cases (e.g. 69.8% vs 10.8%); splitting returns 0 in every ring-16 rep.
* L6 Mitigation study (C48): the "classical RDPO fallback" and "pilot" are synthetic `a·(1+0.005·N)` / `a·(1+0.05·N)` draws around the
  **true** amplitude, not RDPO runs ⇒ guard-2 tail bounds are partly by construction.
* L7 Wall-clock "2.3–11.9× faster" compares serial E2 timings with 4-process-pool timings and different configs.
* L8 Param uncertainty (C01): the EQ posterior is Beta(½+5, ½+4) on raw co-occurrence years (5 of 9 KG-event years, median ≈ 55%), which
  does not remove chance co-occurrence (the bivariate-Poisson MLE that it claims to use gives 48%) ⇒ the 3.12× headline is biased upward
  relative to the 2.75× point-estimate factor.
* L9 E3 "no twist warm start" is a no-op on pair targets (identical rows); not a flaw, but it pads the table.
* L10 E2 result file mixes code versions (pair rows re-run after relevance freezing was added).

## (c) Test → claim coverage

* `test_exact.py` (rates, checkpoints, Table 5 for 5 scenarios, brute force, expm): exact μ behind C09–C12, brief values in C33/C35/C36.
* `test_rdpo.py`: source form = reference values (E1 μ); gradients; unbiasedness; `rdpo_estimate` < 5% on joint exhaustion at 16,384 and
  paths_used == B (C10, L1); fast engine identical to v1 (C21 "identical policy"); spline/stage policies valid, v2 t-interval (C23 machinery);
  exact pair vs dense only (C26; MPS part not tested despite the name); `eq_check` sanity (C32, loose); projected Doob zero variance (C29).
* `test_methods.py`: zero-variance & baselines unbiased (C09 baselines), IQAE accuracy (C42/C46 inputs), MPS = Table 5 and tilted MPS = dense
  (C19, references for C14–C18, C38, C39), structured loading exact (C50), twisted loading exact/bounded (C42/C43), NISQ calibrated weights (C49).
* `test_stronger.py`: replay consistency, SMC < 15% error, flow > 0 (does not test "improves on untwisted"), splitting within ×3 (C31 machinery);
  param-uncertainty point ∈ CI (does not test the "above the brief" headline, C01).
* `test_elt.py`: term law vs brute force, multi-country footprints, YLT round trip (C40 machinery).
* **Not covered by any test:** any headline number (20/25, ×6.3–×171, 0.1–4.2%, 13/16, 16/16, WP2 errors, transfer gains), throughput/timing,
  loss/rain/param-uncertainty outputs, CA/CAREC/AP pipelines, quantum fits, s*, mitigation.

## Mismatches / stale / unbacked claims

1. C31 README "×1.4–×18" vs JSON ×5.4–×7.8 (paper fills "×5.4–×8"); "same seeds" false; Phase 1 numbers under a Phase 2 header.
2. C20 "2% truncation error at tol 10⁻¹²" vs 6.7×10⁻⁴ in `tensor.json` and in REPORT's own §5.2 table.
3. C27 README "EL ×0.6–0.8 instead of ×0.5" fits only B=16,384 (65,536: ×0.45 vs ×0.33).
4. C34/C36 short.md labels profile-likelihood CIs as "posterior"; short.md §7 "2.8× posterior median" (posterior is 3.12×).
5. C11 short.md "94% mean" coverage (stored 94.7% ⇒ 95%).
6. C53 "≈ 7 CPU-h" vs ≈ 10–11 CPU-h from stored timings.
7. Rounding: C26 "≤ 3 ms" (3.3 ms), C32 "every p ≥ 0.12" (0.115), C46 short.md "near 10⁻⁷" (5.6×10⁻⁸).
8. short.md template bug: "9.5×10⁻⁹-probability-probability" (HL_AP_MU already ends in "-probability").
9. No backing file: C22 (50–500×), C30 (×19 spline; value-of-training), C51 (0.2 ms), C53; C47 is derived only.
10. E2 not bit-reproducible (resume-dependent seeds + partial restart); C44 producer has no CLI entry; REPORT.md "35 tests" stale (59).
