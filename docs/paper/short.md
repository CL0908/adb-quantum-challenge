---
title: "Exact and Learned Joint-Tail Estimation for Pooled Disaster Risk Financing"
author: "Tan Chun Loong"
date: "November 2026"
lang: en
---

<style>body{font-size:9.7pt;line-height:1.45} h1{font-size:17pt;margin:14pt 0 8pt;padding-top:14pt} h2{font-size:11.5pt;margin:10pt 0 4pt} figure{margin:6pt 0 8pt} #refs{font-size:7.9pt;line-height:1.3} #refs .csl-entry{margin-bottom:2pt}</style>

::: {.titleblock}
[Exact and Learned Joint-Tail Estimation for Pooled Disaster Risk Financing]{.ttl}
[How often several countries' disaster covers pay out together: computed with zero sampling error where possible,
estimated with a learned sampler where not, and costed on quantum hardware.]{.sub}
[Tan Chun Loong · Nanyang Technological University, Singapore · ADB Quantum Innovation Challenge, Phase 1 proposal ·
November 2026]{.byline}
:::

::: {.pitch}
ADB's US$160m Disaster Relief Bond pays the Kyrgyz Republic and Tajikistan separately; whether such cover can be
pooled, and how much shared capital a pool needs, depends on how often several countries need a payout in the same
three-year term. We compute these joint probabilities exactly, with no sampling error, for pools of up to 28
countries, and estimate the cases exact methods cannot reach with a learned sampler that is more accurate per
simulated scenario than the challenge's reference methods. We then cost the quantum version of the same
calculation end to end and state the hardware it would need to pay.
:::

::: {.motto}
Exact where possible, learned where not, quantum where it can be shown to pay.
:::

::: {.results}
1. **Exact answers at regional scale (quantum-inspired).** A tensor network reproduces every value of the brief's
   Table 5 in {{EQT_EXACT_MS}}, and gives the full joint law of data-calibrated pools of 10 and 28 countries in
   {{CAREC_MPS_MIN}} and {{AP_MPS_SEC}} on one CPU core. Dense dependence, where each country shares events with
   its next four neighbours, is exact up to 28 countries ({{DENSE_BIG_SEC}}).
2. **Higher accuracy per query (learned).** Where exact contraction is not available, Rényi-Doob Policy
   Optimization (RDPO) beats the best of the brief's four methods in {{EQT_A_PATH_BRIEF}} benchmark cells at the
   same number of simulated scenarios, by {{E1_EXH_RANGE_SHORT}} on the rarest target, with training charged and
   no bias.
3. **A quantum stage with a measured bar (quantum).** Amplitude estimation with exact loading needs resources
   linear in countries and dependence links. Against our best classical method it breaks even only on hardware
   {{Q_SSTAR_28}} times faster than a projected fault-tolerant baseline at 28 countries; near-term circuits are
   costed for an IBM device.
:::

For ADB the larger uncertainty is dependence itself: under any dependence consistent with the two countries' own
risks, the chance that the pool pays more than US$80m in a term ranges from 0 to {{FIN_EXC_MAX}} (brief
{{PU_EXC_BRIEF}}). Public data give a stress range centred at {{PU_EXC_X}} times the brief's value (§3.5).

::: {.claims}
::: {.yes}
**What we claim**

* Exact joint tails, zero sampling error, for every benchmark target and for sparse or bounded-treewidth pools of
  up to 28 countries.
* Higher accuracy per query than the brief's baselines in {{EQT_A_PATH_BRIEF}} cells; unbiased estimates with
  95% intervals.
* An end-to-end, falsifiable resource estimate for the quantum stage, and the classical method it must beat.
:::
::: {.no}
**What we do not claim**

* A quantum advantage today, on any hardware.
* Faster wall-clock than simple methods on our CPU engine: at equal CPU time RDPO wins {{EQT_A_TIME_BRIEF}}
  benchmark cells (Table [tab:eqt]).
* A re-pricing of the bond: dependence results use public proxies, not the bond's trigger definitions.
:::
:::

::: {.glossary}
**Term**: the bond's three-year cover period. **Joint tail**: outcomes in which several countries pay together;
**joint exhaustion**: both countries pay their full US$80m in the same term (μ = {{BASE_JE_MU}} on the benchmark). **Exceedance**:
the pool's total payout L exceeds a threshold T, Pr(L > T). **Stop-loss**: the expected payout above T,
E[(L − T)⁺]. **Path** or **query**: one simulated three-year scenario (classical), or one run of the quantum
state-preparation circuit. **μ**: the exact value of a target. **Relative RMSE**: root-mean-square error divided by
μ, over independent repetitions. **Treewidth**: how tangled the dependence graph is; exact cost grows exponentially
in it, not in the number of countries n.
:::

# 1 Problem and classical bottleneck

**The model.** Each country holds US$80m of cover over a three-year term: US$70m against earthquakes, paid in
quarter steps, and US$10m against extreme rain, paid once. Earthquakes and rain arrive as Poisson processes; some
events hit both countries. The brief's five targets (expected loss, both pay, exceedance of US$80m, stop-loss above
US$80m, joint exhaustion) are expectations μ = E[f(L₁, L₂)] of the two term losses, under five dependence
scenarios: 25 cells.

**The bottleneck is accuracy in the joint tail, and it grows with the portfolio.** Joint exhaustion has
μ = {{BASE_JE_MU}}. Plain Monte Carlo with the benchmark's 16,384 paths returns zero in {{MC_ZERO}} of runs. Exact
enumeration is instant for two countries, but the annual state space is 10ⁿ for n countries, and regional pools
cover 10–30 countries, where pool targets fall to {{POOL_RARE_MU}} per term and, on a 28-country ring, to {{Q_MU_28}}. Each new dependence scenario
needs a fresh computation, so a method must be accurate at a fixed, modest budget.

**State of the art.** Catastrophe-model platforms aggregate event and year loss tables by plain Monte Carlo, with
dependence through shared events or copulas [@joe2014; @embrechts1997]. Rare-event methods reduce variance but need
a good proposal [@glasserman2003; @owen1997; @rubinstein2004; @cerou2007; @delmoral2006]. Amplitude estimation
offers a quadratic query speed-up for risk measures [@montanaro2015; @woerner2019; @egger2021], but end-to-end
studies find that state preparation and fault-tolerance overheads decide the outcome [@chakrabarti2021;
@herbert2021; @herman2023]. Section 2.5 positions our method against each approach the brief suggests.

# 2 Scientific approach

## 2.1 One exact representation

The capped multi-year payout process is an exact product of independent Poisson **sources**: for each country a
solo earthquake source and a solo rain source, and for each dependence link (edge) between two countries a shared
earthquake source and a shared rain source. With n countries and |E| edges there are K = 2n + 2|E| sources. Each
source's effect over the term is a capped increment with at most 25 values, and its law is computed exactly from
the brief's rates. Capping commutes with addition, so the term state is the capped sum of K independent draws
(annex, Propositions 1–2). The same representation feeds three solvers.

## 2.2 Exact engine: quantum-inspired tensor networks

A matrix-product state (MPS), the tensor network used to simulate quantum many-body systems and to model
probability distributions [@orus2014; @han2018], stores the joint law of n countries as a chain of small tensors.
Applying the source kernels one by one gives the law exactly; its size is set by the *bond dimension* χ, which
follows the dependence graph, not n.

* **Tilted gauge.** Plain truncation loses the rarest tails: {{TN_PLAIN_ERR9}} error on joint exhaustion at truncation tolerance
  10⁻⁹, against {{TN_TILT_ERR9}} in the tilted gauge. We compress the law multiplied by the inverse of the independent marginals, so rare and common states
  have similar weight, then undo the tilt exactly.
* **Frontier variable elimination** handles dense but narrow graphs (bounded treewidth): countries are added in a
  fixed order and summed out once no remaining link touches them [@dechter1999]. Only non-negative sums occur, so
  no gauge is needed.
* **Pair reduction.** A question about two countries needs only the sources that touch them: exact in
  {{V2_EXACT_MS}} at any pool size.
* **Limits.** Cost grows exponentially with treewidth and with the number of countries a single event can hit.
  Complete graphs are exact to {{TN_COMPLETE_N}} countries only.

## 2.3 RDPO: learned sampling where exact contraction is not available

Importance sampling draws paths from a proposal and reweights them by w = p/q. The zero-variance proposal is the
**Doob h-transform** of the source process: draw each source's increment x in state s with probability
p(x)·V(s ⊕ x)/V(s), where V is the exact conditional value of the target. Every path then returns μ. V is a value
function, so this proposal is the optimal policy of a KL-regularised control problem [@todorov2009; @kappen2005],
and RDPO learns it as a reinforcement-learning policy:

1. **Policy.** A log-linear soft value over local features (country payout levels, link indicators, progress
   towards the threshold), mixed with 5% of the true law so that every likelihood ratio stays bounded. One path
   costs O(n + |E|) and never touches the 10ⁿ joint states.
2. **Training.** Cross-entropy (a KL objective) first, to reach the tail; then a Rényi-2 policy gradient, because
   the estimator's relative variance *is* the χ² divergence from the zero-variance law, while a small KL does not
   bound it.
3. **Estimate.** Training paths count against the budget. Later batches are pooled with weights fixed before they
   are drawn, which makes the estimate exactly unbiased for any learned policy, with a Student-t 95% interval.

RDPO's role is what the exact engine cannot reach: high-treewidth dependence, events that hit three or more
countries (vendor event sets), and any setting where a query is expensive: one catastrophe-model run, or one
quantum state preparation.

## 2.4 Quantum stage: amplitude estimation with exact loading

* **Encoding.** One register per source (1, 3 or 6 qubits), prepared exactly in $\sqrt{q_k(x)}$ with a cascade of
    controlled rotations [@grover2002]; no learned loader, no truncation. Reversible adders form the pool loss.
* **Twisted loading with a bounded weight.** Loading the true law gives amplitude $a = \mu$ and needs of order
  $\mu^{-1/2}$ queries. We load an exponentially twisted law instead and write the bounded importance weight
  $e^{-\eta(U-T)}$ onto the objective qubit one bit of $U - T$ at a time ($U$: uncapped loss, $\eta$: twist,
  $\Lambda$: its cumulant). Then $a = \mu\, e^{\eta T - \Lambda(\eta)}$, which decays with the large-deviation
  gap rather than with $\mu$.
* **Estimation.** Iterative amplitude estimation [@grinko2021] needs about $c_Q/(\varepsilon\sqrt a)$ state
  preparations for relative error $\varepsilon$ ($c_Q$ measured by exact emulation), against $\rho^2/\varepsilon^2$
  paths for a classical sampler with per-path relative variance $\rho^2$.
* **Crossover law.** The quantum stage beats RDPO in wall-clock only on hardware $s^*$ times faster than the stated
  baseline, $s^* = \kappa\,\varepsilon/\rho^2$ with $\kappa = c_Q\tau_A/(\tau_C\sqrt a)$; $\tau_A$ is the
  fault-tolerant time per state preparation, $\tau_C$ RDPO's time per path. Comparing with plain Monte Carlo instead
  would overstate the quantum gain by up to {{Q_MC_OVER}} here.
* **Error correction (projected fault-tolerant baseline).** Surface code at physical error 10⁻³, 1 µs code cycles,
  four magic-state factories, Toffoli = 4 T, Ross–Selinger rotation synthesis [@fowler2012; @litinski2019;
  @rossselinger2016].
* **Error mitigation and a trust guard.** Near-term runs use a noise-aware likelihood across Grover depths. A
  quantum estimate is accepted only if the fit passes a goodness-of-fit test *and* agrees within 3σ with a cheap
  classical RDPO pilot; otherwise the classical estimate is reported.

## 2.5 Position against the approaches the brief suggests

::: {.compact}
| Approach | What it needs on this problem | What we measured or cite | Verdict |
|---|---|---|---|
| Amplitude estimation [@woerner2019; @egger2021] | exact loading, payout oracle, deep circuits | exact source-wise loading, linear resources; break-even needs hardware {{Q_SSTAR_28}}× faster at 28 countries; near-term circuits off by {{HW_BEST_RANGE}} | right primitive; needs fault tolerance |
| QCBM, QGAN, QBM, quantum copula as loaders [@benedetti2019; @zoufal2019; @amin2018; @zhu2022] | learn p from data | p is known exactly here; a learned loader errs in the tail it never sees. As a proposal, an 8-qubit Born machine's noise biased weights by {{QCBM_BIAS}}, {{QCBM_CAL}} after calibration | only as a reweighted proposal |
| Quantum kernels [@havlicek2019] | a surrogate or proposal | payouts are exact and cheap; the optimal proposal is a value function | no role found; not tested |
| Annealing, annealer QBM [@amin2018; @benedetti2016] | samples at unknown effective temperature | unbiased weights need known proposal probabilities | not used |
| Tensor networks [@han2018; @orus2014; @kastoryano2022] | low bond dimension or treewidth | exact to 28 countries in seconds to minutes (Figure [fig:exact]) | **our exact engine** |
| Classical rare-event methods | a good proposal | MC, Sobol, stratified, brief IS, SMC, splitting, flow IS, twisting: all benchmarked | the bar for every claim |
:::

# 3 Prototype and results

All results come from one Python package with {{N_TESTS}} tests, fixed seeds and exact references.

## 3.1 The brief's benchmark, reproduced exactly

{{BRIEF5_TABLE}}

The full law (100 annual states, three-year capped term) takes {{EQT_EXACT_MS}} by dense enumeration; the brief
quotes about 0.2 ms for its own implementation. Two-country questions inside any pool take {{EQT_PAIR_MS}} by pair
reduction. Because the brief lists the published annual expected loss of each note, matching it is a consistency
check: our annual figures are {{FIN_EL_KG}} and {{FIN_EL_TJ}} (published {{FIN_ELC_KG}} and {{FIN_ELC_TJ}}).

## 3.2 Accuracy per query on the benchmark

![**Figure [fig:e1].** Benchmark, 16,384 paths per estimate with RDPO's training included, 40 replications. (a)
RDPO's relative RMSE. (b) Error of the best of the brief's four methods divided by RDPO's (blue < 1: a reference
method is better).](../../figures/paper/e1_heatmap.png){.wide}

* **RDPO wins {{EQT_A_PATH_BRIEF}} cells**, all except expected loss, where scrambled Sobol is better and should be
  used. On joint exhaustion the gain is {{E1_EXH_RANGE_SHORT}}; RDPO's error there is {{E1_BASE_JE}} in the base
  scenario.
* **Intervals.** Nominal 95% intervals cover the exact value in {{E1_COV16_MEAN}} of replications (worst cell
  {{E1_COV16_MIN}}).
* **Stronger baselines.** Against adaptive multilevel splitting, SMC on the optimal exponential twist and a
  flow-based importance sampler, at the same budget and seeds, RDPO is more accurate on every rare target, by
  {{STR_RANGE}}. SMC on the optimal twist is the strongest of the three.

## 3.3 Scaling: exact references, then RDPO beyond two countries

* **Exact references.** Sparse graphs keep χ = {{TN_SPARSE_BOND}} up to 28 countries; the data-calibrated CAREC (10
  countries, χ = {{CAREC_BOND}}) and Asia–Pacific (28 countries, χ = {{AP_BOND}}) pools are exact in minutes;
  {{DENSE_NEXACT}} dense references reach {{DENSE_BIG}}.
* **RDPO against these references.** On the pools, RDPO's relative RMSE on the rarest exceedance is
  {{POOL_RARE_LIST}}, with 65,536 paths; plain Monte Carlo would need {{POOL_MC_PATHS}} paths for 10%. Pair exhaustion
  inside a pool is harder to sample ({{POOLPAIR_ERR}}, coverage {{POOLPAIR_COV}}) and is answered exactly by pair
  reduction instead.
* **Against exponential twisting**, the strongest simple method for thresholds, at the same path budget RDPO is
  more accurate in {{SCALE_TW_WINS}} scaling settings and {{DENSE_NWIN}} dense settings: its per-path variance is
  {{VAR_RING}} lower on rings of 8–28 countries and {{VAR_DENSE}} lower on dense graphs. The gain comes from
  state dependence: the proposal changes with the payouts already drawn, which a single twist cannot do.

![**Figure [fig:exact].** The exact engine. (a) Time to compute the full joint law on one core: tilted MPS for chain,
ring and complete dependence; frontier elimination for band-k graphs (each country linked to its next k neighbours)
and a ladder with diagonals; stars: data-calibrated pools and a 50-country ring. (b) MPS bond dimension: constant
for sparse dependence, far below the √(10ⁿ) a generic law would need.](../../figures/paper/exact_scaling.png){.wide width=92%}

## 3.4 Wall-clock, stated once

Per path is the brief's criterion. Per CPU second the picture is different, because RDPO trains for each estimate
while the simple methods draw paths almost for free.

{{TABLE_EQT_SHORT}}

At equal CPU time on our NumPy engine, cheap methods win most cells. RDPO keeps joint exhaustion on the benchmark
and inside pools, where a single twist does not apply. Two cheaper variants were tried and only tie twisting: a
learned product-form tilt (median error ratio {{V3_TILT_MED}} of twisting's) and an automatic selector
({{V3_SEL_MED}}); the gain is in the state dependence, which costs training. Phase 2 therefore targets the
wall-clock gap directly: a GPU-batched engine, and value functions taken from tensor-network contractions instead of
learned features (§5).

<!-- CUTSET-RESULTS -->

## 3.5 Model risk and the pooling decision

![**Figure [fig:bvd].** The bond's two dependence assumptions against public data, and the effect on a pool. Dots:
point estimates; bars: 95% intervals. Earthquake shares come from years in which both countries recorded losses;
rain shares from {{RAIN_NST}} weather stations at the 1-in-5-year level.](../art/brief_vs_data_clean.svg){.wide}

Expected loss does not depend on dependence; anything that shares capital does. Three results, all exact:

* **Model-free range.** With both countries' term laws fixed at the brief's values, a linear program over all
  couplings bounds Pr(L > 80m) in [0, {{FIN_EXC_MAX}}]; the brief's {{PU_EXC_BRIEF}} sits at {{FIN_EXC_POS_BRIEF}} of
  that range. No sampling accuracy narrows it; only evidence on dependence can.
* **Stress range from public data.** Posterior uncertainty on the two shared shares gives Pr(L > 80m) of
  {{PU_EXC_MED}} (95% credible interval {{PU_EXC_LO}}–{{PU_EXC_HI}}); joint exhaustion is unchanged within its
  interval, because more shared earthquakes and less shared rain cancel.
* **Pooling.** One shared US$80m limit instead of two separate covers keeps {{FIN_POOL_BRIEF}} of expected payouts
  under the brief's dependence and {{FIN_POOL_MED}} ({{FIN_POOL_LO}}–{{FIN_POOL_HI}}) under the data; a US$100m
  shared limit keeps {{FIN_POOL100_MED}}.

**Weaknesses, stated up front.** The earthquake share rests on {{LOSS_NBOTH}} coincident loss years: NOAA
1900–2020 gives {{LOSS_NOAA1900}}, NOAA 1975–2020 gives {{LOSS_NOAA1975}}, and EM-DAT 1990–2020 gives
{{LOSS_EMDAT}}, consistent with the brief's 7.5%. Rain sharing is measured at the 1-in-5-year level ({{RAIN_SHARE}});
the trigger is about 1-in-50, where the stations record too few events. Both proxies are loss years and station
rain, not the bond's cat-in-a-box and gridded-rain triggers. We therefore read the posterior as a stress range for
dependence, not as an estimate, and Phase 2 measures both shares at trigger level.

## 3.6 Catastrophe-model input

A converter turns an event or year loss table into the source form, including events that pay in three or more
countries. On a demonstration event set built from public data ({{ELT_EVENTS}} events, {{ELT_MULTI}} footprints
hitting three or more countries), the rarest pool question read off a 10,000-year year loss table has
{{ELT_RARE_YLT}} relative error; RDPO on the converted sources gives {{ELT_RARE_RDPO}} with 16,384 paths. No vendor
table was available.

## 3.7 Quantum stage, emulated

* **Exactness.** State-vector emulation of every loading circuit reproduces μ to a relative error of
  {{HW_SV_ERR}}; compiled resource counts
  follow {{QS_FITS_INLINE}}, each within {{QS_FIT_ERR}} of the compiled circuits.
* **Where it could pay.** RDPO's measured per-path relative variances are ρ² = {{QS_RHO2}}; break-even at 1% error
  needs ρ² ≈ {{QS_BREAKEVEN_1}} at the projected baseline (Figure [fig:qphase]). The gap closes with portfolio
  size (s* falls from {{Q_SSTAR_2}} at 2 countries to {{Q_SSTAR_28}} at 28), so the test is re-run first where ρ²
  grows fastest: dense graphs and events touching many countries.
* **Near-term run, costed.** The two-country law on {{HW_QUBITS}} qubits, transpiled for an IBM Heron-class device
  model ({{HW_DEVICE}}): one Grover iterate costs {{HW_N2Q_K1}} two-qubit gates. With the device's noise the best
  estimates are off by {{HW_BEST_RANGE}} (the direct joint-exhaustion circuit by {{HW_EXH_DIRECT}}), so the trust
  guard falls back to the classical value. The twisted circuits reach {{HW_TILT_P4}} error at two-qubit error 10⁻⁴
  and {{HW_TILT_P5}} at 10⁻⁵: the error rate at which this calculation becomes usable.
* **Mitigation can hurt.** Noise-aware fitting was worse than none in {{MIT_WORSE0}} of instances without
  miscalibration and {{MIT_WORSE_MAX}} at a control offset of {{MIT_DELTA_MAX}} (annex §7.7); the combined guard
  capped the 99th-percentile error at {{MIT_G2_P99}} with {{MIT_FB}} fallback.

![**Figure [fig:qphase].** (a) Wall-clock phase diagram for amplitude estimation against RDPO. Above the break-even
band the quantum stage is faster; each 100× hardware speed-up moves the band down two decades. Filled markers:
RDPO's measured ρ² at ε = 10%; open markers: plain Monte Carlo on the same targets, which would suggest a misleading
quantum advantage. (b) Logical qubits, Toffolis per state preparation and depth: linear in n and
|E|.](../../figures/paper/quantum_phase.png){.wide width=92%}

# 4 Resource estimation

**Classical** (Phase 1, measured on one core of a 4-core Intel Xeon server CPU, 15 GB RAM, no GPU):

::: {.compact}
| Task | Time | Memory |
|---|---|---|
| Brief's benchmark, exact, 5 scenarios | {{EQT_EXACT_MS}} per scenario | < 1 MB |
| RDPO, one benchmark estimate, 16,384 paths, training included | {{EQT_RDPO_MS}} | < 100 MB |
| Exact tilted MPS: 28-country ring / Asia–Pacific pool / CAREC pool | {{TN_RING28_S}} / {{AP_MPS_SEC}} / {{CAREC_MPS_MIN}} | < 1 GB |
| Exact frontier elimination, {{DENSE_BIG}} | {{DENSE_BIG_SEC}} | < 1 GB |
| RDPO, 28-country ring, 65,536 paths (Phase 2 engine; twisting: {{TW_RING28_SEC}}) | {{V2_RING28_SEC}} | < 1 GB |
| RDPO, Asia–Pacific pool, rarest exceedance, 65,536 paths | {{AP_RARE_SEC}} | < 1 GB |
:::

**Quantum.** Near-term: the costed emulation of §3.7. Fault-tolerant: projected from component formulas for the
twisted loader with the bitwise weight oracle, 10% relative error on Pr(L > T), surface code at physical error 10⁻³.

::: {.compact}
| Quantity | Near-term, 2 countries | Fault-tolerant, 2 countries | Fault-tolerant, 28 countries (ring) |
|---|---|---|---|
| Qubits | {{HW_QUBITS}} physical | {{Q_LQ_2}} logical, {{Q_PHYS_2}} physical (d = {{Q_D_2}}) | {{Q_LQ_28}} logical, {{Q_PHYS_28}} physical (d = {{Q_D_28}}) |
| Circuit depth | {{HW_DEPTH_RANGE}} (Grover power 0–2) | {{Q_DEPTHA_2}} CNOT layers per state preparation | {{Q_DEPTHA_28}} |
| Two-qubit gates | {{HW_N2Q_RANGE}} per circuit | {{Q_CNOTQ_2}} CNOTs per Grover iterate; {{Q_TOFF_2}} Toffolis per preparation | {{Q_CNOTQ_28}}; {{Q_TOFF_28}} |
| Shots / queries | {{HW_SHOTS}} shots, {{HW_QPU_MIN}} QPU minutes | {{Q_CALLS_2}} state preparations, Grover depth ≤ {{Q_KMAX_2}} | {{Q_CALLS_28}}, depth ≤ {{Q_KMAX_28}} |
| Run time | emulator fallback {{HW_DM_MIN}} CPU minutes | {{Q_TIME_2}} | {{Q_TIME_28}} (RDPO: {{Q_CLS_28}}) |
| Error handling | noise-aware likelihood, readout calibration, trust guard | surface code; guard against an RDPO pilot | same |
:::

**Add-on re-pricing (Xu Zi Yu; conditional).** Cheaper rotation synthesis (deterministic or randomised) and a Chebyshev amplitude estimator with an empirical constant lower the required hardware speed-up s* by {{ADD_RED_ALL}}× over the eight settings (ring-28: {{ADD_S28_DECL}} to {{ADD_S28_C}}–{{ADD_S28_M}}); s* remains far above 1, and we claim no quantum advantage (annex §8.3).

**Hardware modality and fit with Phase 2.** Each shared source couples two country registers. On 2D
superconducting lattices SWAP routing inflates two-qubit counts by {{SWAP_RANGE}} in our compilations, so dense
dependence suits all-to-all trapped-ion or neutral-atom machines; sparse, geographically ordered pools map onto a
line. Classical Phase 2 work uses the sandbox's GPU credits (batched rollouts and contractions); the hardware run
needs {{HW_QPU_MIN}} QPU minutes, and the emulator gives the same report if access fails.

# 5 Phase 2 plan of action (30 Nov 2026 – 22 Jan 2027)

**Already done, not repeated in Phase 2.** Exact engine to 28 countries; RDPO with unbiased estimator; all
benchmarks at equal paths and equal time; data-calibrated pools; ELT/YLT converter; quantum resource model and
costed hardware emulation.

**Three work packages, one person.** Planned effort: 20 h/week for all eight weeks (about 160 hours); over the holiday weeks 4–5 these
hours go to long compute jobs and writing, and GPU sweeps run unattended outside them.

::: {.compact}
| Work package | Milestone and measurable criterion (current value) |
|---|---|
| **WP1 Wall-clock (W1–W3).** Port rollouts and training to GPU in the sandbox (array code, batched over replications and scenarios); replace learned features by tensor-network value functions or cutset conditioning on graphs too wide for exact contraction. | **M1 (W3):** at equal wall-clock, RDPO beats the brief's four methods in ≥ 10 of 25 benchmark cells (now {{EQT_A_TIME_BRIEF}}) and is within 2× of twisting on the dense threshold cells (now {{V3_COLD_DENSE_MED}} behind). |
| **WP2 Dependence at trigger level (W2–W5).** Rain sharing from gridded data (ERA5 or GPCP open mirrors) at the 1-in-50-year trigger level; earthquake sharing at trigger magnitudes with a cat-in-a-box proxy; coupling bounds with data constraints. | **M2 (W5):** both shared shares estimated at trigger level with intervals, and a data-constrained range for Pr(L > 80m) narrower than [0, {{FIN_EXC_MAX}}]. |
| **WP3 Quantum run and decision package (W5–W8).** Source-wise twisted loader on the emulator, then a guarded run on IBM hardware; pooling calculator; documentation, classical bottleneck analysis, business case, pitch. | **M3 (W7):** measured fidelity against depth on hardware within the emulator's predicted band, and the guard's verdict. **M4 (22 Jan):** three ADB questions (pooled limit, dependence stress, ELT input) answered end to end from one command. |
:::

**Risks and fallbacks.**

* **GPU port slower than planned** → keep the CPU engine; report the equal-time table as is; M1 becomes "gap
  halved".
* **Tensor-network value functions do not help** → report the negative result; recommend twisting for threshold
  targets and RDPO for joint and pair targets, as Table [tab:eqt] already does.
* **Data access** (ERA5 needs an account; some hosts are blocked in our sandbox) → GPCP and CHIRPS open mirrors,
  then the cached station data with the level mismatch stated.
* **Hardware access or noise** → the emulator with the device noise model produces the same report; the run is the
  first item descoped.
* **Holiday weeks and a one-person team** → weeks 4–5 run long compute jobs only; descoping order: hardware run,
  then cutset conditioning, then the cat-in-a-box proxy. M4 is protected.

**Business case.** An open-source joint-tail engine with exact references and a validation pack, plus paid
integration and support, placed after a catastrophe model: loss tables in, joint-tail answers with intervals out.
First users: ADB's disaster-risk-financing team (trigger design, CAREC pooling studies) and sovereign risk pools;
then reinsurers and model vendors, replacing the plain Monte Carlo aggregation step. A bottom-up estimate gives a serviceable market of about US${{MK_SAM}}m a year (annex,
Appendix B). The same source model fits correlated credit defaults, multi-site infrastructure outages and
compound climate extremes.

# 6 Team

**Tan Chun Loong** (sole member). BAppSc Physics with a Second Major in Quantum Technology, NTU Singapore
(2024–2028, CGPA 4.8/5.0). Incoming Phortify Student Research Fellow at ICFO Barcelona, in Prof. Antonio Acín's
Quantum Information Theory group. URECA Research Fellow with Prof. Mile Gu on relative entropy and belief revision
in model-based reinforcement learning, the mathematics of RDPO; earlier URECA work on stochastic climate models
(Prof. François Gay-Balmaz). Quantum security intern, Fraunhofer Singapore (QKD under device imperfections).
Hands-on quantum software: Clifford+T compilation with the Qiskit transpiler, gridsynth and rmsynth (MIT iQuHACK
2026, sole Singapore representative); warm-start QAOA for wildfire response on satellite data (SEA Quantathon 2025
finalist); quantum reservoir computing for systemic risk (QAI Ventures GenQ Hackathon 2025). Disaster-response
domain: Track Lead for Wildfire & Emergency Response, SEA Quantum Leaders Summit 2026. A short CV is attached.

**One-person team, mitigated.** Every result is regenerated by one command from fixed seeds, and {{N_TESTS}} tests
check each exactness claim, so the work can be audited and handed over. The plan is front-loaded and has a stated
descoping order. A domain contact at ADB or QAI Ventures for the pooling questions would strengthen WP2 and WP3.

# 7 Prototype package, environment and AI use

* **Source code.** The complete repository (code, cached public data, result files, tests and the build of both
  documents) is public at <https://github.com/CL0908/adb-quantum-challenge>; `README.md` gives the
  one-command reproduction.
* **Environment.** Python {{ENV_PY}}, NumPy {{ENV_NUMPY}}, SciPy {{ENV_SCIPY}}, Matplotlib {{ENV_MPL}}, pytest
  {{ENV_PYTEST}} (`requirements.txt`), Linux x86-64, one 4-core Intel Xeon CPU, no GPU.
* **Simulators.** Exact state-vector and resource simulators written for this project; the near-term plan uses
  Qiskit {{HW_QISKIT}} and Qiskit Aer {{HW_AER}} with the fake IBM backend {{HW_DEVICE}} (Heron-class noise
  snapshot). No quantum hardware was used.
* **AI use.** The code, experiments and this document were developed with Claude Code (Anthropic) as an AI
  pair-programmer and writing assistant, following the author's research plan. The author directed the research
  questions and design decisions and reviewed all results. All numbers are produced by the scripts in the
  repository.

::: {.limits}
**Limitations**

* Dependence evidence uses public proxies (loss years, epicentre distance, station rain), not the bond's triggers;
  the earthquake share rests on {{LOSS_NBOTH}} coincident years.
* At equal CPU time RDPO wins only on joint and pair targets (Table [tab:eqt]).
* Exact references for complete dependence stop at 7 countries; high-treewidth pools have no exact check.
* Scrambled Sobol remains better on expected loss.
* No vendor event set and no quantum hardware run in Phase 1.
:::

# References

::: {#refs}
:::
