---
title: "Information-Theoretic Reinforcement Learning for Joint Catastrophe Tails"
subtitle: "Rényi-Doob Policy Optimization — Phase 1 Technical Proposal"
author: "Tan Chun Loong"
affiliation: "Nanyang Technological University, Singapore"
date: "November 2026"
lang: en
---

::: {.frontmatter}

::: {.headline}
## Headline findings

1. **The joint tail of the ADB bond is likely underestimated.** With the two dependence assumptions
   calibrated to public data and the posterior uncertainty propagated through the exact two-country
   model, the chance that the pool pays more than US$80m in a term has posterior median
   **{{PU_EXC_MED}} ({{PU_EXC_X}}× the brief's {{PU_EXC_BRIEF}})** and 95% credible interval
   **[{{PU_EXC_LO}}, {{PU_EXC_HI}}]** — entirely above the brief's point value.
2. **Joint tails that simulation cannot see, at about 1% error.** In a 28-country Asia–Pacific pool
   calibrated from public data, RDPO estimates a {{HL_AP_MU}}-probability pool loss to **{{HL_AP_ERR}}
   relative error** with 65,536 simulated terms ({{HL_AP_SEC}} on one core). Plain Monte Carlo sees
   no such event.
3. **Stress tests in seconds.** All 25 ADB scenario × question cells are re-estimated in **{{HL_SWEEP}}**
   on one core. Overnight policy transfer (sleep phase) cuts small-budget error by {{V2_TRANSFER_GAIN}}.

*Indicative findings. Public data serve as proxies; not a re-pricing of the actual bond. Methods and
caveats in the full proposal §6.6–6.7.*
:::
:::

# 1 Problem

Pooled catastrophe bonds are priced and stress-tested on their **joint tails**: how often several
countries exhaust their cover together. For the ADB bond on the Kyrgyz Republic and Tajikistan the joint
exhaustion probability is about 3×10⁻⁵, and the pool exceedance about 1% per three-year term. These
probabilities are rare, poorly constrained by history, and must be re-computed under many dependence
scenarios. Plain Monte Carlo returns zero in most runs at practical budgets; exact enumeration grows
as 10ⁿ with the number of countries.

A regional pool of 28 ADB developing members has 10²⁸ states, which no exact method can enumerate.
Pricing, trigger design and stress testing all break down at exactly the questions that matter most.

# 2 Method — Rényi-Doob Policy Optimization

**Source-level formulation (Proposition 1).** The capped three-year payout process is an exact product
of commuting Poisson sources with finite increment laws. A scenario is a sequence of K = 2n + 2|E|
independent small draws, one per source. One register per source is all a classical or quantum solver
ever needs.

**Zero-variance sampler = Doob transform = optimal soft-RL policy (Proposition 3).** The best
importance-sampling measure is the Doob h-transform of the source process, equivalently the Boltzmann
policy of a KL-regularised control problem. RDPO learns it with a compact log-linear soft-value
function and a defensive mixture of weight 5%.

**Two-stage training on the right divergence (Proposition 4).**

* **Stage 1: cross-entropy (KL).** Multilevel policy search raises the elite level until the target
  level is reached.
* **Stage 2: Rényi-2 (χ²) policy gradient.** The variance of an IS estimator is a χ² divergence, not a
  KL one — RDPO therefore takes REINFORCE steps on log E_Q[(wf)²].

**Fixed budget, unbiased estimator (Proposition 5).** Training paths count against the total budget B.
Eligible training batches are recycled; a final randomised quasi-Monte Carlo stage spends the rest. The
combined estimator is exactly unbiased for *any* learned policy, with a reported 95% interval.

**Pair reduction.** For targets that depend on two countries only (joint attachment / exhaustion), the
joint law of their 100 states is the exact product of the kernels touching them: computed in ≤ 3 ms
for any portfolio size.

# 3 Results

**Benchmark (all 25 ADB cells, 40 replications, 16,384 paths including training).** RDPO is more
accurate than the best of the brief's four reference methods in **20 of 25 cells**. On the hardest
target (joint exhaustion, μ ≈ 3×10⁻⁵) the median improvement over the best reference method is **×19**,
and ranges from ×6.3 to ×171. Interval coverage is 94% mean (nominal 95%).

**Scaling (8 portfolio sizes up to 28 countries, 24 replications, exact tensor-network references,
same seeds for every method).** On pool-wide questions RDPO beats exponential twisting in **16 of 16
settings**, with the Phase 2 engine 2.3–11.9× faster per estimate than the Phase 1 code.

**Stronger baselines (same budget, same seeds).** Against three strong rare-event methods — adaptive
multilevel splitting [@cerou2007; @aubeck2001], SMC on exponential tilts [@delmoral2006], and a
diagonal normalising-flow importance sampler trained by REINFORCE on χ² [@muller2019], warm-started
from the optimal twist — **RDPO wins on every rare target by factors of {{STR_RANGE}}**. SMC on the
optimal twist is the strongest fixed baseline; RDPO's advantage over it is the gain from a learned
state-dependent policy over the best product proposal.

**Real-world validation.** The bond's assumptions were tested against USGS catalogue data, NOAA
earthquake loss records (1900–2020) and 48 GHCN-Daily weather stations. Earthquake arrivals are
Poisson (dispersion 0.66–1.24, every p ≥ 0.12). The brief's shared-earthquake share of 7.5% is far too
low: 120-year loss data give a posterior with median **48%** and 95% CI **[12%, 79%]**. Extreme-rain
sharing between neighbours is much lower than assumed: pooled posterior median **2%**, CI
**[0.6%, 7.7%]**. Propagating both posteriors through the exact bond model gives the headline finding.

**WP2 at scale: 28-country Asia–Pacific pool and 50-country ring.** RDPO reaches **0.76% / 0.79%** error
at μ ≈ 10⁻⁹ with 65,536 paths — the roadmap's WP2 success criterion (≤ 10% at μ ≤ 10⁻⁸) is met.

**ELT/YLT integration.** A converter turns catastrophe-model event or year loss tables into RDPO's
source form; events paying in any set of countries become one Poisson source with its exact
three-year capped law. On a 10,000-year YLT the standard analyst estimate of a μ = 2×10⁻⁴ pool
exceedance has 132% relative error; RDPO on the converted model gives **5.4%** at the same budget.

# 4 Quantum extension — evidence-first

The quantum stage is designed, not demonstrated: no quantum processor is available in Phase 1, so no
advantage is claimed. We specify it precisely enough to be *tested and falsified*.

**Problem and primitive.** Classical RDPO needs ρ²/ε² paths; iterative amplitude estimation needs
≈ c_Q/(ε √a) applications of a state-preparation operator A, so the wall-clock crossover is

$$s^*(\varepsilon, \rho^2) = \frac{\varepsilon}{\rho^2} \cdot \frac{c_Q \tau_A}{\tau_C \sqrt{a}}.$$

The quantum stage wins only where the *best classical* method is hard (large ρ²) or where very high
precision is required (small ε). Comparing with plain Monte Carlo instead of RDPO overstates the
advantage by up to 10¹¹.

**Algorithm 2: guarded Q-RDPO.** Classical RDPO → choose a loadable product-form measure → predict s*
from the crossover law → compile A and Q = A S₀ A† S_χ → iterative amplitude estimation with
noise-aware likelihood → accept the quantum result only if the fit passes and it agrees with the
classical pilot within 3σ, else report the classical estimate.

**Scalability.** Compiled resources fit {{QS_FITS_INLINE}}, all within 7% of every compiled count,
so register counts and gate depths grow linearly in n and |E|.

**Falsifiable criterion.** At 28 countries the quantum stage needs hardware **8×10⁴** times faster
than the stated fault-tolerant baseline, and a logical error rate near 10⁻⁷. Both are stated as
measurable targets.

**Error mitigation under miscalibration.** Noise-aware mitigation is worse than no mitigation in
21–38% of instances under control-offset miscalibration (consistent with [@errmit2026]). A
pilot-disagreement guard caps p99 error at ≤ 0.17 at a 10–22% fallback rate.

# 5 Prototype deliverable

Open-source Python package (`CL0908/adb-quantum-challenge`), **59 passing tests**, no proprietary
code or data. Runs on one CPU core. Includes:

* RDPO engine (Phase 1 + Phase 2), projected Doob sampler, exact tensor-network references.
* ELT/YLT converter; `api/estimate.py` runs as a Vercel Python function.
* Real-world checks: USGS catalogue, NOAA loss records, GHCN-Daily rainfall, three stronger
  rare-event methods, posterior-uncertainty propagation.
* Quantum stage: structured loading, bitwise weight oracle, IQAE emulation, surface-code resource
  model, NISQ mitigation study.

Full method, proofs, resource tables, Asia–Pacific pool map, ELT demonstration and
market-record comparison in the technical annex.

# 6 Work plan (12 weeks, Deep Development)

| WP | Scope | Milestone |
|---|---|---|
| 1 | GPU-batched RDPO, library API, sandbox onboarding | library v0.1, 10× throughput |
| 2 | 20–50-country pools with real hazard data; ground-motion and exposure models; gridded rainfall | scaling report, ≤ 10% RMSE at μ ≤ 10⁻⁸ in 10⁵ paths (**already achieved** at 28/50 countries) |
| 3 | Heavy-tail diagnostics, model selection, neural soft values; parameter-uncertainty propagation across all deliverables | RDPO v3 |
| 4 | Compiled quantum resource tables; calibrated sampler on sandbox hardware | compiled s* map |
| 5 | Web application, AI analyst agent, MCP tool server | public demo |
| 6 | Validation, documentation, business case, pilot engagement | final deliverables |

Done ahead of schedule: WP1 engineering (table-driven engine); WP2 first results (28 countries); WP3
posterior-uncertainty propagation; WP5 ELT/YLT converter.

# 7 Commercialisation

* **Who buys.** Development banks and sovereign risk pools (ADB, World Bank, CCRIF, ARC); reinsurers;
  ILS funds and cat-bond investors; brokers and catastrophe-model vendors (via Oasis plug-in);
  parametric MGAs. B2B/B2G.
* **How they use it today.** Pricing and stress testing on event loss tables and year loss tables
  from vendor models (Verisk, Moody's RMS). Our ELT/YLT converter plugs in there.
* **Why they would try it.** On the public ADB bond data, a joint-tail finding that no vendor
  pipeline would surface (2.8× posterior median); stress sweeps that would take hours in Oasis in 5
  seconds; pair questions solved exactly.
* **Market.** US$2.1bn software TAM (cat-modelling & exposure management); bottom-up US$30m SAM; target
  US$3.5m ARR by year 5 (28 institutions).

# 8 Team

**Tan Chun Loong** (sole member). BAppSc Physics + Quantum Technology, NTU Singapore (2024–2028,
CGPA 4.8/5.0). Incoming Phortify Student Research Fellow, ICFO Barcelona (hosted by Prof. Antonio
Acín's Quantum Information Theory group; one of three worldwide for this cycle). URECA Research
Fellow with Prof. Mile Gu on relative-entropy foundations of model-based RL; Quantum Security Intern
at Fraunhofer Singapore; prior work on tilted path-integral control, Clifford+T compilation (MIT
iQuHACK), quantum reservoir computing (QAI Ventures GenQ). AI pair-programming by Claude Code; the
author directed all research and reviewed all results. A short CV is attached as an annex.

# 9 Platform

No platform access needed for Phase 1 (all results on one laptop CPU core). In Deep Development the
QAI Ventures sandbox would serve GPU-batched sweeps, compiled loading circuits, and emulator or
hardware amplitude-estimation runs.

::: {.takeaway}
Three deliverables for Phase 2: the pilot with ADB disaster-risk-financing, the first set of real
vendor-model backtests, and a published AI-for-Quantum paper. We would open-source everything that
is not a client's data.
:::

# References {.newpage}

::: {#refs}
:::
