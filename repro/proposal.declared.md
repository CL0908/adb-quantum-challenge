---
title: "Information-Theoretic Reinforcement Learning for Joint Catastrophe Tails"
subtitle: "Quantum Enhanced Rényi-Doob Policy Optimization"
author: "Tan Chun Loong"
affiliation: "Nanyang Technological University, Singapore"
date: "November 2026"
lang: en
---

::: {.frontmatter}

::: {.headline}
## Headline findings

1. **The joint tail of the bond is likely underestimated.** We replaced the two dependence assumptions
   with public data (120 years of earthquake loss records and daily rainfall at 48 weather
   stations) and propagated the posterior uncertainty on both shares through the exact two-country
   model. The posterior for the chance that the pool pays more than US$80m in a term has median
   **3.76% (3.12× the brief's 1.20%)** and 95% credible interval
   **[2.08%, 5.27%]** — the whole interval sits above the brief. Earthquakes hit both
   countries far more often than assumed (about 48% of events, not 7.5%); extreme rain far less
   often (about 2%, not 25%).
2. **Joint tails that simulation cannot see, at about 1% error.** In a 28-country Asia–Pacific pool
   calibrated from public data, RDPO estimates a 9.5×10⁻⁹-probability pool loss to **0.8% relative error**
   with 65,536 simulated terms (21 s on one core). Plain Monte Carlo sees no such event.
3. **Stress tests in seconds.** All 25 scenario × question cells of the bond are re-estimated in
   **5 seconds** on one core. A policy carried over from the previous run cuts the error at small budgets by
   ×1.8–×83.8.

*Indicative findings. Public data serve as proxies: earthquake epicentres and reported losses for payout
triggers, and station rainfall for the rain index. They are not a re-pricing of the actual bond, whose trigger
definitions and modelling agent are not public. Methods and caveats are in §6.6–6.7.*
:::

::: {.abstract}
## Abstract

Pooled catastrophe-risk instruments are priced and stress-tested on their *joint tails*, such as the
probability that two countries exhaust their cover in the same term. These probabilities are rare
(10⁻³ to 10⁻¹³), poorly constrained by historical data, and must be recomputed under many dependence
scenarios. Plain Monte Carlo returns no hit at practical budgets, and exact enumeration grows as 10ⁿ
with the number of countries.

We cast joint-loss estimation as a reinforcement-learning problem. A scenario is a sequence of draws
from independent Poisson *sources* of loss. The zero-variance sampler is the Doob h-transform of this
process, equivalently the optimal policy of a KL-regularised control problem. **Rényi-Doob Policy
Optimization (RDPO)** learns this policy with a compact soft-value function. It is trained first by
cross-entropy (KL) and then by a Rényi-2 (χ²) policy gradient, since χ² is the divergence that sets the
estimator's variance. Training is charged to a fixed path budget, and the estimator is exactly unbiased.

With training included, RDPO attains 1.4% median relative error on the ADB joint-exhaustion
probability at 16,384 paths, a 6.3–171-fold improvement over the best reference method. It keeps
errors of 0.1%–4.2% as the target probability falls to 10⁻¹³ in a 28-country portfolio. The same
learned change of measure loads *exactly* into an amplitude-estimation circuit. On pools calibrated from
public hazard and loss data, RDPO meets the work plan's accuracy target at 28 countries, and the data
suggest that the bond's joint tail is underestimated (Headline findings). We assess this quantum
stage with a falsifiable wall-clock criterion and a calibration trust guard, and we claim no quantum
advantage in Phase 1.
:::

::: {.glance}
::: {.tiles .four}
::: {.tile}
[2.8×]{.big}
pool exceedance with dependence calibrated to public data, relative to the challenge's base case
:::
::: {.tile}
[0.8%]{.big}
relative error at μ ≈ 10⁻⁸ in a 28-country data-calibrated pool (65,536 paths)
:::
::: {.tile}
[×6.3–171]{.big}
lower error than the best reference method on the bond's joint exhaustion
:::
::: {.tile}
[8.2×10⁴]{.big}
hardware speed-up quantum would need to beat RDPO: a falsifiable target
:::
:::
:::

:::

# 1 Introduction {.newpage}

## 1.1 Why now

* **Pooled cover is growing, and its price depends on joint tails.** The Asian Development Bank's
  US$160 million Disaster Relief Bond for the Kyrgyz Republic and Tajikistan pools earthquake and
  precipitation risk across a shared fault and glacier system. Whether pooling lowers cost, and whether
  investors are paid fairly, depends on how often both countries lose *together*.
* **The losses are large and recurrent.** Earthquakes in Central Asia are reported to cause on the order
  of half a billion US dollars of damage per year, with the Kyrgyz Republic among the most affected
  (CABAR.asia, cited in the challenge brief).
* **The data cannot pin down the joint tail.** Only a couple of decades of high-quality disaster data
  exist, so the dependence between countries must be stress-tested across scenarios rather than
  estimated. Every scenario multiplies the number of rare-event computations.
* **Current tools fail exactly where it matters.** At the 3×10⁻⁵ joint-exhaustion probability of the
  benchmark, plain Monte Carlo with 16,384 paths returns *zero* in about 60% of runs. Exact enumeration
  is instant for two countries but grows as 10ⁿ, and regional pools cover 5–30 countries.

![**Figure [fig:bvd].** What public data say about the bond's two dependence assumptions (details in §6.6). Earthquakes hit both countries far more often than the brief assumes, and extreme rain far less often. Together, the chance of a large pool payout rises 2.8-fold. Indicative findings, not a re-pricing of the bond.](../art/brief_vs_data.svg){.wide}

::: {.takeaway}
The bottleneck is accuracy per query in the joint tail, and how it degrades as portfolios grow.
:::

## 1.2 The idea in three cards

![**Figure [fig:cards].** The three ideas behind RDPO. (01) The optimal sampler moves probability
towards the losses and corrects by importance weights. (02) A catastrophe portfolio decomposes into
independent Poisson shocks, so a scenario is a short sequence of small draws. (03) Estimation error is
dominated by a few rare paths, so the sampler must be trained against χ², not only against
KL.](../art/concept_cards.svg){.wide}

## 1.3 System overview

![**Figure [fig:arch].** System architecture. Calibrated hazard parameters, contract terms, a
dependence graph and stress scenarios define an exact source model. The RDPO engine learns a
soft-value sampling policy inside a fixed path budget (top loop) and turns it into an unbiased estimate
with a confidence interval (bottom row). A validation layer checks the method against exact references.
The same learned measure can feed a future quantum estimation stage (dashed).](../art/architecture.svg){.wide}

## 1.4 Contributions

1. **Exact source-level formulation.** The capped multi-year payout process is an exact product of
   commuting Poisson kernels with finite increment laws (Propositions 1–2), so a scenario is a sequence
   of K = 2n + 2|E| independent draws.
2. **RDPO.** The zero-variance sampler is the optimal policy of a soft control problem (Proposition 3).
   We learn it with KL then χ² objectives (Proposition 4) inside a fixed budget, with an exactly unbiased
   estimator (Proposition 5).
3. **Rigorous evaluation.** All 25 benchmark cells, scaling to 28 countries against exact
   tensor-network references, and an ablation. Training is always charged to the budget, and interval
   coverage is reported.
4. **A quantum stage held to an explicit evidence standard.** Exact structured loading of the learned
   measure, a falsifiable wall-clock criterion against our own best classical method, and a calibration
   trust guard for error mitigation.

# 2 Data and Problem Formulation {.newpage}

::: {.opener}
![](../art/cat_peek.svg){.mascot}

This section states what data the prototype uses, then builds the estimation problem step by step,
from the contract terms to an exact source-level representation for any number of countries.
:::

## 2.1 Data and model inputs

The model uses **no historical loss records and no proprietary data**. Its inputs are the calibrated
parameters published in the challenge brief; a public earthquake catalogue is used to *check* them (§6.6):

| Input | Values | Source | Used for |
|---|---|---|---|
| Annual EQ exceedance probabilities (≥25/50/75/100% of sub-limit) | KG: 3.883/2.331/1.790/1.184%; TJ: 3.925/1.743/1.332/1.043% | Brief, Table 1 | compound-Poisson rates (§2.4) |
| Conditional trigger probabilities | Pr(TJ \| KG) = 7.5%, Pr(KG \| TJ) = 7.4% | Brief | shared-event rate (§2.5) |
| Precipitation trigger probability; shared share | 2% per year; 25% of the smaller rate | Brief | rain sources (§2.5) |
| Contract terms | US$80m principal = 70m EQ + 10m rain; 3-year term; T = 80m | Brief | payoffs (§2.3) |
| Stress scenarios | base, independent, common shock (m = 2), rain stress (×2), rain cover (60/20m) | Brief, Table 4 | E1 benchmark |
| Reference values | five targets × five scenarios | Brief, Table 5 | validation (reproduced exactly) |
| n-country extension | KG/TJ parameters alternated on chain, ring, star and complete graphs | constructed (§2.8) | scaling study |
| Earthquake catalogue (real data) | USGS significant earthquakes, M ≥ 5.5, 1965–2016; country boundaries | public mirrors (§6.6) | check of Poisson arrivals, trigger size, shared events |

In Phase 2 the same pipeline will take source laws estimated from public hazard data (for example
earthquake catalogues, regional seismic hazard models and precipitation reanalyses) in place of the
brief's parameters. The method only needs one finite increment law per source.

![**Figure [fig:setup].** The benchmark portfolio. Each country's bond pays in earthquake quarter steps
and a single precipitation trigger; shared events hit both countries at once. All five targets are
expectations of functions of the two term losses.](../art/problem_setup.svg){.wide}

## 2.2 Notation in brief

* Countries $i = 1, \dots, n$; term $\tau = 3$ years; dependence graph $\mathcal G = (V, \mathcal E)$.
* Terminal state of country $i$: EQ quarter steps $a_i \in \{0, \dots, 4\}$ and rain trigger $r_i \in \{0,1\}$.
* Sources $k = 1, \dots, K$ with finite increment laws $p_k$; target functional $f$; target value $\mu = \mathbb E_p[f]$.
* Sampling policy $q_\theta$; importance weight $w = \mathrm dP/\mathrm dQ_\theta$; total path budget $B$.

## 2.3 Contract terms and payout functions

Country $i$ holds a bond with principal $B_i$ split into an earthquake sub-limit $E_i$ and a
precipitation sub-limit $P_i$ (benchmark: $E_i = 70$m, $P_i = 10$m, $B_i = 80$m). Earthquake payouts accrue in
quarter steps and accumulate up to the sub-limit; precipitation pays $P_i$ at most once. The term loss is

$$L_i = E_i \frac{a_i}{4} + P_i\, r_i, \qquad L = \sum_{i=1}^n L_i .$$

## 2.4 Annual earthquake model

From the exceedance probabilities $t_1 > t_2 > t_3 > t_4$, the point probabilities of paying 0–3 quarter
steps are $p_0 = 1 - t_1$, $p_k = t_k - t_{k+1}$ ($k = 1, 2$) and $p_3 = t_3 - t_4$. A compound Poisson model with
total rate $\Lambda$ and size rates $\lambda_k$ matches them when

$$\Lambda = -\ln p_0,\quad \lambda_1 = \frac{p_1}{p_0},\quad \lambda_2 = \frac{p_2}{p_0} - \frac{\lambda_1^2}{2},\quad \lambda_3 = \frac{p_3}{p_0} - \lambda_1\lambda_2 - \frac{\lambda_1^3}{6},\quad \lambda_4 = \Lambda - \lambda_1-\lambda_2-\lambda_3,$$

with per-event size probabilities $w_{i,k} = \lambda_{i,k}/\Lambda_i$.

## 2.5 Dependence through shared events

* **Shared earthquakes.** With conditional trigger probabilities $\rho_{2|1} = 0.075$, $\rho_{1|2} = 0.074$ and
  scenario multiplier $m$,
  $$c = m \cdot \tfrac12\left(\rho_{2|1}\Lambda_1 + \rho_{1|2}\Lambda_2\right).$$
  Solo events arrive at $\Lambda_i - c$, so each country keeps its total rate; shared events draw their
  sizes independently in each country.
* **Generator.** The capped levels $(a_1, a_2)$ form a 25-state Markov chain with generator $G$:
  $$G_{(a,b),(a \oplus k,\, b)} \mathrel{+}= (\Lambda_1 - c)\, w_{1,k}, \qquad G_{(a,b),(a,\, b \oplus j)} \mathrel{+}= (\Lambda_2 - c)\, w_{2,j}, \qquad G_{(a,b),(a\oplus k,\, b \oplus j)} \mathrel{+}= c\, w_{1,k} w_{2,j},$$
  where $x \oplus k = \min(x + k, 4)$ is capped addition.
* **Rain.** Each country triggers with annual probability $q = 2\%$ (rate $\mu = -\ln(1-q)$), of which a
  shared component $c_r = s_r \min(\mu_1, \mu_2)$ with $s_r = 0.25$ triggers both.

## 2.6 Aggregation over the term

::: {.proposition}
**Proposition 1 (term law).** The law of the capped term EQ levels is $e^{\tau G}_{(0,0),\,\cdot}$, and
the term rain law is the annual one with rates multiplied by $\tau$.

*Proof.* For non-negative integers, $\min(\min(x,4) + y, 4) = \min(x + y, 4)$, so capping each year and
then the running total equals capping the total event count over the term. Independent Poisson streams
over three unit intervals superpose to a Poisson stream on $[0,\tau]$, whose capped levels follow $G$.
The rain trigger is the indicator of at least one event. $\square$
:::

Numerically, $e^{3G}$ reproduces the brief's three-fold capped convolution to within $3\times10^{-16}$ in
all five scenarios.

## 2.7 Estimation targets

| Target | Definition | Typical value |
|---|---|---|
| expected principal loss | $\mathbb{E}[L]$ | US$10m (not rare) |
| joint attachment | $\Pr(L_1 > 0,\ L_2 > 0)$ | ≈ 4% |
| portfolio exceedance | $\Pr(L > T)$ | ≈ 1% |
| portfolio stop-loss | $\mathbb{E}[(L - T)^+]$ | tail-driven |
| joint exhaustion | $\Pr(L_1 = B_1,\ L_2 = B_2)$ | **≈ 3×10⁻⁵ (rare)** |

## 2.8 Extension to n countries

* An edge $\{i,j\}$ of the dependence graph means that $i$ and $j$ can be hit by common events (a shared
  fault system, river basin or weather regime).
* Each edge carries a shared EQ source (rate $c_{ij}$, the rule of §2.5) and a shared rain source
  ($c^r_{ij} = s_r\min(\mu_i,\mu_j)/d_{\max}$); solo rates are reduced so that total rates are preserved.
* For $n = 2$ with one edge this *is* the benchmark: all 25 reference values are recovered to
  $4\times10^{-14}$.

![**Figure [fig:topo].** Dependence graphs used in the scaling study. Nodes are countries; edges are
shared-event sources. The number of edges crossing a cut (the cut-width) governs the cost of exact
classical tensor-network methods (§6.4).](../art/topologies.svg){.wide}

## 2.9 Source decomposition

::: {.proposition}
**Proposition 2 (source decomposition).** Let $\mathcal{S}$ be the $K = 2n + 2|\mathcal{E}|$ independent Poisson
sources (solo EQ and solo rain per country; shared EQ and shared rain per edge), with generators $G_k$.
Then the $G_k$ commute and $P^{(\tau)} = \delta_{\mathbf 0} \prod_{k} e^{\tau G_k}$. The terminal state is
$\mathbf{0} \oplus X_1 \oplus \cdots \oplus X_K$ with independent increments $X_k \sim p_k = (e^{\tau G_k})_{\mathbf 0, \cdot}$,
each supported on at most 25 values.

*Proof.* Capped addition is a commutative monoid action, so the kernels of different sources commute,
and independence gives the product. By the identity in Proposition 1, a source's term effect is a capped
increment in $\{0..4\}$, $\{0..4\}^2$ or $\{0,1\}$. $\square$
:::

![**Figure [fig:sources].** Source decomposition of a three-country chain. A scenario is a sequence of
independent draws, one per source; each draw changes only the one or two countries it
touches.](../art/sources.svg){.wide}

## 2.10 Why the problem is hard

* Monte Carlo relative error: $\mathrm{RE}_{\text{MC}} = \sqrt{(1-\mu)/(\mu N)}$, and the probability of no hit
  at all is $(1-\mu)^N$.
* For joint exhaustion ($\mu \approx 3.2\times10^{-5}$) at $N = 16{,}384$: $\mathrm{RE}_{\text{MC}} \approx 1.4$, and about
  60% of runs return exactly zero.
* Exact enumeration takes 0.2 ms for two countries but scales as $10^n$.

::: {.takeaway}
A portfolio scenario is a short sequence of independent, local draws. That is the structure a learned
sampler, and a quantum loader, can exploit.
:::

# 3 Related Work

::: {.opener}
![](../art/cat_books.svg){.mascot}

Where RDPO sits: rare-event simulation, control as inference, and quantum methods for risk, and the evidence standard we hold ourselves to.
:::


* **Catastrophe modelling practice.** Vendor platforms and the open-source Oasis Loss Modelling
  Framework chain hazard, vulnerability and financial modules. They estimate portfolio metrics by
  aggregating simulated event or year loss tables, usually by plain Monte Carlo, and they express
  dependence through shared events or copulas [@joe2014; @embrechts1997].
* **Rare-event simulation.** Stratification, randomised quasi-Monte Carlo and importance sampling are
  classical [@glasserman2003; @owen1997]. The cross-entropy method fits proposals by multilevel KL
  minimisation [@rubinstein2004]. Variance-minimising adaptive importance sampling [@arouna2004] and χ-
  and Rényi-divergence variational inference [@dieng2017; @li2016] motivate our second training stage.
* **Control and RL.** The link between zero-variance measures, Doob transforms and KL control goes back
  to linearly solvable MDPs and path-integral control [@todorov2009; @kappen2005].
* **Quantum methods for risk.** Amplitude estimation offers a quadratic query speed-up over Monte
  Carlo [@brassard2002; @montanaro2015], with risk applications [@woerner2019] and a correlated
  credit-default model [@egger2021] analogous to shared-event catastrophe risk. Practical variants avoid
  phase estimation [@suzuki2020; @grinko2021], and end-to-end studies show that state preparation
  dominates cost [@chakrabarti2021; @herman2023]. Generative models load distributions on near-term
  devices [@benedetti2019; @zoufal2019; @amin2018; @zhu2022].
* **Evidence standards.** A recent position paper argues that quantum machine learning on classical
  data has not earned priority, because its advantage claims rest on strong input assumptions and
  trainability and noise barriers, and lack wall-clock wins; it calls for falsifiable criteria and for
  AI-for-Quantum work [@guo2026stopqml]. Error mitigation can itself increase error under
  miscalibration [@errmit2026]. Section 7 is written to these standards.

**Positioning.** We treat the *change of measure* as the learnable object, train it against the
divergence that governs variance, and carry it unchanged into a quantum loader whose cost follows the
dependence graph.

# 4 Method: Rényi-Doob Policy Optimization {.newpage}

::: {.opener}
![](../art/cat_sitting.svg){.mascot}

One idea: the best possible sampling distribution is a reinforcement-learning policy, and it should be
trained against the quantity that actually determines estimation error.
:::

## 4.1 Importance sampling and the zero-variance measure

For any sampling law $Q$ with $Q \gg P$ on $\{f > 0\}$, the estimator $\hat\mu = \frac1N\sum_{j} w(X_j) f(X_j)$
with $w = \mathrm dP/\mathrm dQ$ is unbiased, and

$$\operatorname{Var}_Q(\hat\mu) = \frac{1}{N}\left(\mathbb{E}_Q[w^2 f^2] - \mu^2\right).$$

For $f \ge 0$ the variance vanishes under $\mathrm dQ^*/\mathrm dP = f/\mu$. In the source representation, with
$s_k$ the state after sources $1..k-1$, define the value function

$$V_k(s) = \mathbb{E}\left[f(S_K) \mid S_k = s\right] = \sum_x p_k(x)\, V_{k+1}(s \oplus_k x), \qquad V_{K+1} = f.$$

::: {.proposition}
**Proposition 3 (Doob transform).** $Q^*$ is the Markov law with transitions
$q^*_k(x \mid s) = p_k(x)\, V_{k+1}(s \oplus_k x) / V_k(s)$, and every path sampled from $Q^*$ has $w \cdot f = \mu$.

*Proof.* The ratios telescope: $\prod_k q^*_k / p_k = V_{K+1}(S_{K+1}) / V_1(\mathbf 0) = f/\mu$. $\square$
:::

For two countries $V$ has only 100 states and can be tabulated exactly. This zero-variance backward
induction is our strongest classical baseline, and it is exact. At scale $V$ has $10^n$ entries and must be
*learned*.

## 4.2 The reinforcement-learning loop

Taking logarithms of the recursion gives the **soft Bellman equation**

$$\log V_k(s) = \log \sum_x p_k(x)\, \exp\!\big(\log V_{k+1}(s \oplus_k x)\big)$$

of a KL-regularised control problem with reference dynamics $p$ and terminal reward $\log f$
[@todorov2009; @kappen2005]. The zero-variance sampler is its Boltzmann policy. Figure [fig:rl] states the
resulting MDP.

![**Figure [fig:rl].** RDPO as one reinforcement-learning loop. The agent chooses the increment of
each Poisson source in turn; the environment applies it with capping. After K steps the payoff and
importance weight are observed. The two learners update the same soft value
function.](../art/rl_loop.svg){.wide}

## 4.3 Policy class

* **Soft value.** $\log \tilde V_\theta(s) = \theta^\top \Phi(s)$; policy
  $$\tilde q_{\theta,k}(x \mid s) \propto p_k(x)\, \exp\!\left(\theta^\top\left[\Phi(s \oplus_k x) - \Phi(s)\right]\right), \qquad q_{\theta,k} = (1-\alpha)\,\tilde q_{\theta,k} + \alpha\, p_k .$$
  The defensive mixture ($\alpha = 0.05$) bounds every per-step likelihood ratio by $1/\alpha$.
* **Features.** Country thresholds $\psi(a_i, r_i) = [\mathbb 1\{a_i \ge 1\}, \dots, \mathbb 1\{a_i \ge 4\}, r_i]$ ($5n$);
  dependence edges $\chi = [\mathbb 1\{L_i>0\}\mathbb 1\{L_j>0\}, \mathbb 1\{a_i = a_j = 4\}, r_i r_j]$ ($3|\mathcal E|$);
  target progress, e.g. $[\min(L,T)/T, \mathbb 1\{L>T\}, (L-T)^+/(L_{\max}-T)]$ (1–3).
* **Locality.** A source touches at most two countries, so $\Phi(s\oplus_k x) - \Phi(s)$ lives on a fixed small
  coordinate set and the normaliser sums over at most 25 values. One path costs $O(n + |\mathcal E|)$ on
  bounded-degree graphs and never touches the $10^n$ joint state space.
* **Target relevance.** If $f$ depends only on a subset $\mathcal R$ of countries, then for every source
  touching no country in $\mathcal R$ the exact Doob policy is $q^*_k = p_k$ (Proposition 3). RDPO freezes such
  sources at $p_k$, with likelihood ratio exactly one. For a pair target inside a 28-country portfolio this
  reduced the error by more than two orders of magnitude in development runs (§6.3).

## 4.4 Training objectives: from KL to Rényi-2

* **Stage 1: cross-entropy policy search (KL).** Minimising $D_{\mathrm{KL}}(Q^*\,\|\,Q_\theta)$ is a weighted
  maximum-likelihood problem, concave in $\theta$ for this log-linear policy. For rare targets, the elite
  level $\gamma = \min(\gamma^*, \text{quantile}_{1-\rho}(\text{score}))$ is raised until the target level is reached
  [@rubinstein2004].
* **Stage 2: Rényi-2 policy gradient (χ²).** Estimation error is governed by a different divergence:

::: {.proposition}
**Proposition 4 (variance is a Rényi-2 divergence).** For $f \ge 0$ and $g = w f$,
$$\frac{N\operatorname{Var}_Q(\hat\mu)}{\mu^2} = \chi^2(Q^*\,\|\,Q) = e^{D_2(Q^*\|Q)} - 1, \qquad D_{\mathrm{KL}}(Q^*\|Q) \le D_2(Q^*\|Q).$$
The bound runs one way only: a policy with small KL can still have very large variance.
:::

  RDPO therefore takes REINFORCE steps on $\log M_2$, $M_2 = \mathbb E_{Q_\theta}[g^2]$, with a constant baseline:
  $$\nabla_\theta \log M_2 = -\,\mathbb{E}_{Q_\theta}\!\left[\left(\frac{g^2}{M_2} - 1\right) \nabla_\theta \log q_\theta(\text{path})\right],$$
  keeping the iterate with the lowest *estimated* χ² and stopping early. No reference value is used in
  training. In a preliminary benchmark study this stage removed every heavy-tailed cross-entropy failure
  (2 of 15 fits) and reduced the exact per-path relative standard deviation of a failed fit from 14.5 to
  0.60. Its role is to repair fits whose KL is small but whose χ² is not.
* **Warm start.** For threshold targets the policy starts at the optimal exponential twist of the total
  loss, $\theta_h \leftarrow \eta^\star T$ with $\eta^\star = \arg\max_\eta\, \eta T - \Lambda(\eta)$ (the large-deviation Doob
  transform).

## 4.5 Algorithm

::: {.algorithm}
**Algorithm 1: RDPO with a fixed total budget B**

1. Initialise $\theta$ (warm start where applicable); batch size $b = \max(500, B/16)$; training cap $B/2$.
2. **Stage 1:** draw $b$ paths; set the elite level $\gamma$; refit $\theta$ by concave weighted log-likelihood. Repeat until the target level is reached, plus one iteration.
3. **Stage 2:** draw $b$ paths; estimate $\hat\chi^2$; take one Adam step on $\log M_2$. Repeat until 3 steps pass without improvement or the cap is reached; keep the best $\theta$.
4. **Stage 3:** spend the rest of the budget with the selected policy, driven by 8 scrambled Sobol sequences.
5. **Estimate:** pool every batch whose policy had reached the target level before it was drawn (weights ∝ batch size); report the 95% interval.
:::

![**Figure [fig:pipeline].** RDPO stages and an illustrative split of the fixed path budget.](../art/pipeline.svg){.wide}

## 4.6 Unbiasedness and uncertainty

::: {.proposition}
**Proposition 5 (recycled estimator is unbiased).** Let batches be drawn with policies $\theta_b$ and sizes
$n_b$ measurable with respect to the history $\mathcal F_{b-1}$, and let $b_0$ be the first batch whose policy had
reached the target level before it was drawn. Then $\hat\mu = \sum_{b \ge b_0} \frac{n_b}{N_e}\,\hat\mu_b$ with
$N_e = B - \sum_{b<b_0} n_b$ satisfies $\mathbb{E}[\hat\mu] = \mu$.

*Proof.* All batches after $b_0$ are pooled, so $N_e$ and every weight $n_b/N_e$ are known before batch $b$ is
drawn. Since $\mathbb{E}[\hat\mu_b \mid \mathcal F_{b-1}] = \mu$, the tower property gives $\mathbb{E}[\hat\mu] = \mu$.
Scrambled Sobol points are marginally uniform, so RQMC blocks are unbiased too. $\square$
:::

The standard error combines within-batch variances (i.i.d. batches) with between-scramble variances
(RQMC blocks). We report the empirical coverage of the resulting 95% intervals.

::: {.takeaway}
RDPO learns the zero-variance sampler as a soft-RL policy: KL to reach the tail, χ² to control the
error, inside one budget, without bias.
:::

# 5 Experimental Protocol

::: {.opener}
![](../art/cat_laptop.svg){.mascot}

Every method gets the same simulation budget, RDPO's training counts against it, and every answer is checked against an exact reference.
:::


* **Budget.** Every method receives the same total number of sampled term scenarios $B$; RDPO's training
  paths count against $B$.
* **Metrics.** Relative RMSE against the exact value over independent replications (40 for the
  benchmark, 6–20 at scale), relative bias, and the empirical coverage of nominal 95% intervals.
* **References.** Benchmark: exact enumeration, which reproduces the brief's Table 5. Scaling: exact
  tensor networks in a tail-preserving tilted gauge (§6.4), validated against dense enumeration for $n \le 6$.
* **Baselines.** Three groups. (i) The challenge's four reference methods: plain Monte Carlo, scrambled
  Sobol (8 scrambles), stratification by first payout year, and the brief's importance sampling. (ii) At
  scale we add classical exponential twisting [@glasserman2003]. (iii) Three stronger rare-event methods,
  compared on the ADB benchmark and the ring-16 scaling case, at the same budget and seeds as RDPO
  (§6.5.5): adaptive multilevel splitting / subset simulation [@cerou2007; @aubeck2001],
  SMC on exponential tilts [@delmoral2006], and a diagonal normalising-flow importance sampler trained by
  REINFORCE on $\chi^2$ (product-of-categoricals, warm-started from the optimal twist) [@muller2019].
* **Compute.** One CPU core per run (Intel Xeon 2.1 GHz); NumPy/SciPy; fixed seeds in every result file.

# 6 Results {.newpage}

::: {.opener}
![](../art/cat_chart.svg){.mascot}

How accurate, how fast, and where it fails: the benchmark, scaling to 50 countries, the Phase 2 engine, real-world data and catastrophe-model output.
:::


## 6.1 Benchmark: five scenarios × five targets

![**Figure [fig:e1].** Benchmark at a total budget of 16,384 paths, training included. Left: RDPO's
relative RMSE per cell. Right: improvement over the best of the four reference methods in the same cell
(>1: RDPO is more accurate).](../../figures/paper/e1_heatmap.png){.wide}

**Table [tab:e1].** RDPO relative RMSE at B = 16,384 paths (training included) and, in parentheses, the improvement factor over the best of the four reference methods in the same cell. Bold: RDPO more accurate.

| Target | Base | Independent | Common shock | Rain stress | Rain cover |
|---|---|---|---|---|---|
| Expected loss | 0.38% (×0.6) | 0.38% (×0.5) | 0.42% (×0.5) | 0.41% (×0.5) | 0.45% (×0.5) |
| Joint attachment | **0.46% (×3.9)** | **0.46% (×5.3)** | **0.48% (×3.2)** | **0.46% (×3.0)** | **0.53% (×3.4)** |
| Pr(L > 80m) | **0.33% (×8.0)** | **0.27% (×11)** | **0.42% (×5.8)** | **0.42% (×5.9)** | **0.42% (×7.9)** |
| Stop-loss | **0.31% (×9.8)** | **0.31% (×10)** | **0.50% (×5.4)** | **0.37% (×8.0)** | **0.35% (×10)** |
| Joint exhaustion | **1.39% (×19)** | **0.35% (×171)** | **1.77% (×15)** | **3.21% (×6.3)** | **0.37% (×73)** |

At a total budget of 16,384 paths, RDPO is more accurate than the best reference method in 20 of the 25 scenario–target cells. The gain grows with the rarity of the target. The median improvement factor is 19 for joint exhaustion, 9.8 for the stop-loss, 7.9 for pool exceedance and 3.4 for joint attachment. For the expected loss, a smooth and non-rare target, scrambled Sobol sampling remains the most accurate method (median factor 0.5), and we recommend it for that quantity. Because the training cost is charged to the budget, the advantage on joint exhaustion depends on the budget: the median factor is 3.5 at 4,096 paths, where the cross-entropy stage consumes most of the budget, and 72 at 65,536 paths (Figure [fig:e1budget]). Across all 75 budget–scenario–target settings the nominal 95% intervals cover the exact value in 95% of replications (range 78%–100%).

![**Figure [fig:e1budget].** Relative RMSE versus total budget in the base scenario, with RDPO's
training included. Grey dashes show the $N^{-1/2}$ slope.](../../figures/paper/e1_budget.png){.wide}

## 6.2 Scaling to large portfolios

![**Figure [fig:e2].** Scaling at a total budget of 65,536 paths. (a) Pool exceedance $\Pr(L > T_n)$ on a
ring: the target falls from $10^{-2}$ to $10^{-13}$. (b) Joint exhaustion of one country pair *inside* the
portfolio. Monte Carlo is shown by its theoretical error, since empirical runs return
zero.](../../figures/paper/e2_scaling.png){.wide}

**Table [tab:e2].** Scaling results at a total budget of 65,536 paths (relative RMSE; training included). MC entries are theoretical relative errors at the same budget.

| Graph | n | μ exceedance | RDPO | Exp. twisting | MC (theory) | RDPO stop-loss | Exp. twisting stop-loss | μ pair exhaustion | RDPO | MC (theory) |
|---|---|---|---|---|---|---|---|---|---|---|
| ring | 2 | 1.2×10⁻² | **0.14%** | 1.10% | 4% | **0.14%** | 0.90% | 3.2×10⁻⁵ | **0.34%** | 69% |
| ring | 8 | 3.9×10⁻⁵ | **0.30%** | 2.18% | 63% | **0.43%** | 1.25% | 1.9×10⁻⁵ | **1.00%** | 90% |
| ring | 16 | 1.5×10⁻⁸ | **0.57%** | 2.91% | 32× | **0.63%** | 2.92% | 1.9×10⁻⁵ | **0.88%** | 90% |
| ring | 20 | 2.9×10⁻¹⁰ | **0.89%** | 2.96% | 229× | **0.57%** | 2.71% | 1.9×10⁻⁵ | **0.56%** | 90% |
| ring | 28 | 1.3×10⁻¹³ | **4.23%** | 5.55% | 10,743× | **5.89%** | 3.48% | 1.9×10⁻⁵ | **0.41%** | 90% |
| complete | 4 | 2.1×10⁻³ | **0.29%** | 1.05% | 9% | **0.28%** | 0.89% | 1.5×10⁻⁵ | **0.42%** | 102% |
| complete | 6 | 6.3×10⁻⁴ | **0.68%** | 1.44% | 16% | **0.48%** | 1.03% | 1.1×10⁻⁵ | **1.14%** | 115% |
| complete | 7 | 2.9×10⁻⁴ | **0.47%** | 1.71% | 23% | **0.59%** | 1.55% | 1.1×10⁻⁵ | **3.35%** | 120% |

As countries are added to the ring, the exceedance probability falls from 1.2e-02 to 1.3e-13, while RDPO's relative error stays within 0.1%–4.2%. At the same budget, plain Monte Carlo would have a relative error of about 10,743 (i.e. 10,743 times the value itself) at 28 countries; it would need on the order of 10¹⁵ paths for 10% accuracy. Against classical exponential twisting, the strongest simple rare-event method for threshold targets, RDPO is more accurate in 15 of 16 settings (median factor 3.5). The exception is stop-loss on the ring with n = 28 (5.9% vs 3.5%): at the largest size, the one-parameter twist is already close to optimal for a smooth loss threshold, and the 5n + 3|E| + 3 parameters of RDPO are not fully trained within the budget. For the joint exhaustion of a single country pair inside the portfolio, a target that depends on two specific countries among many, RDPO reaches 0.3% (n = 2), 1.0% (n = 8), 0.9% (n = 16), 0.6% (n = 20), 0.4% (n = 28) on the ring. Interval coverage across the scaling study averages 93%, but it falls to 80% for pair exhaustion in larger portfolios, with only 6–10 replications; §6.5 re-examines coverage with 24. Replications are fewer at the largest sizes (6–20).

## 6.3 Ablation study

![**Figure [fig:e3].** Ablation at a total budget of 16,384 paths. Each bar removes one component of
RDPO.](../../figures/paper/e3_ablation.png){.wide}

**Table [tab:e3].** Ablation: relative RMSE at B = 16,384 paths (40 replications). Ratios to full RDPO in parentheses.

| Configuration | ADB base, joint exhaustion | ADB base, Pr(L>80m) | ring n=16, Pr(L>T) | ring n=8, joint exhaustion (0,1) |
|---|---|---|---|---|
| full RDPO | 1.85% | 0.33% | 4.15% | 10.25% |
| no edge features | 2.34% (×1.3) | 0.34% (×1.0) | 2.20% (×0.5) | 9.37% (×0.9) |
| no twist warm start | 1.85% (×1.0) | 0.29% (×0.9) | 154.64% (×37.3) | 10.25% (×1.0) |
| no cross-entropy stage | 148.59% (×80.5) | 1.25% (×3.8) | 2.36% (×0.6) | 143.98% (×14.1) |
| no Rényi-2 (χ²) stage | 1.79% (×1.0) | 0.35% (×1.1) | 5.07% (×1.2) | 5.99% (×0.6) |
| no sample recycling | 1.58% (×0.9) | 0.24% (×0.7) | 4.17% (×1.0) | 9.06% (×0.9) |
| no RQMC final stage | 2.15% (×1.2) | 0.42% (×1.3) | 9.03% (×2.2) | 8.21% (×0.8) |
| CE only (v1-style) | 2.45% (×1.3) | 0.43% (×1.3) | 2.28% (×0.5) | 30.46% (×3.0) |
| no target-relevance freezing | 1.85% (×1.0) | 0.33% (×1.0) | 4.15% (×1.0) | 26.79% (×2.6) |

The ablation separates indispensable components from refinements. **Indispensable:** twist warm start (removing it degrades ring n=16, Pr(L>T) by ×37); cross-entropy stage (removing it degrades ADB base, joint exhaustion by ×81); target-relevance freezing (removing it degrades ring n=8, joint exhaustion (0,1) by ×2.6). The cross-entropy stage is what reaches the rarest targets within the budget; the twist warm start and target-relevance freezing are what make the method work at portfolio scale, even though both are irrelevant for two countries. **Refinements** have smaller effects of either sign at this budget. Edge features: removing it was worse in 1, better in 1 and within ±15% in 2 of 4 cases (ratios ×0.5 to ×1.3); Rényi-2 (χ²) stage: removing it was worse in 1, better in 1 and within ±15% in 2 of 4 cases (ratios ×0.6 to ×1.2); Sample recycling: removing it was worse in 0, better in 2 and within ±15% in 2 of 4 cases (ratios ×0.7 to ×1.0); RQMC final stage: removing it was worse in 3, better in 1 and within ±15% in 0 of 4 cases (ratios ×0.8 to ×2.2). With 40 replications the relative RMSE itself carries a sampling error of roughly 10–20% (more in heavy-tailed cases), so ratios within this range are not significant. In particular, sample recycling gave no measurable benefit at these budgets; it is retained because it is unbiased and costless in compute, and whether to pool training batches will be decided by pilot effective sample size in Phase 2 (WP3). The Rényi-2 stage is likewise neutral here; its documented role is the repair of heavy-tailed fits (§4.4). Compared with the first prototype (cross-entropy only, factorised features, no relevance freezing), the full method is more accurate by ×1.3 on ADB base, joint exhaustion, ×1.3 on ADB base, Pr(L>80m), ×3.0 on ring n=8, joint exhaustion (0,1), and less accurate by ×1.8 on ring n=16, Pr(L>T), where a 16-country model with more parameters competes with estimation for a small budget.

## 6.4 Where classical exact methods suffice

* Applying the commuting kernels of Proposition 2 to a matrix-product state gives the full joint law,
  with a bond dimension set by the shared sources crossing each cut.
* Tail accuracy requires compressing in a *tilted gauge*, $\tilde p(x) = p(x)\prod_i g_i(x_i)$ with $g_i$ the inverse
  independent marginal. Plain truncation incurs up to 2% error on the rarest target even at tolerance
  $10^{-12}$.
* For chain, star and ring dependence the bond dimension is constant (9, 10, 58): all targets for 28
  countries are exact in about a minute. For complete dependence it grows (58 → 135 from four to six
  countries).

![**Figure [fig:tn].** Bond dimension of the exact tilted matrix-product state versus portfolio size.
Sparse dependence stays constant; dense dependence grows.](../../figures/paper/tn_bond.png){.wide}

## 6.5 Phase 2 engine: complete results

::: {.opener}
![](../art/cat_stopwatch.svg){.mascot}

Phase 2 engine: the same answers several times faster, a little more accuracy at scale, and exact answers for questions about two countries.
:::


Sections 6.1–6.4 report the **Phase 1 engine** (RDPO v1). We then implemented the first work packages of
the Deep Development plan (WP1 engineering, WP3 RL improvements) as the **Phase 2 engine** (RDPO v2), and
re-ran every study at the same total budgets, with more replications and identical seeds for every method.

**Engineering (WP1).**

* **Table-driven engine.** Each country state is one of 10 codes, so every logit is a table lookup. The
  policy is *identical* to Phase 1: actions, weights and gradients match to machine precision (tested).
  Rollouts are 6.5–9.9× faster on one core for 4–28 countries (Table [tab:v2tp]).
* **Exact compressed training.** Paths share few distinct local states at each source step, so the
  cross-entropy and Rényi-2 objectives are evaluated on unique states with aggregated weights. The
  objective is unchanged, and each evaluation is 50–500× cheaper.

**Table [tab:v2tp].** Rollout throughput (paths per second, one core) of the Phase 1 engine and the table-driven Phase 2 engine. Both sample from the identical policy (tested).

| Graph | n | sources K | Phase 1 | Phase 2 | speed-up |
|---|---|---|---|---|---|
| ring | 2 | 6 | 77,098 | 264,552 | **3.4×** |
| ring | 8 | 32 | 5,600 | 37,371 | **6.7×** |
| ring | 16 | 64 | 2,577 | 18,405 | **7.1×** |
| ring | 28 | 112 | 1,287 | 12,766 | **9.9×** |
| complete | 4 | 20 | 7,436 | 48,143 | **6.5×** |
| complete | 7 | 56 | 1,180 | 10,414 | **8.8×** |

**Reinforcement-learning improvements (WP3).**

* **Stage-dependent soft values.** The exact value function $V_k$ depends on how many sources remain, but
  Phase 1 shares one progress parameter vector across all steps. Phase 2 gives each of 4 contiguous
  stages of source steps its own progress parameters (+9 parameters) for portfolio-wide targets.
* **Flexible soft values for smooth targets.** For expected loss, a piecewise-linear (hat-function)
  basis, a fixed-knot ReLU network in the loss, replaces the hand-made progress features. The
  cross-entropy fit stays concave.
* **Exact reduction for pair targets.** A target that depends on two countries only needs the joint law
  of their 100 states. That law is the product of the kernels of the sources touching them (Proposition 2).
  It is computed exactly in at most 3 ms for any portfolio size, and agrees with the
  tensor-network references to 1.8×10⁻¹⁰. The Phase 2 engine answers such targets exactly and
  reserves RDPO for portfolio-wide targets.
* **Projected Doob sampler: a simplified RDPO for pair targets.** For a target that depends only on a few
  countries R, the exact value function depends only on their 10^|R| joint states. RDPO's learned log-linear
  soft value is then replaced by an exact table per step, computed by backward induction over the sources that
  touch R, while all other sources keep their true law. The policy is the zero-variance Doob sampler: every path
  returns the exact value (relative spread 8.9×10⁻¹⁶; 20,000 paths in 130 ms for the 10-country CAREC
  pool). Its paths are also exact draws from the portfolio *conditional* on the event, which turns the
  estimator into a stress-scenario generator. Given that the Kyrgyz and Tajik bonds are both exhausted, the
  CAREC pool's total payout averages US$194m (95th percentile US$275m), and exceeds US$400m
  with probability 0.04%. Three-country questions also work: all of KG, TJ and UZ exhausted has
  probability 2.1×10⁻⁸.
* **Safeguards.** Student-t intervals with Welch–Satterthwaite degrees of freedom. The initial (twist or
  transferred) policy is selected whenever its estimated χ² beats every trained iterate.

**Scaling study.**

![**Figure [fig:v2].** Phase 1 versus Phase 2 engine on the scaling study (B = 65,536; 24 replications;
identical seeds). (a, b) Relative RMSE for the portfolio-wide targets, with exponential twisting as the
classical reference. (c) Wall-clock time per estimate on one core.](../../figures/paper/v2_results.png){.wide}

*Table [tab:v2scale] in Appendix A lists every setting: method, coverage, twisting and run time.*

* **Accuracy.** On the portfolio-wide targets, at the same budget and seeds, the Phase 2 engine
  is more accurate than the Phase 1 engine in 13 of 16 settings (error lower by up to ×1.7, median ×1.18) and less accurate in 3 (×0.81–×0.95). It beats exponential twisting in 16 of 16 settings.
* **Speed.** One estimate takes 2.3–11.9× less wall-clock time than in the Phase 1 study.
* **Error bars.** With 24 replications, coverage of the Phase 2 intervals on portfolio-wide targets is
  88%–100% (mean 95%). Most low-coverage cells of the Phase 1 study (6–10 replications) were sampling noise.

**Benchmark.**

**Table [tab:v2bench].** Benchmark (25 scenario–target cells, 40 replications): Phase 1 → Phase 2 engine.

| Budget | cells better than best reference | cells where Phase 2 beats Phase 1 | median error ratio Phase 1 / Phase 2 (non-pair cells) | expected loss, median factor vs Sobol | mean coverage, Phase 2 |
|---|---|---|---|---|---|
| 4,096 | 20 → **20** | 14 of 25 | 0.98 | ×0.72 → **×0.74** | 94% |
| 16,384 | 20 → **20** | 15 of 25 | 1.05 | ×0.53 → **×0.62** | 94% |
| 65,536 | 20 → **20** | 18 of 25 | 1.13 | ×0.33 → **×0.45** | 94% |

* The Phase 2 engine keeps the 20/25 wins over the best reference method. On the non-pair cells it is on
  par with Phase 1 at 4,096 paths, and its advantage grows with the budget. Pair targets are answered
  exactly in the product. On expected loss the spline soft values close part of the gap to scrambled
  Sobol, but Sobol remains better.

**Transfer: the sleep phase in practice.** In the planned wake–sleep cycle (§10.2), today's estimate starts
from the policy trained overnight on yesterday's model. We emulate this by training on one model, then
reusing the parameters as the initial policy after a change of scenario or a 20% rate increase.

**Table [tab:v2transfer].** Overnight transfer (sleep phase): today's estimate starts from a policy trained on yesterday's model instead of from scratch. Relative RMSE, 40 replications.

| Change overnight | target | budget | from scratch | with transfer | gain |
|---|---|---|---|---|---|
| ADB base -> common shock | pair exhaustion | 2,048 | 80.81% | **1.14%** | ×71.0 |
| ADB base -> common shock | pair exhaustion | 4,096 | 19.67% | **0.72%** | ×27.3 |
| ADB base -> common shock | stop-loss | 2,048 | 1.55% | **0.61%** | ×2.5 |
| ADB base -> common shock | stop-loss | 4,096 | 0.82% | **0.45%** | ×1.8 |
| ADB base -> rain stress | pair exhaustion | 2,048 | 94.43% | **1.13%** | ×83.8 |
| ADB base -> rain stress | pair exhaustion | 4,096 | 16.28% | **0.72%** | ×22.5 |
| ADB base -> rain stress | stop-loss | 2,048 | 1.60% | **0.71%** | ×2.2 |
| ADB base -> rain stress | stop-loss | 4,096 | 1.14% | **0.46%** | ×2.5 |
| ring 16: rates +20% overnight | Pr(L > T) | 2,048 | 80.41% | **6.24%** | ×12.9 |
| ring 16: rates +20% overnight | Pr(L > T) | 4,096 | 28.07% | **3.57%** | ×7.9 |

* **Small budgets become usable.** Transfer improves accuracy by ×1.8–×83.8. At 2,048 paths, a
  cold start spends most of its budget reaching the tail, while the transferred policy is already there.
  The estimator remains unbiased, because the initial policy is fixed before sampling.

**Negative results (kept for the record).**

* A *value-of-training* stopping rule, which stops training when one more batch is not expected to pay
  for itself, slightly *increased* the error at two countries and was dropped.
* Spline soft values *hurt* the threshold targets (in development probes, up to ×19 on the stop-loss),
  so they are used for expected loss only.
* t-intervals and model selection changed no result in these studies: the effective degrees of freedom
  were large, and the trained policy always beat the initial one. They are kept as safeguards.

### 6.5.5 Comparison with stronger rare-event methods

Reviewers of rare-event methods expect comparisons against adaptive multilevel splitting, SMC on
exponential tilts, and neural importance sampling, not just the brief's four reference methods. We
implemented all three on the identical source-form problem (`src/baselines/stronger.py`) at the same
budget and seeds; the normalising-flow IS is warm-started from the optimal exponential twist, so it
dominates the SMC baseline by construction.

**Table [tab:stronger].** RDPO (Phase 2 engine) vs three stronger rare-event methods on the ADB benchmark and ring-16 scaling case, at the same budget and seeds (20 replications). Relative RMSE.

| Case | Target | Budget | μ | RDPO (Phase 2) | Splitting | SMC (twist) | Flow IS |
|---|---|---|---|---|---|---|---|
| adb-base | joint_exhaustion | 16,384 | 3.2×10⁻⁵ | **1.39%** | 152.51% | 10.77% | 69.81% |
| adb-base | exceedance | 16,384 | 1.2×10⁻² | **0.33%** | 28.75% | 1.78% | 32.67% |
| adb-common-shock | joint_exhaustion | 16,384 | 4.4×10⁻⁵ | **1.77%** | 91.49% | 10.43% | 51.11% |
| ring16 | exceedance | 65,536 | 1.5×10⁻⁸ | **0.57%** | 100.00% | 3.51% | 5.23% |

* **RDPO beats every one of them** on every rare target we tested, by factors of ×5.4–×8.
* **SMC on exponential tilts is the strongest fixed baseline.** It is what good practitioners actually
  deploy; RDPO's advantage against SMC is the gain from a learned *state-dependent* policy over the best
  product proposal.
* **Splitting struggles with the discrete, ties-heavy score** of the capped per-country loss, where the
  default rare-event score (total loss) jumps in large steps. We advance past ties and run Gibbs
  mutations, but acceptance stays low on the joint targets.
* **The flow IS does not beat SMC on joint exhaustion**, because a product proposal in source space can't
  easily force both countries to their respective maxima simultaneously; state-dependent transitions are
  needed, which is exactly what RDPO adds.

## 6.6 Real-world checks

::: {.opener}
![](../art/cat_seismo.svg){.mascot}

Do the bond's assumptions survive contact with data? An earthquake catalogue, 120 years of loss records, 48 weather stations, and calibrated pools.
:::


The benchmark model is calibrated to the brief's parameters, not to observations. We tested its earthquake
assumptions, the peril behind US$70m of each country's US$80m cover, against public data.

* **Data.** The USGS catalogue of significant earthquakes (M ≥ 5.5, 1965–2016) in the public mirror
  `plotly/datasets`, and country boundaries from `datasets/geo-countries`. Aftershocks are removed with
  Gardner–Knopoff windows, which leaves 168 main shocks in the Central Asian window. Both files are cached
  in `data/real/`, and the analysis is `src/realdata/eq_check.py`.
* **Proxy.** An event "reaches" a country if its epicentre lies inside the country or within a given
  distance of its border (25, 50 or 100 km). This is a proxy for damage, not a payout calculation.

![**Figure [fig:rd].** (a) Main shocks of M ≥ 5.5 around the Kyrgyz Republic (blue outline) and Tajikistan
(teal outline), 1965–2016; red events lie within 50 km of both countries. (b) Share of each country's events
that also reach the other country, with 95% intervals, against the brief's 7.5% / 7.4%.](../../figures/paper/realdata_eq.png){.wide}

**Table [tab:rd].** Earthquake main shocks (aftershocks removed) within a given distance of each country, 1965–2016. *Shared share*: fraction of a country's events that are also within that distance of the other country (95% Wilson interval); the brief assumes 7.5% (KG) and 7.4% (TJ). *Dispersion*: variance/mean of yearly counts (Poisson = 1) with the χ² p-value. *Years*: Fisher test of independence between the two countries' event years.

| Distance | M ≥ | events KG / TJ | shared share KG | shared share TJ | dispersion KG / TJ (p) | years: both observed vs independent (p) |
|---|---|---|---|---|---|---|
| 25 km | 5.5 | 32 / 42 | 38% (23%–55%) | 29% (17%–44%) | 0.77 (0.88) / 1.02 (0.43) | 40% vs 29% (p = 0.002) |
| 25 km | 6.0 | 15 / 14 | 33% (15%–58%) | 36% (16%–61%) | 1.00 (0.48) / 0.89 (0.69) | 15% vs 6% (p = 0.001) |
| 25 km | 6.5 | 5 / 4 | 60% (23%–88%) | 75% (30%–95%) | 1.33 (0.06) / 0.94 (0.59) | 6% vs 1% (p = 0.001) |
| 50 km | 5.5 | 41 / 53 | 44% (30%–59%) | 34% (23%–47%) | 0.66 (0.97) / 0.87 (0.74) | 52% vs 41% (p = 0.002) |
| 50 km | 6.0 | 20 / 18 | 40% (22%–61%) | 44% (25%–66%) | 0.83 (0.80) / 0.89 (0.69) | 21% vs 11% (p = 0.001) |
| 50 km | 6.5 | 6 / 6 | 67% (30%–90%) | 67% (30%–90%) | 1.24 (0.12) / 0.90 (0.67) | 8% vs 1% (p = 0.000) |
| 100 km | 5.5 | 57 / 75 | 49% (37%–62%) | 37% (27%–49%) | 0.83 (0.80) / 0.94 (0.60) | 58% vs 52% (p = 0.042) |
| 100 km | 6.0 | 24 / 32 | 42% (24%–61%) | 31% (18%–49%) | 0.89 (0.70) / 1.03 (0.42) | 25% vs 18% (p = 0.046) |
| 100 km | 6.5 | 6 / 13 | 67% (30%–90%) | 31% (13%–58%) | 1.24 (0.12) / 0.92 (0.63) | 8% vs 2% (p = 0.008) |

* **Poisson arrivals: supported.** Yearly main-shock counts have dispersion 0.66–1.24 at 50 km, with
  every p-value ≥ 0.12. This is consistent with the compound-Poisson model.
* **Trigger size: plausible.** The Gutenberg–Richter slope is b = 0.92 ± 0.11. The brief's annual 25%-payout
  probabilities then correspond to events of about M6.9 (Kyrgyz Republic) and M7.0 (Tajikistan): rare, damaging earthquakes, consistent with
  the bond's design.
* **Shared events: the brief's assumption looks too low.** 29%–75% of one country's main shocks lie
  within damaging distance of the other country, against the brief's 7.5%. Seismicity concentrates along
  the shared Alay–Pamir border zone. Examples in the catalogue are 1974 M7.3 (Markansu), 1978 M6.8 (Alay),
  2008 M6.7 (Nura) and 2016 M6.6 (Murghob). Event years are also significantly dependent at 25–50 km.
* **Consequence for the bond.** Setting the shared-event intensity from these data changes the exact joint
  targets as follows:

**Table [tab:rdimp].** Exact two-country targets when the shared-event intensity is set from the data (events within 50 km of both countries, M ≥ 6.0), compared with the brief.

| Shared-event setting | shared share | intensity m | joint attachment | Pr(L > 80m) | joint exhaustion |
|---|---|---|---|---|---|
| brief (base case) | 7.4% | 1.0 | 4.41% | 1.20% | 3.2×10⁻⁵ |
| data, lower 95% bound | 27.9% | 3.7 | 6.18% | 2.30% | 6.6×10⁻⁵ |
| data, point estimate | 42.1% | 5.7 | 7.43% | 3.07% | 8.9×10⁻⁵ |
| data, upper 95% bound | 57.8% | 7.8 | 8.84% | 3.90% | 1.2×10⁻⁴ |

* **Reading.** With the data-based shared intensity, the joint-exhaustion probability is 2.1–3.6× the
  brief's base value, and joint attachment rises to 6.2%–8.8%. The brief's "common shock ×2" scenario sits at
  the mild end of this range. The finding supports the brief's position that dependence must be stress-tested,
  and it suggests testing intensities well above ×2.
* **Caveats.** Epicentre distance is not payout: parametric triggers depend on ground motion at the exposure.
  The catalogue covers 52 years and ends in 2016, and the M ≥ 6.5 rows rest on very few events. Rainfall
  could not be checked: the reanalysis services (ERA5 via Open-Meteo) are blocked in this environment, and
  checking them is a first task of WP2.

**Loss-data backtest.** Epicentre distance is a proxy, so we repeated the test on *reported losses*. Public
records give yearly losses per country: NOAA's significant-earthquake database (1900–2020) and EM-DAT (1990–2020),
both via the Our World in Data mirror. In the bond model, each country's events are a Poisson process with a
country-only part and a shared part. For yearly occurrence this is a bivariate Poisson model, so the 2×2 table
of loss years identifies the shared rate by maximum likelihood (`src/realdata/loss_backtest.py`).

**Table [tab:loss].** Loss-data backtest of the shared-event assumption. Yearly occurrence of reported losses in the Kyrgyz Republic (KG) and Tajikistan (TJ), and the bivariate-Poisson estimate of the shared share (the fraction of KG events that also hit TJ; the brief assumes 7.5%). Profile-likelihood 95% intervals.

| Loss record | years | KG / TJ / both | both, if independent | shared share (95% CI) | test of independence |
|---|---|---|---|---|---|
| NOAA significant earthquakes | 1975–2020 | 8 / 16 / **5** | 2.8 | **45%** (0%–82%) | p = 0.077 |
| NOAA significant earthquakes | 1900–2020 | 9 / 20 / **5** | 1.5 | **48%** (12%–79%) | p = 0.005 |
| EM-DAT earthquake losses | 1990–2020 | 6 / 9 / **2** | 1.7 | **7%** (0%–57%) | p = 0.798 |
| EM-DAT flood/landslide/storm losses | 1990–2020 | 5 / 18 / **3** | 2.9 | **5%** (0%–80%) | p = 0.924 |

* **The longest record rejects the brief's 7.5%.** Over 1900–2020 the shared share is 48% (12%–79%), and
  independence is rejected (p = 0.005). This matches the epicentre analysis above, from independent data.
* **Short records are not informative.** The 31-year EM-DAT windows cannot separate 7.5% from 50%, and the
  flood and landslide record does not constrain the rain assumption. This is the brief's own point about
  short data.
* **Implied bond risk.** With the loss-data share for earthquakes (rain at the brief's value), the
  joint-exhaustion probability is 9.9×10⁻⁵ (95% range 4.0×10⁻⁵–1.5×10⁻⁴): 3.1× the brief's base case.

::: {.opener}
![](../art/cat_umbrella.svg){.mascot}

Rain is the other half of the bond. Is extreme rain shared between neighbours as often as the brief assumes?
:::

**Meteorological calibration of the rain cover.** The rain cover is parametric: it pays when an
extreme-precipitation index passes its 1-in-50-year level. Its frequency is a design choice, but the brief's 25%
shared share is an assumption about the weather. We measured it in daily station data (NOAA GHCN-Daily via its
open-data mirror; 48 stations; `src/realdata/rain_check.py`):

* **Index.** Each station's daily precipitation is divided by its median annual maximum; the country index is
  the daily maximum over its stations.
* **Events.** Days above the country's (1−q) annual-maximum quantile are merged into events less than 3 days
  apart. An event is shared when the neighbour has one within ±1 day.
* **Data cleaning.** Code values of 99.1–99.8 mm, which occur hundreds of times in former-Soviet records, are
  removed.

**Table [tab:rain].** Shared share of extreme-rain events between neighbouring countries (GHCN-Daily stations). Events at annual exceedance probability q; an event is shared when the other country has one within ±1 day. Shared / events of the rarer country; the brief assumes 25%.

| Pair | common years | q = 50% | q = 20% | q = 10% | q = 5% |
|---|---|---|---|---|---|
| Kyrgyz Rep.–Tajikistan | 42 | 1/24 | 0/9 | 0/5 | 0/3 |
| Kyrgyz Rep.–Uzbekistan | 48 | 0/30 | 0/10 | 0/5 | 0/3 |
| Kyrgyz Rep.–Kazakhstan | 45 | 0/26 | 0/10 | 0/5 | 0/3 |
| Tajikistan–Uzbekistan | 45 | 1/33 | 0/9 | 0/5 | 0/3 |
| Uzbekistan–Turkmenistan | 50 | 6/38 | 0/11 | 0/5 | 0/3 |
| Uzbekistan–Kazakhstan | 51 | 0/36 | 0/11 | 0/5 | 0/3 |
| Kazakhstan–Turkmenistan | 47 | 2/35 | 1/10 | 0/4 | 0/3 |
| Azerbaijan–Georgia | 47 | 3/39 | 1/11 | 0/5 | 0/3 |
| Kazakhstan–Mongolia | 37 | 0/25 | 0/9 | 0/4 | 0/2 |
| **pooled** | | **4.5%** (2.7%–7.6%) | **2.2%** (0.6%–7.7%) | **0.0%** (0.0%–8.2%) | **0.0%** (0.0%–12.9%) |

* **Extreme rain is local.** Pooled over neighbouring pairs, the shared share is 2.2% (0.6%–7.7%) at the
  1-in-5-year level and falls towards zero for rarer events. The brief's 25% is well outside the data.
* **Both corrections together.** Earthquakes are more dependent than assumed (loss data) and rain much less
  (station data):

**Table [tab:rainimp].** Exact two-country targets with the dependence assumptions replaced by data.

| Dependence | earthquake intensity m | rain shared share | joint attachment | Pr(L > 80m) | joint exhaustion |
|---|---|---|---|---|---|
| brief | 1.0 | 25.0% | 4.41% | 1.20% | 3.2×10⁻⁵ |
| rain from station data only | 1.0 | 2.2% | 3.43% | 1.13% | 8.7×10⁻⁶ |
| earthquake from loss data only | 6.4 | 25.0% | 7.94% | 3.37% | 9.9×10⁻⁵ |
| both from data | 6.4 | 2.2% | 6.91% | 3.32% | 2.7×10⁻⁵ |

* **Reading.** With both dependence assumptions taken from data, pool exceedance is 2.8× the
  brief's value, while joint exhaustion is 0.85× because exhaustion needs rain in both countries.
  Checking one assumption alone would have been misleading in either direction.
* **Caveats.** Six stations per country sample the rainfall field sparsely, and Kyrgyz and Tajik records thin out
  after the 1990s. Gridded data (ERA5, GPCP, both reachable via open-data mirrors) are the next step.

**Market record of comparable bonds.** The challenge bond has no payout history, and catastrophe bonds are
private placements without public price series, so exchange data such as Yahoo Finance cannot be used. The
closest public record is the World Bank's parametric sovereign bonds. In that record, triggers do occur and
near-misses decide outcomes, but no two countries have ever paid out together:

| Bond (IBRD) | Peril | Modelled annual attachment probability | Outcome |
|---|---|---|---|
| FONDEN 2017, Mexico, Class A (US$150m) | earthquake | not published in the sources found | 100% paid after the M8.1 Chiapas earthquake, Sept 2017 [^mx17] |
| Pacific Alliance 2018, Peru (US$200m) | earthquake | 7.14% | 30% paid after the M8.0 Loreto earthquake, May 2019 [^pe18] |
| FONDEN 2020, Mexico, Class D | Pacific hurricane | not published in the sources found | about 50% loss after Hurricane Otis, 2023 [^mx20] |
| CAR Jamaica 2024 (US$150m) | hurricane | 2.46% (2.34% at launch) | not triggered by Beryl (2024); 100% paid after Melissa, 2025 [^jm24] |

* **What this shows.** Parametric sovereign bonds do pay, and basis risk is real (Beryl). There are too few
  bonds with published attachment probabilities to test the modelled frequencies statistically. The record is far too small to validate a *joint* tail, which is why the joint risk has to come
  from a model of the kind tested here.

[^mx17]: Artemis, "Mexico confirms $150m cat bond payout for quake", https://www.artemis.bm/news/mexico-confirms-150m-cat-bond-payout-for-quake/
[^pe18]: Artemis, "Peru's World Bank cat bond could be triggered by M8.0 earthquake", https://www.artemis.bm/news/perus-world-bank-cat-bond-could-be-triggered-by-m8-0-earthquake/; attachment probability from Artemis, "$1 billion Pacific Alliance earthquake cat bond launched by World Bank", https://www.artemis.bm/news/1-billion-pacific-alliance-earthquake-cat-bond-launched-by-world-bank/
[^mx20]: Artemis, "Mexico expected to renew Pacific coast hurricane cat bond after Otis resolution", https://www.artemis.bm/news/mexico-expected-to-renew-pacific-coast-hurricane-cat-bond-after-otis-resolution/
[^jm24]: World Bank press release, 7 Nov 2025, https://www.worldbank.org/en/news/press-release/2025/11/07/hurricane-melissa-triggers-100-payout-of-150-million-world-bank-catastrophe-bond-for-jamaica; attachment probability from the World Bank case study, https://thedocs.worldbank.org/en/doc/401877c87631461af8ad227793affc5f-0340012025/original/Case-Study-Jamaica-2024-Cat-Bond.pdf; Beryl: Artemis, https://www.artemis.bm/news/jamaicas-disaster-insurance-funds-sufficient-for-beryl-despite-no-cat-bond-payout-cdps-meenan/

**A real five-country pool.** Finally, we calibrated a Central Asian pool from data alone and tested RDPO on
it (`src/realdata/central_asia_pool.py`):

* **Rates.** Rates come from each country's catalogue main shocks within 50 km, with one Gutenberg–Richter slope
  b = 0.86.
* **Payout design.** Each country holds the bond's design. The 25% tier triggers at M7.1, the magnitude at
  which the Kyrgyz and Tajik data reproduce the brief's trigger frequency, and higher tiers follow the brief's
  tier ratios.
* **Dependence graph.** The graph is not assumed: an edge exists wherever main shocks reached two countries.
  The catalogue finds Kyrgyz Rep.–Tajikistan (14 events), Kyrgyz Rep.–Uzbekistan (8 events), Kyrgyz Rep.–Kazakhstan (8 events), Tajikistan–Uzbekistan (3 events), Uzbekistan–Turkmenistan (1 events). A shock reaching three countries is assigned to its two nearest
  countries, so no event is counted twice.

**Table [tab:poolcal].** Data-calibrated five-country pool. Annual probabilities of a 25% and a 100% earthquake payout per country (each country holds US$70m earthquake + US$10m rain cover), and shared-event rates on the edges found in the catalogue.

| Country | main shocks M ≥ 5.5 within 50 km, per year | annual P(≥ 25% payout) | annual P(100% payout) | brief (KG, TJ) |
|---|---|---|---|---|
| Kyrgyz Rep. | 0.79 | 3.44% | 0.99% | 3.88% / 1.18% |
| Tajikistan | 1.02 | 4.43% | 1.28% | 3.92% / 1.04% |
| Uzbekistan | 0.31 | 1.36% | 0.39% | — |
| Kazakhstan | 0.48 | 2.11% | 0.61% | — |
| Turkmenistan | 0.21 | 0.94% | 0.27% | — |

**Table [tab:pool].** RDPO (Phase 2 engine) on the data-calibrated pool against exact answers (dense enumeration), at 16,384 simulated terms, 40 replications. Relative RMSE; interval coverage in parentheses.

| Question | exact value | RDPO | plain Monte Carlo | exp. twisting | RDPO gain over best alternative |
|---|---|---|---|---|---|
| Pr(pool payout > US$200m) | 6.4×10⁻⁴ | **0.71%** (98%) | 24.89% | 2.15% | ×3.0 |
| Pr(pool payout > US$320m) | 2.4×10⁻⁷ | **1.65%** (92%) | 100.00% | 8.93% | ×5.4 |
| expected pool payout | US$16.97m | **0.42%** (98%) | 1.56% | — | ×3.7 |
| KG and TJ both paid | 5.6×10⁻² | **0.61%** (92%) | 3.63% | — | ×5.9 |
| KG and TJ both exhausted | 3.3×10⁻⁵ | **9.53%** (72%) | 141.39% | — | ×14.8 |
| stop-loss above US$200m | US$10,458 | **0.93%** (95%) | 32.91% | 1.57% | ×1.7 |

* **Result.** On the data-calibrated pool, RDPO beats the better of plain Monte Carlo and exponential
  twisting by ×1.7–×15 on the rare questions, against exact answers. On expected pool
  payout, a smooth target, the factor over Monte Carlo is ×3.7.
* **Weak spot.** The joint exhaustion of two countries inside the heterogeneous pool is the hardest question
  at this budget. RDPO's error is 9.5% with 72% interval coverage: far better than Monte
  Carlo, but the intervals are optimistic. The Phase 2 engine therefore answers pair questions exactly (pair
  reduction, §6.5) and uses RDPO for pool-wide questions.

### 6.6.5 Parameter uncertainty: posterior credible intervals for the headline findings

The data give *ranges*, not points. The headline findings compared the brief with the data endpoints,
which hides how much comes from the data itself. We now propagate parameter uncertainty end-to-end. On
each shared share we place a Jeffreys-prior Beta posterior from the bivariate-Poisson occurrence table
(earthquake: NOAA 1900–2020; rain: pooled station analysis at the 1-in-5-year level), draw 2,000
samples, and recompute the exact two-country targets at each sample
(`src/realdata/param_uncertainty.py`).

**Table [tab:pu].** Posterior credible intervals for the ADB bond's two-country targets, from Jeffreys-prior posteriors on the two shared shares. 2000 Monte Carlo samples; exact dense enumeration per sample.

| Target | Brief (point) | Posterior median | 50% credible interval | 95% credible interval | Factor vs brief (median) |
|---|---|---|---|---|---|
| Pr(L > 80m) | 1.20% | **3.76%** | [3.15%, 4.36%] | [2.08%, 5.27%] | **×3.12** |
| Joint attachment | 4.41% | **7.63%** | [6.62%, 8.64%] | [4.93%, 10.23%] | **×1.73** |
| Joint exhaustion | 3.2×10⁻⁵ | **3.1×10⁻⁵** | [2.5×10⁻⁵, 3.8×10⁻⁵] | [1.6×10⁻⁵, 5.6×10⁻⁵] | **×0.99** |
| Stop-loss | US$0.28m | **US$0.94m** | [US$0.79m, US$1.10m] | [US$0.52m, US$1.33m] | **×3.34** |
| Expected loss | US$10.06m | **US$10.06m** | [US$10.06m, US$10.06m] | [US$10.06m, US$10.06m] | **×1.00** |

* **Pool exceedance:** posterior median 3.76% (**3.12× the brief**); 95% credible
  interval **[2.08%, 5.27%]**; the whole interval sits above the brief's 1.20%.
* **Joint attachment:** posterior median 7.63% (1.73× the brief); 95% credible interval
  [4.93%, 10.23%].
* **Joint exhaustion:** posterior median 3.1×10⁻⁵ (0.99× the brief); 95% credible interval
  [1.6×10⁻⁵, 5.6×10⁻⁵]. The two dependence corrections cancel on this target: more earthquakes
  shared, but less rain shared, and exhaustion needs both.
* **Reading.** A credible interval that lies entirely above the brief's value is strong evidence of
  underestimation; one that straddles it is not. By this standard the pool exceedance and joint
  attachment are robust findings and the joint exhaustion is not.

## 6.7 WP2: calibrated regional pools

::: {.opener}
![](../art/cat_globe.svg){.mascot}

From two countries to regional pools built from public data: 5, 10 and 28 real countries, and a 50-country stress test.
:::


The work plan's WP2 asks for realistic portfolios (public hazard data, heterogeneous perils, dense dependence,
10–50 countries). Its success criterion is a relative RMSE of at most 10% at μ ≤ 10⁻⁸ within 10⁵ paths. As a
first step we built the pool of ADB's Central Asia Regional Economic Cooperation programme (CAREC) from public
data alone (`src/realdata/carec_pool.py`):

* **Countries.** Afghanistan, Azerbaijan, Georgia, Kazakhstan, Kyrgyz Republic, Mongolia, Pakistan, Tajikistan,
  Turkmenistan and Uzbekistan. China is excluded because only some of its provinces take part in CAREC.
* **Perils.** Earthquakes come from 731 catalogue main shocks, calibrated as in §6.6. Rain is
  heterogeneous: each country's trigger probability scales the brief's 2% by its observed frequency of major
  hydro-meteorological losses.
* **Dependence.** The catalogue finds 10 shared-event edges: Georgia–Azerbaijan, Turkmenistan–Uzbekistan, Turkmenistan–Afghanistan, Kazakhstan–Kyrgyz Rep., Uzbekistan–Afghanistan, Uzbekistan–Tajikistan, Uzbekistan–Kyrgyz Rep., Afghanistan–Pakistan, Afghanistan–Tajikistan, Tajikistan–Kyrgyz Rep..
* **Exact references.** A tilted matrix-product state with countries ordered west to east has maximum bond
  dimension 588 and builds in 5 min. Results at truncation tolerances 10⁻¹⁰ and 10⁻¹² agree
  to 4.0×10⁻¹⁰.

**Table [tab:carec].** Calibrated CAREC pool (west to east). Each country holds US$70m earthquake + US$10m rain cover. Rain-trigger probabilities scale the brief's 2% by each country's frequency of major flood, landslide and storm losses (EM-DAT 1990–2020).

| Country | main shocks M ≥ 5.5 within 50 km, per year | annual P(≥ 25% EQ payout) | annual P(100% EQ payout) | annual P(rain trigger) |
|---|---|---|---|---|
| Georgia | 0.17 | 0.77% | 0.22% | 0.87% |
| Azerbaijan | 0.12 | 0.51% | 0.15% | 0.87% |
| Turkmenistan | 0.21 | 0.94% | 0.27% | 0.50% |
| Kazakhstan | 0.48 | 2.11% | 0.61% | 1.57% |
| Uzbekistan | 0.31 | 1.36% | 0.39% | 0.50% |
| Afghanistan | 1.31 | 5.64% | 1.64% | 4.35% |
| Pakistan | 1.29 | 5.56% | 1.62% | 4.87% |
| Tajikistan | 1.02 | 4.43% | 1.28% | 3.13% |
| Kyrgyz Rep. | 0.79 | 3.44% | 0.99% | 0.87% |
| Mongolia | 0.27 | 1.19% | 0.34% | 2.09% |

**Table [tab:carecres].** RDPO (Phase 2 engine) on the CAREC pool against exact tensor-network references, 24 replications. Relative RMSE; interval coverage in parentheses.

| Question | budget | exact value | RDPO | plain Monte Carlo | exp. twisting |
|---|---|---|---|---|---|
| Pr(pool payout > US$240m) | 65,536 | 1.3×10⁻³ | **0.36%** (100%) | 9.85% | 0.95% |
| Pr(pool payout > US$500m) | 16,384 | 4.3×10⁻⁹ | **8.28%** (79%) | 100.00% | 18.34% |
| Pr(pool payout > US$500m) | 65,536 | 4.3×10⁻⁹ | **1.25%** (100%) | 100.00% | 6.97% |
| expected pool payout | 65,536 | US$35.27m | **0.13%** (100%) | 0.57% | — |
| KG and TJ both exhausted | 65,536 | 1.6×10⁻⁵ | **5.16%** (75%) | 115.09% | — |
| stop-loss above US$240m | 65,536 | US$37,535 | **0.41%** (96%) | 12.17% | 0.76% |
| stop-loss above US$500m | 65,536 | US$8.4×10⁻² | **0.71%** (100%) | 100.00% | 7.62% |

* **WP2 criterion met.** At μ = 4.3×10⁻⁹ (pool payout above US$500m), RDPO's relative
  RMSE is 1.2% with 65,536 paths, against the 10% target within 10⁵ paths. At 16,384 paths it is
  8.3%.
* **Remaining WP2 scope.** Ground-motion and exposure models in place of epicentre proxies, and gridded
  rainfall.

**Scaling to 28 real countries and 50 synthetic ones.** The second WP2 pool covers the ADB developing member
countries of Asia and the Pacific (`src/realdata/asia_pacific_pool.py`):

* **Design.** Every country buys the same *return-period* cover, as parametric sovereign covers are designed in
  practice. Its 25% earthquake tier attaches at its own 1-in-25.6-year event, so trigger magnitudes range from
  about M5.5 to M8.8. Rain uses the 2% design with the station-calibrated shared share.
* **Data.** 7424 catalogue main shocks in the region. 28 countries had at least one main shock
  within 50 km and form the pool, with 22 data-derived shared-event edges.
* **Exact references.** Ordering the countries by reverse Cuthill–McKee gives bandwidth 5. The exact
  tensor network has maximum bond dimension 367 and builds in 42 s, and a 100× tighter truncation
  agrees to 5.7×10⁻¹⁰. A 50-country ring (bond dimension 64) completes the 10–50-country range of WP2.

![**Figure [fig:apmap].** The data-calibrated Asia–Pacific pool. Countries with at least one main shock within 50 km (CAREC members blue, other ADB developing members teal) and the shared-earthquake links found in the catalogue; line width grows with the number of shared main shocks. Cambodia, Sri Lanka and Samoa had no main shock within 50 km in 1965–2016 and are not in the pool.](../../figures/paper/pool_map.png){.wide}

**Table [tab:ap].** WP2 at scale, 65,536 paths, 24 replications, exact tensor-network references. Relative RMSE; interval coverage in parentheses.

| Pool | question | exact value | RDPO | plain Monte Carlo | exp. twisting |
|---|---|---|---|---|---|
| Asia–Pacific (28 countries) | Pr(pool payout > US$450m) | 1.8×10⁻³ | **0.34%** (96%) | 8.54% | 0.81% |
| Asia–Pacific (28 countries) | Pr(pool payout > US$850m) | 9.5×10⁻⁹ | **0.76%** (96%) | 100.00% | 2.28% |
| Asia–Pacific (28 countries) | stop-loss above US$850m | US$2.7×10⁻¹ | **0.55%** (100%) | 100.00% | 1.84% |
| Asia–Pacific (28 countries) | KG and TJ both exhausted (sampled; exact by pair reduction) | 1.4×10⁻⁵ | **5.93%** (71%) | — | — |
| ring (50 countries) | Pr(pool payout > US$650m) | 1.5×10⁻³ | **0.47%** (96%) | 7.47% | 0.81% |
| ring (50 countries) | Pr(pool payout > US$1150m) | 4.1×10⁻⁹ | **0.79%** (100%) | 100.00% | 1.82% |
| ring (50 countries) | stop-loss above US$1150m | US$1.4×10⁻¹ | **0.95%** (92%) | 100.00% | 1.34% |
| ring (50 countries) | KG and TJ both exhausted (sampled; exact by pair reduction) | 1.9×10⁻⁵ | **0.72%** (92%) | — | — |

* **WP2 criterion at scale met.** Asia–Pacific (28 countries): 0.8% at μ = 9.5×10⁻⁹; ring (50 countries): 0.8% at μ = 4.1×10⁻⁹, all with 65,536 paths (target: ≤ 10% within 10⁵
  paths).



## 6.8 Running on catastrophe-model output (ELT/YLT)

::: {.opener}
![](../art/cat_table.svg){.mascot}

Analysts already have event and year loss tables. The converter lets RDPO run on them directly.
:::


Catastrophe models deliver an *event loss table* (ELT: each stochastic event with its annual rate and the loss it
causes in each country) or a *year loss table* (YLT: simulated years and the events in them). The converter in
`src/elt/convert.py` turns either into RDPO's source form, so the engine runs on model output without changing
the analyst's workflow:

1. **Payouts.** Each event's loss per country is mapped to the cover (earthquake quarter steps, rain trigger).
2. **Footprints.** Events paying in the same set of countries form one compound Poisson source. A footprint may
   contain any number of countries, not only pairs.
3. **Term law.** Each source's capped three-year increment is computed exactly on its joint state space
   (Proposition 1 per source).
4. **Year loss tables.** Event rates are estimated as occurrences divided by the number of simulated years.

![**Figure [fig:eltpipe].** From catastrophe-model output to RDPO. Events are grouped by the countries they pay
in; each group becomes one Poisson source with an exactly computed capped three-year law.](../art/elt_pipeline.svg){.wide}

*Demonstration.* No vendor ELT is public, so we built an illustrative stochastic event set from public data
(`src/elt/demo.py`):

* **Event set.** Every catalogue main shock within 300 km of the five Central Asian countries (198
  locations) generates magnitudes M7.0–8.5 at Gutenberg–Richter rates.
* **Payouts.** An event pays in every country within a magnitude-dependent radius (cat-in-a-circle), with the
  bond's tiers. Rain uses the 2% design and the station-calibrated shared share.
* **Size.** The resulting ELT has 2,411 events. Conversion takes 102 ms and yields 19
  sources, 7 of which pay in three or more countries at once (up to 4).
* **YLT.** A 10,000-year YLT is simulated from the ELT, as a modelling agent would deliver it.

**Table [tab:elt].** RDPO on a converted event loss table against exact answers, 24 replications. *RDPO* and *plain MC*: 16,384 simulated terms from the converted sources. *YLT, 10,000 years*: the estimate an analyst would read directly off a 10,000-year year loss table (3,333 three-year terms).

| Question | exact value | RDPO | plain Monte Carlo | YLT, 10,000 years | exact from the YLT itself |
|---|---|---|---|---|---|
| Pr(pool payout > US$160m) | 3.5×10⁻² | **0.57%** (92%) | 4.25% | 10.75% | 3.6×10⁻² |
| Pr(pool payout > US$320m) | 2.3×10⁻⁴ | **5.38%** (96%) | 42.71% | 131.53% | 2.6×10⁻⁴ |
| stop-loss above US$240m | US$0.26m | **0.56%** (100%) | 8.76% | 29.75% | US$0.25m |
| KG and TJ both exhausted | 1.4×10⁻⁴ | **5.90%** (92%) | 82.98% | 153.06% | 1.2×10⁻⁴ |
| expected pool payout | US$30.26m | **0.36%** (100%) | 1.43% | 3.51% | US$29.93m |

* **The YLT is the bottleneck on rare questions.** For the rarest pool question, reading the answer off a
  10,000-year YLT gives a relative error of 132%. RDPO on the converted sources gives
  5.4% with 16,384 simulated terms.
* **Exact where possible.** On the converted model, pair questions are exact (projected Doob sampler:
  1.4×10⁻⁴ for KG and TJ both exhausted).
* **Next steps.** Next come secondary uncertainty within an event (here the mean loss is used) and Oasis LMF
  file formats.

## 6.9 Limitations

* **Smooth targets.** On expected loss, scrambled Sobol remains more accurate and should be preferred.
* **Level functions.** The cross-entropy stage is indispensable for the rarest targets, so new targets
  need a sensible score function.
* **Error bars under heavy tails.** In the Phase 2 benchmark, coverage is 82%–100% per cell (40 replications); the lowest cells are
  rare targets with heavy-tailed weights. Heavy-tail diagnostics remain part of WP3.
* **Dense dependence.** Complete graphs are limited to 7 countries by the exact references. Larger dense
  portfolios need reference-free validation (WP2).
* **Real-data scope.** Earthquake dependence was checked against a catalogue and two loss records, rain
  dependence against station data, and RDPO was tested on data-calibrated pools of 5, 10 and 28 countries
  (§6.6–6.7). Remaining gaps: loss records are yearly and smaller than bond triggers; the station network is
  sparse; epicentre distance stands in for ground motion and exposure; and no bond payout history exists to
  backtest against.

::: {.takeaway}
RDPO is the most accurate method on every rare portfolio-wide target we tested, up to 50 countries and on
real-data pools of up to 28. The Phase 2 engine is several times faster and up to 1.7× more accurate at scale, overnight transfer makes small
budgets usable, and pair targets are solved exactly. Scrambled Sobol remains better on expected loss.
:::

# 7 Quantum Extension: an Evidence-First Design {.newpage}

::: {.opener}
![](../art/cat_box.svg){.mascot}

Phase 1 provides no quantum processor, so no quantum advantage can be demonstrated, and we claim none.
Instead we specify the quantum stage precisely enough that its advantage can be *tested and falsified*.
:::

## 7.1 Why Q-RDPO is not "QML on classical data"

The position paper of Guo and Pan [@guo2026stopqml] identifies four reasons why quantum machine
learning on classical data has not earned priority. Q-RDPO is designed to avoid three of them by
construction and to measure the fourth:

| Barrier for QML on classical data | Q-RDPO |
|---|---|
| Advantage relies on strong **input-loading assumptions** | Loading is *exact* and cheap: one small register per Poisson source, O(n + \|E\|) gates (Proposition 2, §7.4) |
| **Trainability** (barren plateaus, noisy gradients) | Nothing is trained on the quantum device; the classical RL agent learns the measure (AI-for-Quantum) |
| **Noise** makes deep circuits unreliable | Noise is modelled explicitly, and a calibration trust guard falls back to the classical estimate (§7.7) |
| No convincing **wall-clock** wins | A falsifiable wall-clock criterion against our own best classical method (§7.6) |

## 7.2 The estimation problem and the quantum primitive

* **Task.** Estimate $\mu = \mathbb{E}_p[f]$ for a pool target $f \ge 0$ to relative error $\varepsilon$ with confidence
  $1-\delta$, where $p$ is the source law of Proposition 2 and $f$ is bounded on its support.
* **Classical reference.** RDPO draws paths from a learned measure $q_\theta$ with per-path relative variance $\rho^2$,
  so it needs $N_C = \rho^2 / \varepsilon^2$ paths of cost $\tau_C$ each ($\varepsilon$ measured as relative RMSE).
* **Quantum primitive.** Amplitude estimation on a state-preparation unitary $A$ whose "good" amplitude is
  proportional to $\mu$.

::: {.proposition}
**Proposition 6 (amplitude encoding with bounded weights).** Let $q$ be a law that $A$ loads exactly, with
$w = p/q$, and let $C \ge \sup_x f(x) w(x)$. If $A$ prepares
$$A|0\rangle = \sum_x \sqrt{q(x)}\,|x\rangle\big(\sqrt{1-g(x)}\,|0\rangle + \sqrt{g(x)}\,|1\rangle\big), \qquad g = f w / C \in [0, 1],$$
then the probability of measuring the objective qubit in $|1\rangle$ is $a = \mu / C$. For the exponentially
twisted law $q_\eta \propto p\, e^{\eta U}$ and $f = \mathbb{1}\{L > T\}$, one may take $C = e^{\Lambda(\eta) - \eta T}$, giving
$g = \mathbb{1}\{L>T\}\, e^{-\eta (U - T)}$ and $a = \mu\, e^{\eta T - \Lambda(\eta)}$.

*Proof.* $\Pr(1) = \sum_x q(x) g(x) = \sum_x p(x) f(x) / C = \mu/C$. Under $q_\eta$, $w = e^{-\eta U + \Lambda(\eta)}$, and
$U \ge L > T$ on the target event, so $f w \le e^{\Lambda(\eta) - \eta T}$. $\square$
:::

::: {.proposition}
**Proposition 7 (query complexity).** Iterative amplitude estimation [@grinko2021] returns $\hat a$ with
$|\hat a - a| \le \varepsilon a$ with probability at least $1-\delta$ using
$N_Q = O\big(\varepsilon^{-1} a^{-1/2} \log(1/\delta)\big)$ applications of $A$, against $O(\varepsilon^{-2} a^{-1})$ samples for
classical sampling of the same measure [@brassard2002; @montanaro2015]. We write $N_Q = c_Q/(\varepsilon\sqrt a)$ and
measure $c_Q$ (which absorbs the confidence level, $\delta = 0.05$) by exact emulation of the algorithm.
:::

**Corollary (crossover law).** With $\tau_A$ the fault-tolerant time per application of $A$, the wall-clock
ratio of the quantum and classical estimators is
$$s^*(\varepsilon, \rho^2) = \frac{N_Q \tau_A}{N_C \tau_C} = \kappa\,\frac{\varepsilon}{\rho^2}, \qquad \kappa = \frac{c_Q\,\tau_A}{\tau_C\,\sqrt{a}} .$$
Two consequences follow:

* **What the quantum stage needs.** It wins only where the *best classical* estimator is hard (large $\rho^2$) or
  where very high precision is required (small $\varepsilon$).
* **What a quantum advantage claim must compare against.** Comparing with plain Monte Carlo ($\rho^2 = (1-\mu)/\mu$)
  instead of RDPO overstates the advantage by the factor $\rho^2_{\text{MC}}/\rho^2_{\text{RDPO}}$, which here is up to $10^{11}$.

## 7.3 Algorithm design

::: {.algorithm}
**Algorithm 2: Q-RDPO (guarded quantum estimation stage)**

1. **Classical stage.** Run RDPO (Algorithm 1) on the target. Keep its estimate $\hat\mu_C$, its interval, and its
   per-path variance $\hat\rho^2$.
2. **Choose a loadable measure.** Among product-form (state-independent) laws, choose the twist that
   maximises the encoded amplitude $a$. Each source's law and weight then factorise, so $A$ is a tensor product
   of small exact rotations followed by arithmetic.
3. **Decide whether to go quantum.** Predict $s^*$ from the crossover law with the calibrated hardware
   parameters. If $s^*$ exceeds the hardware's speed-up, stop and report $\hat\mu_C$.
4. **Compile.** Build $A$ (Proposition 6, §7.4) and the iterate $Q = A S_0 A^\dagger S_\chi$. Set the depth cap $k_{\max}$ so
   that the deepest circuit succeeds with probability at least one half at the device's logical error rate.
5. **Estimate.** Run iterative or depth-capped maximum-likelihood amplitude estimation, with Grover powers
   $k \le k_{\max}$ and a noise-aware likelihood.
6. **Guard.** Accept $\hat\mu_Q = C\hat a$ only if the fit passes the goodness-of-fit test across depths *and* $\hat\mu_Q$
   agrees with $\hat\mu_C$ within 3σ. Otherwise report $\hat\mu_C$.
:::

* **Why the learned RDPO policy itself is not loaded.** Its transition probabilities depend on the
  current state (country codes and running loss). Loading it exactly needs multiplexed rotations whose cost
  grows exponentially with the size of each source's context, and its importance weights are bounded only by
  $\alpha^{-K}$. Either cost would destroy the scaling.
* **Product-form laws.** State-independent laws, such as twists and per-source tilts, keep both costs linear and
  the weights bounded (Proposition 6).

| Loadable measure | Gate cost of A | Weight bound C | Status |
|---|---|---|---|
| Untwisted law $p$ | $O(n + \|E\|)$ | 1 ($a = \mu$) | baseline; $N_Q \propto \mu^{-1/2}$ |
| Exponential twist $q_\eta$ | $O(n + \|E\|)$ + one rotation per bit of $D$ | $e^{\Lambda(\eta) - \eta T}$ | **used** |
| Per-source tilts $q_k \propto p_k e^{\eta_k \ell_k}$ (learned) | same | $\sup_{L > T} e^{\sum_k (\Lambda_k(\eta_k) - \eta_k \ell_k)}$ | WP4 |
| Full RDPO policy $q_\theta(\cdot\mid s)$ | exponential in context size | up to $\alpha^{-K}$ | not loadable |

* **The classical agent as the quantum stage's compiler.** The quantum cost $N_Q \propto a^{-1/2}$ depends on the
  loaded measure. Choosing that measure is a reinforcement-learning problem of the same kind as RDPO: maximise
  the encoded amplitude over product-form laws. This is the AI-for-Quantum component of the design.

## 7.4 Construction: exact structured loading with bounded weights

![**Figure [fig:qcirc].** Q-RDPO. Top: the hybrid pipeline of Algorithm 2. Bottom: one Grover iterate. ① Each
source register is loaded exactly in its twisted law. ② Reversible adders form per-country sums. ③ The
uncapped loss minus the threshold is computed. ④ The threshold flag is set. ⑤ The weight $e^{-\eta D}$ is
written onto the objective qubit one bit of $D$ at a time. The oracle $S_\chi$, the inverse $A^\dagger$ and the
reflection $S_0$ complete the iterate.](../art/quantum_v3.svg){.wide}

* **Exact loading.** One register per source (3, 6 or 1 qubits), prepared in $\sum_x \sqrt{q_k(x)}\,|x\rangle$; no
  truncation of Poisson counts. A sparse-statevector simulation reproduces the joint law to $10^{-16}$.
* **Bounded weights.** With twisted registers $q_k \propto p_k e^{\eta \ell_k}$, the weight depends only on the uncapped
  loss $U \ge L$: $w = e^{-\eta U + \Lambda(\eta)}$. On $\{L > T\}$ it is bounded, and the amplitude becomes
  $$a = \mu\, e^{\eta T - \Lambda(\eta)}, \qquad \max_\eta a = \mu\, e^{I(T)},$$
  which decays with the large-deviation gap, not with $\mu$.
* **Bitwise oracle.** Since $e^{-\eta D} = \prod_b (e^{-\eta 2^b})^{d_b}$ for $D = U - T$, the weight needs one controlled
  rotation per bit of $D$ (about 12 at 28 countries) instead of a lookup table with thousands of entries.
* **Query effect.** At 28 countries ($\mu \approx 1.3\times10^{-13}$) the projected number of amplitude-estimation
  calls falls from $2.8\times10^9$ (untwisted) to about $3.4\times10^4$, against $7.6\times10^{14}$ Monte Carlo paths
  (Figure [fig:queries]).

![**Figure [fig:queries].** Projected query counts for 10% relative error on $\Pr(L > T_n)$ (ring dependence).
Quantum curves are projections from an exact emulation of iterative amplitude estimation, not hardware
runs.](../../figures/paper/quantum_queries.png){.wide}

## 7.5 Scalability

* **Resource laws.** Least-squares fits over the eight compiled settings (rings and complete graphs, 2–28
  countries) give the laws below, each within 7% of every compiled count:

**Table [tab:qscale].** Fitted resource laws, value ≈ c₀ + c_n·n + c_E·|E|, over eight compiled settings.

| Resource | c₀ | per country c_n | per edge c_E | max deviation |
|---|---|---|---|---|
| logical qubits | 38 | 13.2 | 8.2 | 7.1% |
| Toffoli gates per A | -26 | 106.0 | 39.0 | 6.2% |
| single-qubit rotations per A | 9 | 8.0 | 64.1 | 0.9% |
| CNOTs per Grover iterate Q | 181 | 1600.0 | 783.0 | 5.0% |
| CNOT depth of A | 378 | 0.3 | 14.7 | 5.8% |

* **Depth.** Source registers load in parallel and their adders commute, so the depth of $A$ grows with the
  number of shared-event edges but not with the number of countries.
* **Connectivity.** A shared source couples two country registers. On a 2D nearest-neighbour lattice, a dense
  dependence graph needs long-range gates: SWAP routing inflates CNOT counts by 10–40×, and lattice surgery
  inflates time instead. Sparse geographic graphs (bandwidth 5 for the 28-country pool, §6.7) map onto a
  line of country blocks.
* **Parallelism.** Amplitude-estimation rounds at different Grover powers are independent, so $m$ QPUs cut the
  wall-clock by up to $m$. A single Grover iterate cannot be parallelised.
* **Where the quantum stage could win.** The phase diagram (Figure [fig:qphase]) places every measured
  portfolio. RDPO's measured relative variances are $\rho^2 =$ 0.13–117, while break-even at 1% error needs
  $\rho^2 \approx$ 5.8×10⁴–9.6×10⁵ at the stated fault-tolerant baseline. Against plain Monte Carlo, every
  rare target would sit deep in the quantum region. This is the misleading comparison that Proposition 7's
  corollary rules out.

![**Figure [fig:qphase].** (a) Wall-clock phase diagram. Quantum amplitude estimation beats classical RDPO above
the break-even band (which spans the eight compiled settings), which moves down by two orders of magnitude
for every 100× of hardware speed-up. Filled markers: RDPO's measured variance at ε = 10%. Open markers:
plain Monte Carlo on the same targets. (b) Logical qubits, Toffolis per state preparation and depth of $A$
versus portfolio size.](../../figures/paper/quantum_phase.png){.wide}

## 7.6 A falsifiable wall-clock criterion

Query counts are not wall-clock time. For a target relative error $\varepsilon$ we compare measured
classical time with projected fault-tolerant time:

$$T_{\text{RDPO}}(n) = \frac{\rho^2(n)}{\varepsilon^2}\, t_{\text{path}}(n), \qquad T_{Q}(n) = \text{runtime of IQAE on the surface-code model}, \qquad s^*(n) = \frac{T_Q(n)}{T_{\text{RDPO}}(n)} .$$

Here $\rho^2$ is RDPO's measured per-path relative variance and $t_{\text{path}}$ its measured time per path, both
with training included. **Q-RDPO beats RDPO in wall-clock if and only if the quantum machine is $s^*$ times
faster than the stated baseline** (physical error $10^{-3}$, 1 µs code cycles, 4 magic-state factories).

![**Figure [fig:qadv].** (a) Hardware speed-up $s^*$ required for the quantum stage to match classical RDPO
in wall-clock time at 10% relative error. (b) Logical two-qubit error rate required for the deepest
amplitude-estimation circuit to succeed with probability one half.](../../figures/paper/quantum_advantage.png){.wide}

**Table [tab:qadv].** Falsifiable wall-clock comparison at 10% relative error on Pr(L > Tₙ). Classical: measured single-core RDPO time (training included). Quantum: projected IQAE runtime on the stated surface-code baseline. s*: hardware speed-up required for break-even. p_L: logical two-qubit error rate required by the deepest circuit.

| Graph | n | μ | twisted amplitude a | RDPO paths needed | RDPO time | AE calls | max Grover depth | quantum time | s* | p_L required |
|---|---|---|---|---|---|---|---|---|---|---|
| ring | 2 | 1.2×10⁻² | 0.072 | 13 | 0.6 ms | 4,290 | 16 | 0.7 h | **4.4×10⁶** | 9.9×10⁻⁶ |
| ring | 8 | 3.9×10⁻⁵ | 0.025 | 60 | 16.2 ms | 7,075 | 27 | 9.0 h | **2.0×10⁶** | 1.3×10⁻⁶ |
| ring | 16 | 1.5×10⁻⁸ | 0.006 | 210 | 92.7 ms | 8,650 | 32 | 24.3 h | **9.4×10⁵** | 5.5×10⁻⁷ |
| ring | 20 | 2.9×10⁻¹⁰ | 0.003 | 520 | 320.2 ms | 11,015 | 42 | 39.2 h | **4.4×10⁵** | 3.4×10⁻⁷ |
| ring | 28 | 1.3×10⁻¹³ | 0.001 | 11,717 | 11.7 s | 46,685 | 183 | 11 d | **8.2×10⁴** | 5.6×10⁻⁸ |
| complete | 4 | 2.1×10⁻³ | 0.042 | 55 | 13.0 ms | 6,075 | 23 | 5.4 h | **1.5×10⁶** | 2.5×10⁻⁶ |
| complete | 6 | 6.3×10⁻⁴ | 0.035 | 303 | 277.2 ms | 6,280 | 24 | 14.7 h | **1.9×10⁵** | 1.3×10⁻⁶ |
| complete | 7 | 2.9×10⁻⁴ | 0.025 | 144 | 202.8 ms | 7,585 | 29 | 24.9 h | **4.4×10⁵** | 8.4×10⁻⁷ |

* **Result.** In every setting the quantum stage is slower: break-even needs a hardware speed-up between 8.2×10⁴ and 4.4×10⁶. The reason is that RDPO needs only 13–11,717 paths for 10% error, so the quadratic query advantage has almost nothing to act on.
* **Trend.** The gap closes with portfolio size: on the ring, s* falls from 4.4×10⁶ (n = 2) to 8.2×10⁴ (n = 28), because RDPO's per-path variance grows (ρ² from 0.13 to 117) while the twisted amplitude decays only slowly.
* **Hardware requirement.** The deepest circuit at n = 28 has Grover depth 183, 635 logical qubits and about 1.2 million physical qubits, and needs a logical error rate of 5.6×10⁻⁸.
* **Falsification statements.** The quantum stage is *not* useful for a portfolio class if (i) s* stays above the projected speed-up of available fault-tolerant hardware, or (ii) RDPO's per-path relative variance ρ² stays below about 10³ as portfolios grow. Conversely, a target class on which RDPO's ρ² grows quickly (dense dependence, many-country compound triggers) is where the stage should be re-tested first (WP4).
* **What the comparison already includes.** The classical side is our *best* method, timed with training; the quantum side includes state preparation, the weight oracle and all IQAE rounds. Comparing against plain Monte Carlo instead would show a misleading quantum advantage of several orders of magnitude.

## 7.7 Noise, mitigation and a calibration trust boundary

Error mitigation is not monotone: under miscalibration it can increase error [@errmit2026]. We test this
directly on low-depth maximum-likelihood amplitude estimation.

* **Model.** Grover powers $k = 0, 1, 2, 4, \dots$ (depth capped by circuit fidelity), 100 shots each, with
  depolarising decay $f$ and a coherent control offset $\delta$ per Grover iterate:
  $P_k = \tfrac12 - \tfrac12 f^{2k+1}\cos\big((4k+2)\theta + 2k\delta\big)$.
* **Estimators.** (i) no mitigation ($f = 1$, $\delta = 0$); (ii) noise-aware mitigation (fit $f$, assume $\delta = 0$);
  (iii) + goodness-of-fit guard (likelihood-ratio test across depths); (iv) + pilot-disagreement guard,
  which also rejects the quantum estimate when it disagrees by more than 3σ with a cheap RDPO pilot
  (5% error). Rejected estimates fall back to the full classical RDPO estimate.
* **Instances.** 250 per offset level; amplitude $a \in [10^{-3}, 10^{-1}]$, two-qubit error $p_2 \in [10^{-7}, 10^{-5}]$.

![**Figure [fig:mit].** Tail of the relative error over random instances at three levels of control-offset
miscalibration. Mitigation removes the worst raw errors but is worse than no mitigation in a growing
fraction of instances; the combined guard bounds the tail.](../../figures/paper/mitigation.png){.wide}

**Table [tab:mit].** Error mitigation under control-offset miscalibration (250 random instances per level; relative error of the estimated amplitude). Guard 2 combines the goodness-of-fit and pilot-disagreement guards with fallback to RDPO.

| offset δ_max | mitigation worse than none | p99 error: none | mitigated | guard 2 | max error: none | mitigated | guard 2 | fallback rate |
|---|---|---|---|---|---|---|---|---|
| 0 | 21% | 0.91 | 0.45 | **0.17** | 1.37 | 0.63 | **0.20** | 10% |
| 0.003 | 24% | 1.63 | 0.41 | **0.15** | 3.21 | 1.00 | **0.16** | 12% |
| 0.01 | 38% | 1.66 | 0.41 | **0.17** | 2.58 | 0.47 | **0.20** | 22% |

* **Mitigation is not monotone.** Noise-aware fitting removes the worst raw errors, but it is worse than no mitigation in 21% of instances without miscalibration, rising to 38% at δ_max = 0.01. This reproduces the qualitative finding of [@errmit2026] on our own workload.
* **Goodness of fit alone is not enough.** The likelihood-ratio guard triggers in only 2%–4% of instances: a coherent offset is largely absorbed by a shifted amplitude, so the fit still looks good.
* **An independent classical check bounds the tail.** Adding the pilot-disagreement test caps the 99th-percentile error at 0.15–0.17 and the maximum at 0.20 across all offset levels, at a fallback rate of 10%–22%. The median error is unchanged or better.
* **Design rule.** The quantum estimate is accepted only when it is consistent with an unbiased classical pilot. This is where RDPO and the quantum stage meet: the classical RL estimator is the calibration anchor, not a competitor to be hidden.

## 7.8 Verification, near-term component and AI-for-Quantum roadmap

* **Verification protocol.** (i) On instances small enough for exact classical simulation (up to three
  countries), hardware estimates are compared with the exact value. (ii) Every production run is checked
  against the classical RDPO pilot (Algorithm 2, step 6). (iii) Raw counts per Grover power and the fitted
  noise parameters are published with each estimate.
* **Near-term sampler.** An 8-qubit circuit Born machine (depth 9, 32 entangling gates) can serve as a
  proposal sampler. Hardware noise biases the importance weights by 9–60%. Calibrating the *sampled* law
  with $10^5$ shots reduces the bias to at most 0.6%; inverting readout errors is the wrong mitigation in
  this setting.
* **AI-for-Quantum work in Phase 2** (in the spirit of [@guo2026stopqml]): an RL agent that chooses Grover
  depth schedules under a noise budget; learned trust guards trained on calibration drift; and RL-driven
  compilation of the source registers. All three use the same machinery as RDPO.

::: {.takeaway}
The quantum stage is exact, its data loading is cheap, and nothing is trained on the device. It still needs
a machine about 8.2×10⁴ faster than today's fault-tolerant assumptions, and a logical error
rate near 5.6×10⁻⁸ (28 countries). We state both as testable targets.
:::

# 8 Resource Estimation

::: {.opener}
![](../art/cat_stopwatch.svg){.mascot}

What it costs to run: seconds on one CPU core today; a fault-tolerant quantum computer far beyond today's assumptions.
:::


## 8.1 Classical (the Phase 1 deliverable and the core of Phase 2)

| Task | Hardware | Wall time | Memory |
|---|---|---|---|
| Exact benchmark model (5 scenarios) | 1 CPU core | 0.2 ms / scenario | < 1 MB |
| RDPO estimate, benchmark, B = 16,384 (training included) | 1 core | 0.4 s | < 100 MB |
| RDPO estimate, 28 countries, B = 65,536 | 1 core | 65 s | < 1 GB |
| Exact tilted tensor network, 28 countries (ring) | 1 core | 68 s | < 1 GB |
| Full reproduction of all experiments | 4 cores, 15 GB RAM | ≈ 7 CPU-hours (≈ 3 h wall-clock) | < 4 GB |

## 8.2 Quantum (projected; future feature)

Logical-level estimates from explicit component formulas, not compiled circuits. Assumptions: Toffoli = 4 T
= 6 CNOT; Ross–Selinger rotation synthesis; surface code at physical error $10^{-3}$, 1 µs cycles, 4
magic-state factories; twisted loading with the bitwise oracle; 10% target relative error.

**Table [tab:qres].** Projected logical and physical resources per amplitude-estimation run (10% relative error, twisted loading with the bitwise weight oracle). A: one state preparation; Q: one Grover iterate.

| Graph | n | logical qubits | Toffoli (A) | rotations (A) | CNOT (Q) | lookup Toffolis avoided | code distance | physical qubits | runtime |
|---|---|---|---|---|---|---|---|---|---|
| ring | 2 | 68 | 236 | 88 | 4,212 | 272 | 25 | 109,000 | 0.7 h |
| ring | 8 | 209 | 1,092 | 586 | 18,720 | 1,556 | 27 | 328,722 | 9.0 h |
| ring | 16 | 381 | 2,280 | 1,163 | 38,148 | 3,094 | 29 | 664,842 | 24.3 h |
| ring | 20 | 465 | 2,840 | 1,451 | 47,428 | 3,862 | 29 | 806,130 | 39.2 h |
| ring | 28 | 635 | 4,074 | 2,028 | 67,360 | 5,400 | 31 | 1,244,470 | 11 d |
| complete | 4 | 145 | 674 | 426 | 11,872 | 1,044 | 27 | 235,410 | 5.4 h |
| complete | 6 | 242 | 1,212 | 1,019 | 21,752 | 2,326 | 29 | 431,044 | 14.7 h |
| complete | 7 | 300 | 1,520 | 1,411 | 27,596 | 3,158 | 29 | 528,600 | 24.9 h |

* **Error mitigation:** noise-aware likelihood plus the two-part trust guard (§7.4); calibrated
  sampled-law weights for the near-term sampler; post-selection of invalid register codes.
* **Hardware modality:** dense dependence graphs need all-to-all coupling between country registers,
  which favours trapped-ion and reconfigurable neutral-atom platforms. SWAP routing on 2D superconducting
  lattices inflates CNOT counts by 10–40×.

# 9 Work Plan for the Deep Development Phase {.newpage}

::: {.opener}
![](../art/cat_calendar.svg){.mascot}

Twelve weeks, six work packages, and a measurable success criterion for each.
:::


![**Figure [fig:gantt].** Work plan in weeks from the start of the Deep Development phase. The indicative
calendar maps W1 to November 2026; the plan shifts one-to-one if the phase starts later.](../../figures/paper/gantt.png){.wide}

| Work package | Milestone | Success criterion |
|---|---|---|
| WP1 Engineering (W1–2): GPU-batched RDPO, library API, sandbox onboarding. *CPU part done (§6.5): 6.5–9.9× rollout throughput, identical policy* | M1 library v0.1 | prototype results reproduced; ≥ 10× throughput |
| WP2 Realistic portfolios (W3–6): public hazard data, heterogeneous perils, dense dependence, 10–50 countries. *First results in §6.6–6.7: calibrated 5- and 10-country pools* | M2 scaling report | ≤ 10% relative RMSE at μ ≤ 10⁻⁸ within 10⁵ paths |
| WP3 RL improvements (W5–8): small neural soft values, adaptive levels, robust error bars, model selection. *First results in §6.5 (Phase 2 engine)* | M3 Phase 2 engine | ≥ 2× over Phase 1 on dense graphs; coverage ≥ 93% |
| WP4 Quantum stage and AI-for-Quantum (W8–10): compiled loader; guarded estimation on emulators or hardware; RL depth schedules | M4 compiled resource tables + updated s* map | compiled counts within 2× of estimates; guard caps tail error |
| WP5 Web application and agents (W9–11): dashboard, news-scout and data agents, advisory agent, MCP tool server | M5 public demo | three end-to-end analyst workflows |
| WP6 Validation and delivery (W11–12): independent checks, documentation, business case | M6 final deliverables | all deliverables of the call |

| Risk | Mitigation and fallback |
|---|---|
| Dense-dependence cases also turn out to be classically easy | Publish the crossover map either way; RDPO remains the product. |
| Training instability on new targets | Warm starts, Rényi-2 repair, model selection; estimates stay unbiased regardless. |
| Quantum hardware unavailable or too noisy | The trust guard falls back to RDPO; the quantum stage remains a tested design. |
| Single-person team capacity | Modular packages; WP4 is descoped first, then WP5. |

# 10 Commercialisation and Product {.newpage}

::: {.opener}
![](../art/cat_briefcase.svg){.mascot}

Who would use this, how it fits their day, and how it reaches the market.
:::


::: {.advisory}

::: {.exec}
## Executive summary

* **What we sell.** A joint-tail analytics service: open hazard data and daily news flow into an RDPO
  estimation engine. An advisory agent turns the results into evidence-backed recommendations,
  delivered as a web application and as Model Context Protocol (MCP) tools.
* **Who buys.** Business-to-business and business-to-government. The buyers are development banks and
  sovereign risk pools, reinsurers, ILS investors, brokers, model vendors and parametric insurers. We do
  not sell to consumers; they benefit indirectly through cheaper pooled and parametric cover.
* **How big.** Within a US$2.1bn catastrophe-modelling software category, we size a bottom-up serviceable
  market of about US$30m a year and target US$3.5m of annual recurring revenue (ARR) from
  28 institutions by year 5.
* **Why we win.** We are the only offering that pairs an *unbiased, validated* rare-event engine with an
  agent interface. Every number the agent reports comes from a tool call to the engine, never from the
  language model.
:::

## 10.1 From data to decision: end-to-end data flow

![**Figure [fig:flow].** End-to-end data flow of the planned product. Phase 1 has built the source model, the
RDPO engine, the exact checks and the diagnostics. Data agents, calibration from open data, the
advisory agent and the delivery layer are planned. Data sources are examples of open or licensable datasets, subject to licence terms.](../art/data_flow.svg){.wide}

| Stage | What happens | Status |
|---|---|---|
| 1 Data | Rainfall (ERA5, GPM IMERG, CHIRPS); ensemble forecasts (Google WeatherNext 2); earthquake catalogues and hazard models (USGS, GEM); terrain (Copernicus DEM); exposure (WorldPop, OpenStreetMap); contract terms | contract terms used; open data planned |
| 2 Ingest | *News-scout agent*: scrapes daily news and disaster reports, extracts event type, place and severity, geocodes and scores credibility. *Data agent*: pulls feeds, runs quality checks, keeps versioned snapshots | planned (WP2, WP5) |
| 3 Calibrate | Fit event rates and severities per peril and country; build the source laws $p_k$ and the dependence graph; generate stress and forecast-conditioned scenarios | source laws and graph built; fitting planned |
| 4 Estimate | RDPO learned change of measure; unbiased estimate with 95% interval; exact checks and diagnostics; GPU batching; guarded quantum back end in the future | **built** (GPU planned, quantum future) |
| 5 Advise | Advisory agent plans stress tests and drafts recommendations, calling the MCP tools `estimate_joint_tail`, `stress_test`, `validate_exact`, `event_impact` | planned (WP5) |
| 6 Deliver | Joint-tail dashboard, morning risk brief, recommendation memos, API and MCP access for Excel, Oasis and other agents | planned (WP5) |

* **Same-day event response.** When the news-scout agent flags a breaking event (for example a
  magnitude-6 earthquake near a covered region), the advisory agent re-estimates the affected joint tails
  and trigger probabilities the same day.
* **Governance by design.** A human approves every recommendation. Data, model and policy versions are
  logged, every estimate carries its uncertainty, and a validation pack supports model review by
  regulators and rating agencies.

## 10.2 Operating cycle: wake by day, sleep by night

![**Figure [fig:wake].** Wake–sleep operating cycle, adapted from the sleep framework for continual learning
[@behrouz2026sleep]. By day the system serves frozen, validated policies and writes a short-term memory.
Overnight it consolidates that memory into the risk model, retrains policies on self-generated stress
scenarios, and promotes them only after passing exact regression checks. Planned for WP3 and WP5; Phase 1 already provides the exact references and the warm-start mechanism.](../art/wake_sleep.svg){.wide}

* **Why a sleep phase.** Hazard information arrives continuously, but a risk number used for pricing must be
  reproducible. Separating *serving* (day) from *learning* (night) gives both.
* **Wake phase.** Agents ingest news and feeds and answer questions with the current policy library. Events,
  queries and sampled paths are logged as short-term memory. Nothing in the risk model changes during the day.
* **Consolidation.** Credible events update source rates through credibility-weighted (Bayesian) updates that
  an analyst approves. Newly observed shared events add dependence edges, and buffered paths warm-start the
  next policies.
* **Dreaming.** The system generates stress scenarios where the day's estimated χ² was highest (an RL
  curriculum) and retrains RDPO (KL → χ²) from yesterday's parameters. Capacity grows (more features) only
  when χ² stalls.
* **Gate against forgetting.** Before promotion, every new policy must reproduce the 25 exact ADB reference
  values and the tensor-network references within its intervals, and it must pass coverage checks.
  Otherwise the system rolls back.
* **Safety property.** RDPO is unbiased for *any* learned policy (Proposition 5), so a bad night can only
  widen intervals, never bias an estimate, and the gate catches it.

## 10.3 Market opportunity

::: {.tiles .market}
::: {.tile}
[US$61.3bn]{.big}
catastrophe bonds outstanding, end-2025: a record, up 24% in one year (Artemis)
:::
::: {.tile}
[US$19bn]{.big}
parametric insurance premiums, 2025, growing ≈ 12% a year (Global Market Insights)
:::
::: {.tile}
[US$181bn]{.big}
uninsured natural-catastrophe losses in 2024, the protection gap (Swiss Re Institute)
:::
:::

![**Figure [fig:market].** Total, serviceable and obtainable market. The TAM is a third-party estimate of the
catastrophe-modelling and exposure-management software market. The SAM and SOM are bottom-up:
institutions per segment × annual contract value, with all assumptions in
`src/business/market.py`. Sources: TrendX Insights and Market Intelo (software, 2025); Artemis (cat bonds, end-2025); Global Market Insights (parametric, 2025); Swiss Re Institute sigma 1/2025.](../art/market_sizing.svg){.wide}

* **TAM, US$2.1bn (2025).** Catastrophe-modelling and exposure-management software. Third-party estimates
  range from US$2.1bn to US$2.8bn, growing 9–12% a year.
* **SAM, ≈ US$30m a year.** 247 institutions that price, structure or hold
  *multi-region joint* catastrophe risk, at annual contract values of US$50–300k.
* **SOM, ≈ US$3.5m ARR by year 5.** 28 customers, about 11% of the SAM.
* **Upside not counted.** Correlated credit and supply-chain risk, compound climate extremes, and
  advisory fees on bond structuring.

## 10.4 Customers and business model

| Segment | Buyer | Pain point today | What we offer | Model |
|---|---|---|---|---|
| Development banks and sovereign pools (ADB, World Bank, CCRIF, ARC, SEADRIF) | disaster-risk-financing teams | joint tails of pooled bonds cannot be estimated from data; slow scenario studies | portfolio analytics, stress tests, trigger monitoring | **B2G**: technical-assistance contracts, enterprise licence |
| Reinsurers | catastrophe analytics, pricing | Monte Carlo on event loss tables is noisy in the joint tail | engine plug-in, faster and tighter tail metrics | **B2B**: enterprise subscription |
| ILS funds and cat-bond investors | portfolio managers, risk | correlation between bonds; pricing new issues | joint-tail dashboard, agent, morning brief | **B2B**: professional subscription |
| Brokers and model vendors | structuring, product teams | aggregation step is the accuracy bottleneck | licensed estimation layer, Oasis plug-in | **B2B**: licence and royalty |
| Parametric insurers and MGAs | product, actuarial | pricing multi-site and multi-peril triggers | API and MCP tools | **B2B**: usage-based API |

* **Why B2B and B2G, not B2C.** Joint-tail analytics are bought by institutions that hold risk capital.
  Consumers do not buy them, but they benefit *indirectly* (B2B2C): parametric and micro-insurance
  products priced with tighter tails can carry lower uncertainty loadings.
* **Land and expand.** We start with public institutions, where the challenge's use case lives and
  where validation and openness matter most. We then expand to private risk capital, where the budget
  sits.

| Tier | For whom | Includes | Indicative price |
|---|---|---|---|
| Open core | researchers, model validators | Python library, exact references, documentation | free (open source) |
| Professional | ILS funds, MGAs | web app, dashboard, advisory agent, 5 seats, daily brief | ≈ US$50–100k / year |
| Enterprise | reinsurers, development banks, pools | on-premise or private cloud, MCP server, validation pack, SLA, custom perils | ≈ US$150–300k / year |
| Advisory | bond issuers, sponsors | structuring and trigger-design studies with independent validation | per project |

## 10.5 Fit with the analyst's daily workflow

![**Figure [fig:day].** A day with the product. Agents prepare (news scan, hazard refresh, re-estimation);
the analyst decides (morning brief, questions, recommendation, sign-off). Illustrative; times are indicative.](../art/workflow_day.svg){.wide}

* **No new habit required.** The product produces a morning brief, answers questions in plain language,
  and drafts the memo the analyst already writes.
* **Fits existing systems.** The MCP and API layer exposes the same tools to Excel, Oasis pipelines and
  in-house agents.
* **Decision rights stay with people.** The agent never executes a trade, a trigger or a price. Every
  recommendation is approved and logged with its data and model versions.

## 10.6 Go-to-market

![**Figure [fig:gtm].** Go-to-market in three phases, from public pilots to private-capital
scale. Phases are indicative and follow the work plan of Section 9.](../art/gtm.svg){.wide}

| Competitor type | Examples | Their strength | Our position |
|---|---|---|---|
| Catastrophe-model vendors | commercial vendor platforms, Oasis LMF | hazard science, exposure data, market acceptance | **complement**: we replace the plain-Monte-Carlo aggregation step and plug in via event or source output |
| Risk analytics and ILS platforms | in-house broker and fund tools | distribution, client relationships | licence partner; joint-tail accuracy as a differentiator |
| General AI assistants | chat-based copilots | language interface | our agent only reports engine outputs with intervals; no hallucinated numbers |

* **Defensibility.** The method and its validation suite (unbiasedness proof, exact references,
  coverage tests); the source-level data model; and the agent protocol, which is hard to replicate
  without the engine underneath.

## 10.7 Commercial risks and impact

| Risk | Mitigation |
|---|---|
| Long institutional sales cycles | start with grant-funded pilots; open core builds trust and adoption |
| Data licensing for commercial use | open datasets first; licensed feeds per client contract |
| Model-risk scrutiny of AI agents | numbers only from validated tools; human sign-off; audit trail |
| Incumbent vendors build the same capability | partner and licence early; Oasis plug-in keeps us inside the workflow |

* **Impact.** Tighter joint-tail estimates reduce uncertainty loadings in pricing, so pooled cover becomes
  cheaper for vulnerable countries, and trigger decisions become easier to defend (UN SDG 13: climate
  action).
* **Broader applicability.** Correlated credit portfolios, parametric pandemic and supply-chain cover,
  compound climate extremes, infrastructure reliability.

::: {.takeaway}
Open data and daily news in; validated joint-tail numbers and human-approved recommendations out. We sell
business-to-business and business-to-government, starting with public pilots.
:::

:::

# 11 Team {.newpage}

::: {.opener}
![](../art/cat_team.svg){.mascot}

One researcher, with a background that spans quantum information, reinforcement learning and stochastic climate models.
:::


::: {.profile}
**Tan Chun Loong**, sole member: research, implementation, quantum resource estimation, product and
business case.

Bachelor of Applied Physics with a Second Major in Quantum Technology, Nanyang Technological University
(2024–2028; CGPA 4.8/5.0, First Class Honours).
:::

* **Incoming Phortify Student Research Fellow, ICFO – The Institute of Photonic Sciences (Barcelona).** One of
  three students selected worldwide for this funded research fellowship. Hosted by the Quantum Information Theory
  group of ICREA Professor Antonio Acín, a leading group in quantum information and device-independent quantum
  cryptography.
* *URECA Research Fellow, "The Thermodynamics of Mitigating Ignorance" (Prof. Mile Gu, NTU, 2026–2027).*
  This project studies the cost of acting on beliefs that differ from reality, measured by the relative
  entropy between belief and truth. It also covers optimal belief revision by dynamic programming for
  model-based reinforcement learning. RDPO rests on the same mathematics: backward induction for value
  functions, and divergences between a policy and the optimal measure.
* *Quantum Security Research Intern, QUASAR-CREATE Thrust 2, Fraunhofer Singapore (2026–).* Security
  evaluation of quantum key distribution under device imperfections.
* *URECA Research Fellow, atmosphere–ocean interaction models (Prof. François Gay-Balmaz, NTU, 2025–2026).*
  Fokker–Planck equations and stochastic thermodynamics for reduced climate models.
* *Selected projects:*
  * Clifford+T compilation and T-count optimisation (MIT iQuHACK 2026, sole Singapore representative).
  * Quantum optimisation for wildfire response (SEA Quantathon 2025 finalist).
  * Quantum reservoir computing for sustainable-finance systemic risk (QAI Ventures GenQ Hackathon 2025).
  * AI risk scoring for maritime fuel verification (winner, SuperAI NEXT Hackathon 2026).
* *Community:* founder of Quantum Wings (quantum outreach to more than 500 students); Country Lead
  (Singapore), Qiskit Fall Fest 2026.
* **Why this profile fits.** Quantum information theory (ICFO, QKD security) underpins the quantum stage.
  Relative entropy and dynamic programming (NTU) underpin RDPO. Stochastic climate models connect to the
  hazard side, and hackathon product delivery connects to the agent product.

A short CV is attached as an annex.

# 12 Platform Requirement, Prototype Package and AI Use

::: {.opener}
![](../art/cat_laptop.svg){.mascot}

Everything runs on a laptop-class CPU. The code, data and this document are open.
:::


* **Platform.** No platform access is required at this stage; all results were produced on the author's
  own compute (4-core CPU, no GPU, no quantum hardware). In Deep Development the QAI Ventures sandbox
  would serve GPU-batched RDPO sweeps, compilation of the loading circuits, and emulator or hardware runs.
* **Prototype package.** Source code, results and this document: <https://github.com/CL0908/adb-quantum-challenge>.
  Python 3.13, NumPy 2.5, SciPy 1.18, Matplotlib 3.11, pytest 9.2 (`requirements.txt`); Linux x86-64. All
  quantum circuits are simulated by exact simulators written for this project; no quantum SDK or hardware
  was used. Every experiment uses fixed seeds recorded in its output, and a test suite checks every
  exactness claim.
* **AI use.** The code, experiments and this document were developed with Claude Code (Anthropic) as an AI
  pair-programmer and writing assistant, following the author's research plan. The author directed the
  research questions and design decisions and reviewed all results. All numbers are produced by the
  scripts in the repository.

# Appendix A: Full Phase 2 scaling table {.newpage}

**Table [tab:v2scale].** Scaling study at B = 65,536 with 24 replications per cell and identical seeds for every method. Relative RMSE, with empirical 95%-interval coverage in parentheses. *Phase 1 study*: the original results (6–20 replications). Pair exhaustion is answered exactly by the Phase 2 engine (pair reduction); its sampled estimate is shown for comparison.

| Graph | n | target | Phase 1 study | Phase 1 engine | Phase 2 engine | exp. twisting | time per estimate |
|---|---|---|---|---|---|---|---|
| ring | 2 | Pr(L > T) | 0.14% (95%) | 0.11% (100%) | **0.11%** (96%) | 0.79% | 3 s → 0.3 s |
| ring | 2 | pair exhaustion | 0.34% (90%) | 0.19% (96%) | **exact** (2 ms); sampled 0.19% | — | 4 s → 0.3 s |
| ring | 2 | stop-loss | 0.14% (95%) | 0.14% (100%) | **0.14%** (96%) | 0.66% | 3 s → 0.3 s |
| ring | 8 | Pr(L > T) | 0.30% (100%) | 0.50% (92%) | **0.44%** (96%) | 1.45% | 18 s → 5.7 s |
| ring | 8 | pair exhaustion | 1.00% (90%) | 0.36% (92%) | **exact** (1 ms); sampled 0.36% | — | 4 s → 1.0 s |
| ring | 8 | stop-loss | 0.43% (100%) | 0.54% (88%) | **0.35%** (92%) | 1.14% | 18 s → 6.3 s |
| ring | 16 | Pr(L > T) | 0.57% (100%) | 0.69% (100%) | **0.64%** (96%) | 2.70% | 29 s → 11.4 s |
| ring | 16 | pair exhaustion | 0.88% (100%) | 0.64% (96%) | **exact** (2 ms); sampled 0.64% | — | 6 s → 1.5 s |
| ring | 16 | stop-loss | 0.63% (90%) | 0.79% (92%) | **0.47%** (100%) | 2.36% | 29 s → 12.5 s |
| ring | 20 | Pr(L > T) | 0.89% (100%) | 0.97% (96%) | **0.95%** (96%) | 3.76% | 40 s → 16.7 s |
| ring | 20 | pair exhaustion | 0.56% (100%) | 0.51% (100%) | **exact** (3 ms); sampled 0.51% | — | 6 s → 1.8 s |
| ring | 20 | stop-loss | 0.57% (100%) | 0.81% (96%) | **0.72%** (96%) | 3.48% | 42 s → 18.4 s |
| ring | 28 | Pr(L > T) | 4.23% (100%) | 1.74% (92%) | **1.32%** (100%) | 7.10% | 65 s → 23.8 s |
| ring | 28 | pair exhaustion | 0.41% (83%) | 0.49% (100%) | **exact** (2 ms); sampled 0.49% | — | 9 s → 2.2 s |
| ring | 28 | stop-loss | 5.89% (83%) | 1.24% (96%) | **0.79%** (96%) | 4.90% | 67 s → 24.5 s |
| complete | 4 | Pr(L > T) | 0.29% (100%) | 0.44% (83%) | **0.28%** (96%) | 1.38% | 15 s → 2.9 s |
| complete | 4 | pair exhaustion | 0.42% (95%) | 1.00% (83%) | **exact** (3 ms); sampled 1.00% | — | 4 s → 1.0 s |
| complete | 4 | stop-loss | 0.28% (95%) | 0.36% (96%) | **0.30%** (92%) | 0.78% | 16 s → 3.0 s |
| complete | 6 | Pr(L > T) | 0.68% (60%) | 0.38% (92%) | **0.28%** (96%) | 1.05% | 60 s → 14.3 s |
| complete | 6 | pair exhaustion | 1.14% (100%) | 4.13% (92%) | **exact** (3 ms); sampled 4.13% | — | 8 s → 1.8 s |
| complete | 6 | stop-loss | 0.48% (90%) | 0.34% (100%) | **0.43%** (92%) | 0.83% | 54 s → 14.0 s |
| complete | 7 | Pr(L > T) | 0.47% (100%) | 0.47% (96%) | **0.50%** (88%) | 1.30% | 92 s → 25.3 s |
| complete | 7 | pair exhaustion | 3.35% (80%) | 2.92% (83%) | **exact** (3 ms); sampled 2.92% | — | 9 s → 2.3 s |
| complete | 7 | stop-loss | 0.59% (90%) | 0.54% (96%) | **0.42%** (100%) | 1.05% | 91 s → 25.5 s |

# References {.newpage}

::: {#refs}
:::

::: {.closing}
![](../art/cat_sleep.svg){.mascot-end}
:::
