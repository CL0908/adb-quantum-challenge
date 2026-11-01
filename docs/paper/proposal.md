---
title: "Information-Theoretic Reinforcement Learning for Joint Catastrophe Tails"
subtitle: "Quantum Enhanced Rényi-Doob Policy Optimization"
author: "Tan Chun Loong"
affiliation: "Nanyang Technological University, Singapore"
date: "November 2026"
lang: en
---

::: {.frontmatter}

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

With training included, RDPO attains {{E1_EXH_MED}} median relative error on the ADB joint-exhaustion
probability at 16,384 paths, a {{E1_EXH_RANGE}} improvement over the best reference method. It keeps
errors of {{E2_RANGE}} as the target probability falls to 10⁻¹³ in a 28-country portfolio. The same
learned change of measure loads *exactly* into an amplitude-estimation circuit. We assess this quantum
stage with a falsifiable wall-clock criterion and a calibration trust guard, and we claim no quantum
advantage in Phase 1.
:::

::: {.glance}
## At a glance

::: {.tiles}
::: {.tile}
[{{E1_EXH_RANGE_SHORT}}]{.big}
lower error than the best reference method on joint exhaustion (training included)
:::
::: {.tile}
[10⁻¹³]{.big}
smallest target probability estimated, at {{E2_RING28_EXC}} relative RMSE (28 countries)
:::
::: {.tile}
[{{E1_COVERAGE}}]{.big}
empirical coverage of nominal 95% intervals over 75 benchmark settings; the estimator is provably unbiased
:::
::: {.tile}
[{{Q_SSTAR_28}}]{.big}
hardware speed-up a fault-tolerant machine would need for quantum to beat RDPO in wall-clock (28 countries): a falsifiable target
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

* **Budget.** Every method receives the same total number of sampled term scenarios $B$; RDPO's training
  paths count against $B$.
* **Metrics.** Relative RMSE against the exact value over independent replications (40 for the
  benchmark, 6–20 at scale), relative bias, and the empirical coverage of nominal 95% intervals.
* **References.** Benchmark: exact enumeration, which reproduces the brief's Table 5. Scaling: exact
  tensor networks in a tail-preserving tilted gauge (§6.4), validated against dense enumeration for $n \le 6$.
* **Baselines.** The challenge's four reference methods: plain Monte Carlo, scrambled Sobol (8 scrambles),
  stratification by first payout year, and the brief's importance sampling. At scale we add classical
  exponential twisting [@glasserman2003].
* **Compute.** One CPU core per run (Intel Xeon 2.1 GHz); NumPy/SciPy; fixed seeds in every result file.

# 6 Results {.newpage}

## 6.1 Benchmark: five scenarios × five targets

![**Figure [fig:e1].** Benchmark at a total budget of 16,384 paths, training included. Left: RDPO's
relative RMSE per cell. Right: improvement over the best of the four reference methods in the same cell
(>1: RDPO is more accurate).](../../figures/paper/e1_heatmap.png){.wide}

{{TABLE_E1}}

{{E1_TEXT}}

![**Figure [fig:e1budget].** Relative RMSE versus total budget in the base scenario, with RDPO's
training included. Grey dashes show the $N^{-1/2}$ slope.](../../figures/paper/e1_budget.png){.wide}

## 6.2 Scaling to large portfolios

![**Figure [fig:e2].** Scaling at a total budget of 65,536 paths. (a) Pool exceedance $\Pr(L > T_n)$ on a
ring: the target falls from $10^{-2}$ to $10^{-13}$. (b) Joint exhaustion of one country pair *inside* the
portfolio. Monte Carlo is shown by its theoretical error, since empirical runs return
zero.](../../figures/paper/e2_scaling.png){.wide}

{{TABLE_E2}}

{{E2_TEXT}}

## 6.3 Ablation study

![**Figure [fig:e3].** Ablation at a total budget of 16,384 paths. Each bar removes one component of
RDPO.](../../figures/paper/e3_ablation.png){.wide}

{{E3_TEXT}}

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

Sections 6.1–6.4 report the **Phase 1 engine** (RDPO v1). We then implemented the first work packages of
the Deep Development plan (WP1 engineering, WP3 RL improvements) as the **Phase 2 engine** (RDPO v2), and
re-ran every study at the same total budgets, with more replications and identical seeds for every method.

**Engineering (WP1).**

* **Table-driven engine.** Each country state is one of 10 codes, so every logit is a table lookup. The
  policy is *identical* to Phase 1: actions, weights and gradients match to machine precision (tested).
  Rollouts are {{V2_SPEEDUP}} faster on one core for 4–28 countries (Table [tab:v2tp]).
* **Exact compressed training.** Paths share few distinct local states at each source step, so the
  cross-entropy and Rényi-2 objectives are evaluated on unique states with aggregated weights. The
  objective is unchanged, and each evaluation is 50–500× cheaper.

{{V2_TP_TABLE}}

**Reinforcement-learning improvements (WP3).**

* **Stage-dependent soft values.** The exact value function $V_k$ depends on how many sources remain, but
  Phase 1 shares one progress parameter vector across all steps. Phase 2 gives each of 4 contiguous
  stages of source steps its own progress parameters (+9 parameters) for portfolio-wide targets.
* **Flexible soft values for smooth targets.** For expected loss, a piecewise-linear (hat-function)
  basis, a fixed-knot ReLU network in the loss, replaces the hand-made progress features. The
  cross-entropy fit stays concave.
* **Exact reduction for pair targets.** A target that depends on two countries only needs the joint law
  of their 100 states. That law is the product of the kernels of the sources touching them (Proposition 2).
  It is computed exactly in at most {{V2_EXACT_MS}} for any portfolio size, and agrees with the
  tensor-network references to {{V2_EXACT_ERR}}. The Phase 2 engine answers such targets exactly and
  reserves RDPO for portfolio-wide targets.
* **Safeguards.** Student-t intervals with Welch–Satterthwaite degrees of freedom. The initial (twist or
  transferred) policy is selected whenever its estimated χ² beats every trained iterate.

**Scaling study.**

![**Figure [fig:v2].** Phase 1 versus Phase 2 engine on the scaling study (B = 65,536; 24 replications;
identical seeds). (a, b) Relative RMSE for the portfolio-wide targets, with exponential twisting as the
classical reference. (c) Wall-clock time per estimate on one core.](../../figures/paper/v2_results.png){.wide}

{{V2_SCALE_TABLE}}

* **Accuracy.** On the portfolio-wide targets, at the same budget and seeds, the Phase 2 engine
  {{V2_GAIN}}. It beats exponential twisting in {{V2_TW_WINS}} settings.
* **Speed.** One estimate takes {{V2_TIME}} less wall-clock time than in the Phase 1 study.
* **Error bars.** With 24 replications, coverage of the Phase 2 intervals on portfolio-wide targets is
  {{V2_COV}}. Most low-coverage cells of the Phase 1 study (6–10 replications) were sampling noise.

**Benchmark.**

{{V2_BENCH_TABLE}}

* The Phase 2 engine keeps the 20/25 wins over the best reference method. On the non-pair cells it is on
  par with Phase 1 at 4,096 paths, and its advantage grows with the budget. Pair targets are answered
  exactly in the product. On expected loss the spline soft values close part of the gap to scrambled
  Sobol, but Sobol remains better.

**Transfer: the sleep phase in practice.** In the planned wake–sleep cycle (§10.2), today's estimate starts
from the policy trained overnight on yesterday's model. We emulate this by training on one model, then
reusing the parameters as the initial policy after a change of scenario or a 20% rate increase.

{{V2_TRANSFER_TABLE}}

* **Small budgets become usable.** Transfer improves accuracy by {{V2_TRANSFER_GAIN}}. At 2,048 paths, a
  cold start spends most of its budget reaching the tail, while the transferred policy is already there.
  The estimator remains unbiased, because the initial policy is fixed before sampling.

**Negative results (kept for the record).**

* A *value-of-training* stopping rule, which stops training when one more batch is not expected to pay
  for itself, slightly *increased* the error at two countries and was dropped.
* Spline soft values *hurt* the threshold targets (in development probes, up to ×19 on the stop-loss),
  so they are used for expected loss only.
* t-intervals and model selection changed no result in these studies: the effective degrees of freedom
  were large, and the trained policy always beat the initial one. They are kept as safeguards.

## 6.6 Real-world check of the model's assumptions

The benchmark model is calibrated to the brief's parameters, not to observations. We tested its earthquake
assumptions, the peril behind US$70m of each country's US$80m cover, against public data.

* **Data.** The USGS catalogue of significant earthquakes (M ≥ 5.5, 1965–2016) in the public mirror
  `plotly/datasets`, and country boundaries from `datasets/geo-countries`. Aftershocks are removed with
  Gardner–Knopoff windows, which leaves {{RD_N}} main shocks in the Central Asian window. Both files are cached
  in `data/real/`, and the analysis is `src/realdata/eq_check.py`.
* **Proxy.** An event "reaches" a country if its epicentre lies inside the country or within a given
  distance of its border (25, 50 or 100 km). This is a proxy for damage, not a payout calculation.

![**Figure [fig:rd].** (a) Main shocks of M ≥ 5.5 around the Kyrgyz Republic (blue outline) and Tajikistan
(teal outline), 1965–2016; red events lie within 50 km of both countries. (b) Share of each country's events
that also reach the other country, with 95% intervals, against the brief's 7.5% / 7.4%.](../../figures/paper/realdata_eq.png){.wide}

{{RD_TABLE}}

* **Poisson arrivals: supported.** Yearly main-shock counts have dispersion {{RD_DISP}} at 50 km, with
  every p-value ≥ {{RD_DISP_P}}. This is consistent with the compound-Poisson model.
* **Trigger size: plausible.** The Gutenberg–Richter slope is b = {{RD_B}}. The brief's annual 25%-payout
  probabilities then correspond to events of about {{RD_MSTAR}}: rare, damaging earthquakes, consistent with
  the bond's design.
* **Shared events: the brief's assumption looks too low.** {{RD_SHARE}} of one country's main shocks lie
  within damaging distance of the other country, against the brief's 7.5%. Seismicity concentrates along
  the shared Alay–Pamir border zone. Examples in the catalogue are 1974 M7.3 (Markansu), 1978 M6.8 (Alay),
  2008 M6.7 (Nura) and 2016 M6.6 (Murghob). Event years are also significantly dependent at 25–50 km.
* **Consequence for the bond.** Setting the shared-event intensity from these data changes the exact joint
  targets as follows:

{{RD_IMP_TABLE}}

* **Reading.** With the data-based shared intensity, the joint-exhaustion probability is {{RD_EXH_FACTOR}} the
  brief's base value, and joint attachment rises to {{RD_ATT}}. The brief's "common shock ×2" scenario sits at
  the mild end of this range. The finding supports the brief's position that dependence must be stress-tested,
  and it suggests testing intensities well above ×2.
* **Caveats.** Epicentre distance is not payout: parametric triggers depend on ground motion at the exposure.
  The catalogue covers 52 years and ends in 2016, and the M ≥ 6.5 rows rest on very few events. Rainfall
  could not be checked: the reanalysis services (ERA5 via Open-Meteo) are blocked in this environment, and
  checking them is a first task of WP2.

## 6.7 Limitations

* **Smooth targets.** On expected loss, scrambled Sobol remains more accurate and should be preferred.
* **Level functions.** The cross-entropy stage is indispensable for the rarest targets, so new targets
  need a sensible score function.
* **Error bars under heavy tails.** In the Phase 2 benchmark, coverage is {{V2_BENCH_COV}} per cell (40 replications); the lowest cells are
  rare targets with heavy-tailed weights. Heavy-tail diagnostics remain part of WP3.
* **Dense dependence.** Complete graphs are limited to 7 countries by the exact references. Larger dense
  portfolios need reference-free validation (WP2).
* **Synthetic scaling and partial real-data check.** Scaling results use a calibrated synthetic extension,
  and only the earthquake assumptions were checked against data (§6.6). Realistic multi-peril portfolios
  follow in WP2.

::: {.takeaway}
RDPO is the most accurate method on every rare portfolio-wide target we tested, up to 28 countries. The
Phase 2 engine is several times faster and up to {{V2_GAIN_MAX}} more accurate at scale, overnight transfer makes small
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
| Advantage relies on strong **input-loading assumptions** | Loading is *exact* and cheap: one small register per Poisson source, O(n + \|E\|) gates (Proposition 2, §7.2) |
| **Trainability** (barren plateaus, noisy gradients) | Nothing is trained on the quantum device; the classical RL agent learns the measure (AI-for-Quantum) |
| **Noise** makes deep circuits unreliable | Noise is modelled explicitly, and a calibration trust guard falls back to the classical estimate (§7.4) |
| No convincing **wall-clock** wins | A falsifiable wall-clock criterion against our own best classical method (§7.3) |

## 7.2 Construction: exact structured loading with bounded weights

![**Figure [fig:qcirc].** State-preparation operator A and the guarded estimation loop. ① Each source
register is prepared in the twisted law $\sqrt{q_k}$. ② In-place adders form per-country sums. ③ The loss
and threshold flag are computed. ④ The importance weight $e^{-\eta D}$ is written bit by bit onto ancillas.
⑤ Good states are marked. Amplitude estimation then runs with a trust guard.](../art/quantum_circuit_v2.svg){.wide}

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

## 7.3 A falsifiable wall-clock criterion

Query counts are not wall-clock time. For a target relative error $\varepsilon$ we compare measured
classical time with projected fault-tolerant time:

$$T_{\text{RDPO}}(n) = \frac{\rho^2(n)}{\varepsilon^2}\, t_{\text{path}}(n), \qquad T_{Q}(n) = \text{runtime of IQAE on the surface-code model}, \qquad s^*(n) = \frac{T_Q(n)}{T_{\text{RDPO}}(n)} .$$

Here $\rho^2$ is RDPO's measured per-path relative variance and $t_{\text{path}}$ its measured time per path, both
with training included. **Q-RDPO beats RDPO in wall-clock if and only if the quantum machine is $s^*$ times
faster than the stated baseline** (physical error $10^{-3}$, 1 µs code cycles, 4 magic-state factories).

![**Figure [fig:qadv].** (a) Hardware speed-up $s^*$ required for the quantum stage to match classical RDPO
in wall-clock time at 10% relative error. (b) Logical two-qubit error rate required for the deepest
amplitude-estimation circuit to succeed with probability one half.](../../figures/paper/quantum_advantage.png){.wide}

{{Q_ADV_TABLE}}

{{Q_ADV_TEXT}}

## 7.4 Noise, mitigation and a calibration trust boundary

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

{{MIT_TABLE}}

{{MIT_TEXT}}

## 7.5 Near-term component and AI-for-Quantum roadmap

* **Near-term sampler.** An 8-qubit circuit Born machine (depth 9, 32 entangling gates) can serve as a
  proposal sampler. Hardware noise biases the importance weights by 9–60%. Calibrating the *sampled* law
  with $10^5$ shots reduces the bias to at most 0.6%; inverting readout errors is the wrong mitigation in
  this setting.
* **AI-for-Quantum work in Phase 2** (in the spirit of [@guo2026stopqml]): an RL agent that chooses Grover
  depth schedules under a noise budget; learned trust guards trained on calibration drift; and RL-driven
  compilation of the source registers. All three use the same machinery as RDPO.

::: {.takeaway}
The quantum stage is exact, its data loading is cheap, and nothing is trained on the device. It still needs
a machine about {{Q_SSTAR_28}} faster than today's fault-tolerant assumptions, and a logical error
rate near {{Q_PL_28}} (28 countries). We state both as testable targets.
:::

# 8 Resource Estimation

## 8.1 Classical (the Phase 1 deliverable and the core of Phase 2)

| Task | Hardware | Wall time | Memory |
|---|---|---|---|
| Exact benchmark model (5 scenarios) | 1 CPU core | 0.2 ms / scenario | < 1 MB |
| RDPO estimate, benchmark, B = 16,384 (training included) | 1 core | {{E1_SEC_16K}} s | < 100 MB |
| RDPO estimate, 28 countries, B = 65,536 | 1 core | {{E2_SEC_28}} s | < 1 GB |
| Exact tilted tensor network, 28 countries (ring) | 1 core | 68 s | < 1 GB |
| Full reproduction of all experiments | 4 cores, 15 GB RAM | ≈ 7 CPU-hours (≈ 3 h wall-clock) | < 4 GB |

## 8.2 Quantum (projected; future feature)

Logical-level estimates from explicit component formulas, not compiled circuits. Assumptions: Toffoli = 4 T
= 6 CNOT; Ross–Selinger rotation synthesis; surface code at physical error $10^{-3}$, 1 µs cycles, 4
magic-state factories; twisted loading with the bitwise oracle; 10% target relative error.

{{Q_RES_TABLE}}

* **Error mitigation:** noise-aware likelihood plus the two-part trust guard (§7.4); calibrated
  sampled-law weights for the near-term sampler; post-selection of invalid register codes.
* **Hardware modality:** dense dependence graphs need all-to-all coupling between country registers,
  which favours trapped-ion and reconfigurable neutral-atom platforms. SWAP routing on 2D superconducting
  lattices inflates CNOT counts by 10–40×.

# 9 Work Plan for the Deep Development Phase {.newpage}

![**Figure [fig:gantt].** Work plan in weeks from the start of the Deep Development phase. The indicative
calendar maps W1 to November 2026; the plan shifts one-to-one if the phase starts later.](../../figures/paper/gantt.png){.wide}

| Work package | Milestone | Success criterion |
|---|---|---|
| WP1 Engineering (W1–2): GPU-batched RDPO, library API, sandbox onboarding. *CPU part done (§6.5): 6.5–9.9× rollout throughput, identical policy* | M1 library v0.1 | prototype results reproduced; ≥ 10× throughput |
| WP2 Realistic portfolios (W3–6): public hazard data, heterogeneous perils, dense dependence, 10–50 countries | M2 scaling report | ≤ 10% relative RMSE at μ ≤ 10⁻⁸ within 10⁵ paths |
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
  market of about US${{MK_SAM}}m a year and target US${{MK_SOM}}m of annual recurring revenue (ARR) from
  {{MK_CUST}} institutions by year 5.
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
* **SAM, ≈ US${{MK_SAM}}m a year.** {{MK_INST}} institutions that price, structure or hold
  *multi-region joint* catastrophe risk, at annual contract values of US$50–300k.
* **SOM, ≈ US${{MK_SOM}}m ARR by year 5.** {{MK_CUST}} customers, about {{MK_SHARE}} of the SAM.
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

# References {.newpage}

::: {#refs}
:::

::: {.closing}
![](../art/cat_sleep.svg){.mascot-end}
:::
