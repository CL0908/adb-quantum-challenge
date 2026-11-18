# 给作者的备注（2026-11-18，xzy 分支）

这些是 add-on 工作里得到、但需要你决定是否写进论文的东西。`src/` 和论文源文件都没有改。引用只核到摘要或检索结果的都标了「待核」，写进论文前请按原文核对。

## 1. 可以直接用的数字

| 内容 | 数字 | 来源文件 |
|---|---|---|
| 量子所需硬件加速 s*（28 国 ring） | 8.2e4 → 3.3e3（保守档）/ 2.9e3（mixed 档），8 个设定缩小 10.7–29.3 倍 | `results/combined.json` |
| 余震按单独事件赔付时，两债同时耗尽 | 2.5 倍（区间上限 4.7 倍） | `results/quality_eq_cluster.json` |
| 加上基础触发率的数据不确定性后，两债同时耗尽的 95% 区间 | 比只算共享比例宽约 3 倍 | `results/quality_rates_unc.json` |
| 两债同时耗尽的方差来源 | 81% 来自两国降雨 1-in-50 水平 | `results/voi.json` |
| CAREC 池超额 > US$240m | 点值 1.33e-3，后验预测 1.47e-3，95% [0.85e-3, 2.37e-3] | `results/model_risk.json` |
| 亚太池逐年更新 2017–2025 | 见 `README.md` 的风险孪生一节 | `results/twin.json` |
| 期望损失区间对应的利差带 | 见 `README.md` 的市场换算一节 | `results/spread_band.json` |
| 孪生的逐年回测 | 下一年主震数的 90% 预测区间覆盖率 91.0%（各国，252 个预测）/ 92.1%（国家对，198 个） | `results/twin_backtest.json` |
| 一次地震之后的再震预报 | 样本外对数得分增益 +0.38（30 天）/ +0.18（一年）；2026-11-03 边境 M 5.2 使到期前赔付级地震概率多 0.07 个百分点 | `results/event_response.json` |
| 各项 add-on 折成价格点 | 地震触发率数据区间 +7 / −15 点；余震读法 −3 点；+2°C −2 点；11 月地震 −0.1 点 | `results/live_mark.json` |

全部在你的 e6572b4 上重跑过，日志在 `repro/logs_upstream/`。

## 2. 为什么等 CPU 时间下 twisting 不输 RDPO：一个可以写进工作计划的假设

你在 §3 里坦白了等时间下 twisting 在多数格子不输。我们另外量了四个更重的设计，结果一致：

| 设计 | 等时间下相对 twisting |
|---|---|
| cutset 条件化（你的 `src/tensor/cutset.py`，按已有原始结果推算） | 约差 10 倍 |
| 全边条件化（只采样边源，各国精确卷积） | 效率 0.1–0.4 倍 |
| 简化图精确解作控制变量 | 方差最多降 1.1–1.8 倍 |
| 重加权孪生里用 RDPO 锚点代替 twisting 锚点 | 等 CPU 下误差大 1.2–1.7 倍 |

文献给出的解释（待核）：固定的指数倾斜会失效、必须用状态依赖方案的条件是

1. 稀有事件集合不是半空间，到达失效有多条差别很大的路径（Glasserman & Wang 1997 的反例）；
2. 动力学依赖状态（Dupuis & Wang 的 subsolution 框架；Dupuis, Sezer & Wang 的排队网络例子里标准启发式方差可以无穷）；
3. 合同有路径依赖条款，例如多年期巨灾债的年度重置（reset）或一个季节内多次事件累计的聚合触发。

现在的模型是独立源、有上限的轻尾增量、目标是总损失超阈值，三条都不满足，所以解析倾斜本身已接近最优的 Doob 变换。

建议的写法：把「当合同含年度重置或聚合触发，或者目标是多国同时耗尽这类非半空间事件时，固定倾斜退化，RDPO 才必要」写成第二阶段的可检验假设。这同时解释了 §3 的等时间结果。

## 3. 量子部分建议补的两篇直接相关的 2026 年预印本（待核）

- arXiv:2603.15664：用振幅估计做巨灾保险尾部定价（仅模拟，4–9 qubit）。等预算下重要性采样和 QMC 远胜振幅估计，离散化误差大于估计误差。
- arXiv:2608.26840：图模型稀有事件，用 belief propagation 加 Chow–Liu 树做态制备。按查询计的优势 1.2–2.1 倍。它把统计误差、态制备偏差、oracle 偏差分开报告，并同时给 per-shot 和 per-query 两种口径，这个呈现方式值得参考。

两篇的结论都支持「不声称量子优势」的写法。

评委可能的反驳：张量网络能高效加载的分布，经典收缩通常也能精确算出尾概率。所以 s* 的适用范围应明确限定在稠密图区间，即你的精确引擎不可行的地方。

## 4. 两件需要你确认的事

- `HANDOFF.md` 写着仓库是公开的，同时写着比赛规则要求提交内容在结果公布前保密。
- 合同对「一次事件」的定义（余震是否单独赔付）公开资料里没有，`eq_cluster` 的 2.5 倍取决于它。

## 5. 参考文献核对（2026-11-18，逐条打开原页面核对；对应 TEAMMATE.md 第 3 节 a、b）

| 条目 | 结果 |
|---|---|
| `kastoryano2022` | 一致。Kastoryano & Pancotti，A highly efficient tensor network algorithm for multi-asset Fourier options pricing，arXiv:2203.02804（2022-03-05），无期刊号 |
| `benedetti2016` | 一致。Benedetti, Realpe-Gómez, Biswas, Perdomo-Ortiz，Phys. Rev. A 94, 022308 (2016)，DOI 10.1103/PhysRevA.94.022308 |
| `herbert2021` | 一致。发表版标题就是 bib 里的这个（arXiv 版标题不同：The Problem with Grover-Rudolph State Preparation for Quantum Monte-Carlo），Phys. Rev. E 103, 063302 (2021) |
| `errmit2026` | **要改两处**。作者：Guilin Zhang, Kai Zhao, Xiquan Cui, Henry Heng, Xu Chu, Aletta Johanna Blanken（Workday AI Research），arXiv:2609.38713（2026-09-30）。发表处不是 NeurIPS 正会论文集，而是 **NeurIPS 2026 Workshop on Secure and Trustworthy Quantum Machine Learning (SaTQuML)** 的短口头报告；bib 里的 `booktitle={Advances in Neural Information Processing Systems}` 应改成该 workshop。论文里的用法（校准失配下误差缓解可能使误差变大）和摘要一致：ZNE 在 38–63% 的模拟实例上不如不做缓解 |
| arXiv:2603.15664 | 存在。Alexis Kirke，Quantum Amplitude Estimation for Catastrophe Insurance Tail-Risk Pricing: Empirical Convergence and NISQ Noise Analysis（2026-03-10）。正文：3–8 个索引量子比特加 1 个辅助；预算 8192 下 RMSE：振幅估计 484，重要性采样 45，条件尾采样 32，QMC 37；oracle 模型下优势 2.2–3.7 倍；「Discretisation, not estimation, is the current bottleneck」 |
| arXiv:2608.26840 | 存在。Hernández Vera, Jamialahmadi, Mete, Schneider, De Pascale（Munich Quantum Valley；SAP Labs Munich），Quantum Rare-Event Estimation for Ising Graphical Models with Belief-Propagation State Preparation（2026-08-27）。正文：20 量子比特、38 个 CX；按查询计 RMSE 优势 1.2–2.1 倍；拟合斜率 −0.50（理想为 −1）；作者自己写明该基准树宽为 2、经典精确推断可行 |
| Kliuchnikov 等，Quantum 7, 1208 (2023) | 一致。Shorter quantum circuits via single-qubit gate approximation，DOI 10.22331/q-2023-12-18-1208。表 1（Clifford+T，均值，ε 为 diamond 距离，ε < 1e-4 的线性拟合）：fallback 1.03·log₂(1/ε) + 5.75，mixed fallback 0.53·log₂(1/ε) + 4.86，diagonal 3.02·log₂(1/ε) + 1.77。和 `rotations.py` 用的常数相同 |

没核对的：`benedetti2019`（npj Quantum Information 5, 45）这次没打开原页面。
