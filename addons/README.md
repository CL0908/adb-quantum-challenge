# ADB RDPO 计算提升 add-on

原则：只做**新增**或**同类替换**，不碰主轴。主轴包括两条：量子侧是「倾斜加载 → 振幅估计」，经典侧是 RDPO 的「CE → Rényi-2 → RQMC」。朋友的 `src/` 只当库来 import，不做任何修改，也没有改写其中的函数。所有输出只写到 `addons/results/`。

运行方式：`addons/` 放在朋友仓库里时，直接用同级的 `src/` 和 `results/`，也就是作者提交的结果；放在仓库外时，默认读取 `../adb-repro`，也就是我们的复现克隆。要指向别处，设置环境变量 `ADB_REPO`。`addons/results/` 里提交的 JSON 来自我们复现的结果，所以在仓库里直接重跑时，读到的是作者的结果，数字会略有不同。

```
python rotations.py   # Q1  旋转合成：同类替换
python q2_ae.py       # Q2  振幅估计算法：同类替换（ae.py 由 QContagion session 起草，本 session 审核）
python router.py      # C2  新增精确参考值，并列附在 RDPO 旁边
python psis.py        # C1  新增 k̂ 诊断（14 进程，约 16 分钟）
python measure.py     # Q3  评估会碰主轴的替换（只评估，不采用）
python combine.py     # 合计表
python -m pytest -q test_addons.py
```

正式运行记录在 `repro/run.py` 第 6 阶段（`repro/runs.jsonl`、`repro/logs/addon_*.log`），数字汇总在 `repro/RDPO_experiment_log.xlsx` 的 Addons 表。

## 结果（ε = 10%，朋友 quantum_advantage.json 的 8 个设定）

**量子所需硬件加速 s***：越小对量子越有利。Q1+Q2 两项合计后，s* 缩小 **10.7–25.9 倍（保守档）** 或 **12.3–29.3 倍（mixed 档）**。

| 设定 | 原 s* | Q1 保守 + Q2 | Q1 mixed + Q2 |
|---|---|---|---|
| ring-2 | 8.3e6 | 5.2e5 | 4.4e5 |
| ring-8 | 8.0e5 | 4.3e4 | 3.8e4 |
| ring-16 | 3.0e5 | 2.7e4 | 2.3e4 |
| ring-20 | 1.0e5 | 9.4e3 | 8.1e3 |
| ring-28 | 5.8e5 | 2.3e4 | 2.0e4 |
| complete-4 | 8.6e5 | 3.7e4 | 3.3e4 |
| complete-6 | 9.3e4 | 3.6e3 | 3.2e3 |
| complete-7 | 2.0e5 | 8.5e3 | 7.5e3 |

ring-28 若改用 I25 的合并 ρ²，s* 如下：
- E2 配置（ρ² ≈ 38）：原 9.7e4，保守档 3.9e3，mixed 档 3.4e3；
- v2-final（ρ² ≈ 13）：原 2.8e5，保守档 1.1e4，mixed 档 9.9e3。

**各项说明**
- **Q1 旋转合成**：ring-28 的 T-count 里约 92% 来自旋转。按 Kliuchnikov 等，Quantum 7, 1208（2023）表 1，取 Clifford+T 下的均值：
  - fallback：1.03·log₂(1/δ)+5.75，不需要随机化；
  - mixed fallback：0.53·log₂(1/δ)+4.86。

  两档都再加上 QROM 加相位梯度的态制备层，T-count 分别缩小 2.8–4.4 倍和 3.4–5.0 倍。朋友原来的 3·log₂(1/δ) 模型能逐位复现他的全部 8 个 T-count 和 s*。
- **Q2 ChebAE**：只用奇数次多项式，所以仍是 Grover 幂电路，可以沿用朋友的电路族和资源账。常数 c ≈ 23–25（IQAE 是 58–138），调用次数缩小 2.5–5.8 倍，覆盖率 ≥ 0.97，没有出现 k=0 的情况。
- **C2 精确参考值**：E2 的 24 格都能用朋友已有的精确引擎算出（exact_pair、exact_dense、MPS），和参考值的相对差 ≤ 1e-8（complete-7 为 3e-5）。耗时从毫秒到 5 秒，complete-7 约 80 秒。ε = 10% 时 RDPO 通常更快；要求更高精度时精确引擎更快，交叉点 ring 约 4–9%，complete-4/6 约 12–17%，complete-7 约 0.8%。
- **C1 PSIS**：ring-28 上做了 28 次拟合，k̂ 全部 < 0.7（0.18–0.54）。平滑只把 rel RMSE 从 2.84% 降到 2.67%，说明重尾不是误差的来源，PSIS 只作诊断用。朋友的估计器 rel RMSE 为 1.91%，落在 I25 区间内。
- **Q3（不采用）**：
  - 把加载换成 RDPO 策略：调用次数能降到 0.075–0.30 倍，但每次 A 的 Toffoli 数至少多 324–820 倍，净工作量至少是原来的 32–226 倍；即使改用 QROM 查表，也仍 ≥ 16 倍。
  - 同一电路改用方差型估计：最多只能把调用次数降到 0.52–0.70 倍，文献里没有给出常数，判定为没有收益。

  这也说明朋友用指数倾斜做加载是对的。

## 前提与限制

- **mixed fallback 要求随机化**：每个门实例、每个 shot 都必须独立重新抽样，否则误差变成相干的，2 倍收益会消失。用的是平均 T-count。ε 是 diamond 距离，在任意组合下次可加，所以朋友的 Σδ = 1e-3 预算仍然成立。
- **运行时间**：沿用朋友的 T-factory 吞吐模型，没有计入反应时间下界（T-depth × 10 µs）。以后如果增加 factory 数量，需要复查。
- **ChebAE 的常数**：c 是模拟实测值，没有理论证明。它的 kmax 比 IQAE 大 5–40%，所以对逻辑比特错误率的要求略严：ring-28 的 pL_required 从 5.6e-8 降到 4.1e-8。
- **还没用上的节省**：按电路分配误差预算，估计还能再省 10–25% 的 T，但需要先单独推导误差传播。
- **s* 的经典侧**：沿用朋友的口径，用的是我们复现得到的 ρ²；ring-28 另外给了用 I25 合并值的两行。
