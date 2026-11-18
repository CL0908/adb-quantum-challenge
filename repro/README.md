# repro：RDPO 复现工具与实验记录

这个目录用来复现本仓库 README 和 proposal 里声明的实验数字，并记录复现结果、发现的问题，以及我们能怎样协助。作者的原文和数字一律不改。

- `RDPO_experiment_log.xlsx`：总表，各 sheet 内容如下：
  - Summary、Claims、PaperNumbers、StatTests、Files、Runs：声明与复现的逐项对比；
  - Assist：协助项；
  - I25：n = 28 的 ρ² 补充实验；
  - Addons：计算提升 add-on，代码在 `../addons`；
  - Quality：风险 add-on（地震余震读法、基础触发率的不确定性）；
  - Value：RDPO 价值实验（E3 一次运行出敏感度、E2 嵌套不确定性 + 策略迁移）；
  - Climate：Clausius–Clapeyron 气候情景；
  - Env：运行环境。
- `run.py`：按阶段跑全部驱动脚本，每次运行写入 `runs.jsonl` 和 `logs/`。
- `claims_inventory.md`、`claims.py`、`paper_values.py`、`verdicts.py`：声明清单、取数方法和判定结果。
- `i25_rho2.py`、`i25/`：I25 补充实验，含 366 次拟合的逐次原始数据。
- `build_log.py`：根据以上内容生成 xlsx；公式用 LibreOffice 重算。

布局说明：这些脚本按「仓库外一个独立克隆」的布局设计，即 `../adb-repro`，这样原仓库不会被写入。复现时 `results/` 会被清空重跑，所以不要直接在本仓库里运行 `run.py`，否则会覆盖作者提交的 `results/`。这个目录放在这里是为了存档和查阅。
