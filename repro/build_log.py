"""Build repro/RDPO_experiment_log.xlsx: declared claims vs our reproduction, with run log, issues and environment.

    python repro/build_log.py
Then recalculate formulas with the xlsx skill's recalc.py (LibreOffice).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import claims  # noqa: E402
import compare  # noqa: E402
import paper_values  # noqa: E402
from verdicts import HELP, ISSUES, VERDICTS  # noqa: E402

REPO = HERE.parent / "adb-repro"
OUT = HERE / "RDPO_experiment_log.xlsx"
FONT = "Arial"
HEAD = PatternFill("solid", start_color="1F3864")
ID_FIELDS = ("case", "scenario", "topology", "n", "target", "threshold_musd", "method", "variant", "budget", "transfer")
RAW_SUMMARY = {"transfer": "rdpo_v2_transfer.json", "ca_pool": "realdata_pool.json", "elt_demo": "elt_demo.json",
               "scale": "rdpo_v2_scale.json", "bench": "rdpo_v2_bench.json", "wp2_carec": "wp2_carec.json",
               "wp2_asia_pacific": "wp2_asia_pacific.json"}


# ------------------------------------------------------------------------------------------------ helpers
def sheet(wb, title, headers, widths):
    ws = wb.create_sheet(title)
    ws.append(headers)
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for c in ws[1]:
        c.font = Font(name=FONT, bold=True, color="FFFFFF")
        c.fill = HEAD
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"
    return ws


def finish(ws):
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name=FONT, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.auto_filter.ref = ws.dimensions


def norm(v):
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, (int, float)):
        v = v / 1e6 if abs(v) >= 1e6 else v
        return f"{v:.6g}"
    return str(v)


def raw_rows(tag, name):
    p = REPO / tag / "raw" / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def stat_tests(rng=np.random.default_rng(0), nboot=2000):
    """Per (file, job key): rel RMSE from raw per-replication estimates, declared vs ours, bootstrap 95% CI of declared."""
    out = []
    for name, summ in RAW_SUMMARY.items():
        D, O = raw_rows("results_declared", name), raw_rows("results", name)
        if not D or not O:
            continue
        rows = json.loads((REPO / "results_declared" / summ).read_text()).get("rows", [])
        lut = [({norm(r[k]) for k in ID_FIELDS if k in r}, r["mu"]) for r in rows if "mu" in r]
        od = {r["id"]: r["est"] for r in O}
        groups = {}
        for r in D:
            groups.setdefault(tuple(r["key"]), []).append((r["est"], od.get(r["id"])))
        for key, pairs in groups.items():
            ks = {norm(k) for k in key}
            mus = {mu for ids, mu in lut if ks <= ids}
            if len(mus) != 1 or any(o is None for _, o in pairs):
                continue
            mu = mus.pop()
            d = np.array([p[0] for p in pairs]) / mu - 1
            o = np.array([p[1] for p in pairs]) / mu - 1
            boot = np.sqrt((d[rng.integers(0, len(d), (nboot, len(d)))] ** 2).mean(1))
            out.append({"file": name, "key": " / ".join(map(str, key)), "mu": mu, "reps": len(d),
                        "decl": float(np.sqrt((d ** 2).mean())), "ours": float(np.sqrt((o ** 2).mean())),
                        "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
                        "same": float(np.mean([a == b for a, b in pairs]))})
    return out


SUMMARY_KEYS = {"rdpo_e2.json": ("topology", "n", "target"), "rdpo_e3.json": ("case", "ablation")}


def summary_tests():
    """E2/E3 store no per-replication data: compare summary rel RMSE by key, with the normal-theory band
    for the ratio of two R-rep RMSEs, sqrt(F(R, R)) quantiles. Approximate: IS errors are heavy-tailed."""
    from scipy.stats import f as fdist
    out = []
    for f, keys in SUMMARY_KEYS.items():
        D, O = ({tuple(r[k] for k in keys): r for r in json.loads((REPO / t / f).read_text())["rows"]}
                for t in ("results_declared", "results"))
        for key in sorted(D, key=str):
            for part in ("rdpo", "twisted") if "rdpo" in D[key] else (None,):
                d = D[key].get(part) if part else D[key]
                o = (O.get(key) or {}).get(part) if part else O.get(key)
                if not d or not o or not d.get("rel_rmse"):
                    continue
                R = d["reps"]
                lo, hi = (float(np.sqrt(fdist.ppf(q, R, R))) * d["rel_rmse"] for q in (0.025, 0.975))
                out.append({"file": f.removesuffix(".json") + "（汇总，近似区间）",
                            "key": " / ".join(map(str, key)) + (f" / {part}" if part else ""), "mu": D[key]["mu"],
                            "reps": R, "decl": d["rel_rmse"], "ours": o["rel_rmse"], "lo": lo, "hi": hi,
                            "same": 1.0 if d["rel_rmse"] == o["rel_rmse"] else None})
    return out


def keyed(doc):
    """Sort 'rows' by identifying fields so files written in completion order compare row-to-row."""
    if isinstance(doc, dict) and isinstance(doc.get("rows"), list) and all(isinstance(r, dict) for r in doc["rows"]):
        ks = ID_FIELDS + ("ablation",)
        doc = {**doc, "rows": sorted(doc["rows"], key=lambda r: tuple(str(r.get(k)) for k in ks))}
    return doc


FILE_NOTES = {**{v: "随机实验：按 StatTests 统计口径判断" for v in RAW_SUMMARY.values()},
              "rdpo_e1.json": "随机实验：按 StatTests / Claims 判断", "rdpo_e2.json": "随机实验：StatTests 汇总近似区间",
              "rdpo_e3.json": "随机实验：StatTests 汇总近似区间",
              "quantum_advantage.json": "量子资源逐位相同；差异来自 E2 的 ρ² 与计时（I25 / I26）",
              "quantum_scaling.json": "同上，由 quantum_advantage 派生"}


def i25_sheet(wb, s):
    """Our supplementary experiment for Assist I25: rho^2 at ring-28 / ring-20 exceedance, and what it implies."""
    ws = sheet(wb, "I25", ["网络", "配置", "数据来源", "重复次数", "rel RMSE", "预算 B", "ρ² = B·rel²", "95% CI 下限 (rel)",
                           "95% CI 上限 (rel)", "ρ² 下限", "ρ² 上限", "离群比例 (|err|>5%)", "最大 |err|", "说明"],
               [8, 10, 30, 9, 10, 9, 10, 11, 11, 10, 10, 11, 10, 50])
    bold = Font(name=FONT, bold=True)
    pooled_row, heads = {}, set()

    def head(values):
        ws.append(values)
        heads.add(ws.max_row)
        for c in ws[ws.max_row]:
            c.font = Font(name=FONT, bold=True, color="FFFFFF")
            c.fill = HEAD
            c.alignment = Alignment(wrap_text=True, vertical="top")

    def add(n, cfg, src, reps, rel, lo=None, hi=None, outl=None, mx=None, note=""):
        i = ws.max_row + 1
        ws.append([f"ring-{n}", cfg, src, reps, rel, s["budget"], f"=F{i}*E{i}^2", lo, hi,
                   f'=IF(H{i}="","",F{i}*H{i}^2)', f'=IF(I{i}="","",F{i}*I{i}^2)', outl, mx, note])
        for c in "EHIM":
            ws[f"{c}{i}"].number_format = "0.00%"
        ws[f"L{i}"].number_format = "0.0%"
        for c in "GJK":
            ws[f"{c}{i}"].number_format = "0.0"
        return i

    for c in s["cells"]:
        n, cfg, nw, pr = c["n"], c["config"], c["new"], c["pooled_per_rep"]
        for src in c["sources"]:
            if src["source"].startswith("补充实验"):
                add(n, cfg, src["source"], nw["reps"], nw["rel_rmse"], *nw["ci"], nw["outlier_share"], nw["max_abs_err"],
                    f"CI 覆盖率 {nw['ci_coverage']:.0%}；单次拟合中位耗时 {nw['median_sec']:.0f} s（14 进程并行）")
            else:
                add(n, cfg, src["source"], src["reps"], src["rel_rmse"],
                    note="" if src["per_rep"] else "只有汇总值，计入「合并（含 E2 汇总）」")
        pooled_row[(n, cfg)] = add(n, cfg, "合并（全部逐次数据）", pr["reps"], pr["rel_rmse"], *pr["ci"], pr["outlier_share"],
                                   pr["top_abs_err"][0], f"去掉离群重复后 rel RMSE {pr['rel_rmse_without_outliers']:.2%}；"
                                   f"最大 5 个 |err|：{', '.join(f'{v:.1%}' for v in pr['top_abs_err'])}")
        ws.cell(pooled_row[(n, cfg)], 3).font = bold
        add(n, cfg, "合并（含 E2 汇总）", c["pooled_all"]["reps"], c["pooled_all"]["rel_rmse"])

    ws.append([])
    ws.append(["量子比较（只换 ρ²；Tq 与 τ_C 用作者 quantum_advantage.json 的值，计时问题见 I26）"])
    ws.cell(ws.max_row, 1).font = bold
    head(["网络", "配置", "量子耗时 Tq (s)", "经典每路径 τ_C (s)", "ε", "ρ² 点估计", "ρ² 下限", "ρ² 上限", "s* 点估计",
          "s* 下（ρ² 上限）", "s* 上（ρ² 下限）", "作者 ρ²", "作者 s*", "(1−μ)/μ", "MC 夸大倍数", "作者 MC 夸大倍数"])
    for c in s["cells"]:
        q, p, i = c["quantum"], pooled_row[(c["n"], c["config"])], ws.max_row + 1
        ws.append([f"ring-{c['n']}", c["config"], q["Tq_s"], q["t_path_s"], s["eps_rel"], f"=I25!G{p}", f"=I25!J{p}",
                   f"=I25!K{p}", f"=C{i}/(F{i}/E{i}^2*D{i})", f"=C{i}/(H{i}/E{i}^2*D{i})", f"=C{i}/(G{i}/E{i}^2*D{i})",
                   q["declared_rho2"], q["declared_s_star"], q["mc_ratio"], f"=N{i}/F{i}", f"=N{i}/L{i}"])
        for col in "CDIJKMNOP":
            ws[f"{col}{i}"].number_format = "0.00E+00"
        for col in "FGHL":
            ws[f"{col}{i}"].number_format = "0.0"

    ws.append([])
    ws.append(["6 次重复会抽到什么（从合并逐次数据里有放回抽 6 次，20 万次模拟）"])
    ws.cell(ws.max_row, 1).font = bold
    head(["网络", "配置", "作者 E2 值", "P(6 次 ≥ 作者值)", "我们 E2 值", "P(6 次 ≤ 我们值)", "5% 分位", "中位", "95% 分位"])
    for c in s["cells"]:
        x = c["six_rep_sim"]
        ws.append([f"ring-{c['n']}", c["config"], x["author_e2"], x["p_ge_author_e2"], x["ours_e2"], x["p_le_ours_e2"],
                   *x["q05_q50_q95"]])
        for col in "CDEFGHI":
            ws[f"{col}{ws.max_row}"].number_format = "0.0%" if col in "DF" else "0.00%"
    a = s.get("author_seeds")
    if a:
        ws.append([])
        ws.append(["作者 E2 的原种子在我们机器上重跑（ring-28 exceedance，E2 配置）"])
        ws.cell(ws.max_row, 1).font = bold
        head(["种子", *a["seeds"], "rel RMSE", "作者 E2 rel RMSE"])
        ws.append(["相对误差", *a["rel_err"], a["rel_rmse"], a["author_e2_rel_rmse"]])
        for c in ws[ws.max_row][1:]:
            c.number_format = "0.00%"
    for row in ws.iter_rows(min_row=2):
        for c in row:
            if c.row not in heads:
                c.font = Font(name=FONT, size=10, bold=c.font.bold)
                c.alignment = Alignment(wrap_text=True, vertical="top")


def addons_sheet(wb, d):
    """Computation add-ons (addons/): before -> after on the friend's own operating points. Friend's src/ untouched."""
    ws = sheet(wb, "Addons", ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
               [14, 8, 13, 13, 13, 13, 13, 13, 13, 13, 13, 40])
    ws.delete_rows(1)                     # this sheet is a stack of titled tables, not one header row
    ws.freeze_panes = None
    bold, heads = Font(name=FONT, bold=True), set()

    def title(text):
        ws.append([])
        ws.append([text])
        ws.cell(ws.max_row, 1).font = bold

    def head(values):
        ws.append(values)
        heads.add(ws.max_row)
        for c in ws[ws.max_row]:
            c.font = Font(name=FONT, bold=True, color="FFFFFF")
            c.fill = HEAD
            c.alignment = Alignment(wrap_text=True, vertical="top")

    def fmt(i, cols, f):
        for col in cols:
            ws[f"{col}{i}"].number_format = f

    comb = d["combined"]["rows"]
    title("合计：量子所需硬件加速 s*（ε = 10%，越小对量子越有利）。全部是同类替换，主轴（倾斜加载 → 振幅估计；RDPO 的 CE → Rényi-2 → RQMC）不动。"
          "Q1 = 旋转合成方法的同类替换（保守 = fallback+QROM，无需随机化；mixed = mixed fallback+QROM，每个门实例每个 shot 须独立重抽）；"
          "Q2 = 振幅估计算法的同类替换（IQAE → ChebAE，同一 Grover 幂电路族）；经典侧沿用作者口径（我们复现的 ρ²）")
    head(["网络", "n", "原 s*", "Q2 后", "Q1 保守 + Q2", "Q1 mixed + Q2", "缩小倍数（保守）", "缩小倍数（mixed）", "精确引擎",
          "精确耗时 (s)", "精确快于 RDPO 的 ε", "说明"])
    for r in comb:
        i, s = ws.max_row + 1, r["s_star"]
        ws.append([r["topology"], r["n"], s["declared"], s["Q2"], s["Q1cQ2"], s["Q1Q2"], f"=C{i}/E{i}", f"=C{i}/F{i}",
                   r["exact_engine"], r["exact_seconds"], r["crossover_eps_exact_beats_rdpo"],
                   "原 s* 由 rotations.price('rs') 逐位复现作者 quantum_advantage.json"])
        fmt(i, "CDEF", "0.00E+00"); fmt(i, "GH", "0.0"); fmt(i, "J", "0.000"); fmt(i, "K", "0.0%")
    r28 = next(r for r in comb if r["topology"] == "ring" and r["n"] == 28)
    for cfg in ("e2", "v2-final"):
        if f"s_star_Q1Q2_rho2_i25_{cfg}" in r28:
            i = ws.max_row + 1
            ws.append(["ring", 28, r28[f"s_star_declared_rho2_i25_{cfg}"], None, r28[f"s_star_Q1cQ2_rho2_i25_{cfg}"],
                       r28[f"s_star_Q1Q2_rho2_i25_{cfg}"], f"=C{i}/E{i}", f"=C{i}/F{i}", None, None, None,
                       f"经典 ρ² 改用 I25 合并值（{cfg} 配置）"])
            fmt(i, "CEF", "0.00E+00"); fmt(i, "GH", "0.0")

    title("Q1 旋转合成降本（rotations.py）：同一电路、同一 Toffoli/旋转个数、同一 surface-code 模型，只换每个旋转的 T 价格。"
          "ε 均为 diamond 距离，误差预算 Σδ = 1e-3 仍成立；运行时间沿用作者的 factory 吞吐模型，未计反应时间下界")
    head(["网络", "n", "A 调用次数", "T（作者 3·log₂(1/δ)）", "T（fallback）", "T（fallback+QROM）", "T（mixed fallback）",
          "T（mixed+QROM）", "旋转占 T 比例", "T 缩小（fallback+QROM）", "T 缩小（mixed+QROM）", "来源"])
    for r in d["q1"]["rows"]:
        i = ws.max_row + 1
        ws.append([r["topology"], r["n"], r["calls"], r["rs"]["T_count"], r["fallback"]["T_count"],
                   r["qrom_fallback"]["T_count"], r["mixed"]["T_count"], r["qrom_mixed"]["T_count"],
                   r["rs"]["rotation_share"], f"=D{i}/F{i}", f"=D{i}/H{i}",
                   "Kliuchnikov et al., Quantum 7, 1208 (2023) 表 1，Clifford+T 均值：fallback 1.03·log₂(1/δ)+5.75；"
                   "mixed fallback 0.53·log₂(1/δ)+4.86"])
        fmt(i, "CDEFGH", "0.00E+00"); fmt(i, "I", "0%"); fmt(i, "JK", "0.00")

    title("Q2 振幅估计算法的同类替换（q2_ae.py，ae.py 由 QContagion session 起草、本 session 审核）：IQAE 保留为对照；"
          "c = 调用次数·ε·√a，200 次模拟，α = 0.05")
    head(["网络", "n", "a", "IQAE c", "ChebAE c", "MIQAE c", "IQAE 覆盖率", "ChebAE 覆盖率", "调用缩小（ChebAE）",
          "kmax IQAE", "kmax ChebAE", "说明"])
    for r in d["q2"]["rows"]:
        i = ws.max_row + 1
        ws.append([r["topology"], r["n"], r["a"], r["iqae"]["c"], r["chebae_odd"]["c"], r["miqae_chernoff"]["c"],
                   r["iqae"]["coverage"], r["chebae_odd"]["coverage"], f"=D{i}/E{i}", r["iqae"]["kmax_max"],
                   r["chebae_odd"]["kmax_max"], "ChebAE 只用奇次（Grover 幂电路），沿用作者电路族与资源账；MIQAE 为有证明的 Chernoff 版"])
        fmt(i, "C", "0.00E+00"); fmt(i, "DEF", "0.0"); fmt(i, "GH", "0.0%"); fmt(i, "I", "0.00")

    title("Q3 会碰主轴的替换（只评估、不采用）——加载测度（measure.py）：b0 = 同电路改方差型估计（需 c_var/c < 1/调用比 才不亏，文献无显式常数，按算法结构判为无收益）；"
          "b1 = 加载 RDPO 学到的策略（Toffoli 增量取下界；即便改用 QROM 查表把加载减半，净工作量仍 ≥ 16 倍）。比值 < 1 才划算")
    head(["网络", "n", "a", "E[g²]/a", "ρ² 倾斜", "1/a − 1", "ρ² RDPO", "b0 调用比", "b1 调用比", "b1 每次 A Toffoli 倍数",
          "b1 净工作量比（下界）", "结论"])
    for r in d["q3"]["rows"]:
        i = ws.max_row + 1
        ws.append([r["topology"], r["n"], r["a_sampled"], r["E_g2_over_a"], r["rho2_twisted"], r["inv_a_minus_1"],
                   r["rho2_rdpo"], r["b0_calls_ratio"], r["b1_calls_ratio"], r["toffoli_A_rdpo_loading_lb"] / r["toffoli_A_now"],
                   f"=I{i}*J{i}", f'=IF(K{i}>1,"b1 不划算","b1 划算")'])
        fmt(i, "C", "0.00E+00"); fmt(i, "DEGHI", "0.000"); fmt(i, "F", "0.0"); fmt(i, "JK", "0.0")

    title("C2 新增精确参考值（router.py）：用朋友已有的精确引擎，把精确值并列附在 RDPO 估计旁边（朋友的 api/estimate.py 已对 pair 目标和 n ≤ 5 这样做）；"
          "RDPO 仍是主估计器，不替换。E2 全部 24 格：与作者参考值的相对差，以及与 RDPO 单次拟合的耗时对比")
    head(["网络", "n", "目标", "引擎", "精确耗时 (s)", "与参考值相对差", "RDPO rel RMSE", "RDPO 单次耗时 (s)", "耗时缩小", "", "",
          "路由理由"])
    for r in d["c2"]["rows"]:
        i = ws.max_row + 1
        ws.append([r["topology"], r["n"], r["target"], r["engine"], r["seconds"], r["rel_diff_vs_ref"], r["rdpo_rel_rmse"],
                   r["rdpo_seconds_per_fit"], f"=H{i}/E{i}", None, None, r["why"]])
        fmt(i, "E", "0.000"); fmt(i, "F", "0.0E+00"); fmt(i, "G", "0.00%"); fmt(i, "H", "0.0"); fmt(i, "I", "0")

    p = d.get("c1")
    if p:
        title(f"C1 新增诊断 k̂（psis.py，只加诊断、不替换朋友的估计器）：ring-28 exceedance，E2 配置，{p['fits']} 次独立拟合；每次拟合用其 θ 新抽 "
              f"{p['n_est']} 条 iid 路径；误差对精确 MPS 值")
        head(["估计器", "", "rel RMSE", "rel 偏差", "最大 |err|", "", "", "", "", "", "", "说明"])
        notes = {"friend_est": "作者 rdpo_estimate 自己的估计（含 RQMC 阶段）", "raw": "同一 θ，iid 路径普通均值",
                 "psis": "同一 θ，iid 路径 Pareto 平滑后均值"}
        for k in ("friend_est", "raw", "psis"):
            s = p["summary"][k]
            ws.append([k, None, s["rel_rmse"], s["rel_bias"], s["max_abs"], None, None, None, None, None, None, notes[k]])
            fmt(ws.max_row, "CDE", "0.00%")
        ws.append(["k̂ ≥ 0.7 的拟合数", None, p["summary"]["khat_ge_0.7"]])
        ws.append(["k̂ 与 |raw 误差| 相关系数", None, p["summary"]["corr_khat_abs_raw_err"]])
        fmt(ws.max_row, "C", "0.00")

    for row in ws.iter_rows(min_row=1):
        for c in row:
            if c.row not in heads:
                c.font = Font(name=FONT, size=10, bold=c.font.bold)
                c.alignment = Alignment(wrap_text=True, vertical="top")


# ------------------------------------------------------------------------------------------------ build
def build():
    paper = paper_values.collect()   # chdirs into adb-repro; every path below is absolute
    wb = Workbook()
    wb.remove(wb.active)

    # README -------------------------------------------------------------------------------------
    ws = sheet(wb, "README", ["项目", "内容"], [22, 120])
    readme = [
        ("目的", "复现 CL0908/adb-quantum-challenge（commit d29bd43）里 README / proposal 声明的实验数字，并记录我们这边的复现结果与发现的问题。"),
        ("原仓库", "D:\\adb-quantum-challenge\\adb-quantum-challenge（未改动）"),
        ("复现副本", "D:\\adb-quantum-challenge\\adb-repro：results_declared/ = 作者提交的结果；results/ = 我们从空目录重新跑出的结果"),
        ("运行方式", "repro/run.py 按依赖顺序分 3 个阶段运行全部驱动脚本；每次运行写入 repro/runs.jsonl 与 repro/logs/"),
        ("取数方式", "paper 里的数字用作者自己的 src/viz/build_paper.py 取值函数分别对两套结果计算（PaperNumbers 表）；其余声明用 repro/claims.py 中的小提取函数"),
        ("Claims 判定", "一致 = 逐位或浮点末位一致；统计一致 = 随机实验，数值落在作者结果的抽样误差内且结论不变；"
                       "结论成立但数字变 = 结论仍成立，但文中具体数字超出抽样误差；需核对文字 = 原文的数字/表述与数据（作者自己的或我们复现的）对不上，原文不改，列给作者核对；"
                       "机器相关 = 计时类；无法复现 = 没有生成该数字的代码或数据"),
        ("StatTests 判定", "对有逐次原始数据（results/raw/*.jsonl）的随机实验：用作者的逐次估计做 2000 次 bootstrap 得到 rel RMSE 的 95% 区间，看我们的 rel RMSE 是否落入。"
                           "E2 / E3 没有逐次数据：按 key 对齐汇总 rel RMSE，区间取两组各 R 次重复 RMSE 之比的正态理论区间 √F(R,R)（重尾时只是近似）"),
        ("相对差", "Claims!M 与 StatTests!K 为公式：|复现 − 原值| / |原值|"),
        ("跨平台说明", "作者环境 Linux x86-64；我们 Windows 11、16 核，Python/numpy/scipy/matplotlib/pytest 版本与 requirements.txt 完全一致。"
                     "RDPO 训练会把 1e-16 级浮点差放大，跨平台不能逐位复现，随机实验需按统计口径判断"),
        ("计算提升 add-on", "D:\\adb-quantum-challenge\\addons：只做新增或同类替换，不碰主轴；只 import 朋友的 src（只读），结果写 addons/results/；"
                         "repro/run.py 第 6 阶段登记运行；数字见 Addons 表。ae.py 由 QContagion session 起草、本 session 审核集成"),
        ("颜色", "Claims 判定列：绿 = 一致/统计一致；黄 = 结论成立但数字变/机器相关；红 = 需核对文字；灰 = 无法复现"),
    ]
    for r in readme:
        ws.append(r)
    finish(ws)

    # Claims -------------------------------------------------------------------------------------
    ev = claims.evaluate(paper)
    related = {}
    for row in ISSUES:
        ids = set()
        for a, b in re.findall(r"C(\d+)(?:–C(\d+))?", row[6]):
            ids.update(range(int(a), int(b or a) + 1))
        for k in ids:
            related.setdefault(f"C{k:02d}", []).append(row[0])
    ws = sheet(wb, "Claims", ["ID", "原文（逐字摘自作者文档，未改动）", "原文位置", "声明摘要（审计用）", "文档声明值",
                              "结果文件 / key", "生成脚本", "档位", "原作者结果算出的值", "我们复现算出的值",
                              "主指标(原)", "主指标(复现)", "相对差", "判定", "说明", "相关协助项", "静态审计备注（子 agent，未运行代码）"],
               [6, 60, 22, 28, 26, 28, 20, 6, 42, 42, 11, 11, 9, 14, 50, 12, 36])
    fills = {"一致": "C6EFCE", "统计一致": "C6EFCE", "结论成立但数字变": "FFEB9C", "机器相关": "FFEB9C",
             "需核对文字": "FFC7CE", "无法复现": "D9D9D9", "部分复现": "FFEB9C"}
    for i, c in enumerate(ev, start=2):
        verdict, note = VERDICTS.get(c["id"], ("待判定", ""))
        ws.append([c["id"], c["quote"], c["quote_where"], c["claim"], c["declared"], c["source"], c["producer"], c["tier"],
                   c["stored_text"], c["ours_text"], c["stored_num"], c["ours_num"],
                   f'=IF(OR(K{i}="",L{i}=""),"",ABS(L{i}-K{i})/MAX(ABS(K{i}),1E-300))', verdict, note,
                   ", ".join(related.get(c["id"], [])), c["audit_match"]])
        ws[f"N{i}"].fill = PatternFill("solid", start_color=fills.get(verdict, "FFFFFF"))
        ws[f"M{i}"].number_format = "0.00%"
    finish(ws)

    # PaperNumbers -------------------------------------------------------------------------------
    ws = sheet(wb, "PaperNumbers", ["占位符", "来源函数", "原作者结果算出", "我们复现算出", "是否相同"], [22, 18, 60, 60, 10])
    d, o = paper["results_declared"], paper["results"]
    for i, k in enumerate(sorted(d), start=2):
        ws.append([k, d[k]["group"], str(d[k]["value"])[:32000], str(o.get(k, {}).get("value"))[:32000],
                   f'=IF(C{i}=D{i},"相同","不同")'])
    finish(ws)

    # StatTests ----------------------------------------------------------------------------------
    st = stat_tests() + summary_tests()
    ws = sheet(wb, "StatTests", ["原始数据", "分组 key", "真值 mu", "重复次数", "原 rel RMSE", "复现 rel RMSE",
                                 "原 95% CI 下限", "原 95% CI 上限", "逐次估计相同比例", "判定", "相对差"],
               [16, 46, 12, 8, 12, 12, 12, 12, 12, 12, 10])
    for i, s in enumerate(st, start=2):
        ws.append([s["file"], s["key"], s["mu"], s["reps"], s["decl"], s["ours"], s["lo"], s["hi"], s["same"],
                   f'=IF(I{i}=1,"逐位一致",IF(AND(F{i}>=G{i},F{i}<=H{i}),"统计一致","超出95%CI"))',
                   f'=IF(E{i}=0,"",ABS(F{i}-E{i})/E{i})'])
        for col in "EFGHK":
            ws[f"{col}{i}"].number_format = "0.000%"
        ws[f"I{i}"].number_format = "0%"
        ws[f"C{i}"].number_format = "0.000E+00"
    finish(ws)

    # Files --------------------------------------------------------------------------------------
    ws = sheet(wb, "Files", ["结果文件", "状态", "相同数值", "比较数值", "最大相对差（非计时）", "计时字段", "原有我们缺", "我们多出", "备注"],
               [26, 18, 10, 10, 14, 10, 10, 10, 44])
    for f in sorted(p.name for p in (REPO / "results_declared").glob("*.json")):
        ours = REPO / "results" / f
        if not ours.exists():
            ws.append([f, "未重跑（不在复现范围）", None, None, None, None, None, None, None])
            continue
        c = compare.compare(keyed(json.loads((REPO / "results_declared" / f).read_text())), keyed(json.loads(ours.read_text())))
        shape = not c["missing"] and not c["extra"]
        status = ("逐位一致" if not c["diffs"] else "浮点末位一致" if c["max_rel"] <= compare.PREC else "数值有差异") \
            if shape else "结构不同"
        ws.append([f, status, c["equal"], c["n"], c["max_rel"], len(c["timing"]), len(c["missing"]), len(c["extra"]),
                   FILE_NOTES.get(f, "")])
        ws.cell(ws.max_row, 5).number_format = "0.00E+00"
    finish(ws)

    # Runs ---------------------------------------------------------------------------------------
    recs = [json.loads(l) for l in (HERE / "runs.jsonl").read_text().splitlines()]
    ws = sheet(wb, "Runs", ["任务", "阶段", "命令", "开始", "结束", "耗时(秒)", "退出码", "占用核数", "日志", "commit", "data/src/tests 改动"],
               [18, 6, 46, 18, 18, 10, 7, 8, 18, 12, 30])
    for r in recs:
        ws.append([r["name"], r["phase"], r["cmd"], r["start"], r["end"], r["seconds"], r["exit"], min(r["cpu_cost"], 4),
                   r["log"], r["commit"][:7], r["data_changed"] or "无"])
    n = ws.max_row
    ws.append(["合计", None, None, None, None, f"=SUM(F2:F{n})", f'=COUNTIF(G2:G{n},"<>0")&" 个非零"', None,
               "CPU 小时（约）", None, f"=SUMPRODUCT(F2:F{n},H2:H{n})/3600"])
    finish(ws)

    # Issues -------------------------------------------------------------------------------------
    ws = sheet(wb, "Assist", ["ID", "类别", "优先级", "发现", "证据", "核实状态", "相关声明", "我们可以怎么协助", "可能的改法（供作者参考）"],
               [6, 12, 8, 50, 46, 18, 12, 46, 36])
    for row in ISSUES:
        ws.append(list(row[:7]) + [HELP.get(row[0], ""), row[7]])
    finish(ws)

    # I25 supplementary experiment -----------------------------------------------------------------
    i25 = HERE / "i25" / "summary.json"
    if i25.exists():
        i25_sheet(wb, json.loads(i25.read_text(encoding="utf-8")))

    # computation add-ons (addons/results/*.json) ---------------------------------------------------
    ar = HERE.parent / "addons" / "results"
    files = {"combined": "combined.json", "q1": "q1_rotations.json", "q2": "q2_ae.json", "q3": "q3_measure.json",
             "c2": "c2_router.json", "c1": "c1_psis.json"}
    if (ar / "combined.json").exists():
        addons_sheet(wb, {k: json.loads((ar / f).read_text(encoding="utf-8")) for k, f in files.items() if (ar / f).exists()})

    # Env ----------------------------------------------------------------------------------------
    ws = sheet(wb, "Env", ["项目", "作者（README / proposal 声明）", "我们"], [16, 40, 60])
    env = json.loads((HERE / "env_phase1.json").read_text())
    for k, decl in [("python", "3.13"), ("numpy", "2.5.3"), ("scipy", "1.18.1"), ("matplotlib", "3.11.2"),
                    ("pytest", "9.2.0"), ("platform", "Linux x86-64"), ("cpu", "4-core laptop CPU"), ("cores", "4"),
                    ("commit", "d29bd43")]:
        ws.append([k, decl, str(env.get(k))])
    finish(ws)

    # Summary (first sheet) ----------------------------------------------------------------------
    ws = sheet(wb, "Summary", ["指标", "值", "说明"], [34, 14, 70])
    wb.move_sheet("Summary", offset=-(len(wb.sheetnames) - 1))
    nc, ns, nf, nr = len(ev) + 1, len(st) + 1, ws.parent["Files"].max_row, len(recs) + 1
    rows = [("声明总数", f"=COUNTA(Claims!A2:A{nc})", "来自 claims_inventory.md"),
            *[(f"判定：{v}", f'=COUNTIF(Claims!N2:N{nc},"{v}")', "") for v in
              ("一致", "统计一致", "结论成立但数字变", "部分复现", "需核对文字", "机器相关", "无法复现", "待判定")],
            ("统计检验分组数", f"=COUNTA(StatTests!A2:A{ns})", "随机实验：有逐次数据的用 bootstrap 95% CI；E2 / E3 用汇总近似区间"),
            ("  逐位一致", f'=COUNTIF(StatTests!J2:J{ns},"逐位一致")', ""),
            ("  统计一致（落在原 95% CI）", f'=COUNTIF(StatTests!J2:J{ns},"统计一致")', ""),
            ("  超出原 95% CI", f'=COUNTIF(StatTests!J2:J{ns},"超出95%CI")', "多为少数重复分叉导致的重尾，见 Assist I13 / I25 / I28"),
            ("结果文件：浮点末位或逐位一致", f'=COUNTIF(Files!B2:B{nf},"逐位一致")+COUNTIF(Files!B2:B{nf},"浮点末位一致")', ""),
            ("结果文件：数值有差异", f'=COUNTIF(Files!B2:B{nf},"数值有差异")+COUNTIF(Files!B2:B{nf},"结构不同")', ""),
            ("运行任务数", f"=COUNTA(Runs!A2:A{nr})", ""),
            ("运行总耗时（小时）", f"=Runs!F{nr + 1}/3600", "各任务耗时之和（并行运行，墙钟时间更短）"),
            ("协助项", f"=COUNTA(Assist!A2:A{len(ISSUES) + 1})", "发现 + 我们可以怎么协助，作者原文不改")]
    for r in rows:
        ws.append(list(r))
        if r[0].startswith("运行总耗时"):
            ws.cell(ws.max_row, 2).number_format = "0.00"
    finish(ws)

    wb.save(OUT)
    print(f"wrote {OUT}  claims={len(ev)} placeholders={len(d)} stat_groups={len(st)} runs={len(recs)} issues={len(ISSUES)}")


if __name__ == "__main__":
    build()
