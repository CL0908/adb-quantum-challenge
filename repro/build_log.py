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
          "Q2 = 振幅估计算法的同类替换（IQAE → ChebAE，同一 Grover 幂电路族）；经典侧沿用作者口径（ρ² 取自所读的 quantum_advantage.json；2026-11-18 起是作者提交的结果）")
    head(["网络", "n", "原 s*", "Q2 后", "Q1 保守 + Q2", "Q1 mixed + Q2", "缩小倍数（保守）", "缩小倍数（mixed）", "", "", "", "说明"])
    for r in comb:
        i, s = ws.max_row + 1, r["s_star"]
        ws.append([r["topology"], r["n"], s["declared"], s["Q2"], s["Q1cQ2"], s["Q1Q2"], f"=C{i}/E{i}", f"=C{i}/F{i}",
                   None, None, None, "原 s* 由 rotations.price('rs') 逐位复现作者 quantum_advantage.json"])
        fmt(i, "CDEF", "0.00E+00"); fmt(i, "GH", "0.0")
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

    for row in ws.iter_rows(min_row=1):
        for c in row:
            if c.row not in heads:
                c.font = Font(name=FONT, size=10, bold=c.font.bold)
                c.alignment = Alignment(wrap_text=True, vertical="top")


def quality_sheet(wb, d):
    """Quality add-ons (methods borrowed from QContagion's CDS work), additive; friend's src/ read-only."""
    ws = wb.create_sheet("Quality")
    for i, w in enumerate([16, 10, 12, 12, 12, 12, 12, 12, 12, 12, 46], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    bold, heads = Font(name=FONT, bold=True), set()

    def title(text):
        ws.append([]); ws.append([text]); ws.cell(ws.max_row, 1).font = bold

    def head(values):
        ws.append(values); heads.add(ws.max_row)
        for c in ws[ws.max_row]:
            c.font = Font(name=FONT, bold=True, color="FFFFFF"); c.fill = HEAD
            c.alignment = Alignment(wrap_text=True, vertical="top")

    def fmt(cols, f):
        for col in cols:
            ws[f"{col}{ws.max_row}"].number_format = f

    e = d.get("eq")
    if e:
        title("1 地震触发聚集（eq_cluster.py）：KG/TJ 50 km 内，原始目录 vs Gardner–Knopoff 去余震；ν = 每个主震带来的同级余震数")
        head(["震级 ≥", "原始", "去余震", "余震占比", "ν", "ν 95% 下", "ν 95% 上", "离散度（原始）", "p（原始）",
              "离散度（去余震）", "相邻年依赖 r（p）"])
        for M, x in e["by_magnitude"].items():
            y, i = x["KG_or_TJ"], ws.max_row + 1
            ws.append([float(M), y["raw"], y["declustered"], f"=1-C{i}/B{i}", y["nu"], *y["nu_ci95"],
                       y["dispersion_raw"][0], y["dispersion_raw"][1], y["dispersion_declustered"][0],
                       f"{y['lag1_declustered'][0]:.2f}（p={y['lag1_declustered'][1]:.2f}）"])
            fmt("D", "0%"); fmt("EFGHJ", "0.00"); fmt("I", "0.000")
        h = e["hawkes_raw_m55"]
        ws.append([f"Hawkes（原始 M≥5.5，日分辨率，单指数核）：分支比 {h['n_hat']:.2f}，95% [{h['n_ci95'][0]:.2f}, {h['n_ci95'][1]:.2f}]；"
                   "单指数核抓不到 Omori 长尾，余震占比以 Gardner–Knopoff 为准"])
        head(["合同读法", "ν", "两债同时耗尽", "相对读法 A", "联合附着", "EL (US$m)", "", "", "", "", "说明"])
        K = "term_both_bonds_exhausted_probability"
        base_row = None
        for lab, v in e["impact_adb_base"].items():
            i = ws.max_row + 1
            base_row = base_row or i
            ws.append([lab, v.get("nu", 0.0), v[K], f"=C{i}/C${base_row}", v["term_joint_attachment_probability"],
                       v["term_expected_principal_loss_usd"] / 1e6, None, None, None, None,
                       "A = 余震并入同一事件（朋友模型）" if lab.startswith("reading_A") else "B = 余震单独赔付、brief 尾部只含主震"])
            fmt("B", "0.00"); fmt("C", "0.00E+00"); fmt("D", "0.00"); fmt("E", "0.0000"); fmt("F", "0.00")
    u = d.get("rates")
    if u:
        title("2 基础触发率的数据不确定性（rates_unc.py）：地震 = G-R 外推（发生率 Poisson + b 值误差），降雨 = GEV（GML + Martins–Stedinger 先验）bootstrap")
        head(["倍数", "2.5%", "中位", "97.5%", "", "", "", "", "", "", "说明"])
        for k, q in u["multiplier_q"].items():
            ws.append([k, *q]); fmt("BCD", "0.00")
        head(["抽样范围", "两债同时耗尽中位", "95% 下", "95% 上", "区间宽度/朋友", "EL 中位 (US$m)", "EL 95% 下", "EL 95% 上", "", "",
              "说明"])
        f = u["friend_shares_only_joint_exhaustion"]
        fr = ws.max_row + 1
        ws.append(["朋友 param_uncertainty（只含共享比例）", f["median"], f["ci95"][0], f["ci95"][1], f"=(D{fr}-C{fr})/(D${fr}-C${fr})"])
        fmt("BCD", "0.00E+00"); fmt("E", "0.00")
        for lab, x in u["sweeps"].items():
            s_, el, i = x["summary"]["joint_exhaustion"], x["summary"]["expected_loss"], ws.max_row + 1
            ws.append([lab, s_["median"], s_["ci95"][0], s_["ci95"][1], f"=(D{i}-C{i})/(D${fr}-C${fr})", el["median"] / 1e6,
                       el["ci95"][0] / 1e6, el["ci95"][1] / 1e6, None, None,
                       {"shares_only": "复现朋友的口径", "rates_only": "只加发生率不确定性", "all": "两者都加"}[lab]])
            fmt("BCD", "0.00E+00"); fmt("E", "0.00"); fmt("FGH", "0.00")
    for row in ws.iter_rows(min_row=1):
        for c in row:
            if c.row not in heads:
                c.font = Font(name=FONT, size=10, bold=c.font.bold)
                c.alignment = Alignment(wrap_text=True, vertical="top")


def climate_sheet(wb, d):
    """Climate add-on: climate_cc (Clausius–Clapeyron scenarios)."""
    ws = wb.create_sheet("Climate")
    for i, w in enumerate([24, 20, 13, 13, 13, 13, 13, 13, 13, 13, 40], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    bold, heads = Font(name=FONT, bold=True), set()
    J, A = "term_both_bonds_exhausted_probability", "term_joint_attachment_probability"

    def title(text):
        ws.append([]); ws.append([text]); ws.cell(ws.max_row, 1).font = bold

    def head(values):
        ws.append(values); heads.add(ws.max_row)
        for c in ws[ws.max_row]:
            c.font = Font(name=FONT, bold=True, color="FFFFFF"); c.fill = HEAD
            c.alignment = Alignment(wrap_text=True, vertical="top")

    def fmt(i, cols, f):
        for col in cols:
            ws[f"{col}{i}"].number_format = f

    ws.append(["气候：Clausius–Clapeyron 情景（情景，不是预测）。在朋友模型外层精确计算，主轴不动"])
    ws.cell(1, 1).font = bold

    cc = d.get("cc")
    if cc:
        title("1 Clausius–Clapeyron 情景（climate_cc.py）：强度 × 1.07^ΔT，用 GEV（GML + Martins–Stedinger 先验）换算成朋友 2% 阈值的超越频率；情景，不是预测")
        head(["场景", "ΔT", "KG 降雨率倍数", "TJ 降雨率倍数", "两国同时触发", "× 朋友", "两债同时耗尽", "× 朋友", "", "", "说明"])
        for sn, r in cc["targets"].items():
            f0 = ws.max_row + 1
            ws.append([sn, "朋友原模型", 1.0, 1.0, r["friend"][A], None, r["friend"][J]])
            fmt(f0, "CD", "0.000"); fmt(f0, "E", "0.0000"); fmt(f0, "G", "0.000E+00")
            for lab, v in r.items():
                if lab == "friend":
                    continue
                i = ws.max_row + 1
                ws.append([sn, lab.replace("CC dT=", "+") + " °C", v["mults"][0], v["mults"][1], v["point"][A], f"=E{i}/E${f0}",
                           v["point"][J], f"=G{i}/G${f0}"])
                fmt(i, "CD", "0.000"); fmt(i, "E", "0.0000"); fmt(i, "G", "0.000E+00"); fmt(i, "FH", "0.00")

    for row in ws.iter_rows(min_row=1):
        for c in row:
            if c.row not in heads:
                c.font = Font(name=FONT, size=10, bold=c.font.bold)
                c.alignment = Alignment(wrap_text=True, vertical="top")


def value_sheet(wb, d):
    """RDPO value experiments (addons/sensitivity.py E3, nested_uq.py E2); each block has its pre-set rule."""
    ws = wb.create_sheet("Value")
    for i, w in enumerate([26, 16, 13, 13, 13, 13, 13, 13, 13, 13, 40], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    bold, heads = Font(name=FONT, bold=True), set()
    verdict = lambda v: "强" if v == "strong" else "不强"

    def title(text):
        ws.append([]); ws.append([text]); ws.cell(ws.max_row, 1).font = bold

    def head(values):
        ws.append(values); heads.add(ws.max_row)
        for c in ws[ws.max_row]:
            c.font = Font(name=FONT, bold=True, color="FFFFFF"); c.fill = HEAD
            c.alignment = Alignment(wrap_text=True, vertical="top")

    def fmt(i, cols, f):
        for col in cols:
            ws[f"{col}{i}"].number_format = f

    ws.append(["提升 RDPO 价值的实验：朋友的 rdpo_estimate / ELT 转换 / 策略迁移原样调用，只在外层新增；每项都有事先定的判定标准"])
    ws.cell(1, 1).font = bold

    e3 = d.get("e3")
    if e3:
        title(f"E3 一次运行同时给出敏感度（sensitivity.py）：同一批路径给出估计值和每个源发生率的弹性（score function + 有限状态半群导数）；"
              f"判定：|弹性| ≥ 0.05 的源，中位相对误差 ≤ 20%。结论：{verdict(e3['verdict'])}")
        head(["测试床", "源", "精确弹性", "RDPO 均值", "RDPO RMSE", "相对误差", "", "", "", "", "说明"])
        for bed, v in e3["beds"].items():
            ex, est, rm = v["elasticity_exact"], v["elasticity_rdpo_mean"], v["elasticity_rmse"]
            for k in sorted(range(len(ex)), key=lambda k: -abs(ex[k]))[:6]:
                i = ws.max_row + 1
                ws.append([bed, v["sources"][k], ex[k], est[k], rm[k], f'=IF(ABS(C{i})>0,E{i}/ABS(C{i}),"")', None, None, None, None,
                           f"μ = {v['mu_exact']:.3g}，K = {v['K']} 个源" if k == max(range(len(ex)), key=lambda j: abs(ex[j])) else ""])
                fmt(i, "CDE", "+0.000;-0.000"); fmt(i, "F", "0.0%")
            ws.append([f"{bed} 汇总", f"中位 {v['median_rel_rmse_big']:.1%}，最大 {v['max_rel_rmse_big']:.1%}", None, None, None, None, None, None, None, None,
                       f"路径：拟合 {v['paths']['rdpo_fit']} + 敏感度批 {v['paths']['sensitivity_batch']}；有限差分需 {v['paths']['finite_difference_would_need_fits']} 次拟合"])

    e2 = d.get("e2")
    if e2:
        ex = e2["exact_summary"]
        title(f"E2 嵌套不确定性 + 策略迁移（nested_uq.py）：{e2['bed']}，朋友的共享比例后验抽 {e2['draws']} 次，每次精确值由 MPS 给出；"
              f"精确后验：均值 {ex['mean']:.3g}，95% [{ex['q025']:.3g}, {ex['q975']:.3g}]。判定：transfer 的后验摘要误差都 ≤ 10%、路径 ≤ refit 的 1/5、且优于同预算 cold。"
              f"结论：{verdict(e2['verdict'])}")
        head(["方法", "每次预算", "后验均值误差", "2.5% 分位误差", "97.5% 分位误差", "逐次 rel RMSE", "覆盖率", "每轮路径", "每轮 CPU (s)", "", "说明"])
        notes = {"refit": "每次从零拟合", "cold": "每次从零拟合，同 transfer 的小预算", "transfer": "中位数处拟合一次（65536），之后每次用其策略起步"}
        for m, v in e2["methods"].items():
            e, i = v["summary_rel_rmse"], ws.max_row + 1
            ws.append([m, v["per_draw_budget"], e["mean"], e["q025"], e["q975"], v["per_draw_rel_rmse"], v["coverage"], v["paths_per_rep"],
                       v["cpu_sec_per_rep"], None, notes.get(m, "")])
            fmt(i, "CDEF", "0.0%"); fmt(i, "G", "0%"); fmt(i, "H", "0.00E+00"); fmt(i, "I", "0")

    vo = d.get("voi")
    if vo:
        title(f"V 全局敏感度 + 信息价值（voi.py）：朋友两国模型精确计算，{vo['n']} 行 pick-freeze；输入 = 朋友的两个共享比例后验 + rates_unc 的地震 / 降雨触发率不确定性；"
              "输出取 log10。S = 一阶指数（完全知道该输入后平均消掉的方差比例），T = 总指数。判定：方法自检通过、两债同时耗尽的前两名总指数区间不重叠、"
              f"第一名 S ≥ 0.5。结论：{verdict(vo['verdict'])}")
        head(["输出", "输入组", "S", "S 95% 下", "S 95% 上", "T", "T 95% 下", "T 95% 上", "完全信息：区间收窄", "多 10 / 30 / 100 年记录消掉的方差", "说明"])
        names = {"joint_exhaustion": "两债同时耗尽", "expected_loss": "期望损失 EL", "joint_attachment": "两国同时触发"}
        groups = {"s_eq": "地震共享比例", "s_rain": "降雨共享比例", "eq_rates": "地震触发率（KG、TJ）", "rain_rates": "降雨 1-in-50 水平（KG、TJ）"}
        for o, r in vo["outputs"].items():
            s, q = r["sobol"], r["log10_quantiles"]
            for g in groups:
                i, v = ws.max_row + 1, r["voi"][g]
                note = ""
                if g == "s_eq":
                    c = r["contract_reading_B"]
                    note = (f"95% 范围 10^[{q['0.025']:.2f}, {q['0.975']:.2f}]；合同读法 B（余震单独赔付）×{c['factor']:.2f}，"
                            f"若当作五五开的不确定，占方差 {c['share_if_coin_flip']:.0%}（读合同即可消除）")
                ws.append([names[o], groups[g], s["S"][g], *s["S_ci"][g], s["T"][g], *s["T_ci"][g], f"=1-SQRT(MAX(0,1-C{i}))",
                           " / ".join(f"{x:.0%}" for x in v["years"].values()), note])
                fmt(i, "CDEFGHI", "0.000"); fmt(i, "I", "0%")

    mr = d.get("mr")
    if mr:
        title(f"R 模型风险引擎（model_risk.py）：朋友的 CAREC 10 国池，每次抽样重跑他的标定公式（各国地震事件数 + G-R 斜率、共享地震事件数、"
              f"EM-DAT 水文灾害年数、降雨共享比例）；精确目标由按独立分量的稠密枚举 + 卷积给出（与他的 MPS 参考值相对差 < 1e-6，单次 1.3 秒，MPS 约 130 秒）；"
              f"{mr['n']} 行 pick-freeze。判定同 V（以 US${mr['thresholds_musd'][0]}m 超额概率为主）。结论：{verdict(mr['verdict'])}")
        head(["目标", "朋友的值", "数据中心值", "后验预测", "95% 下", "95% 上", "P(高于朋友)", "地震费率 S", "共享地震 S", "降雨费率 S", "说明"])
        names = {"exc_moderate": f"池超额 > US${mr['thresholds_musd'][0]}m", "exc_rare": f"池超额 > US${mr['thresholds_musd'][1]}m",
                 "expected_loss": "期望损失 EL (USD)", "joint_exhaustion_KG_TJ": "KG–TJ 两债同时耗尽"}
        for k, t in mr["targets"].items():
            s, i = t["sobol"], ws.max_row + 1
            ws.append([names.get(k, k), t["friend"], mr["data_center_point"][k], t["posterior_predictive"], *t["range95"], t["P_above_friend"],
                       s["S"]["eq_rates"], s["S"]["eq_shared"], s["S"]["rain_rates"],
                       f"总指数 T：地震费率 {s['T']['eq_rates']:.2f}、共享地震 {s['T']['eq_shared']:.2f}、降雨费率 {s['T']['rain_rates']:.2f}、"
                       f"降雨共享 {s['T']['rain_share']:.2f}"])
            fmt(i, "BCDEF", "0.000E+00"); fmt(i, "G", "0%"); fmt(i, "HIJ", "0.00")

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
        ("运行方式", "repro/run.py 按依赖顺序分阶段运行全部驱动脚本（1–3 复现，5 补充实验，6–9 add-on）；每次运行写入 repro/runs.jsonl 与 repro/logs/"),
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
        ("风险 add-on", "addons/eq_cluster.py（地震余震读法）、rates_unc.py（基础触发率的数据不确定性）：只新增、不碰主轴；"
                       "repro/run.py 第 7 阶段登记运行；数字见 Quality 表"),
        ("气候 add-on", "addons/climate_cc.py（Clausius–Clapeyron 情景）；repro/run.py 第 9 阶段；数字见 Climate 表；"
                      "下载的数据都在 E:\\data，按领域分类"),
        ("RDPO 价值实验", "addons/nested_uq.py（E2 嵌套不确定性 + 策略迁移）、sensitivity.py（E3 一次运行出敏感度）、voi.py（V 全局敏感度 + 信息价值）、model_risk.py（R CAREC 池模型风险引擎）；"
                        "repro/run.py 第 11、14、17 阶段；数字见 Value 表"),
        ("已撤销（非强项）", "router、risk_measures、rain_lag、joint_k、data_gaps、boost、climate_regime、cascade 于 2026-11-07 移入 "
                          "D:\\adb-quantum-challenge\\archive\\2026-11-07_non_strong；更早撤销的 measure、psis 在 xzy 分支 4031fc5；"
                          "rain_tail（日数据 POT）测试后撤销：两债同时耗尽区间只缩到年最大值方法的 0.65–0.77 倍（标准 ≤ 0.5），"
                          "9 国共用尾部形状反而把 TJ 拉偏到 5.2%、区间变宽 1.79 倍；"
                          "NEX-GDDP-CMIP6 33 个模型的三项测试后撤销：climate_ens（KG 观测尾部比所有模型都重、TJ 相反，观测点估计不在集合区间内）；"
                          "E1 rain_elt（16 区 61 个源，最稀有 μ = 6e-7：65536 路径误差 1.3%、覆盖 0.93、比普通 MC 少约 2600 倍，"
                          "但事先定的规则看最小达标预算 16384 处的覆盖率，只有 0.73）；"
                          "M2 rain_share_gcm（降雨越严重越局部：KG–TJ 共享比例 6.2% → 3.5% → 1.3% → 0.8% → 1-in-50 为 0/66，"
                          "但区间宽度是朋友后验的 0.57 倍，标准 ≤ 0.5，且不排除朋友的 2.2%）；"
                          "A 精确收尾 rb_tail（Rao–Blackwell 化：少数国家内的源精确积分；精确核对误差 1e-15、同一策略下方差降 4.6–12 倍，"
                          "但策略是为原目标训练的，前缀权重仍是重尾，只有 complete-4 达标；E1 在 B = 4096 时 z = −5.1、覆盖 0.40）；"
                          "M1 eq_footprint（地震物理足迹：ComCat 带深度目录 + Allen–Wald–Worden 2012 烈度公式，ShakeMap 验证 182 次地震："
                          "档位完全一致 33%、差一档内 67%，标准 70% / 90%；朋友 50 km 规则 24%；境内最大烈度在 MMI 8 附近饱和，4 档阈值挤在 7.76–8.12，"
                          "分不出档位）及其后续多保真控制变量（前提不成立：物理事件集里只有 1.7% 的地震触发 3 国以上，朋友的两两截断几乎无损，未实现）；"
                          "降雨尾部数据计划（GSOD 延长朋友指数 + GHCN 站点部分合并形状参数，R1–R3）经功效分析未做：形状完全已知且多 25 年时，"
                          "KG / TJ 的 2% 触发概率区间也只缩到 0.68 / 0.64 倍（标准 ≤ 0.5），降雨 1-in-50 的不确定性是约 46 年记录决定的结构性下限"
                          "（rain_power.py 与输出在 archive）；"
                          "方向 2 量子加载 q_loading（Q1+Q2 计价下比当前最好 s*）：格点界免费但只降 1.02–1.19 倍，两参数倾斜 0.72–1.18 倍，"
                          "局部可加载 RDPO 策略即使完美也只有 0.20–1.80 倍（多路受控加载太贵，未训练），O(σ/ε) 均值估计要降一半需常数 ≤ 3.5–19"
                          "（ChebAE 实测约 23，论文无常数）；8 个设置无一减半；"
                          "量子嵌套期望 nested（Blanchet 等 NeurIPS 2025 只覆盖 Lipschitz g，目标改为 E[(μ−L)⁺]；complete-7 稠密枚举 4 秒，经典无嵌套惩罚）："
                          "照证明计价 s* = 1.7e7–2.6e7，是单一期望的 2300–3500 倍；去掉 2L⁴ = 1250 后仍为 1.2e4–1.8e4（按 θ 加载每次多 290–610 万 Toffoli）；"
                          "只有常数全去掉且加载便宜时 s* 才到 18–27；"
                          "地震数据升级 eq_data（ComCat 1965–2025、只用 Mw 估 b、ISC-GEM 样本外检验）：两国 EL 区间只缩 19%、池超额缩 26%（标准 30%），"
                          "KG/TJ 区域 b 随完备震级变动 0.115（标准 0.1）；但两国 EL 的不确定性 88–90% 来自 b，池子则来自各国事件数（b 被对齐 brief 抵消）；"
                          "1918–1964 年 M≥6.5 实测 27、预测 [17, 43]，百年数据也不需要有上限的 G-R（p = 0.72），朋友的外推经得起检验；"
                          "模糊溢价→利差 ambiguity（smooth ambiguity 定价，用两只 ADB 债券都是 600 bp 风险利差校准）：点估计 α = 0.016/bp、m₀ = 1.76，"
                          "模糊溢价占利差 30%（KG）/ 37%（TJ），但 200 次 bootstrap 有 59% 根本解不出校准，同价无法用数据不确定性稳健解释；"
                          "亚太 28 国 RDPO + MIS 模型风险引擎 ap_risk：精度（中位误差 0.2% / 0.8%）和算力（MPS 的 0.022）达标，"
                          "但 US$850m 只有 39% 的后验点有效样本量过线（标准 90%）；"
                          "全边条件化 edge_cond（等时间效率是 twisting 的 0.1–0.4 倍）、简化图控制变量（方差最多降 1.1–1.8 倍）、"
                          "规模边界图 scale_frontier（被作者新版的稠密精确解覆盖）"),
        ("2026-11-18 新增", "风险孪生 twin（亚太 28 国池，后验逐年更新 2017–2025，重加权固定路径）：twisting 锚点版为强——"
                          "30 个 MPS 核对点中位误差 0.1–0.4%，总 CPU 是逐点重跑 twisting 的 0.062，k̂ < 0.7 占 100%；"
                          "RDPO 锚点版等 CPU 下误差大 1.2–1.7 倍，所以孪生不需要 RDPO。9 年新数据后池超额 > US$450m 的后验预测 1.90e-3 → 1.88e-3。"
                          "利差带 spread_band（计算，不计入强项）：三条市场曲线在发行期望损失上给出 5.3–6.9%，实付 6%；"
                          "地震触发率的数据不确定性对应 400–600 bp 宽的利差带。"
                          "十个强项已在作者 main e6572b4 上重跑，结论不变；Addons 表的 s* 现在用作者提交的基线。详见 addons/README.md"),
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
    import run as runner   # runs.jsonl stays the full append-only history; the sheet lists jobs still in run.py
    recs = [r for r in map(json.loads, (HERE / "runs.jsonl").read_text().splitlines()) if r["name"] in runner.BY]
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
    files = {"combined": "combined.json", "q1": "q1_rotations.json", "q2": "q2_ae.json"}
    if (ar / "combined.json").exists():
        addons_sheet(wb, {k: json.loads((ar / f).read_text(encoding="utf-8")) for k, f in files.items() if (ar / f).exists()})
    qf = {"eq": "quality_eq_cluster.json", "rates": "quality_rates_unc.json"}
    if any((ar / f).exists() for f in qf.values()):
        quality_sheet(wb, {k: json.loads((ar / f).read_text(encoding="utf-8")) for k, f in qf.items() if (ar / f).exists()})
    vf = {"e2": "nested_uq.json", "e3": "sensitivity.json", "voi": "voi.json", "mr": "model_risk.json"}
    if any((ar / f).exists() for f in vf.values()):
        value_sheet(wb, {k: json.loads((ar / f).read_text(encoding="utf-8")) for k, f in vf.items() if (ar / f).exists()})
    cf = {"cc": "climate_cc.json"}
    if any((ar / f).exists() for f in cf.values()):
        climate_sheet(wb, {k: json.loads((ar / f).read_text(encoding="utf-8")) for k, f in cf.items() if (ar / f).exists()})

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
