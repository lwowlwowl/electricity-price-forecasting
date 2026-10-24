#!/usr/bin/env python
"""生成可直接向老师汇报的电价尾部不足与专家路由进展PDF。

本脚本只读取已冻结的2023/2024预测与已完成的单因素尾部加权结果，
不训练模型，不读取2025作为方法选择依据。
"""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
TAIL_MODULE_PATH = ROOT / "src/decision_aware/tail_diagnostics.py"
TAIL_SPEC = importlib.util.spec_from_file_location("tail_diagnostics_standalone", TAIL_MODULE_PATH)
if TAIL_SPEC is None or TAIL_SPEC.loader is None:
    raise RuntimeError(f"无法加载{TAIL_MODULE_PATH}")
TAIL_MODULE = importlib.util.module_from_spec(TAIL_SPEC)
sys.modules[TAIL_SPEC.name] = TAIL_MODULE
TAIL_SPEC.loader.exec_module(TAIL_MODULE)
daily_peak_report = TAIL_MODULE.daily_peak_report
event_detection_report = TAIL_MODULE.event_detection_report


OUTPUT = ROOT / "output/pdf/电价尾部不足诊断补充汇报_2026-10-21.pdf"
DIAGNOSTIC_JSON = ROOT / "data/results/tail_profit_diagnostics_v1/tail_profit_diagnostics.json"
CONFIRMATION_JSON = (
    ROOT
    / "data/results/universal_lomo_ercot_tailprice_da_outer_confirmation_summary_v1.json"
)
CURRENT_2024_NPZ = (
    ROOT / "data/results/universal_lomo_da_outer_frozen_v1/ercot_da_v1.npz"
)
CANDIDATE_2024_NPZ = (
    ROOT
    / "data/results/universal_lomo_ercot_tailprice_da_predictions_v1"
    / "tail_price_da_da_outer_seed0.npz"
)
ROUTER_AUDIT_JSON = ROOT / "data/results/tail_expert_router_v1/followup_2025_audit_v1.json"
ROUTER_SPEC_JSON = ROOT / "data/results/tail_expert_router_v1/followup_router.json"
CROSS_MARKET_JSON = ROOT / "data/results/tail_expert_router_cross_market_summary_v1.json"
STRICT_ROUTER_JSON = ROOT / "data/results/crossmarket_regret_router_summary_v1.json"

FONT = "MicrosoftYaHei"
FONT_BOLD = "MicrosoftYaHei-Bold"
NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2E6FBB")
CYAN = colors.HexColor("#DCEFFC")
RED = colors.HexColor("#C74440")
ORANGE = colors.HexColor("#E99537")
GREEN = colors.HexColor("#2F7D5B")
LIGHT = colors.HexColor("#F4F7FA")
MID = colors.HexColor("#D9E2EA")
GRAY = colors.HexColor("#5C6773")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont(FONT, r"C:\Windows\Fonts\msyh.ttc"))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, r"C:\Windows\Fonts\msyhbd.ttc"))


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=FONT_BOLD,
            fontSize=25,
            leading=34,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=10 * mm,
            wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=13,
            leading=21,
            textColor=GRAY,
            spaceAfter=7 * mm,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=18,
            leading=25,
            textColor=NAVY,
            spaceBefore=2 * mm,
            spaceAfter=5 * mm,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=13,
            leading=19,
            textColor=BLUE,
            spaceBefore=3 * mm,
            spaceAfter=2 * mm,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=10.5,
            leading=17,
            textColor=colors.HexColor("#263442"),
            spaceAfter=2.5 * mm,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=8,
            leading=12,
            textColor=GRAY,
            wordWrap="CJK",
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base["BodyText"],
            fontName=FONT_BOLD,
            fontSize=14,
            leading=23,
            textColor=NAVY,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
        "card_value": ParagraphStyle(
            "card_value",
            parent=base["Normal"],
            fontName=FONT_BOLD,
            fontSize=19,
            leading=22,
            textColor=RED,
            alignment=TA_CENTER,
        ),
        "card_label": ParagraphStyle(
            "card_label",
            parent=base["Normal"],
            fontName=FONT,
            fontSize=8.5,
            leading=12,
            textColor=GRAY,
            alignment=TA_CENTER,
            wordWrap="CJK",
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def metric_cards(items: list[tuple[str, str]], sty: dict) -> Table:
    cells = [
        [para(value, sty["card_value"]), para(label, sty["card_label"])]
        for value, label in items
    ]
    table = Table([cells], colWidths=[43 * mm] * len(items), rowHeights=[27 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.7, MID),
                ("INNERGRID", (0, 0), (-1, -1), 0.7, MID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
            ]
        )
    )
    return table


def report_table(data: list[list], widths: list[float], sty: dict) -> Table:
    converted = []
    for r, row in enumerate(data):
        converted.append(
            [
                para(str(value), sty["small"] if r else ParagraphStyle(
                    f"header_{r}_{c}",
                    parent=sty["small"],
                    fontName=FONT_BOLD,
                    textColor=colors.white,
                    alignment=TA_CENTER,
                ))
                for c, value in enumerate(row)
            ]
        )
    table = Table(converted, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.45, MID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2 * mm),
            ]
        )
    )
    return table


def line_chart(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    title: str,
    subtitle: str,
    threshold: float | None = None,
) -> Drawing:
    width, height = 170 * mm, 68 * mm
    left, bottom, plot_w, plot_h = 18 * mm, 13 * mm, 143 * mm, 42 * mm
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 5 * mm, title, fontName=FONT_BOLD, fontSize=11, fillColor=NAVY))
    drawing.add(String(0, height - 10 * mm, subtitle, fontName=FONT, fontSize=7.5, fillColor=GRAY))
    values = np.concatenate([truth, prediction])
    y_min = float(min(0.0, np.min(values)))
    y_max = float(max(0.0, np.max(values)))
    pad = max((y_max - y_min) * 0.08, 1.0)
    y_min -= pad
    y_max += pad

    def xy(hour: int, value: float) -> tuple[float, float]:
        x = left + plot_w * hour / max(len(truth) - 1, 1)
        y = bottom + plot_h * (value - y_min) / max(y_max - y_min, 1e-12)
        return x, y

    drawing.add(Rect(left, bottom, plot_w, plot_h, fillColor=colors.white, strokeColor=MID, strokeWidth=0.7))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom + plot_h * frac
        value = y_min + (y_max - y_min) * frac
        drawing.add(Line(left, y, left + plot_w, y, strokeColor=colors.HexColor("#E8EDF2"), strokeWidth=0.4))
        drawing.add(String(left - 2 * mm, y - 1.5 * mm, f"{value:.0f}", fontName=FONT, fontSize=6.5, fillColor=GRAY, textAnchor="end"))
    for hour in (0, 6, 12, 18, 23):
        x, _ = xy(hour, 0)
        drawing.add(String(x, bottom - 4 * mm, str(hour), fontName=FONT, fontSize=6.5, fillColor=GRAY, textAnchor="middle"))
    if threshold is not None and y_min <= threshold <= y_max:
        _, ty = xy(0, threshold)
        drawing.add(Line(left, ty, left + plot_w, ty, strokeColor=ORANGE, strokeWidth=0.8, strokeDashArray=[3, 2]))
    for series, color in ((truth, RED), (prediction, BLUE)):
        for hour in range(1, len(series)):
            x1, y1 = xy(hour - 1, float(series[hour - 1]))
            x2, y2 = xy(hour, float(series[hour]))
            drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.5))
    drawing.add(Line(left + 100 * mm, height - 5 * mm, left + 108 * mm, height - 5 * mm, strokeColor=RED, strokeWidth=1.7))
    drawing.add(String(left + 110 * mm, height - 6.5 * mm, "真实价格", fontName=FONT, fontSize=7.2, fillColor=GRAY))
    drawing.add(Line(left + 126 * mm, height - 5 * mm, left + 134 * mm, height - 5 * mm, strokeColor=BLUE, strokeWidth=1.7))
    drawing.add(String(left + 136 * mm, height - 6.5 * mm, "预测", fontName=FONT, fontSize=7.2, fillColor=GRAY))
    return drawing


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files if key in {"nodes", "dates", "prediction", "target"}}


def choose_example(data: dict[str, np.ndarray], kind: str) -> tuple[str, str, np.ndarray, np.ndarray]:
    target = np.asarray(data["target"], dtype=np.float64)
    prediction = np.asarray(data["prediction"], dtype=np.float64)
    groups = target.reshape(-1, target.shape[-1])
    pred_groups = prediction.reshape(-1, prediction.shape[-1])
    if kind == "high":
        group_index = int(np.argmax(np.max(groups, axis=1)))
    elif kind == "negative":
        group_index = int(np.argmin(np.min(groups, axis=1)))
    else:
        raise ValueError(kind)
    days = target.shape[1]
    node_index, day_index = divmod(group_index, days)
    return (
        str(data["nodes"][node_index]),
        str(data["dates"][day_index]),
        groups[group_index],
        pred_groups[group_index],
    )


def page_decor(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, A4[1] - 6 * mm, A4[0], 6 * mm, fill=1, stroke=0)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(GRAY)
    canvas.drawString(18 * mm, 10 * mm, "Electricity Price Forecasting | Tail Diagnostic")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    register_fonts()
    sty = styles()
    diagnostic = json.loads(DIAGNOSTIC_JSON.read_text(encoding="utf-8"))
    confirmation = json.loads(CONFIRMATION_JSON.read_text(encoding="utf-8"))
    current = load_npz(CURRENT_2024_NPZ)
    candidate = load_npz(CANDIDATE_2024_NPZ)
    router = json.loads(ROUTER_AUDIT_JSON.read_text(encoding="utf-8"))
    router_spec = json.loads(ROUTER_SPEC_JSON.read_text(encoding="utf-8"))
    cross_market = json.loads(CROSS_MARKET_JSON.read_text(encoding="utf-8"))
    strict_router = json.loads(STRICT_ROUTER_JSON.read_text(encoding="utf-8"))
    if not np.array_equal(current["nodes"].astype(str), candidate["nodes"].astype(str)):
        raise ValueError("当前模型与尾部加权候选的节点不一致")
    if not np.array_equal(current["dates"].astype(str), candidate["dates"].astype(str)):
        raise ValueError("当前模型与尾部加权候选的日期不一致")
    if not np.allclose(current["target"], candidate["target"], rtol=0, atol=1e-3):
        raise ValueError("当前模型与尾部加权候选的真值不一致")

    threshold = float(confirmation["thresholds_frozen_from_2020_2022"]["da_upper_1pct"])
    current_positive = event_detection_report(
        current["prediction"], current["target"],
        threshold=threshold, direction="above", tolerance_hours=1,
    )
    current_negative = event_detection_report(
        current["prediction"], current["target"],
        threshold=0.0, direction="below", tolerance_hours=1,
    )
    peak = daily_peak_report(
        current["prediction"], current["target"], positive_threshold=threshold
    )
    candidate_positive = event_detection_report(
        candidate["prediction"], candidate["target"],
        threshold=threshold, direction="above", tolerance_hours=1,
    )
    candidate_negative = event_detection_report(
        candidate["prediction"], candidate["target"],
        threshold=0.0, direction="below", tolerance_hours=1,
    )
    current_summary = confirmation["forecast"]["current"]
    candidate_summary = confirmation["forecast"]["tail_price_da"]
    spread_2024 = diagnostic["forecast_slices"]["ERCOT"]["transformer_2024"]["da_rt_at_da_spread"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=17 * mm,
        title="电价尾部不足诊断与专家路由补充汇报",
        author="Electricity Price Forecasting Project",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=page_decor)])
    story = []

    # 1. Executive summary
    story.extend([
        Spacer(1, 10 * mm),
        para("电价尾部不足诊断<br/>与专家路由进展", sty["title"]),
        para(
            "补充汇报 | 从尾部失败审计到可解释经济专家路由 | 2026-10-21",
            sty["subtitle"],
        ),
        Table(
            [[para(
                "<b>一句话结论：</b>当前模型不是整体 MAE 太差，而是把少数极端价格系统性压平。"
                "负价大多数没识别，正尖峰全部漏报，极端 DA-RT 价差幅度只保留约 2%。",
                sty["callout"],
            )]],
            colWidths=[174 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), CYAN),
                ("BOX", (0, 0), (-1, -1), 1.0, BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 7 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7 * mm),
            ]),
        ),
        Spacer(1, 9 * mm),
        metric_cards([
            ("0%", "2024正尖峰精确召回率"),
            ("22.2%", "2024负价精确召回率"),
            ("79×", "最高尾部MAE / 普通时段MAE"),
            ("1.8%", "2024极端价差幅度保留比"),
        ], sty),
        Spacer(1, 8 * mm),
        para("老师的判断可以被具体化为三个问题：", sty["h2"]),
        para(
            "1）<b>事件判断不足</b>：尖峰和负价有没有发生都判断不好；"
            "2）<b>幅度还原不足</b>：即使方向偶尔正确，价格也被明显压向中心；"
            "3）<b>决策信号不足</b>：DA-RT 极端价差几乎被抹平，影响 DA 充放电计划。",
            sty["body"],
        ),
        para(
            "汇报口径：这是对已冻结预测的分层诊断，不是通过重新调参获得的结论。",
            sty["small"],
        ),
        PageBreak(),
    ])

    # 2. Protocol and current model
    high_ratio = current_summary["highest_1pct"]["mae"] / current_summary["ordinary"]["mae"]
    story.extend([
        para("1. 怎么定义和检验“尾部”", sty["h1"]),
        para(
            f"正尖峰阈值为 <b>{threshold:.2f} USD/MWh</b>，由 ERCOT 2020-2022 训练历史的DA价格最高1%冻结；"
            "负价按经济定义固定为价格低于0。评价用未参与阈值拟合的2024冻结时间外样本，"
            f"共15个节点、{current_summary['all']['samples']:,}个小时样本。",
            sty["body"],
        ),
        report_table([
            ["切片", "样本数", "MAE", "偏差", "关键现象"],
            ["普通时段", f"{current_summary['ordinary']['samples']:,}", f"{current_summary['ordinary']['mae']:.2f}", f"{current_summary['ordinary']['bias']:.2f}", "总体看起来较好"],
            ["负价", f"{current_summary['negative_price']['samples']:,}", f"{current_summary['negative_price']['mae']:.2f}", f"{current_summary['negative_price']['bias']:+.2f}", "系统性高估，多数被拉回正价"],
            ["最高1%", f"{current_summary['highest_1pct']['samples']:,}", f"{current_summary['highest_1pct']['mae']:.2f}", f"{current_summary['highest_1pct']['bias']:+.2f}", "所有误差都是低估"],
        ], [31 * mm, 25 * mm, 26 * mm, 27 * mm, 65 * mm], sty),
        Spacer(1, 5 * mm),
        para(
            f"最高1%尾部 MAE 是普通时段的 <b>{high_ratio:.1f}倍</b>。"
            f"正尖峰精确召回率为 <b>{100*current_positive['exact']['recall']:.1f}%</b>，"
            f"允许前合1小时仍只有 <b>{100*current_positive['within_1h']['recall']:.1f}%</b>。"
            f"负价精确召回率为 <b>{100*current_negative['exact']['recall']:.1f}%</b>，"
            f"即约 <b>{100*(1-current_negative['exact']['recall']):.1f}%</b> 的负价小时未被正确标出。",
            sty["body"],
        ),
        para("峰值时点与幅度分解", sty["h2"]),
        metric_cards([
            (f"{peak['mean_absolute_timing_error_hours']:.1f}h", "含尖峰节点-日的平均峰值时点误差"),
            (f"{100*peak['within_1h_rate']:.1f}%", "日峰值时点在±1小时内"),
            (f"{100*peak['predicted_daily_peak_to_true_peak_ratio']:.1f}%", "预测日峰值 / 真实日峰值"),
            (f"{100*current_positive['magnitude_ratio_at_true_events']:.1f}%", "真实尖峰时刻的超阈幅度保留比"),
        ], sty),
        Spacer(1, 4 * mm),
        para(
            "因此问题不只是“峰值错了1小时”：模型对是否发生正尖峰的阈值判断为零召回，"
            "并且峰值幅度显著压缩。",
            sty["body"],
        ),
        PageBreak(),
    ])

    # 3. Examples
    high_node, high_date, high_truth, high_pred = choose_example(current, "high")
    low_node, low_date, low_truth, low_pred = choose_example(current, "negative")
    story.extend([
        para("2. 典型日：曲线被明显压平", sty["h1"]),
        line_chart(
            high_truth, high_pred,
            title="正尖峰案例",
            subtitle=f"ERCOT {high_node} | {high_date} | 当日真实峰值 {np.max(high_truth):.1f}，预测峰值 {np.max(high_pred):.1f} USD/MWh",
            threshold=threshold,
        ),
        Spacer(1, 4 * mm),
        line_chart(
            low_truth, low_pred,
            title="负价案例",
            subtitle=f"ERCOT {low_node} | {low_date} | 当日真实最低 {np.min(low_truth):.1f}，预测最低 {np.min(low_pred):.1f} USD/MWh",
            threshold=0.0,
        ),
        Spacer(1, 3 * mm),
        para(
            "这两个案例仅用于把统计结果可视化：正尖峰时，预测曲线基本没有跟随真实幅度；"
            "负价时，预测容易停留在零附近或正价区间。正式结论仍以上一页全样本统计为准。",
            sty["small"],
        ),
        PageBreak(),
    ])

    # 4. Broad evidence
    story.extend([
        para("3. 不是一个市场或一种骨干的偶然", sty["h1"]),
        para(
            "下表都使用各2020-2022训练历史冻结阈值，在2023留出市场评价。"
            "它支持“当前模型和已评价TSFM普遍压平尾部”，但不应扩展为“所有Transformer必然失败”。",
            sty["body"],
        ),
    ])
    market_rows = [["市场", "DA普通MAE", "DA最高1% MAE", "倍率", "正尖峰召回", "负价召回"]]
    for market in ("CAISO", "ERCOT", "NYISO", "PJM"):
        item = diagnostic["forecast_slices"][market]["transformer_2023"]["da"]
        market_rows.append([
            market,
            f"{item['ordinary']['mae']:.2f}",
            f"{item['highest_1pct']['mae']:.2f}",
            f"{item['highest_1pct']['mae']/item['ordinary']['mae']:.1f}×",
            f"{100*item['all']['spike_recall']:.1f}%",
            f"{100*item['all']['negative_recall']:.1f}%",
        ])
    story.extend([
        report_table(market_rows, [24 * mm, 29 * mm, 34 * mm, 20 * mm, 33 * mm, 30 * mm], sty),
        Spacer(1, 7 * mm),
        para("ERCOT 2023同信息合同跨模型对照", sty["h2"]),
    ])
    system_labels = {
        "transformer_2023": "当前Transformer",
        "xgboost_pooled_2023": "pooled XGBoost",
        "timesfm3_2023": "TimesFM-3",
        "timesfm25_lora_2023": "TimesFM-2.5 LoRA",
    }
    system_rows = [["系统", "DA普通MAE", "DA最高1% MAE", "正尖峰召回", "负价召回", "极端价差幅度比"]]
    for key, label in system_labels.items():
        item = diagnostic["forecast_slices"]["ERCOT"][key]
        system_rows.append([
            label,
            f"{item['da']['ordinary']['mae']:.2f}",
            f"{item['da']['highest_1pct']['mae']:.2f}",
            f"{100*item['da']['all']['spike_recall']:.1f}%",
            f"{100*item['da']['all']['negative_recall']:.1f}%",
            f"{100*item['da_rt_at_da_spread']['abs_spread_top_1pct']['magnitude_ratio']:.1f}%",
        ])
    story.extend([
        report_table(system_rows, [39 * mm, 27 * mm, 34 * mm, 28 * mm, 26 * mm, 31 * mm], sty),
        Spacer(1, 5 * mm),
        para(
            "关键解读：原生 TimesFM-3 并没有自动解决尾部；普通 LoRA 适配有所改善，"
            "但对正尖峰和极端价差幅度仍然很弱。因此后续应验证“针对尾部的机制”，而不是只换骨干。",
            sty["body"],
        ),
        PageBreak(),
    ])

    # 5. Existing simple intervention
    paired = confirmation["profit"]["paired_vs_current"]
    story.extend([
        para("4. 补充对照：简单尾部加权只解决了一部分", sty["h1"]),
        para(
            "唯一改动是把训练期DA价格最低/最高1%样本的总MAE权重从1变为2。"
            "模型结构、数据、seed、RT两条腿和LP/MPC合同不变。",
            sty["body"],
        ),
        report_table([
            ["指标（2024冻结确认）", "当前Transformer", "尾部加权DA", "变化"],
            ["全体DA MAE", f"{current_summary['all']['mae']:.2f}", f"{candidate_summary['all']['mae']:.2f}", f"{candidate_summary['all']['mae']-current_summary['all']['mae']:+.2f}"],
            ["普通时段DA MAE", f"{current_summary['ordinary']['mae']:.2f}", f"{candidate_summary['ordinary']['mae']:.2f}", f"{candidate_summary['ordinary']['mae']-current_summary['ordinary']['mae']:+.2f}"],
            ["负价MAE", f"{current_summary['negative_price']['mae']:.2f}", f"{candidate_summary['negative_price']['mae']:.2f}", f"{candidate_summary['negative_price']['mae']-current_summary['negative_price']['mae']:+.2f}"],
            ["负价精确召回", f"{100*current_negative['exact']['recall']:.1f}%", f"{100*candidate_negative['exact']['recall']:.1f}%", f"{100*(candidate_negative['exact']['recall']-current_negative['exact']['recall']):+.1f} pp"],
            ["正尖峰精确召回", f"{100*current_positive['exact']['recall']:.1f}%", f"{100*candidate_positive['exact']['recall']:.1f}%", "无改善"],
            ["最高1% DA MAE", f"{current_summary['highest_1pct']['mae']:.2f}", f"{candidate_summary['highest_1pct']['mae']:.2f}", f"{candidate_summary['highest_1pct']['mae']-current_summary['highest_1pct']['mae']:+.2f}"],
        ], [60 * mm, 39 * mm, 39 * mm, 32 * mm], sty),
        Spacer(1, 6 * mm),
        metric_cards([
            (f"+{paired['mean_daily_difference']:.2f}", "日均净收益差（USD/节点·日）"),
            (f"[{paired['seven_day_block_bootstrap_95ci'][0]:.2f}, {paired['seven_day_block_bootstrap_95ci'][1]:.2f}]", "7日块Bootstrap 95%区间"),
            (f"{100*candidate_negative['exact']['recall']:.1f}%", "加权后负价精确召回"),
            (f"{100*candidate_positive['exact']['recall']:.1f}%", "加权后正尖峰精确召回"),
        ], sty),
        Spacer(1, 6 * mm),
        para(
            "结论不是“加权无效”：它改善了负价识别和下游收益。但它仍未召回任何正尖峰，"
            "也没有降低最高1%价格的MAE。这说明单一的固定权重不足以完成老师所说的尾部修复。",
            sty["body"],
        ),
        PageBreak(),
    ])

    # 6. Interpretable expert router
    static_cmp = router["comparisons"]["static_hybrid"]
    transformer_cmp = router["comparisons"]["transformer"]
    xgb_cmp = router["comparisons"]["per_node_xgboost"]
    router_tail = router["tail_attribution"]["prediction_metrics"]["router"]
    story.extend([
        para("5. 从尾部修复转向可解释专家路由", sty["h1"]),
        para(
            "固定尾部加权改善了负价识别，却没有恢复正尖峰；局部峰值损失只在一个seed有效；日级尖峰概率门控"
            "又因跨市场概率校准失败而停止。因此后续没有继续堆叠损失或调阈值，而是利用已经存在的模型互补："
            "DLinear负责部分DA/RT-at-DA节点—日，Transformer负责其余日前任务并始终负责rolling RT。",
            sty["body"],
        ),
        para("路由器只使用三个日前可见信号", sty["h2"]),
        report_table([
            ["条件", "选择", "经济解释"],
            ["Transformer价差q10≤-0.10且最大价差≤6.71", "DLinear", "Transformer日前价差整体偏弱"],
            ["Transformer价差q10≤-0.10且最大价差>6.71", "Transformer", "保留明确正价差机会"],
            ["q10>-0.10且DLinear含负价差小时", "DLinear", "利用DLinear负价差敏感性"],
            ["其余节点—日", "Transformer", "避免无依据切换"],
        ], [78 * mm, 35 * mm, 61 * mm], sty),
        Spacer(1, 5 * mm),
        para(
            "规则由2023日期阻塞折外记录选择，在2024不重选参数地确认；随后保持树深、特征全集和0.70阈值不变，"
            "用2023+2024样本外记录重拟合。2025不参与参数拟合，且之后不允许继续移动阈值。",
            sty["small"],
        ),
        Spacer(1, 6 * mm),
        para("ERCOT 2025完整结算", sty["h2"]),
        report_table([
            ["系统", "净收益", "路由优势", "7日块95%区间", "改善节点"],
            ["原Transformer", f"{transformer_cmp['baseline_mean']:.2f}", f"+{transformer_cmp['mean_difference']:.2f}",
             f"[{transformer_cmp['difference_7day_block_bootstrap_95ci'][0]:.2f}, {transformer_cmp['difference_7day_block_bootstrap_95ci'][1]:.2f}]", "15/15"],
            ["静态DLinear日前＋Transformer RT", f"{static_cmp['baseline_mean']:.2f}", f"+{static_cmp['mean_difference']:.2f}",
             f"[{static_cmp['difference_7day_block_bootstrap_95ci'][0]:.2f}, {static_cmp['difference_7day_block_bootstrap_95ci'][1]:.2f}]", "13/15"],
            ["逐节点XGBoost", f"{xgb_cmp['baseline_mean']:.2f}", f"+{xgb_cmp['mean_difference']:.2f}",
             f"[{xgb_cmp['difference_7day_block_bootstrap_95ci'][0]:.2f}, {xgb_cmp['difference_7day_block_bootstrap_95ci'][1]:.2f}]", "12/15"],
            ["专家路由", f"{router['router_macro_mean_daily_revenue']:.2f}", "—", "自身[47.81, 79.74]", "—"],
        ], [52 * mm, 24 * mm, 27 * mm, 43 * mm, 28 * mm], sty),
        Spacer(1, 5 * mm),
        para(
            "路由是当前点估计最高的系统；但与静态组合和XGBoost的配对区间仍跨0，所以只能说“点估计领先、"
            "改善覆盖多数节点”，不能说统计显著战胜强基线。收益增加主要来自DA计划腿，并已扣除更高吞吐带来的"
            "退化成本与偏差罚金。",
            sty["body"],
        ),
        PageBreak(),
    ])

    # 7. Cross-market transfer
    cross_rows = [["市场", "Transformer", "静态组合", "专家路由", "路由差", "DLinear切换"]]
    for market in ("CAISO", "ERCOT", "NYISO", "PJM"):
        item = cross_market["markets"][market]
        value = item["standardized_mean_daily_value"]
        cross_rows.append([
            market,
            f"{value['transformer']:.2f}",
            f"{value['static_hybrid']:.2f}",
            f"{value['frozen_router']:.2f}",
            f"{item['router_minus_best_static']:+.2f}",
            f"{100*item['dlinear_route_share']:.1f}%",
        ])
    market_equal = cross_market["market_equal"]
    market_ci = market_equal["paired_7day_block_bootstrap_95ci"]
    cross_rows.append([
        "市场等权", "—", f"{market_equal['best_static_per_market']:.2f}",
        f"{market_equal['frozen_router']:.2f}", f"{market_equal['difference']:+.2f}", "—",
    ])
    story.extend([
        para("6. 冻结规则能否跨市场工作", sty["h1"]),
        para(
            "把ERCOT 2023+2024得到的同一路由器原封不动应用到CAISO、NYISO、PJM和ERCOT 2025。"
            "目标市场不重拟合、不移动阈值，也不只挑有利市场报告。",
            sty["body"],
        ),
        report_table(cross_rows, [29 * mm, 28 * mm, 31 * mm, 31 * mm, 27 * mm, 28 * mm], sty),
        Spacer(1, 6 * mm),
        metric_cards([
            ("3/4", "路由点估计超过最强静态系统的市场"),
            (f"+{market_equal['difference']:.2f}", "市场等权标准化价值差"),
            (f"[{market_ci[0]:.2f}, {market_ci[1]:.2f}]", "市场等权7日块95%区间"),
            (f"{100*router_spec['selection_rate_hybrid_2025']:.1f}%", "ERCOT切换到DLinear的节点—日"),
        ], sty),
        Spacer(1, 6 * mm),
        para("如何理解这组跨市场结果", sty["h2"]),
        para(
            "• CAISO、NYISO和ERCOT的极端价差方向相对静态组合有所改善；PJM的尾部识别没有同步改善。<br/>"
            "• 四市场的总体DA与RT-at-DA误差都比始终使用DLinear更低，但仍高于纯Transformer。<br/>"
            "• 经济点估计在ERCOT、NYISO、PJM为正，CAISO为-0.15；市场等权区间仍跨0。<br/>"
            "• 这支持“什么时候选哪个专家”具有一定跨市场规律，不支持“路由普遍修复了尾部”。",
            sty["body"],
        ),
        para(
            "重要边界：CAISO、NYISO、PJM尚未冻结真实偏差费用与储能结算规则，因此表中数字是同一电池、LP/MPC、"
            "退化与偏差罚金下的标准化研究价值，不是三个市场的真实可实现利润。",
            sty["small"],
        ),
        PageBreak(),
    ])

    # 8. Strict no-target-label LOMO gate
    strict_rows = [["留出市场", "最强静态", "经济路由", "收益差", "95%区间", "DLinear切换"]]
    for market in ("CAISO", "ERCOT", "NYISO", "PJM"):
        item = strict_router["markets"][market]
        family = (
            "transformer"
            if item["best_static_family"].startswith("da_transformer__")
            else "static_hybrid"
        )
        baseline = item["systems"][family]["mean_daily_net_revenue"]
        route = item["systems"]["economic_regret_tree"]
        ci = route["difference_7day_block_bootstrap_95ci"]
        strict_rows.append([
            market, f"{baseline:.2f}", f"{route['mean_daily_net_revenue']:.2f}",
            f"{route['difference_vs_best_static']:+.2f}",
            f"[{ci[0]:.2f}, {ci[1]:.2f}]",
            f"{100*item['router']['dlinear_selected_node_day_rate']:.1f}%",
        ])
    strict_equal = strict_router["market_equal_summary"]
    strict_ci = strict_equal["hierarchical_7day_block_bootstrap_95ci"]
    strict_pass = strict_router["pre_registered_gate"]["main_method_passes_pre_registered_core_gate"]
    story.extend([
        para("7. 更严格的无ID跨市场检验", sty["h1"]),
        para(
            "前一页是把ERCOT规则固定迁移；本页进一步要求两套基础预测器和路由器都不看留出市场标签。"
            "每一折只用另外三个市场2020—2022训练预测器，并用三个源市场2023完整LP/MPC收益学习一棵深度2"
            "经济遗憾树，再原样应用于第四个市场。主标签是DLinear相对Transformer多赚的完整净收益，MAE只作消融。",
            sty["body"],
        ),
        report_table(strict_rows, [29 * mm, 28 * mm, 28 * mm, 26 * mm, 36 * mm, 27 * mm], sty),
        Spacer(1, 6 * mm),
        metric_cards([
            (f"{strict_equal['positive_markets']}/4", "严格留出市场中收益差为正"),
            (f"{strict_equal['mean_daily_profit_difference_vs_best_static']:+.2f}", "市场等权收益差"),
            (f"[{strict_ci[0]:.2f}, {strict_ci[1]:.2f}]", "分层7日块95%区间"),
            ("通过" if strict_pass else "未通过", "预注册核心门槛"),
        ], sty),
        Spacer(1, 6 * mm),
        para("为什么这不是单纯调参", sty["h2"]),
        para(
            "• 树深、叶节点大小、切换条件和特征在读取留出市场结果前固定。<br/>"
            "• 同时比较经济收益标签、MAE标签和去掉尾部特征三条路线；主方法不能按市场更换。<br/>"
            "• 只有经济标签优于MAE标签且规则实际使用尾部/价差信号，才能说明改进来自可解释的经济机制。<br/>"
            "• CAISO、NYISO、PJM仍是统一研究合同下的标准化价值，不冒充当地真实利润。",
            sty["body"],
        ),
        para(
            (
                "严格门槛通过：结果支持在不读取目标市场标签时，日前专家的经济互补仍具有跨市场可迁移性。"
                if strict_pass else
                "严格门槛未通过：结果只能保留为负证据，不能继续根据留出市场移动阈值或挑选规则。"
            ),
            sty["small"],
        ),
        PageBreak(),
    ])

    # 9. Final conclusions
    story.extend([
        para("8. 最终结论", sty["h1"]),
        para("已经改善的部分", sty["h2"]),
        para(
            "• 路由器利用预测价差的低分位数、最大值和负价差小时数，改善了DA专家选择和完整净收益。<br/>"
            "• ERCOT 2025点估计达到63.53 USD/节点·日，高于静态组合61.80、DLinear 61.53和逐节点XGBoost 59.99。<br/>"
            "• 冻结规则在3/4市场取得正向标准化价值，说明模型互补不完全是ERCOT偶然。",
            sty["body"],
        ),
        Spacer(1, 5 * mm),
        para("仍未解决的部分", sty["h2"]),
        para(
            f"• ERCOT绝对正尖峰召回仍为{100*router_tail['positive_spike_recall']:.1f}%，峰值幅度平均只保留"
            f"{100*router_tail['positive_spike_peak_amplitude_ratio_mean']:.1f}%。<br/>"
            "• 路由相对最近强基线以及四市场等权增益的95%区间仍跨0，不能声称统计显著。<br/>"
            "• PJM尾部指标没有随经济点估计同步改善，路由不是通用尖峰预测器。",
            sty["body"],
        ),
        Spacer(1, 6 * mm),
        para("对老师汇报时的准确表述", sty["h2"]),
        Table(
            [[para(
                "我们没有继续把单个模型的尖峰幅度硬拉高，而是用一个只有三个预测价差信号的浅树，在每天、每个节点"
                "选择Transformer或DLinear的日前专家。它在ERCOT取得当前最高收益点估计，并在其他市场表现出一定"
                "迁移方向；但统计区间仍跨0，绝对尖峰召回仍未解决。",
                sty["callout"],
            )]],
            colWidths=[174 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F5EE")),
                ("BOX", (0, 0), (-1, -1), 1.0, GREEN),
                ("LEFTPADDING", (0, 0), (-1, -1), 8 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 7 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7 * mm),
            ]),
        ),
        Spacer(1, 8 * mm),
        para("证据边界", sty["h2"]),
        para(
            "该方法是在原2025基线结果已经被观察后提出。2025没有参与参数拟合，但它属于固定事后机制验证，"
            "不是首次盲测；后续不得再根据2025移动路由阈值、删改特征或选择市场。",
            sty["small"],
        ),
    ])

    doc.build(story)
    print(json.dumps({
        "output": str(OUTPUT),
        "pages_expected": 9,
        "current_positive": current_positive,
        "current_negative": current_negative,
        "daily_peak": peak,
        "spread_top1_magnitude_ratio_2024": spread_2024["abs_spread_top_1pct"]["magnitude_ratio"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build_pdf()
