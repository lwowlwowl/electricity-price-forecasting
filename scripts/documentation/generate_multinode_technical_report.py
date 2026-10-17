#!/usr/bin/env python
"""Generate the multi-node technical report as a polished PDF.

The report is deliberately evidence-backed: current zero-ID results are read from
machine-readable JSON artifacts, while historical ablations are clearly marked as
historical and are never mixed into the current deployment claim.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "多节点无ID通用电价预测与储能双结算技术报告_2026-10-13.pdf"

NAVY = HexColor("#14213D")
BLUE = HexColor("#2F6BFF")
TEAL = HexColor("#00A6A6")
GREEN = HexColor("#2A9D6F")
ORANGE = HexColor("#F4A261")
RED = HexColor("#D1495B")
PURPLE = HexColor("#6C5CE7")
INK = HexColor("#1F2937")
MUTED = HexColor("#64748B")
LIGHT = HexColor("#F4F7FB")
GRID = HexColor("#D9E2F0")
PALE_BLUE = HexColor("#EAF1FF")
PALE_GREEN = HexColor("#E9F8F1")
PALE_ORANGE = HexColor("#FFF4E8")
PALE_RED = HexColor("#FDECEF")


def load_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def load_json_if_exists(rel: str) -> dict | None:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def locked_2025_comparison_rows(summary: dict) -> list[list[str]]:
    """把锁定测试汇总转成可直接进入论文的比较表，并加粗每列最佳可部署系统。"""
    specs = (
        ("transformer_ensemble", "当前Transformer固定集成", "da_ensemble__rt_at_da_ensemble__rt_ensemble"),
        ("timesfm3", "TimesFM-3 frozen", "da_timesfm3__rt_at_da_timesfm3__rt_timesfm3"),
        ("timesfm25", "TimesFM-2.5 LoRA", "da_timesfm25__rt_at_da_timesfm25__rt_timesfm25"),
        ("pooled_xgboost", "pooled XGBoost", "da_pooled_xgboost__rt_at_da_pooled_xgboost__rt_pooled_xgboost"),
        ("per_node_xgboost", "逐节点XGBoost系统", "da_per_node_xgboost__rt_at_da_per_node_xgboost__rt_per_node_xgboost"),
        ("dlinear", "DLinear", "da_dlinear__rt_at_da_dlinear__rt_dlinear"),
        ("seasonal", "lag-168", "da_seasonal__rt_at_da_seasonal__rt_seasonal"),
    )
    tasks = ("da", "rt_at_da", "rt")
    records = []
    derived = summary["ercot_settlement_derived"]
    for system, label, settlement_label in specs:
        task_mae = {}
        for task in tasks:
            values = [
                float(summary["markets"][market]["systems"][system][task]["all"]["mae"])
                for market in ("CAISO", "ERCOT", "NYISO", "PJM")
            ]
            task_mae[task] = sum(values) / len(values)
        settlement = derived[settlement_label]
        records.append({
            "label": label,
            **task_mae,
            "revenue": float(settlement["macro_mean_daily_revenue"]),
            "oracle_fraction": settlement["fraction_of_oracle"],
        })
    best = {task: min(row[task] for row in records) for task in tasks}
    best_revenue = max(row["revenue"] for row in records)
    rows: list[list[str]] = [[
        "系统", "四市场DA MAE", "四市场RT-at-DA MAE", "四市场滚动RT MAE",
        "ERCOT日均净收益", "Oracle占比",
    ]]
    for row in records:
        values = []
        for task in tasks:
            text = f"{row[task]:.2f}"
            values.append(B(text) if math.isclose(row[task], best[task]) else text)
        revenue = f"{row['revenue']:.2f}"
        if math.isclose(row["revenue"], best_revenue):
            revenue = B(revenue)
        fraction = row["oracle_fraction"]
        rows.append([
            row["label"], *values, revenue,
            "—" if fraction is None else f"{100.0 * float(fraction):.1f}%",
        ])
    oracle = derived["da_oracle__rt_at_da_oracle__rt_oracle"]
    rows.append(["Oracle（受限先知上界）", "—", "—", "—", f"{oracle['macro_mean_daily_revenue']:.2f}", "100.0%"])
    return rows


def locked_2025_profit_evidence(summary: dict) -> dict:
    """提取事前冻结的主盈利比较；MAE不得改变该结论。"""
    transformer_label = "da_ensemble__rt_at_da_ensemble__rt_ensemble"
    comparator_label = (
        "da_pooled_xgboost__rt_at_da_pooled_xgboost__rt_pooled_xgboost"
    )
    derived = summary["ercot_settlement_derived"]
    transformer = derived[transformer_label]
    comparator = derived[comparator_label]
    comparison = comparator["transformer_ensemble_minus_system"]
    mean_difference = float(comparison["mean_daily_difference"])
    ci = [float(value) for value in comparison["paired_7day_block_bootstrap_95ci"]]
    if mean_difference > 0.0 and ci[0] > 0.0:
        classification = "stable_superiority"
        conclusion = (
            "固定三seed Transformer集成在预注册ERCOT合同下稳定优于pooled XGBoost；"
            "盈利差的7日块95%区间整体高于0。"
        )
        tone = "good"
    elif mean_difference > 0.0:
        classification = "positive_point_estimate_only"
        conclusion = (
            "固定三seed Transformer集成的盈利点估计高于pooled XGBoost，但7日块95%区间跨0；"
            "只能称为盈利改善信号，不能称为稳定胜出。"
        )
        tone = "warn"
    else:
        classification = "no_superiority"
        conclusion = (
            "固定三seed Transformer集成未在ERCOT 2025盈利上超过pooled XGBoost；"
            "即使部分MAE更低，也不能写成经济目标达成。"
        )
        tone = "risk"
    return {
        "classification": classification,
        "conclusion": conclusion,
        "tone": tone,
        "transformer": transformer,
        "comparator": comparator,
        "mean_difference": mean_difference,
        "ci": ci,
    }


def register_fonts() -> None:
    regular = Path(r"C:\Windows\Fonts\Deng.ttf")
    bold = Path(r"C:\Windows\Fonts\Dengb.ttf")
    if not regular.exists() or not bold.exists():
        regular = Path(r"C:\Windows\Fonts\simhei.ttf")
        bold = regular
    pdfmetrics.registerFont(TTFont("CJK", str(regular)))
    pdfmetrics.registerFont(TTFont("CJK-Bold", str(bold)))
    pdfmetrics.registerFontFamily(
        "CJK", normal="CJK", bold="CJK-Bold", italic="CJK", boldItalic="CJK-Bold"
    )


register_fonts()


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="CJK", fontSize=9.4,
            leading=15.0, textColor=INK, alignment=TA_JUSTIFY,
            spaceAfter=4.5, wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="CJK", fontSize=7.7,
            leading=11.2, textColor=MUTED, wordWrap="CJK",
        ),
        "tiny": ParagraphStyle(
            "Tiny", parent=base["BodyText"], fontName="CJK", fontSize=6.6,
            leading=9.2, textColor=MUTED, wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "Heading1", parent=base["Heading1"], fontName="CJK-Bold",
            fontSize=18, leading=25, textColor=NAVY, spaceBefore=4, spaceAfter=10,
            keepWithNext=True, wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "Heading2", parent=base["Heading2"], fontName="CJK-Bold",
            fontSize=12.5, leading=18, textColor=BLUE, spaceBefore=10, spaceAfter=5,
            keepWithNext=True, wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "Heading3", parent=base["Heading3"], fontName="CJK-Bold",
            fontSize=10.2, leading=15, textColor=NAVY, spaceBefore=7, spaceAfter=3,
            keepWithNext=True, wordWrap="CJK",
        ),
        "title": ParagraphStyle(
            "Title", fontName="CJK-Bold", fontSize=24, leading=36,
            textColor=NAVY, alignment=TA_CENTER, wordWrap="CJK",
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", fontName="CJK", fontSize=12, leading=20,
            textColor=MUTED, alignment=TA_CENTER, wordWrap="CJK",
        ),
        "callout": ParagraphStyle(
            "Callout", fontName="CJK", fontSize=9.2, leading=14.2,
            textColor=INK, wordWrap="CJK",
        ),
        "equation": ParagraphStyle(
            "Equation", fontName="CJK", fontSize=9.0, leading=14,
            alignment=TA_CENTER, textColor=NAVY, wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "Table", fontName="CJK", fontSize=7.4, leading=10.2,
            textColor=INK, wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "TableHead", fontName="CJK-Bold", fontSize=7.5, leading=10.5,
            textColor=colors.white, alignment=TA_CENTER, wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "Caption", fontName="CJK", fontSize=7.4, leading=11,
            textColor=MUTED, alignment=TA_CENTER, spaceBefore=3, spaceAfter=7,
            wordWrap="CJK",
        ),
        "toc1": ParagraphStyle(
            "TOC1", fontName="CJK-Bold", fontSize=10.2, leading=16,
            leftIndent=0, firstLineIndent=0, textColor=NAVY,
        ),
        "toc2": ParagraphStyle(
            "TOC2", fontName="CJK", fontSize=8.4, leading=13,
            leftIndent=16, firstLineIndent=0, textColor=MUTED,
        ),
    }


S = make_styles()


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
            topMargin=18 * mm, bottomMargin=16 * mm,
            title="多节点无ID通用电价预测与储能双结算技术报告",
            author="Electricity Price Forecasting Project",
            subject="模型架构、训练、消融、零样本预测与储能盈利评估",
        )
        frame = Frame(
            self.leftMargin, self.bottomMargin, self.width, self.height,
            id="normal", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=page_decor)])

    def afterFlowable(self, flowable: Flowable) -> None:
        if isinstance(flowable, Paragraph):
            name = flowable.style.name
            if name in {"Heading1", "Heading2"}:
                level = 0 if name == "Heading1" else 1
                text = flowable.getPlainText()
                key = f"h{level}-{self.page}-{abs(hash(text))}"
                self.canv.bookmarkPage(key)
                if level == 0:
                    self.canv.addOutlineEntry(text, key, level=0, closed=False)
                self.notify("TOCEntry", (level, text, self.page, key))


def page_decor(canvas, doc) -> None:
    page = canvas.getPageNumber()
    if page == 1:
        return
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, A4[1] - 13 * mm, A4[0] - 18 * mm, A4[1] - 13 * mm)
    canvas.setFont("CJK", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, A4[1] - 10 * mm, "多节点无ID通用电价预测与储能双结算技术报告")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"{page}")
    canvas.drawString(18 * mm, 9 * mm, "技术报告 - 2026-10-18")
    canvas.restoreState()


class CoverBand(Flowable):
    def __init__(self, width: float, height: float = 92 * mm):
        super().__init__()
        self.width, self.height = width, height

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        c.setFillColor(NAVY)
        c.rect(-18 * mm, -4 * mm, A4[0], self.height + 26 * mm, stroke=0, fill=1)
        c.setFillColor(BLUE)
        c.circle(self.width - 8 * mm, self.height - 4 * mm, 32 * mm, stroke=0, fill=1)
        c.setFillColor(TEAL)
        c.circle(self.width - 43 * mm, self.height + 14 * mm, 10 * mm, stroke=0, fill=1)
        c.setFillColor(ORANGE)
        c.rect(self.width - 18 * mm, 9 * mm, 34 * mm, 6 * mm, stroke=0, fill=1)
        c.restoreState()


class MetricCard(Flowable):
    def __init__(self, width: float, value: str, label: str, accent=BLUE, note: str = ""):
        super().__init__()
        self.width = width
        self.height = 28 * mm
        self.value, self.label, self.accent, self.note = value, label, accent, note

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        c.setFillColor(LIGHT)
        c.setStrokeColor(GRID)
        c.roundRect(0, 0, self.width, self.height, 3 * mm, stroke=1, fill=1)
        c.setFillColor(self.accent)
        c.roundRect(0, 0, 3.2 * mm, self.height, 2 * mm, stroke=0, fill=1)
        c.setFillColor(NAVY)
        c.setFont("CJK-Bold", 16)
        c.drawString(7 * mm, 16 * mm, self.value)
        c.setFont("CJK", 7.6)
        c.setFillColor(INK)
        c.drawString(7 * mm, 9 * mm, self.label)
        if self.note:
            c.setFont("CJK", 6.5)
            c.setFillColor(MUTED)
            c.drawString(7 * mm, 4 * mm, self.note)
        c.restoreState()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def B(text: str) -> str:
    """Bold a winning value inside a ReportLab paragraph cell."""
    return f"<b>{text}</b>"


def H1(text: str) -> Paragraph:
    return Paragraph(text, S["h1"])


def H2(text: str) -> Paragraph:
    return Paragraph(text, S["h2"])


def H3(text: str) -> Paragraph:
    return Paragraph(text, S["h3"])


def bullets(items: list[str], level: int = 0) -> ListFlowable:
    return ListFlowable(
        [ListItem(P(x, "body"), leftIndent=10) for x in items],
        bulletType="bullet", start="circle", leftIndent=14 + 8 * level,
        bulletFontName="CJK", bulletFontSize=6, spaceAfter=5,
    )


def callout(title: str, body: str, kind: str = "info") -> Table:
    palette = {
        "info": (PALE_BLUE, BLUE), "good": (PALE_GREEN, GREEN),
        "warn": (PALE_ORANGE, ORANGE), "risk": (PALE_RED, RED),
    }
    bg, accent = palette[kind]
    data = [[P(f"<b>{title}</b><br/>{body}", "callout")]]
    t = Table(data, colWidths=[174 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), 0.8, accent),
        ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def equation(text: str) -> Table:
    t = Table([[P(text, "equation")]], colWidths=[164 * mm], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.7, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def styled_table(rows: list[list], widths: list[float], font_size: float = 7.4,
                 header_bg=NAVY, alignments: dict[int, str] | None = None) -> Table:
    cooked = []
    for r, row in enumerate(rows):
        cooked.append([
            cell if isinstance(cell, Flowable) else P(str(cell), "table_head" if r == 0 else "table")
            for cell in row
        ])
    t = Table(cooked, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in range(1, len(rows)):
        if r % 2 == 0:
            style.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    if alignments:
        for col, align in alignments.items():
            style.append(("ALIGN", (col, 1), (col, -1), align))
    t.setStyle(TableStyle(style))
    return t


def arrow(d: Drawing, x1: float, y1: float, x2: float, y2: float, color=MUTED) -> None:
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.4))
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 5
    pts = []
    for delta in (2.55, -2.55):
        pts.extend([x2 + size * math.cos(ang + delta), y2 + size * math.sin(ang + delta)])
    d.add(Polygon([x2, y2, *pts], fillColor=color, strokeColor=color))


def box(d: Drawing, x: float, y: float, w: float, h: float, title: str,
        subtitle: str, fill, stroke, title_size=8.2, sub_size=6.5) -> None:
    d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fill, strokeColor=stroke, strokeWidth=1.1))
    d.add(String(x + w / 2, y + h - 12, title, fontName="CJK-Bold", fontSize=title_size,
                 fillColor=NAVY, textAnchor="middle"))
    lines = subtitle.split("\n")
    for i, line in enumerate(lines):
        d.add(String(x + w / 2, y + h - 24 - i * 9, line, fontName="CJK", fontSize=sub_size,
                     fillColor=INK, textAnchor="middle"))


def system_architecture() -> Drawing:
    d = Drawing(500, 290)
    box(d, 5, 206, 100, 62, "DA模型", "D-1一次起报\n输出24小时DA价", PALE_BLUE, BLUE)
    box(d, 135, 206, 100, 62, "RT-at-DA模型", "同一D-1信息时点\n输出24小时RT预期", PALE_ORANGE, ORANGE)
    box(d, 265, 206, 100, 62, "滚动RT模型", "交付日逐小时\n预测未来4小时RT", PALE_GREEN, GREEN)
    box(d, 395, 206, 100, 62, "真实市场数据", "动作后才读取\nDA/RT真实价格", LIGHT, MUTED)
    box(d, 50, 112, 150, 60, "日前联合连续LP", "读取DA与RT-at-DA\n生成并锁定u_DA[24]", PALE_BLUE, BLUE)
    box(d, 260, 112, 150, 60, "逐小时连续RT MPC", "读取滚动RT、锁定计划与SOC\n4小时求解，只执行第1步", PALE_GREEN, GREEN)
    box(d, 145, 20, 210, 58, "完整双结算账本", "DA腿 + RT偏差腿 - 退化成本 - 偏差罚金\n节点内SOC连续，节点间严格隔离", PALE_ORANGE, ORANGE)
    arrow(d, 55, 206, 105, 172, BLUE)
    arrow(d, 185, 206, 165, 172, ORANGE)
    arrow(d, 315, 206, 325, 172, GREEN)
    arrow(d, 200, 142, 260, 142, MUTED)
    arrow(d, 125, 112, 205, 78, BLUE)
    arrow(d, 335, 112, 295, 78, GREEN)
    arrow(d, 445, 206, 355, 78, MUTED)
    return d


def model_architecture(task: str) -> Drawing:
    d = Drawing(500, 245)
    title = {"da": "DA：24小时日前价格", "rtad": "RT-at-DA：24小时实时价格预期", "rt": "滚动RT：未来4小时实时价格"}[task]
    fusion = "F0 残差MLP" if task == "rtad" else "F1 来源注意力"
    target = "目标日8维日历" if task != "rt" else "未来4小时已知DA价 + 8维日历"
    box(d, 5, 164, 100, 60, "168小时历史", "DA价 / RT价 / 负荷\n风+光 / 8维日历", PALE_BLUE, BLUE)
    box(d, 135, 164, 105, 60, "五流独立编码", "价格/负荷/系统：Transformer\n日历：MLP；d=128", LIGHT, PURPLE)
    box(d, 270, 164, 100, 60, fusion, "同小时跨来源交互\n再建模时间关系", PALE_ORANGE, ORANGE)
    box(d, 400, 164, 95, 60, target, "仅使用起报时\n已经合法可见的信息", PALE_GREEN, GREEN)
    box(d, 150, 72, 120, 58, "Query Decoder", "历史Memory + 目标查询\n不读未来真实价格/负荷", PALE_BLUE, BLUE)
    box(d, 330, 72, 115, 58, title, "Linear(128,1)\n反归一化 + 价格截断", PALE_GREEN, GREEN)
    arrow(d, 105, 194, 135, 194)
    arrow(d, 240, 194, 270, 194)
    arrow(d, 320, 164, 235, 130)
    arrow(d, 447, 164, 260, 130)
    arrow(d, 270, 101, 330, 101)
    return d


def protocol_diagram() -> Drawing:
    d = Drawing(500, 205)
    steps = [
        (5, "训练", "2020-2022\n3个市场"),
        (105, "验证", "2023\n留出第4市场"),
        (205, "预测对照", "lag-168\n目标市场内XGB"),
        (305, "ERCOT决策", "连续LP/MPC\n15节点"),
        (405, "冻结检查", "2024只推理\n不重训"),
    ]
    cols = [BLUE, PURPLE, ORANGE, GREEN, RED]
    fills = [PALE_BLUE, LIGHT, PALE_ORANGE, PALE_GREEN, PALE_RED]
    for i, (x, title, sub) in enumerate(steps):
        box(d, x, 90, 90, 70, title, sub, fills[i], cols[i])
        if i < len(steps) - 1:
            arrow(d, x + 90, 125, x + 105, 125)
    d.add(String(250, 180, "四折留一市场（LOMO）零样本协议", fontName="CJK-Bold",
                 fontSize=12, fillColor=NAVY, textAnchor="middle"))
    d.add(String(250, 52, "四个留出市场 × 三个独立任务 = 12个预测结果，全部不读取node_id/market_id",
                 fontName="CJK", fontSize=7.5, fillColor=INK, textAnchor="middle"))
    d.add(String(250, 34, "2025保持锁定；CAISO/NYISO/PJM尚未形成ISO级完整盈利结论",
                 fontName="CJK", fontSize=7.5, fillColor=RED, textAnchor="middle"))
    return d


def bar_chart(categories: list[str], series: list[tuple[str, list[float], colors.Color]],
              y_min: float, y_max: float, width=500, height=220) -> Drawing:
    d = Drawing(width, height)
    chart = VerticalBarChart()
    chart.x, chart.y = 46, 40
    chart.width, chart.height = width - 75, height - 75
    chart.data = [x[1] for x in series]
    chart.categoryAxis.categoryNames = categories
    chart.categoryAxis.labels.fontName = "CJK"
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.angle = 0
    chart.valueAxis.valueMin = y_min
    chart.valueAxis.valueMax = y_max
    chart.valueAxis.valueStep = (y_max - y_min) / 5
    chart.valueAxis.labels.fontName = "CJK"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.gridStrokeColor = GRID
    chart.valueAxis.gridStrokeWidth = 0.4
    chart.barSpacing = 1.5
    chart.groupSpacing = 7
    for i, (_, _, color) in enumerate(series):
        chart.bars[i].fillColor = color
        chart.bars[i].strokeColor = color
    d.add(chart)
    legend = Legend()
    legend.x, legend.y = 55, height - 20
    legend.fontName = "CJK"
    legend.fontSize = 7
    legend.colorNamePairs = [(x[2], x[0]) for x in series]
    legend.dx = 7
    legend.dy = 7
    legend.deltax = 75
    legend.columnMaximum = 1
    d.add(legend)
    return d


def chapter(story: list, number: str, title: str, lead: str = "") -> None:
    story.append(PageBreak())
    story.append(H1(f"{number}  {title}"))
    story.append(HRFlowable(width="100%", thickness=1.3, color=BLUE, spaceAfter=8))
    if lead:
        story.append(P(lead, "callout"))
        story.append(Spacer(1, 3 * mm))


def build_story() -> list:
    locked_2025 = load_json_if_exists(
        "data/results/allmarket_locked_2025_summary_v2.json"
    )
    lomo: dict[tuple[str, str], float] = {}
    for task_dir, task in [
        ("universal_lomo_da_v1", "DA"),
        ("universal_lomo_rt_at_da_v1", "RT-at-DA"),
        ("universal_lomo_rt_v1", "滚动RT"),
    ]:
        for path in sorted((ROOT / "data" / "results" / task_dir).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["uses_node_id_for_prediction"] is False
            assert payload["uses_market_id_for_prediction"] is False
            assert payload["normalization_mode"] == "context_robust"
            lomo[(payload["holdout_market"], task)] = float(payload["node_macro_mae"])

    profit23 = load_json("data/results/universal_lomo_ercot_profit_continuous_mpc_v1.json")
    profit24 = load_json("data/results/universal_lomo_ercot_outer_frozen_continuous_mpc_v1.json")
    fair_xgb = load_json("data/results/fair_xgboost_comparison_v1/summary.json")
    tail = load_json("data/results/tail_profit_diagnostics_v1/tail_profit_diagnostics.json")
    tsfm_profit = load_json(
        "data/results/universal_lomo_timesfm25_lora_ercot_continuous_mpc_v1.json"
    )

    def external_mae(directory: str, filename: str) -> float:
        return float(load_json(f"data/results/{directory}/{filename}")["node_macro_mae"])

    tsfm3_mae = {}
    tsfm25_mae = {}
    pooled_mae = {}
    task_slug = {"DA": "da", "RT-at-DA": "rt_at_da", "滚动RT": "rt"}
    for market in ("CAISO", "ERCOT", "NYISO", "PJM"):
        for label, slug in task_slug.items():
            tsfm3_mae[(market, label)] = external_mae(
                "universal_lomo_timesfm3_v1", f"{market.lower()}_{slug}_v1.json"
            )
            tsfm25_mae[(market, label)] = external_mae(
                "universal_lomo_timesfm25_lora_v1",
                f"holdout_{market.lower()}_{slug}_inner_seed0.json",
            )
    fair_task_label = {"da": "DA", "rt_at_da": "RT-at-DA", "rt": "滚动RT"}
    for row in fair_xgb["forecast_rows"]:
        pooled_mae[(row["market"], fair_task_label[row["task"]])] = float(
            row["pooled_lomo_xgboost_mae"]
        )
    key_u = "da_universal__rt_at_da_universal__rt_universal"
    key_x = "da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
    key_s = "da_seasonal__rt_at_da_seasonal__rt_seasonal"
    key_t3 = "da_timesfm3_frozen__rt_at_da_timesfm3_frozen__rt_timesfm3_frozen"
    key_t25 = "da_timesfm25_lora__rt_at_da_timesfm25_lora__rt_timesfm25_lora"
    u23, x23, s23 = profit23["systems"][key_u], profit23["systems"][key_x], profit23["systems"][key_s]
    u24, x24 = profit24["systems"][key_u], profit24["systems"][key_x]
    t3_profit = tsfm_profit["systems"][key_t3]
    t25_profit = tsfm_profit["systems"][key_t25]
    pooled_profit = next(
        row for row in fair_xgb["profit_rows"] if row["system"] == "xgboost_pooled"
    )
    ensemble_profit = next(
        row for row in fair_xgb["profit_rows"] if row["system"] == "transformer_ensemble"
    )

    story: list = []

    # Academic title page
    story.append(Spacer(1, 43 * mm))
    story.append(P("多节点无 ID 通用电价预测<br/>与储能双结算系统", "title"))
    story.append(Spacer(1, 7 * mm))
    story.append(HRFlowable(width="45%", thickness=0.8, color=NAVY, spaceBefore=0, spaceAfter=8 * mm,
                            hAlign="CENTER"))
    story.append(P("研究问题、方法、实验结果与讨论", "subtitle"))
    story.append(PageBreak())

    # Abstract
    story.append(H1("摘要"))
    story.append(P(
        "本研究面向电池储能参与日前—实时双结算市场时的净收益问题，建立了从价格预测、约束优化到真实价格结算的完整流程。"
        "系统按照信息到达时点分别训练三个模型：日前价格模型（DA）预测次日24小时日前价格；日前时点实时价格模型（RT-at-DA）"
        "在相同截止时点预测次日24小时实时价格；滚动实时模型（rolling RT）在交付日每小时预测未来4小时实时价格。"
        "三个模型独立训练，预测结果先进入日前线性规划（linear programming, LP）确定全天计划，再进入模型预测控制"
        "（model predictive control, MPC）逐小时滚动修正，最终用真实日前与实时价格计算净收益。"
    ))
    story.append(P(
        "为检验未见市场上的迁移能力，采用四折留一市场（Leave-One-Market-Out, LOMO）协议：每一折仅用三个市场训练，"
        "将第4个市场全部留作零样本测试。最终模型不读取节点或市场身份。2023年四个留出市场、三个任务共12项预测的平均绝对误差"
        "（mean absolute error, MAE）均优于168小时前同小时复制基线。盈利只在结算规则已完整实现的ERCOT上比较："
        "同训练数据比较中，当前Transformer在12项MAE中的9项优于pooled LOMO XGBoost，也在9项上优于源市场监督的TimesFM-2.5 LoRA；"
        "frozen TimesFM-3仅在四个滚动RT任务上占优。ERCOT 2023中，当前seed 0收益377.84，固定三seed集成407.17，"
        "仍未稳定超过同数据pooled XGBoost的418.06；2024冻结检查又落后于目标市场监督XGBoost。"
        "结果表明跨市场预测迁移已成立，但稳定盈利优势尚未成立。"
    ))
    story.append(P(
        "训练阶段出现MAE和Pseudo-Huber，是因为利润并非逐小时可直接监督的标签，而要经过跨时段电量状态、效率、偏差罚金和优化器才形成；"
        "因此价格误差用于稳定训练，净收益始终是最终系统评价指标。结构代号F0/F1、两类损失及其选择依据见第4.1—4.3节，"
        "随机种子与训练范围见第3.5和第4.4节。"
    ))
    story.append(P("关键词：电价预测；跨市场迁移；储能套利；日前—实时双结算；模型预测控制；LOMO", "small"))
    story.append(Spacer(1, 3 * mm))
    story.append(styled_table([
        ["评价层", "评价范围", "自有系统", "XGBoost/基线", "结论"],
        ["预测", "四市场 × 三任务", B("12/12优于lag-168"), "lag-168", "跨市场预测迁移成立"],
        ["预测", "同数据零样本", B("9/12优于pooled XGBoost"), "TimesFM-2.5仅胜3/12", "需结合价差与收益"],
        ["盈利", "ERCOT 2023", B("377.84"), "292.29", "点估计+29.3%，区间跨0"],
        ["盈利", "ERCOT 2024冻结", "73.38", B("90.69"), "自有系统-19.1%，区间低于0"],
    ], [24 * mm, 36 * mm, 33 * mm, 33 * mm, 48 * mm]))
    story.append(P("表1  核心预测与盈利结果；加粗表示该比较中的较优结果。", "caption"))

    story.append(PageBreak())
    story.append(H1("目录"))
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    story.append(toc)
    story.append(Spacer(1, 6 * mm))
    story.append(callout(
        "实验口径",
        "LOMO即Leave-One-Market-Out（留一市场）：四个市场轮流作为完全未参与训练的测试市场，因此形成四折。"
        "“当前”指无ID LOMO seed 0最终系统；“历史”指单节点、带ID或尚未复现到当前四市场协议的实验。",
        "info",
    ))

    chapter(story, "1", "引言与研究目标",
            "本文研究严格信息边界下的多市场电价预测，以及这些预测能否通过可执行的储能决策形成稳定净收益。")
    story.append(H2("1.1 研究问题与贡献"))
    story.append(styled_table([
        ["研究问题", "本文处理方式", "可验证产出"],
        ["不同信息时点如何预测DA与RT价格", "建立DA、RT-at-DA、滚动RT三个独立任务", "三个冻结checkpoint与逐任务预测指标"],
        ["模型能否迁移到未见市场", "移除node_id/market_id并采用四折LOMO", "四市场 × 三任务零样本结果"],
        ["预测能否转化为经济价值", "连续DA LP + 滚动RT MPC + 双结算账本", "ERCOT 2023/2024净收益与风险指标"],
    ], [46 * mm, 70 * mm, 58 * mm]))
    story.append(P("表2  研究问题、技术路线与可验证产出。", "caption"))
    story.append(H2("1.2 系统闭环"))
    story.append(system_architecture())
    story.append(P("图1  三个独立价格模型与统一决策/结算层。", "caption"))
    story.append(H2("1.3 三个预测任务"))
    story.append(styled_table([
        ["任务", "运行时点", "输出", "决策用途", "当前结构/损失"],
        ["DA", "交付日前一日，按市场冻结起报时刻", "目标日24小时DA价", "与RT-at-DA共同形成日前计划", "F1来源注意力 / MAE"],
        ["RT-at-DA", "与DA完全相同的信息截止时点", "目标日24小时RT价格预期", "决定DA-RT预期价差方向", "F0残差融合 / Pseudo-Huber"],
        ["滚动RT", "交付日每小时", "未来4小时RT价", "连续RT MPC，只执行第1步", "F1来源注意力 / MAE"],
    ], [24 * mm, 38 * mm, 34 * mm, 43 * mm, 35 * mm]))
    story.append(P(
        "F0和F1是五类输入信息的两种融合方式：F0先拼接再经残差多层感知机，F1用注意力学习各来源之间的关系；"
        "完整定义和历史消融见第4.1节。MAE是预测值与真实值绝对误差的平均，Pseudo-Huber是平滑且抗尖峰的误差函数；"
        "两者是训练价格模型的代理损失，不是最终业务目标，详见第4.2—4.3节。"
    ))
    story.append(H2("1.4 ‘联合’的准确含义"))
    story.append(P(
        "当前最终系统的‘联合’发生在决策层，不发生在神经网络参数层。三模型不共享权重、优化器或checkpoint，"
        "也没有把MAE与Huber简单相加成一个总损失。DA和RT-at-DA先进入日前连续LP，滚动RT再进入小时级MPC；"
        "这是一种预测后优化（predict-then-optimize）的模块化组合。储能在价格不确定下进行日前—实时联合套利，以及使用"
        "前瞻滚动优化形成动作，分别可见相关储能市场研究[6][8]。"
    ))
    story.append(callout(
        "为什么不直接端到端按利润联合训练？",
        "早期单节点已做过零阶收益导向联合微调。V1因非光滑排名、扰动噪声和全参数更新而显著退化；V2只取得很小改善；"
        "罚金版active联合训练在开发期小幅改善，但独立报告期反而下降，定量结果见第7.1节。该训练使用随机扰动估计利润方向，属于零阶/无梯度优化思路[9]。"
        "基于这些证据，多节点最终版优先采用可审计的独立预测 + LP/MPC，而不是把尚不稳定的业务梯度带入四市场训练。",
        "warn",
    ))
    story.append(H2("1.5 论文结构"))
    story.append(styled_table([
        ["部分", "章节", "核心内容"],
        ["实验基础", "第2章", "数据、信息时点、无ID部署合同与评价协议"],
        ["方法", "第3-6章", "预测网络、训练目标、比较基线、储能决策与结算"],
        ["实验", "第7-9章", "消融选择、跨市场预测结果与ERCOT盈利结果"],
        ["讨论与结论", "第10-11章", "跨年稳定性、统计边界、结论与后续工作"],
        ["复现", "第12章及附录", "配置、机器可读结果、节点映射与结论边界"],
    ], [30 * mm, 32 * mm, 112 * mm]))

    chapter(story, "2", "数据与实验设计",
            "数据合同的作用是规定‘模型在某个起报时点到底知道什么’，它比网络结构更优先；任何未来真实价格、未来实际负荷或事后修订数据都不得进入输入。")
    story.append(H2("2.1 数据规模与市场构成"))
    story.append(styled_table([
        ["市场", "节点数", "每节点小时数", "总行数", "时区", "D-1起报时刻"],
        ["CAISO", "3", "56,255", "168,765", "America/Los_Angeles", "10:00"],
        ["ERCOT", "15", "56,255", "843,825", "America/Chicago", "10:00"],
        ["NYISO", "11", "56,255", "618,805", "America/New_York", "05:00"],
        ["PJM", "35", "56,255", "1,968,925", "America/New_York", "11:00"],
        ["合计", "64", "-", "3,600,320", "-", "按市场冻结"],
    ], [23 * mm, 18 * mm, 27 * mm, 30 * mm, 48 * mm, 28 * mm]))
    story.append(P(
        "各节点本地时间覆盖2020-01-01至2026-06-01；必要字段缺失率为0，节点×UTC时间无重复，UTC序列无非1小时间隔。"
        "夏令时产生的23/25小时本地日不静默补成24小时，而是在日级任务中显式排除。"
    ))
    story.append(H2("2.2 五条历史流与目标日已知信息"))
    story.append(styled_table([
        ["流", "维度", "168小时历史内容", "编码器"],
        ["price_da", "1", "历史日前价格", "2层Transformer + RoPE"],
        ["price_rt", "1", "历史实时价格", "2层Transformer + RoPE"],
        ["load", "1", "起报前实际负荷", "2层Transformer + RoPE"],
        ["system", "2", "起报前实际风电、光伏", "2层Transformer + RoPE"],
        ["calendar", "8", "小时/星期/月周期、周末、节假日", "MLP"],
    ], [28 * mm, 18 * mm, 76 * mm, 52 * mm]))
    story.append(styled_table([
        ["日历含义", "输入字段", "维度", "编码目的"],
        ["小时周期", "hour_sin、hour_cos", "2", "连续表示0-23时的循环位置"],
        ["星期周期", "dow_sin、dow_cos", "2", "连续表示周内位置"],
        ["月份周期", "month_sin、month_cos", "2", "连续表示年内季节位置"],
        ["周末标记", "is_weekend", "1", "区分工作日与周末"],
        ["节假日标记", "is_holiday", "1", "标记已知公共假日"],
        ["合计", "", "8", "全部在起报时已知"],
    ], [35 * mm, 57 * mm, 19 * mm, 63 * mm]))
    story.append(P(
        "目标日历投影把每小时8维日历通过Linear(8,128)+GELU+LayerNorm映射到模型隐空间，作为Decoder查询条件；"
        "它不包含未来真实价格或负荷。滚动RT额外拼入已公布的目标小时DA价格，形成9维目标上下文。"
    ))
    story.append(H2("2.3 无ID部署合同与稳健归一化"))
    story.append(equation(
        "center = median(x<sub>1:168</sub>)； scale = max(1.4826 × MAD, 同窗口std回退, 1)； "
        "x' = (x - center) / scale"
    ))
    story.append(P(
        "最终模型设置global_use_entity_embeddings=false、global_num_nodes=0、global_num_markets=0。"
        "node_id和market_id仍用于平衡采样、指标汇总、SOC隔离和文件审计，但不得进入前向传播、损失或门控。"
        "context_robust只使用当前样本起报前的168小时，因此新市场不需要预先注册节点scaler。"
    ))
    story.append(callout(
        "历史带ID模型、pre-encoder与当前无ID模型的区别",
        "历史带ID模型也是一个跨节点共享参数的Transformer，并不是每个节点各训一个模型。训练时，系统为每个已知节点学习一个node embedding，"
        "为每个已知市场学习一个market embedding；二者相加后得到该样本的身份向量。所谓pre-encoder，是指这个身份向量不是在预测头末端才加入，"
        "而是在价格、负荷、风光和日历等每条历史流进入各自编码器之前，就加到每个小时的输入表示中。这样，编码器从第一层开始便能根据“哪个节点、"
        "哪个市场”采用不同的表示方式。它适合训练集内已知实体，因为所有节点仍共享骨干网络，同时可利用已学习的身份差异；但对完全未见的新市场，"
        "不存在对应的已学习embedding，强行分配或平均身份向量都不是真正的零样本部署。当前无ID模型因此把node/market embedding及其pre-encoder注入"
        "全部删除，只依据起报前168小时可观测历史和目标日已知日历预测。故历史带ID结果表示“已知实体条件下的性能上界”，不能作为未见市场的零样本证据。",
        "info",
    ))
    story.append(H2("2.4 数据划分与评价层级"))
    story.append(styled_table([
        ["层级", "训练/开发数据", "评价数据", "是否用于调参", "主要指标"],
        ["零样本预测", "其他三个市场，2020-2022", "留出市场，2023", "是：内部开发评价", "节点等权MAE、市场宏平均"],
        ["ERCOT盈利", "预测器按各自冻结合同训练", "ERCOT 2023共同日", "是：系统开发验证", "净收益、区间、回撤、收益分解"],
        ["时间外检查", "保持模型和策略不变", "ERCOT 2024共同日", "否：冻结推理", "同口径净收益与配对区间"],
        ["最终年度", "尚未执行", "2025锁定", "否", "预注册后一次性评价"],
    ], [28 * mm, 53 * mm, 42 * mm, 27 * mm, 24 * mm]))

    chapter(story, "3", "预测方法：三任务Transformer",
            "三个模型复用同一套模块代码和张量接口，但每次实例化都会建立独立参数。‘都是Transformer’的准确说法是：价格、负荷和系统历史由Transformer编码；日历流与输出头仍包含MLP/Linear。")
    story.append(H2("3.1 共享骨干设计"))
    story.append(styled_table([
        ["模块", "冻结配置", "作用"],
        ["历史窗口", "168小时", "覆盖最近7天的价格、负荷、风光和日历"],
        ["隐空间", "d_model=128", "统一各来源编码维度"],
        ["编码器", "2层；4头注意力；FFN=512", "学习各历史流的时序依赖"],
        ["来源融合", "4头注意力（F1）或残差MLP（F0）", "按任务融合五类来源"],
        ["正则化", "dropout=0.1", "降低训练期过拟合"],
        ["输出", "24或4个Query；Linear(128,1)", "生成目标时段价格并截断至[-500,10000] USD/MWh"],
    ], [32 * mm, 57 * mm, 85 * mm]))
    story.append(bullets([
        "每条连续历史流先独立编码，避免在输入层过早混合量纲和含义不同的变量。",
        "旋转位置编码（Rotary Position Embedding, RoPE）[4]只用于按时间排列的token；来源注意力不把五个来源误当成时间序列。",
        "Decoder从历史Memory中读取与目标小时相关的信息；骨干采用Transformer的自注意力结构[3]。",
    ]))
    story.append(H2("3.2 DA模型"))
    story.append(model_architecture("da"))
    story.append(P("图2  当前无ID DA架构：F1来源注意力 + 24 Query Decoder。", "caption"))
    story.append(P(
        "F1先将同一小时的五个来源堆叠为[B×168,5,128]，加入来源embedding后做跨来源自注意力；"
        "再通过可学习softmax权重汇聚为[B,168,128]，最后做时间Transformer。目标日历提供24个查询上下文。"
    ))
    story.append(H2("3.3 RT-at-DA模型"))
    story.append(model_architecture("rtad"))
    story.append(P("图3  当前无ID RT-at-DA架构：F0残差融合 + 24 Query Decoder。", "caption"))
    story.append(P(
        "RT-at-DA在DA截止时点预测同一交付日的RT价格，因此不能使用交付日真实DA、RT、负荷或风光。"
        "当前最终模型采用direct_price参数化和F0：五流编码结果按同小时拼接，投影回128维，经残差MLP和时间Transformer形成Memory。"
        "经济状态头不在当前无ID最终模型中；它是历史已知ERCOT实体实验。"
    ))
    story.append(H2("3.4 滚动RT模型"))
    story.append(model_architecture("rt"))
    story.append(P("图4  当前无ID滚动RT架构：F1来源注意力 + 已知未来DA/日历查询。", "caption"))
    story.append(P(
        "每个交付小时重新构造168小时历史，输出未来4小时RT价格。目标上下文由已公布DA价与8维日历拼接为9维后投影到128维。"
        "模型可以看见未来4小时DA，是因为日前市场已出清；不能看见对应未来RT真值。MPC求解4小时但只执行第1小时，下一小时重新预测。"
    ))
    story.append(H2("3.5 模型数量、随机种子与最终冻结关系"))
    story.append(P(
        "随机种子（seed）控制参数初始化、训练样本顺序等随机过程。多个seed不是多个任务模型，而是对同一配置进行独立重复，"
        "用于判断结论是否依赖一次偶然初始化。项目在不同阶段使用种子的范围不同，不能把历史三seed误写成当前最终模型已完成三seed。"
    ))
    story.append(styled_table([
        ["实验阶段", "模型/配置范围", "重复设置", "用途"],
        ["早期单节点结构消融", "DA、RT-at-DA、滚动RT的F0/F1/F1b/F2候选", "每个候选3 seeds", "选择任务对应的融合结构"],
        ["早期单节点联合微调", "三模型组成的完整双结算系统", "3 seeds", "检查收益导向联合训练的稳定性"],
        ["历史多节点经济状态头", "冻结DA与RT-at-DA骨干，仅训练方向分类头", "3 seeds", "检查方向分类是否稳定"],
        ["当前无ID LOMO最终系统", "4个留出市场 × 3个任务 = 12个checkpoint", "每折seed 0", "形成当前正式零样本预测与盈利结果"],
    ], [39 * mm, 69 * mm, 30 * mm, 36 * mm]))
    story.append(P(
        "因此，本报告既保留历史多seed证据，也明确把当前无ID系统的多seed复现实验列为待完成项；二者不能互相替代。"
    ))

    chapter(story, "4", "训练方法：融合结构、损失与优化",
            "本章回答三个容易混淆的问题：F0/F1到底比较了什么；MAE和Pseudo-Huber是否同一概念；为什么盈利目标没有直接替代预测损失。")
    story.append(H2("4.1 F0 / F1 / F1b / F2定义"))
    story.append(styled_table([
        ["结构", "跨来源处理", "时间处理", "特点"],
        ["F0", "同小时五流concat→Linear→残差MLP", "时间Transformer", "无来源attention，强简单基线"],
        ["F1", "同小时五来源自注意力→softmax加权汇聚", "时间Transformer", "显式学习来源关系"],
        ["F1b", "来源注意力后保留五token，再concat投影", "时间Transformer", "避免过早加权求和"],
        ["F2", "168×5 token全局注意力", "与来源共同建模", "表达强但计算更重"],
    ], [20 * mm, 60 * mm, 40 * mm, 54 * mm]))
    story.append(P(
        "早期单节点阶段把候选结构分别装入三个预测任务，并用完整双结算日均净收益选型，而不是只比较MAE。"
        "已归档的三seed比较如下。"
    ))
    story.append(styled_table([
        ["任务", "候选结构", "日均净收益（USD/日）", "最终选型", "解释"],
        ["DA", "F0", "105.84 ± 0.99", "", ""],
        ["DA", "F1", B("107.57 ± 2.80"), B("F1"), "点估计更高"],
        ["RT-at-DA", "F0", "110.68 ± 0.83", B("F0"), "与F1b接近，方差更小、结构更简单"],
        ["RT-at-DA", "F1b", B("110.92 ± 1.52"), "", "点估计仅高0.24，不构成稳定优势"],
        ["滚动RT", "F0", B("108.56 ± 6.80"), "历史选F0", "早期单节点合同下较优"],
        ["滚动RT", "F1", "显著下降（原记录未给固定数值）", B("当前多节点选F1"), "后续多节点数据与合同下重新冻结"],
    ], [25 * mm, 25 * mm, 41 * mm, 32 * mm, 51 * mm]))
    story.append(P(
        "表3  早期单节点结构消融。加粗数值表示同一任务中已报告的最高点估计；加粗结构表示最终保留配置。"
        "完整F0/F1/F1b/F2比较来自早期单节点Huber阶段，不是当前无ID四市场、任务专属损失下的新全量消融。",
        "caption",
    ))
    story.append(callout(
        "论文级结构结论仍需补强",
        "如果论文要主张‘DA/RT使用F1而RT-at-DA使用F0是跨市场普遍最优’，建议补一次预算受控的无ID三任务消融："
        "至少比较当前结构与F0/F1替代，固定数据、seed、损失、训练轮数和LP/MPC合同。若只做阶段工程收尾，则现有结构可冻结，但应把该项列为局限。",
        "warn",
    ))
    story.append(H2("4.2 MAE、Pseudo-Huber与训练损失"))
    story.append(equation(
        "MAE = mean(|ŷ-y|)； Pseudo-Huber<sub>δ</sub>(e) = δ²(√(1+(e/δ)²)-1)，当前δ=1（作用于标准化价格）"
    ))
    story.append(P(
        "两者不是同一概念，但都属于监督预测损失。MAE对每个误差施加线性惩罚，抗尖峰且目标直观；"
        "Pseudo-Huber在误差很小时近似二次、误差很大时近似线性，处处可导，可在稳定梯度的同时降低少数极端价对训练的支配。"
        "RT-at-DA更直接影响DA-RT价差，历史实验显示其尖峰/幅度更难训练，因此最终采用Pseudo-Huber；DA与滚动RT采用MAE。"
    ))
    story.append(H2("4.3 盈利目标与监督损失的关系"))
    story.append(P(
        "利润不是逐小时独立标签，而是由24小时计划、跨时段SOC、效率、偏差罚金和求解器共同决定。"
        "动作中还包含方向选择、约束投影和滚动重优化，映射非光滑且跨时段耦合。直接用负利润训练需要对优化器与离散门控求导，"
        "或使用高方差零阶估计。项目已验证这条路线可运行，但尚未在跨时间/跨市场上稳定泛化。"
        "因此当前策略是用稳定监督损失学习价格表示，再用完整盈利作为最终系统评价，而不是把MAE误当作业务目标。"
    ))
    story.append(H2("4.4 训练流程与超参数"))
    story.append(styled_table([
        ["步骤", "操作", "目的"],
        ["1、平衡采样", "按市场、节点等权抽取训练样本", "避免PJM等大市场支配梯度"],
        ["2、样本归一化", "仅用起报前168小时计算中位数与MAD", "适配未见节点并杜绝目标期泄漏"],
        ["3、前向预测", "输出24小时或4小时价格", "形成三个任务各自的价格预测"],
        ["4、参数更新", "计算MAE/Pseudo-Huber并由AdamW更新", "稳定学习价格表示"],
        ["5、验证选模", "先算节点MAE，再做市场宏平均", "避免节点数量差异改变模型排序"],
        ["6、冻结评价", "保留最优checkpoint，再经LP/MPC结算", "将训练代理损失与最终盈利评价分离"],
    ], [32 * mm, 78 * mm, 64 * mm]))
    story.append(P(
        "完整年度样本外评价、强基线与统计检验遵循电价预测文献的基本建议[1][2]；2024冻结检查不参与重新选模。"
    ))
    story.append(styled_table([
        ["配置项", "冻结值", "具体含义"],
        ["训练 / 验证期", "2020-2022 / 2023", "2023用于内部开发评价，不与2024混池"],
        ["每市场样本数", "16,350", "配合市场-节点平衡采样"],
        ["优化器", "AdamW", "更新全部可训练参数"],
        ["初始学习率", "0.0002（2 × 10^-4）", "验证停滞2轮后减半，最低0.00001（10^-5）"],
        ["权重衰减", "0.01", "限制参数规模"],
        ["批量大小", "16", "每次参数更新的样本数"],
        ["最大轮数 / 早停", "20 / 7", "连续7轮无改善则停止"],
        ["梯度与精度", "clip=1.0；AMP", "抑制异常梯度并降低显存"],
        ["保存策略", "验证最优top-3", "最终LOMO采用seed 0；多seed仍待补"],
    ], [37 * mm, 58 * mm, 79 * mm]))

    chapter(story, "5", "比较方法：XGBoost与其他基线",
            "强基线不是陪衬。它用于检验Transformer是否真的利用了跨来源和长历史结构，并揭示复杂模型在小样本、强日历特征环境下可能不如树模型。")
    story.append(H2("5.1 XGBoost特征与配置"))
    story.append(styled_table([
        ["特征组", "DA / RT-at-DA", "滚动RT", "信息合法性"],
        ["最近24小时历史", "DA价、RT价、负荷、风、光、8维日历", "相同", "均早于对应起报时点"],
        ["目标时段日历", "目标日24小时 × 8维", "未来4小时 × 8维", "日历在起报时已知"],
        ["已公布DA价格", "不使用", "未来4小时DA价", "滚动RT运行时日前市场已出清"],
        ["身份与未来真值", "不使用未来价格/负荷；基线按节点独立训练", "相同", "防止目标泄漏"],
    ], [38 * mm, 53 * mm, 39 * mm, 44 * mm]))
    story.append(P(
        "特征直接展平后输入标准XGBoost[5]；没有Transformer表示、embedding、注意力或额外神经预训练。"
        "多步任务通过MultiOutputRegressor为每个预测维度拟合独立回归器。"
    ))
    story.append(callout(
        "比较单位：3个通用Transformer对比一整套逐节点XGBoost系统",
        "本文不是把3个Transformer只与其中某一个XGBoost回归器比较。Transformer侧由DA、RT-at-DA和滚动RT共3个独立训练的通用模型组成；"
        "同一组3个checkpoint复用于全部ERCOT节点。XGBoost侧则是大量彼此独立的本地回归器：每个节点的DA任务包含24个小时专属回归器，"
        "RT-at-DA包含24个，滚动RT包含4个，因此每节点共有52个XGBoost回归器。ERCOT盈利评价使用15个共同节点，合计为15 × 52 = 780个"
        "独立回归器；每个回归器内部又由180棵提升树组成，780不是树的总数。评价时，各节点只调用自己的XGBoost模型组；两侧分别生成全部节点的三任务预测，"
        "再进入完全相同的LP/MPC和双结算流程，最后汇总所有共同节点与日期的净收益。因此准确表述是“3个跨节点共享的通用Transformer系统”对比"
        "“由780个节点、任务和预测时距专属回归器构成的本地XGBoost基线系统”。",
        "info",
    ))
    story.append(styled_table([
        ["超参数", "冻结值", "作用"],
        ["树数量", "180", "控制迭代轮数与模型容量"],
        ["最大深度", "5", "限制单棵树的交互复杂度"],
        ["学习率", "0.04", "缩小每棵树的增量贡献"],
        ["样本采样率", "0.8", "每棵树使用80%训练样本"],
        ["特征采样率", "0.7", "每棵树使用70%候选特征"],
        ["L2正则", "1.0", "抑制叶节点权重过大"],
        ["目标函数", "逐节点三任务均为绝对误差（L1）", "避免跨市场逐节点Pseudo-Huber数值外推"],
        ["训练方式", "每节点独立；seed 0；单模型单线程", "节点间可并行，模型内保持可复现"],
    ], [42 * mm, 43 * mm, 89 * mm]))
    story.append(H2("5.2 XGBoost的优化机制"))
    story.append(P(
        "XGBoost不使用神经网络反向传播。它在每轮boosting中基于损失相对于当前预测的一阶梯度和二阶曲率，"
        "选择分裂并增加一棵树；树结构本身对输入不可导并不妨碍其原生训练算法。项目没有用‘把树求导’的办法解决问题，"
        "而是调用XGBoost的标准梯度提升过程。当前最终逐节点系统的三个任务统一冻结为绝对误差目标。"
    ))
    story.append(callout(
        "为什么最终逐节点RT-at-DA不用Pseudo-Huber",
        "在打开2025以前，训练期数值审计发现原生reg:pseudohubererror用于CAISO、NYISO逐节点多输出拟合时会产生约3万—4.7万美元/MWh的病态外推；"
        "CAISO验证MAE一度约为6401。改为绝对误差后，同一CAISO任务MAE为7.17，预测恢复合理量级。因此最终逐节点强基线统一采用L1。"
        "pooled LOMO XGBoost的RT-at-DA Pseudo-Huber已在四折中保持稳定，仍按原冻结协议保留。该修复是2025解封前的数值正确性处理，不使用最终测试结果选模。",
        "risk",
    ))
    story.append(callout(
        "比较公平性",
        "本文同时保留两种XGBoost。pooled LOMO XGBoost与Transformer都只用三个源市场2020-2022，"
        "不读取留出市场标签，是同数据公平基线；逐节点本地XGBoost看过目标市场，是实际监督强上界。"
        "2024盈利仅有后者的冻结外推，仍存在训练年限与市场可见性差异，因此不将其写成纯架构因果比较。",
        "risk",
    ))
    story.append(H2("5.3 其他基线与历史证据"))
    story.append(P("项目还实现了lag-168 Seasonal、cutoff-safe LEAR、GRU以及单节点统一完整系统对照。历史收益比较统一列于下表。"))
    story.append(styled_table([
        ["历史合同", "系统", "日均净收益（USD/日）", "比较结论"],
        ["旧无罚金，182日", "Transformer-Huber", "123.06", ""],
        ["旧无罚金，182日", "V2联合微调", B("123.83 ± 0.33"), "该合同点估计最高"],
        ["旧无罚金，182日", "XGBoost", "89.99 ± 11.17", ""],
        ["旧无罚金，182日", "GRU", "88.63 ± 11.57", ""],
        ["旧无罚金，182日", "Seasonal", "36.76", ""],
        ["偏差罚金合同", "XGBoost", B("97.32 ± 0.65"), "该合同验证冠军"],
        ["偏差罚金合同", "Transformer-Huber", "83.70", "低于XGBoost"],
        ["偏差罚金合同", "三路active联合微调", "50.45", "未晋级"],
    ], [38 * mm, 45 * mm, 48 * mm, 43 * mm]))
    story.append(P(
        "表4  历史单节点基线。加粗表示同一结算合同中的较优结果。合同变化会改变模型排名，因此这些结果不与当前多节点结果混排。",
        "caption",
    ))
    story.append(H2("5.4 各基线在实验中的角色"))
    story.append(styled_table([
        ["方法", "基线类型", "主要用途", "与最终系统的关系"],
        ["lag-168 Seasonal", "周季节复制", "检验模型是否超过最简单的周周期规律", "四市场零样本预测的统一底线"],
        ["cutoff-safe LEAR", "稀疏线性电价模型", "提供经典统计学习参照", "仅用于适用的日前任务"],
        ["GRU", "轻量神经序列模型", "区分Transformer收益是否来自注意力结构", "历史单节点对照"],
        ["pooled LOMO XGBoost", "源市场树模型", "同训练数据公平对照", "Transformer胜9/12项MAE，收益未胜"],
        ["逐节点XGBoost", "目标市场监督树模型", "提供实际监督强上界", "15节点共780个任务/时距回归器"],
        ["TimesFM-3 frozen", "时序基础模型", "检验通用预训练迁移", "滚动RT占优，完整收益失败"],
        ["TimesFM-2.5 LoRA", "源市场监督适配", "检验TSFM微调迁移", "单变量价格历史，不是frozen"],
        ["无ID Transformer", "跨市场零样本模型", "检验不依赖身份的可迁移表示", "本文研究对象"],
    ], [37 * mm, 39 * mm, 58 * mm, 40 * mm]))

    chapter(story, "6", "决策方法：连续DA LP、滚动RT MPC与结算",
            "三模型输出只有转化为满足功率、SOC和偏差成本约束的动作，才具备可比较的经济含义。")
    story.append(H2("6.1 电池合同"))
    story.append(styled_table([
        ["参数", "数值", "含义"],
        ["Pmax", "1 MW", "最大充/放功率"],
        ["Emax", "4 MWh", "额定能量容量"],
        ["SOC范围", "0.4-3.6 MWh", "保留10%安全边界"],
        ["初始SOC", "2 MWh", "每个真实连续段开始时"],
        ["ηc, ηd", "0.95, 0.95", "充/放电分量效率"],
        ["Ecyc", "4 MWh/日", "每日放电量上限"],
        ["κ", "5.7 USD/MWh", "实际充电+放电吞吐退化成本"],
        ["偏差罚金", "2×|pRT|×超容忍偏差", "容忍带为3%×|uDA|"],
    ], [34 * mm, 38 * mm, 102 * mm]))
    story.append(P(
        "5.7来自项目冻结的电池退化/交易成本合同，不是把5.7除以效率后再作为成本重复扣除。"
        "5.7/0.95=6.0 USD/MWh只在历史经济状态头中作为近似盈亏平衡价差阈值：一次电量通过效率损失后，"
        "价差至少需要覆盖有效放电侧成本。将循环老化成本显式纳入电池市场调度的建模动机见[7]。当前无ID最终RT-at-DA没有经济状态头。"
    ))
    story.append(H2("6.2 双结算账本"))
    story.append(equation(
        "R = Σ<sub>t</sub>[ p<sup>DA</sup><sub>t</sub>u<sup>DA</sup><sub>t</sub> + "
        "p<sup>RT</sup><sub>t</sub>(u<sup>RT</sup><sub>t</sub>-u<sup>DA</sup><sub>t</sub>) "
        "- κ·throughput<sub>t</sub> ] - P<sub>dev</sub>"
    ))
    story.append(equation(
        "P<sub>dev</sub> = Σ<sub>t</sub> 2|p<sup>RT</sup><sub>t</sub>·max(|u<sup>RT</sup><sub>t</sub>-u<sup>DA</sup><sub>t</sub>| "
        "- 0.03|u<sup>DA</sup><sub>t</sub>|, 0)"
    ))
    story.append(P(
        "正动作表示放电，负动作表示充电。DA计划在交付前锁定；实际SOC只由实时执行动作更新。"
        "同一节点连续日期继承SOC，不同节点绝不互相继承。终端SOC只报告MWh，不货币化；真实价格只在动作确定后用于结算。"
    ))
    story.append(H2("6.3 模块化预测-优化框架的选择依据"))
    story.append(bullets([
        "LP/MPC显式保证功率、SOC、效率、日循环和偏差罚金可行性；纯神经网络输出动作需要再投影，仍绕不开优化层。",
        "三个任务的信息时点不同：D-1模型一天运行一次，滚动RT每小时更新；强行共享编码器会混淆可见信息和样本频率。",
        "独立模型便于替换、诊断和公平比较；某一任务退化时可定位到DA、RT-at-DA或滚动RT，而不是只看到总利润变化。",
        "端到端盈利训练仍可作为后续研究，但必须先解决非光滑方向门控、solver梯度、跨时段信用分配和外层稳定性。",
    ]))

    chapter(story, "7", "消融实验与模型选择依据",
            "本章只保留影响当前系统设计的关键阶段，说明每一阶段解决的问题、得到的证据以及最终保留了什么。")
    story.append(H2("7.1 关键阶段"))
    story.append(styled_table([
        ["阶段", "当时要解决的问题", "实施与证据", "对当前系统的影响"],
        ["1、单节点结构与闭环验证", "三个任务是否应采用同一种融合结构；盈利是否能直接训练", "F0/F1/F1b/F2按任务做三seed比较；收益导向联合微调可运行，但跨设置提升很小或不稳定", "保留任务专属结构；最终不采用端到端联合训练"],
        ["2、多节点共享模型", "共享参数能否扩大有效样本，同时保持不同节点的可分性", "引入市场/节点embedding、平衡采样和节点独立SOC；已知实体模型最高接近或略超部分基线，但不支持未见市场", "保留多节点数据与训练基础设施；区分‘已知实体’和‘零样本’结论"],
        ["3、任务专属训练", "同一Huber损失是否适合三个价格任务", "DA与滚动RT改用MAE；RT-at-DA使用Pseudo-Huber；同时完善连续LP/MPC与罚金结算", "形成当前三任务独立训练合同，盈利统一在冻结后评价"],
        ["4、无ID跨市场模型", "移除身份后，模型能否迁移到完全未见市场", "删除node_id/market_id前向输入，使用样本内稳健归一化；四折LOMO的12/12项MAE优于lag-168", "形成当前正式零样本预测系统"],
        ["5、时间外盈利检查", "2023优势能否跨年保持", "ERCOT 2023点估计领先XGBoost；2024冻结检查落后且区间低于0", "多节点工程可收尾，但稳定盈利目标尚未闭环"],
    ], [31 * mm, 48 * mm, 58 * mm, 37 * mm]))
    story.append(P(
        "其余未晋级尝试（如简单尖峰加权、显式价差回归和高回撤残差路线）不再逐项展开，因为它们没有改变最终架构选择。"
        "这里保留的因果链是：先建立可执行闭环，再扩展到多节点，随后移除身份形成零样本协议，最后用跨年盈利检查检验业务目标。"
    ))
    story.append(styled_table([
        ["收益导向训练实验", "冻结预测基线", "联合训练结果", "相对变化", "结论"],
        ["旧无罚金V2", "123.06", B("123.83 ± 0.33"), "+0.77（约+0.63%）", "提升很小"],
        ["罚金版active：开发期", "冻结基线", B("冻结基线 + 0.78"), "+0.78", "仅小幅改善"],
        ["罚金版active：独立报告期", B("冻结基线"), "冻结基线 - 5.00", "-5.00", "未晋级"],
    ], [42 * mm, 34 * mm, 42 * mm, 30 * mm, 26 * mm]))
    story.append(P("表5  历史收益导向联合训练比较，单位USD/日；加粗为同一行较优结果。", "caption"))
    story.append(H2("7.2 历史经济状态头及其适用范围"))
    story.append(P(
        "DA计划的方向直接由p_DA-p_RT-at-DA决定，历史实验中主要瓶颈是这条价差的经济方向，而不是滚动RT的局部价格。"
        "因此只在RT-at-DA侧增加三分类头：中性、DA高、RT高；上游DA和RT-at-DA骨干全部冻结，"
        "输入为128维冻结表征加1维标准化基础价差，Linear(129,3)约390参数。它只翻转方向，不改变基础价差绝对幅度。"
    ))
    story.append(styled_table([
        ["历史带ID方案", "seed 0", "seed 1", "seed 2", "多数票/基线", "统计结论"],
        ["经济状态头", "296.99", "298.53", B("300.30"), B("298.53"), "三seed范围较窄；多数票点估计较高"],
        ["XGBoost", "-", "-", "-", "292.29", "经济状态头相对差异95%区间[-9.35, 24.98]，跨0"],
    ], [40 * mm, 22 * mm, 22 * mm, 22 * mm, 34 * mm, 34 * mm]))
    story.append(P("表6  历史经济状态头的三seed结果，单位USD/节点·日；加粗分别表示最高seed和多数票点估计领先。", "caption"))
    story.append(callout(
        "与当前最终系统的关系：历史经济状态头未被使用",
        "历史经济状态头没有进入当前无ID模型的checkpoint、训练、LP/MPC或结算；377.84/73.38仅来自DA、RT-at-DA和滚动RT三套基础价格模型。"
        "当前由DA模型（F1/MAE）和RT-at-DA模型（F0/Pseudo-Huber）分别预测价格，日前LP计算“预测DA价 - 预测RT-at-DA价”，隐式决定买卖方向；"
        "效率、5.7 USD/MWh吞吐成本、SOC/循环约束和偏差罚金过滤弱价差，交付日再由滚动RT与MPC逐小时修正RT动作。"
        "当前没有显式价差方向头或方向损失，错误方向仍可能发生；若重新引入该头，必须在无ID骨干上重新训练并完成四市场及2024冻结验证，不能沿用历史数字。",
        "risk",
    ))

    chapter(story, "8", "实验结果：四市场零样本预测",
            "留一市场（Leave-One-Market-Out, LOMO）是指每次用三个市场训练，把第4个市场完全留作测试。"
            "CAISO、ERCOT、NYISO、PJM轮流留出，因此一共四折；每折都重新训练三项任务，网络前向不读取node_id或market_id，"
            "归一化也不使用留出市场的训练期scaler。")
    story.append(protocol_diagram())
    story.append(P("图5  当前四折LOMO验证协议与外层边界。", "caption"))
    seasonal = {
        ("CAISO", "DA"): 20.20, ("CAISO", "RT-at-DA"): 23.28, ("CAISO", "滚动RT"): 23.30,
        ("ERCOT", "DA"): 47.99, ("ERCOT", "RT-at-DA"): 50.60, ("ERCOT", "滚动RT"): 50.72,
        ("NYISO", "DA"): 12.28, ("NYISO", "RT-at-DA"): 16.17, ("NYISO", "滚动RT"): 16.17,
        ("PJM", "DA"): 10.10, ("PJM", "RT-at-DA"): 13.00, ("PJM", "滚动RT"): 12.99,
    }
    rows = [["留出市场", "任务", "无ID模型MAE", "lag-168 MAE", "相对改善"]]
    for market in ["CAISO", "ERCOT", "NYISO", "PJM"]:
        for task in ["DA", "RT-at-DA", "滚动RT"]:
            value, base = lomo[(market, task)], seasonal[(market, task)]
            gain = 100 * (base - value) / base
            rows.append([market, task, B(f"{value:.2f}"), f"{base:.2f}", f"+{gain:.1f}%"])
    story.append(styled_table(rows, [30 * mm, 32 * mm, 38 * mm, 38 * mm, 36 * mm], alignments={2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}))
    story.append(P("表7  2023节点等权MAE，单位USD/MWh；加粗为同一行中更低的误差，12项均优于lag-168。", "caption"))
    story.append(callout(
        "预测指标与盈利指标的关系",
        "本表回答的是‘模型是否在未见市场学到可迁移的价格规律’，不是最终业务结论。MAE越小通常有利于决策，但错误发生在什么小时、"
        "是否改变充放电方向、是否错过尖峰，都会使相近MAE产生不同利润。因此最终排序必须以同一LP/MPC合同下的净收益为准。",
        "info",
    ))
    story.append(styled_table([
        ["盈利评价范围", "自有系统", "XGBoost", "差额/结论"],
        ["ERCOT 2023（开发验证）", B("377.84"), "292.29", "+85.55（+29.3%）；95%区间跨0"],
        ["ERCOT 2024（冻结检查）", "73.38", B("90.69"), "-17.32（-19.1%）；95%区间低于0"],
        ["CAISO / NYISO / PJM", "—", "—", "结算合同未冻结，只报告预测结果"],
    ], [48 * mm, 30 * mm, 30 * mm, 66 * mm], alignments={1: "RIGHT", 2: "RIGHT"}))
    story.append(P("表8  日均净收益，单位USD/节点·日；加粗为同年度较高收益，详细分布和分解见第9章。", "caption"))
    story.append(bar_chart(
        ["CAISO", "ERCOT", "NYISO", "PJM"],
        [("DA", [9.9, 22.3, 30.4, 38.6], BLUE),
         ("RT-at-DA", [9.7, 31.2, 33.6, 31.1], ORANGE),
         ("滚动RT", [58.2, 43.1, 55.9, 49.4], GREEN)], 0, 65,
    ))
    story.append(P("图6  相对lag-168的MAE改善（%）。该图只评价预测迁移，不替代第9章盈利结论。", "caption"))
    story.append(H2("8.1 与目标市场内XGBoost的边界比较"))
    story.append(styled_table([
        ["市场", "无ID DA MAE", "目标市场XGBoost MAE", "较优系统", "比较边界"],
        ["CAISO", "18.20", B("12.17"), B("XGBoost"), "XGBoost见过目标市场"],
        ["ERCOT", "37.27", B("35.51"), B("XGBoost"), "差距1.76"],
        ["NYISO", "8.55", B("6.80"), B("XGBoost"), "XGBoost见过目标市场"],
        ["PJM", "6.20", B("5.88"), B("XGBoost"), "差距0.32"],
    ], [29 * mm, 34 * mm, 43 * mm, 31 * mm, 37 * mm]))
    story.append(P(
        "表9  DA预测与目标市场内监督XGBoost的历史MAE比较，单位USD/MWh；加粗为较低误差。"
        "该结果不是相同训练数据下的纯架构因果比较。最终2020—2024逐节点基线统一采用L1目标；旧的非ERCOT RT-at-DA Pseudo-Huber异常结果不参与最终比较。",
        "caption",
    ))
    story.append(H2("8.2 同数据pooled XGBoost与时序基础模型"))
    comparison_rows = [[
        "留出市场", "任务", "当前Transformer",
        "pooled XGBoost", "TimesFM-3", "TimesFM-2.5 LoRA",
    ]]
    for market in ("CAISO", "ERCOT", "NYISO", "PJM"):
        for task in ("DA", "RT-at-DA", "滚动RT"):
            values = [
                lomo[(market, task)], pooled_mae[(market, task)],
                tsfm3_mae[(market, task)], tsfm25_mae[(market, task)],
            ]
            best = min(values)
            cells = [B(f"{value:.2f}") if value == best else f"{value:.2f}" for value in values]
            comparison_rows.append([market, task, *cells])
    story.append(styled_table(
        comparison_rows,
        [24 * mm, 25 * mm, 32 * mm, 31 * mm, 30 * mm, 32 * mm],
        alignments={2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT"},
    ))
    story.append(P(
        "表9A  2023四折LOMO节点等权MAE，单位USD/MWh；每行加粗为最低误差。"
        "当前Transformer相对pooled XGBoost和TimesFM-2.5均胜9/12项，相对TimesFM-3胜8/12项；"
        "TimesFM-3的4项优势全部来自滚动RT。",
        "caption",
    ))
    story.append(P(
        "TimesFM-3是不使用本项目监督标签的frozen时序基础模型[10]。TimesFM-2.5 LoRA仅用三个源市场的"
        "2020-2022监督数据适配，目标市场仍为零样本；其输入仅含目标价格历史。表9A比较的是完整系统，"
        "不将差异归因于单一结构。"
    ))
    story.append(PageBreak())
    story.append(H2("8.3 结果解释与适用边界"))
    story.append(bullets([
        "能证明：模型学习到了跨市场可迁移规律，而不是依赖身份查找表；12/12均优于周季节复制。",
        "不能证明：任意世界市场都可直接使用；新市场仍须满足同字段、小时频率、信息发布时间和目标定义。",
        "能证明：在同源市场和年份协议下，当前Transformer的MAE多数优于pooled XGBoost和TimesFM-2.5。",
        "不能证明：零样本模型已经胜过所有目标市场监督模型；逐节点本地XGBoost仍是另一类监督强上界。",
        "不能由MAE直接推出盈利；只有ERCOT完成了同口径连续LP/MPC结算。",
    ]))

    chapter(story, "9", "实验结果：ERCOT 2023与2024盈利",
            "盈利结果按节点等权、共同交付日、相同电池合同与求解器比较。2023是内部开发验证；2024是冻结时间外检查，不能混池选模。")
    story.append(bar_chart(
        ["2023", "2024"],
        [("无ID三模型", [u23["macro_mean_daily_revenue"], u24["macro_mean_daily_revenue"]], BLUE),
         ("稳健XGBoost", [x23["macro_mean_daily_revenue"], x24["macro_mean_daily_revenue"]], ORANGE),
         ("Seasonal", [s23["macro_mean_daily_revenue"], 0], MUTED)], 0, 420,
    ))
    story.append(P("图7  ERCOT日均净收益（USD/节点·日）。2024未在冻结流程中重算Seasonal，图中不作比较。", "caption"))
    story.append(styled_table([
        ["年份", "共同日", "无ID三模型", "XGBoost", "差额", "相对变化", "配对95%区间"],
        ["2023", "362", B("377.84"), "292.29", "+85.55", "+29.3%", "[-12.70, 204.87]"],
        ["2024冻结", "363", "73.38", B("90.69"), "-17.32", "-19.1%", "[-30.86, -5.06]"],
    ], [27 * mm, 20 * mm, 28 * mm, 28 * mm, 24 * mm, 24 * mm, 43 * mm]))
    story.append(P("表10  ERCOT年度日均净收益，单位USD/节点·日；加粗为同年度较高收益。", "caption"))
    story.append(H2("9.1 2023同口径多系统盈利比较"))
    story.append(styled_table([
        ["系统", "训练/适配边界", "日均净收益", "与相关基线的统计结论"],
        ["pooled LOMO XGBoost", "三源市场、不看ERCOT", B(f"{pooled_profit['ercot_2023_mean_daily_net_revenue']:.2f}"), B("同数据最高点估计")],
        ["当前Transformer三seed集成", "三源市场、不看ERCOT", f"{ensemble_profit['ercot_2023_mean_daily_net_revenue']:.2f}", "相对pooled差-10.90，区间[-28.07, 7.43]"],
        ["当前Transformer seed 0", "三源市场、不看ERCOT", f"{u23['macro_mean_daily_revenue']:.2f}", "相对pooled差-40.23，区间[-61.44, -21.95]"],
        ["逐节点本地XGBoost", "看过ERCOT，监督强上界", f"{x23['macro_mean_daily_revenue']:.2f}", "与同数据架构对照分开解释"],
        ["TimesFM-2.5 LoRA", "三源市场监督适配", f"{t25_profit['macro_mean_daily_revenue']:.2f}", "相对Transformer seed 0差-115.05，区间[-225.13, -28.04]"],
        ["TimesFM-3 frozen", "不用项目监督标签", f"{t3_profit['macro_mean_daily_revenue']:.2f}", "DA计划与偏差罚金使完整系统失败"],
    ], [38 * mm, 43 * mm, 32 * mm, 61 * mm]))
    story.append(P(
        "表10A  ERCOT 2023同一LP/MPC与双结算下的日均净收益，单位USD/节点·日；"
        "加粗为同训练数据主对照中的较高点估计。",
        "caption",
    ))
    story.append(P(
        "这一结果修正了‘当前模型已打败XGBoost’的粗略表述：seed 0确实高于逐节点本地XGBoost，"
        "但低于同数据pooled XGBoost；固定三seed集成将差距缩小到10.90，配对区间仍跨0，"
        "因此只能说已有竞争性，不能说稳定胜出。"
    ))
    story.append(H2("9.2 2023：点估计领先目标市场基线，但收益偏重尾部"))
    story.append(styled_table([
        ["系统", "均值", "中位数", "正收益日", "去最高5日", "最大回撤"],
        ["无ID三模型", B("377.84"), "19.21", "72.9%", B("251.43"), B("195.01")],
        ["XGBoost", "292.29", B("38.13"), B("79.3%"), "210.51", "206.07"],
        ["Seasonal", "213.78", "-3.25", "46.7%", "117.55", "9733.25"],
    ], [36 * mm, 27 * mm, 27 * mm, 27 * mm, 30 * mm, 27 * mm]))
    story.append(P("表11  2023收益分布；加粗表示较优结果，其中最大回撤越低越好。", "caption"))
    story.append(styled_table([
        ["2023配对比较", "自有系统减基线", "7日块bootstrap 95%区间", "统计判断"],
        ["相对Seasonal", B("+164.06"), "[51.05, 305.50]", B("区间高于0")],
        ["相对XGBoost", "+85.55", "[-12.70, 204.87]", "区间跨0"],
    ], [45 * mm, 39 * mm, 52 * mm, 38 * mm]))
    story.append(P("表12  2023配对收益差，单位USD/节点·日；加粗表示具有明确正向区间的比较。", "caption"))
    story.append(P(
        "表12表明，自有系统相对Seasonal的优势具有明确正向区间，但相对XGBoost只能表述为点估计领先。"
        "其最高10日贡献全年约55.1%，说明平均值受尖峰日显著影响；但去除最高5日后仍高于XGBoost，优势并非完全由极少数日构成。"
    ))
    story.append(H2("9.3 2024：冻结优势反转"))
    story.append(P("同一seed 0 checkpoint保持2020-2022、CAISO/NYISO/PJM训练，不使用2023重训；2024仅生成ERCOT预测并结算。"))
    story.append(styled_table([
        ["2024指标", "无ID三模型", "XGBoost", "较优系统"],
        ["日均净收益", "73.38", B("90.69"), B("XGBoost")],
        ["日收益中位数", "18.79", B("29.15"), B("XGBoost")],
        ["正收益日比例", "77.96%", B("78.51%"), B("XGBoost")],
        ["去最高5日后均值", "52.88", B("67.28"), B("XGBoost")],
        ["最高单日收益", "3063.65", B("4003.98"), B("XGBoost")],
        ["节点均值胜出数", "1 / 15", B("14 / 15"), B("XGBoost")],
    ], [52 * mm, 38 * mm, 38 * mm, 46 * mm]))
    story.append(P(
        "表13  2024冻结收益比较，单位除比例和节点数外均为USD/节点·日；加粗表示较优结果。"
        "去最高5日后结论不变，两套系统均无SOC越界。",
        "caption",
    ))
    story.append(H2("9.4 2024收益分解"))
    story.append(bar_chart(
        ["DA腿", "RT偏差腿", "退化成本", "偏差罚金"],
        [("无ID三模型", [101.86, 0.31, 21.65, 7.15], BLUE),
         ("XGBoost", [131.60, 0.07, 30.46, 10.52], ORANGE)], 0, 145,
    ))
    story.append(P("图8  2024收益分解。成本项以正数显示，净收益为DA腿+RT腿-成本。", "caption"))
    story.append(styled_table([
        ["2024收益分量", "无ID三模型", "XGBoost", "判定方向"],
        ["DA结算腿", "101.86", B("131.60"), "收入越高越好"],
        ["RT偏差腿", B("0.31"), "0.07", "收入越高越好"],
        ["退化成本", B("21.65"), "30.46", "成本越低越好"],
        ["偏差罚金", B("7.15"), "10.52", "成本越低越好"],
        ["净收益", "73.38", B("90.69"), "最终评价指标"],
    ], [48 * mm, 38 * mm, 38 * mm, 50 * mm]))
    story.append(P("表14  2024日均收益分解，单位USD/节点·日；加粗表示该分量的较优值。", "caption"))
    story.append(P(
        "表14显示，自有系统较低的退化成本和偏差罚金不足以抵消DA结算腿收入缺口。"
        "下一阶段应优先检查DA/RT-at-DA价差对锁仓的影响，但不能单凭收益分解断言某个网络模块失效。"
    ))

    chapter(story, "10", "讨论：跨年稳定性与统计边界",
            "2024收益显著下降既有市场机会收缩，也有模型相对XGBoost捕捉能力下降；二者必须分开陈述。")
    story.append(H2("10.1 市场机会收缩"))
    story.append(styled_table([
        ["数据级诊断（ERCOT 15节点）", "2023", "2024", "变化"],
        ["平均日内DA最高-最低差", "225.15", "93.54", "-58.5%"],
        ["平均|DA-RT|", "34.67", "13.33", "-61.6%"],
        ["DA>1000 USD/MWh小时占比", "0.810%", "0.087%", "约降至1/9"],
        ["无ID系统最好单日收益", "11865.21", "3063.65", "-74.2%"],
        ["无ID系统日收益中位数", "19.21", "18.79", "基本不变"],
    ], [70 * mm, 32 * mm, 32 * mm, 40 * mm]))
    story.append(P("表15  ERCOT跨年市场机会与收益尾部变化；该表描述市场环境，不将较高数值直接定义为模型胜者。", "caption"))
    story.append(P(
        "2024大幅减少了可套利尖峰，因此所有系统均值下降是合理的；而中位数几乎不变，说明下降主要发生在高收益尾部。"
        "但表10显示同年XGBoost仍领先，因此不能把相对落后全部归因于市场变平。"
    ))
    story.append(H2("10.2 负价、尖峰与极端价差诊断"))
    ercot_tail = tail["forecast_slices"]["ERCOT"]
    tail_sources = [
        ("当前Transformer", ercot_tail["transformer_2023"]),
        ("pooled XGBoost", ercot_tail["xgboost_pooled_2023"]),
        ("TimesFM-3", ercot_tail["timesfm3_2023"]),
        ("TimesFM-2.5 LoRA", ercot_tail["timesfm25_lora_2023"]),
    ]
    metric_values = []
    for _, report in tail_sources:
        metric_values.append([
            report["da"]["highest_1pct"]["spike_recall"],
            report["rt_at_da"]["highest_1pct"]["spike_recall"],
            report["rt"]["highest_1pct"]["spike_recall"],
            report["da"]["negative_price"]["negative_recall"],
            report["da_rt_at_da_spread"]["abs_spread_top_1pct"]["direction_accuracy"],
            report["da_rt_at_da_spread"]["abs_spread_top_1pct"]["magnitude_ratio"],
        ])
    best_by_metric = [max(row[index] for row in metric_values) for index in range(6)]
    tail_rows = [[
        "系统", "DA尖峰召回", "RT-at-DA尖峰召回", "滚动RT尖峰召回",
        "DA负价召回", "极端价差方向", "极端价差幅度比",
    ]]
    for (name, _), values in zip(tail_sources, metric_values):
        cells = []
        for index, value in enumerate(values):
            shown = f"{100 * value:.1f}%"
            cells.append(B(shown) if value == best_by_metric[index] else shown)
        tail_rows.append([name, *cells])
    story.append(styled_table(
        tail_rows,
        [34 * mm, 21 * mm, 25 * mm, 25 * mm, 21 * mm, 24 * mm, 24 * mm],
    ))
    story.append(P(
        "表15A  ERCOT 2023尾部识别与价差指标；阈值由2020-2022历史预先冻结，加粗为每列较高值。"
        "幅度比为预测绝对价差与真实绝对价差的比值，100%表示幅度不收缩。",
        "caption",
    ))
    story.append(P(
        "当前Transformer的DA和RT-at-DA最高1%尖峰召回率均为0，对应MAE为1329.15和1758.20 USD/MWh；"
        "在|DA-RT|最高1%时段，方向准确率为56.5%，但幅度只保留真实价差的2.0%。"
        "TimesFM-3的滚动RT尖峰召回率达55.1%，但DA价差几乎消失，导致完整收益为-91.14；"
        "TimesFM-2.5改善了DA尖峰召回，但完整收益仍低于当前Transformer。"
        "因此后续不应只继续扩大滚动RT模型，而应在信息合法的DA/RT-at-DA两腿做单因素尾部损失或分层采样消融。"
    ))
    story.append(H2("10.3 统计解释"))
    story.append(bullets([
        "收益按同日配对，并用7日块bootstrap保留短期时间相关性；区间不是独立日假设下的普通t区间。",
        "2023对XGBoost区间跨0：只能说点估计领先，不能说统计显著。",
        "2024区间整体低于0：在当前冻结比较设置下，自有模型明确落后。",
        "2023用于开发验证、2024用于冻结检查；不能合并两年后重新选结构，否则会污染外层评价。",
        "ERCOT三任务seed 0/1/2及固定等权集成已完成；其他三个留出市场的seed 1/2作为非硬门槛暂缓。",
    ]))
    story.append(callout(
        "训练数据不等同",
        "2024无ID模型没见过ERCOT且只用2020-2022；XGBoost见过ERCOT并用到2023。"
        "因此结果能回答‘现有两套可用系统谁在2024赚得更多’，不能严格回答‘Transformer架构是否因果上弱于XGBoost’。"
        "2023已增pooled LOMO XGBoost作为相同源市场与年份的公平对照；2024仍只能报告现有两套可用系统的冻结工程比较。",
        "risk",
    ))

    chapter(story, "11", "结论与后续工作",
            "多节点阶段可以作为基础设施与验证协议收尾，但科学结论必须保留盈利稳定性和公平对照上的缺口。")
    story.append(H2("11.1 已经完成"))
    story.append(bullets([
        "四市场64节点数据合同、时区、DST、字段完整性和哈希审计。",
        "DA、RT-at-DA、滚动RT三套独立模型及无ID部署合同。",
        "四折LOMO共12个正式预测结果，checkpoint哈希12/12一致。",
        "市场-节点平衡采样、节点等权/市场宏平均选模、SOC按节点隔离。",
        "连续DA LP、逐小时4小时RT MPC、偏差罚金、退化成本和双结算账本。",
        "ERCOT 2023内部完整盈利与2024冻结时间外检查；全部结果和输入哈希归档。",
        "ERCOT留出折三任务seed 0/1/2、固定等权集成及当前无ID协议下的受控结构消融。",
        "同训练数据pooled LOMO XGBoost、官方frozen TimesFM-3和源市场监督TimesFM-2.5 LoRA对照。",
        "五套系统统一尾部诊断：负价、最高/最低1%、极端DA-RT价差及同合同收益分解。",
        "历史单节点、带ID、经济状态头和失败实验均保留，可追溯而未混入最终零样本口径。",
    ]))
    if locked_2025 is None:
        story.append(H2("11.2 最终闭环状态与剩余环节"))
        story.append(styled_table([
            ["待完成项", "当前状态", "是否增加研究路线"],
            ["最终基线训练", "201/201项产物已经完成并通过逐文件哈希审计", "已完成"],
            ["2025一次性测试", "三份预注册核验后才允许打开；不得据结果回选模型", "否"],
            ["三外部市场结算", "缺少已冻结的真实结算合同，本轮仅报告预测与尾部指标", "非投稿阻塞项"],
            ["最终成本与Oracle", "已接入统一2025入口，随最终评价自动生成", "否"],
        ], [45 * mm, 94 * mm, 35 * mm]))
        story.append(callout(
            "事前冻结的盈利主判据",
            "主经济终点为ERCOT 2025固定三seed Transformer集成相对四市场pooled XGBoost的节点等权日均净收益差。"
            "只有差值大于0且配对7日块Bootstrap 95%区间下界大于0，才称为稳定盈利胜出；"
            "若点估计为正但区间跨0，只称改善信号。MAE仅用于预测与迁移诊断，不能覆盖盈利结论。",
            "info",
        ))
        story.append(H2("11.3 冻结执行与停止规则"))
        story.append(bullets([
            "201项最终基线训练产物审计已经通过；三份预注册核验后再显式授权一次性打开2025。",
            "2025统一评价固定报告四市场预测、尾部指标及ERCOT完整盈利、Oracle和风险。",
            "结果无论胜出、接近或落后，均不再追加结构、seed、损失或尾部机制。",
            "只有机器中断、文件损坏或明确代码错误，才允许在同一冻结哈希下恢复运行。",
        ]))
        story.append(callout(
            "阶段定性",
            "研究路线与最终验收协议已经冻结；剩余工作是执行既定基线和一次性2025测试，而不是继续选模。",
            "good",
        ))
    else:
        story.append(H2("11.2 2025一次性锁定测试"))
        story.append(P(
            "下表由锁定汇总JSON自动生成。MAE为四个市场节点宏平均后的市场平均；盈利仅报告具有真实冻结结算合同的ERCOT，"
            "单位为USD/节点·日。每列最佳可部署系统加粗，Oracle仅作为受限先知上界。"
        ))
        story.append(styled_table(
            locked_2025_comparison_rows(locked_2025),
            [39 * mm, 25 * mm, 29 * mm, 29 * mm, 28 * mm, 24 * mm],
        ))
        profit = locked_2025_profit_evidence(locked_2025)
        story.append(H2("11.3 预注册盈利主终点"))
        story.append(styled_table([
            ["主系统", "主要公平对照", "收益差", "配对7日块95%区间", "判定"],
            [f"Transformer集成 {profit['transformer']['macro_mean_daily_revenue']:.2f}",
             f"pooled XGBoost {profit['comparator']['macro_mean_daily_revenue']:.2f}",
             f"{profit['mean_difference']:+.2f}",
             f"[{profit['ci'][0]:.2f}, {profit['ci'][1]:.2f}]",
             {
                 "stable_superiority": "稳定胜出",
                 "positive_point_estimate_only": "仅点估计改善",
                 "no_superiority": "未胜出",
             }[profit["classification"]]],
        ], [38 * mm, 38 * mm, 26 * mm, 39 * mm, 33 * mm]))
        story.append(callout("盈利结论", profit["conclusion"], profit["tone"]))

        transformer_label = "da_ensemble__rt_at_da_ensemble__rt_ensemble"
        pooled_label = (
            "da_pooled_xgboost__rt_at_da_pooled_xgboost__rt_pooled_xgboost"
        )
        settlement_systems = locked_2025["ercot_settlement"]["systems"]
        transformer_settlement = settlement_systems[transformer_label]
        pooled_settlement = settlement_systems[pooled_label]
        story.append(P(
            "下列风险与去极端日结果用于解释盈利是否依赖少数尖峰，不参与2025后的模型回选。",
            "body",
        ))
        story.append(styled_table([
            ["盈利与风险诊断", "Transformer固定集成", "pooled XGBoost"],
            ["去最高10日后日均收益", f"{profit['transformer']['macro_mean_daily_revenue_without_top_10_days']:.2f}",
             f"{profit['comparator']['macro_mean_daily_revenue_without_top_10_days']:.2f}"],
            ["最大回撤（USD）", f"{profit['transformer']['macro_maximum_drawdown_usd']:.2f}",
             f"{profit['comparator']['macro_maximum_drawdown_usd']:.2f}"],
            ["最差节点日均收益", f"{transformer_settlement['worst_node_mean_daily_revenue']:.2f}",
             f"{pooled_settlement['worst_node_mean_daily_revenue']:.2f}"],
            ["正收益日比例", f"{100 * transformer_settlement['macro_positive_day_rate']:.1f}%",
             f"{100 * pooled_settlement['macro_positive_day_rate']:.1f}%"],
            ["受限先知收益占比", f"{100 * profit['transformer']['fraction_of_oracle']:.1f}%",
             f"{100 * profit['comparator']['fraction_of_oracle']:.1f}%"],
        ], [68 * mm, 53 * mm, 53 * mm]))
        stability = locked_2025["ercot_transformer_seed_stability"]
        story.append(styled_table([
            ["Transformer盈利稳定性", "seed 0", "seed 1", "seed 2", "均值±样本标准差", "固定集成"],
            ["ERCOT 2025", f"{stability['seed_values']['seed0']:.2f}",
             f"{stability['seed_values']['seed1']:.2f}", f"{stability['seed_values']['seed2']:.2f}",
             f"{stability['mean']:.2f} ± {stability['sample_std']:.2f}",
             f"{stability['fixed_ensemble']:.2f}"],
        ], [38 * mm, 22 * mm, 22 * mm, 22 * mm, 43 * mm, 27 * mm]))
        story.append(H2("11.4 最终停止规则"))
        story.append(callout(
            "实验已经闭环",
            "2025只按预注册协议评价一次，selection_allowed=false。该结果用于确定论文能够支持的主张强度，"
            "不再用于选择结构、损失、seed、尾部权重或新增模型。",
            "good",
        ))

    chapter(story, "12", "复现性说明与审计路径",
            "本报告的关键数字均可由下列配置、代码和机器可读结果复核。")
    story.append(H2("12.1 当前最终配置与代码"))
    story.append(styled_table([
        ["类别", "路径"],
        ["LOMO DA/RT配置", "configs/decision_aware/universal_lomo_holdout_*_mae_v1.yaml"],
        ["LOMO RT-at-DA配置", "configs/decision_aware/universal_lomo_holdout_*_rt_at_da_v1.yaml"],
        ["DA / RT-at-DA模型", "src/decision_aware/model_da.py；model_rt_da.py"],
        ["滚动RT模型", "src/decision_aware/model_rt.py"],
        ["无ID稳健归一化", "src/decision_aware/portable_normalization.py"],
        ["训练入口", "scripts/decision_aware/train_global_forecaster.py"],
        ["连续LP/MPC", "src/decision_aware/continuous_mpc.py"],
        ["完整结算入口", "scripts/decision_aware/evaluate_global_continuous_mpc.py"],
    ], [48 * mm, 126 * mm]))
    story.append(H2("12.2 机器可读结果"))
    story.append(styled_table([
        ["内容", "路径"],
        ["四市场DA LOMO", "data/results/universal_lomo_da_v1/*.json"],
        ["四市场RT-at-DA LOMO", "data/results/universal_lomo_rt_at_da_v1/*.json"],
        ["四市场滚动RT LOMO", "data/results/universal_lomo_rt_v1/*.json"],
        ["ERCOT 2023盈利", "data/results/universal_lomo_ercot_profit_continuous_mpc_v1.json"],
        ["ERCOT 2024冻结盈利", "data/results/universal_lomo_ercot_outer_frozen_continuous_mpc_v1.json"],
        ["2024冻结摘要", "data/results/universal_lomo_ercot_outer_frozen_summary_v1.json"],
        ["公平pooled XGBoost", "data/results/fair_xgboost_comparison_v1/summary.json"],
        ["TimesFM-3正式对照", "data/results/universal_lomo_timesfm3_v1/*.json"],
        ["TimesFM-2.5 LoRA对照", "data/results/universal_lomo_timesfm25_lora_v1/*.json"],
        ["统一尾部与盈利诊断", "data/results/tail_profit_diagnostics_v1/tail_profit_diagnostics.json"],
        ["2025盈利分析预注册", "data/results/allmarket_final_2025_profit_analysis_preregistration_v1/protocol.json"],
        ["最终2025统一汇总", "data/results/allmarket_locked_2025_summary_v2.json（完成后自动纳入本报告）"],
        ["最终盈利主张审计", "data/results/allmarket_locked_2025_profit_claim_audit_v1.json（完成后生成）"],
        ["历史经济状态头", "data/results/global_economic_regime_multiseed_continuous_mpc_v1.json"],
        ["最终口径文档", "docs/progress/46_多节点结果口径修订与无ID通用模型路线.md"],
    ], [48 * mm, 126 * mm]))
    story.append(H2("12.3 当前架构图"))
    story.append(P(
        "docs/下八张draw.io分别覆盖三模型架构/训练和联合系统架构/训练：da_model_architecture、da_training_loop、"
        "rt_at_da_model_architecture、rt_at_da_training_loop、rt_model_architecture、rt_training_loop、"
        "joint_model_architecture、joint_training_loop。旧model_architecture与training_loop已标记为历史归档。"
    ))

    chapter(story, "附录A", "64节点清单与ID的实际用途")
    story.append(P(
        "全量稳定映射为market_id：CAISO=0、ERCOT=1、NYISO=2、PJM=3；node_id按该市场顺序、市场内节点名字典序编码为0-63。"
        "在当前无ID模型中，它们只用于采样、汇总、SOC隔离和审计，不输入网络。"
    ))
    node_rows = [
        ["市场", "节点清单"],
        ["CAISO\nmarket_id=0", "0=TH_NP15_GEN-APND；1=TH_SP15_GEN-APND；2=TH_ZP26_GEN-APND"],
        ["ERCOT\nmarket_id=1", "3=HB_BUSAVG；4=HB_HOUSTON；5=HB_HUBAVG；6=HB_NORTH；7=HB_PAN；8=HB_SOUTH；9=HB_WEST；10=LZ_AEN；11=LZ_CPS；12=LZ_HOUSTON；13=LZ_LCRA；14=LZ_NORTH；15=LZ_RAYBN；16=LZ_SOUTH；17=LZ_WEST"],
        ["NYISO\nmarket_id=2", "18=CAPITL；19=CENTRL；20=DUNWOD；21=GENESE；22=HUD VL；23=LONGIL；24=MHK VL；25=MILLWD；26=N.Y.C.；27=NORTH；28=WEST"],
        ["PJM\nmarket_id=3", "29=HUB:AEP GEN HUB；30=HUB:AEP-DAYTON HUB；31=HUB:ATSI GEN HUB；32=HUB:CHICAGO GEN HUB；33=HUB:CHICAGO HUB；34=HUB:DOMINION HUB；35=HUB:EASTERN HUB；36=HUB:N ILLINOIS HUB；37=HUB:NEW JERSEY HUB；38=HUB:OHIO HUB；39=HUB:WEST INT HUB；40=HUB:WESTERN HUB；41=ZONE:AECO；42=ZONE:AEP；43=ZONE:APS；44=ZONE:ATSI；45=ZONE:BGE；46=ZONE:COMED；47=ZONE:DAY；48=ZONE:DEOK；49=ZONE:DOM；50=ZONE:DPL；51=ZONE:DUQ；52=ZONE:EKPC；53=ZONE:JCPL；54=ZONE:METED；55=ZONE:MID-ATL/APS；56=ZONE:OVEC；57=ZONE:PECO；58=ZONE:PENELEC；59=ZONE:PEPCO；60=ZONE:PJM-RTO；61=ZONE:PPL；62=ZONE:PSEG；63=ZONE:RECO"],
    ]
    story.append(styled_table(node_rows, [30 * mm, 144 * mm]))

    chapter(story, "附录B", "结论边界速查")
    story.append(styled_table([
        ["表述", "是否允许", "严谨写法"],
        ["模型实现了四市场零样本预测", "允许", "在四折LOMO与当前字段合同下，12/12优于lag-168"],
        ["模型稳定击败XGBoost", "不允许", "2023点估计领先且区间跨0；2024冻结检查显著落后"],
        ["经济状态头是最终无ID模型组成", "不允许", "它是历史带ID ERCOT上界实验"],
        ["三模型联合训练", "当前不允许", "当前三模型独立训练，只在LP/MPC决策层联合"],
        ["CAISO/NYISO/PJM盈利领先", "不允许", "三市场目前只有预测层零样本证据"],
        ["2025是最终测试", "允许" if locked_2025 is not None else "尚不允许",
         "已按冻结协议一次性评价且不得回选" if locked_2025 is not None else "2025仍锁定、未评价"],
        ["F1/F0已在四市场完整消融", "不允许", "完整消融来自早期单节点Huber阶段"],
        ["Transformer全面优于TimesFM", "不允许", "多数DA/RT-at-DA占优；TimesFM-3在四个滚动RT任务上更准"],
    ], [62 * mm, 28 * mm, 84 * mm]))

    chapter(story, "参考文献", "方法来源与实验规范")
    story.append(P(
        "下列文献均在正文以编号标注其具体作用。仅在reference/README中列为‘待读’、尚未完成核对的候选文献，"
        "不作为本报告的方法或实验结论依据。"
    ))
    refs = [
        "[1] J. Lago et al. Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. Applied Energy, 2021. DOI: 10.1016/j.apenergy.2021.116983.",
        "[2] R. Weron. Electricity price forecasting: A review of the state-of-the-art with a look into the future. International Journal of Forecasting, 2014. DOI: 10.1016/j.ijforecast.2014.08.008.",
        "[3] A. Vaswani et al. Attention Is All You Need. NeurIPS, 2017.",
        "[4] J. Su et al. RoFormer: Enhanced Transformer with Rotary Position Embedding. Neurocomputing, 2024; arXiv:2104.09864.",
        "[5] T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. KDD, 2016. DOI: 10.1145/2939672.2939785.",
        "[6] D. Krishnamurthy et al. Energy Storage Arbitrage Under Day-Ahead and Real-Time Price Uncertainty. IEEE Transactions on Power Systems, 2018. DOI: 10.1109/TPWRS.2017.2685347.",
        "[7] B. Xu et al. Factoring the Cycle Aging Cost of Batteries Participating in Electricity Markets. IEEE Transactions on Power Systems, 2018. DOI: 10.1109/TPWRS.2017.2733339.",
        "[8] Y. Wang et al. Look-Ahead Bidding Strategy for Energy Storage. IEEE Transactions on Sustainable Energy, 2017. DOI: 10.1109/TSTE.2017.2656800.",
        "[9] Y. Nesterov and V. Spokoiny. Random Gradient-Free Minimization of Convex Functions. Foundations of Computational Mathematics, 2017. DOI: 10.1007/s10208-015-9296-2.",
        "[10] A. Das, W. Kong, R. Sen, and Y. Zhou. A Decoder-Only Foundation Model for Time-Series Forecasting. ICML, 2024.",
    ]
    for ref in refs:
        story.append(P(ref, "small"))
    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDocTemplate(str(OUTPUT))
    doc.multiBuild(build_story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
