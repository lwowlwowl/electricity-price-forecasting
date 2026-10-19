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
OUTPUT = ROOT / "output" / "pdf" / "多节点无ID通用电价预测与储能双结算技术报告_2026-10-19.pdf"

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
    """把锁定测试汇总转成论文表；LEAR只进入其覆盖的两个日级任务。"""
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
    lear_mae = {}
    for task in ("da", "rt_at_da"):
        values = [
            float(summary["markets"][market]["systems"]["lear"][task]["all"]["mae"])
            for market in ("CAISO", "ERCOT", "NYISO", "PJM")
        ]
        lear_mae[task] = sum(values) / len(values)
    records.insert(1, {
        "label": "LEAR（日级任务）",
        "da": lear_mae["da"],
        "rt_at_da": lear_mae["rt_at_da"],
        "rt": None,
        "revenue": None,
        "oracle_fraction": None,
    })
    best = {
        task: min(row[task] for row in records if row[task] is not None)
        for task in tasks
    }
    best_revenue = max(
        row["revenue"] for row in records if row["revenue"] is not None
    )
    rows: list[list[str]] = [[
        "系统", "四市场DA MAE", "四市场RT-at-DA MAE", "四市场滚动RT MAE",
        "ERCOT日均净收益", "相对受限先知参照",
    ]]
    for row in records:
        values = []
        for task in tasks:
            if row[task] is None:
                values.append("—")
                continue
            text = f"{row[task]:.2f}"
            values.append(B(text) if math.isclose(row[task], best[task]) else text)
        revenue = "—" if row["revenue"] is None else f"{row['revenue']:.2f}"
        if row["revenue"] is not None and math.isclose(row["revenue"], best_revenue):
            revenue = B(revenue)
        fraction = row["oracle_fraction"]
        rows.append([
            row["label"], *values, revenue,
            "—" if fraction is None else f"{100.0 * float(fraction):.1f}%",
        ])
    oracle = derived["da_oracle__rt_at_da_oracle__rt_oracle"]
    rows.append(["受限先知参照（非上界）", "—", "—", "—", f"{oracle['macro_mean_daily_revenue']:.2f}", "100.0%"])
    return rows


def locked_2025_market_task_rows(summary: dict, task: str) -> list[list[str]]:
    """生成单个任务的四市场完整基线表，并逐列加粗最低MAE。"""
    specs = [
        ("transformer_ensemble", "Transformer固定集成"),
        ("timesfm3", "TimesFM-3 frozen"),
        ("timesfm25", "TimesFM-2.5 LoRA"),
        ("pooled_xgboost", "pooled XGBoost"),
        ("per_node_xgboost", "逐节点XGBoost"),
        ("dlinear", "DLinear"),
        ("seasonal", "lag-168"),
    ]
    if task in ("da", "rt_at_da"):
        specs.insert(1, ("lear", "LEAR"))

    markets = ("CAISO", "ERCOT", "NYISO", "PJM")
    records: list[dict] = []
    for system, label in specs:
        values = {
            market: float(
                summary["markets"][market]["systems"][system][task]["all"]["mae"]
            )
            for market in markets
        }
        records.append({
            "label": label,
            "values": values,
            "macro": sum(values.values()) / len(values),
        })

    best_by_market = {
        market: min(record["values"][market] for record in records)
        for market in markets
    }
    best_macro = min(record["macro"] for record in records)
    rows: list[list[str]] = [["系统", *markets, "四市场宏平均"]]
    for record in records:
        row = [record["label"]]
        for market in markets:
            value = record["values"][market]
            text = f"{value:.2f}"
            row.append(B(text) if math.isclose(value, best_by_market[market]) else text)
        macro = f"{record['macro']:.2f}"
        row.append(B(macro) if math.isclose(record["macro"], best_macro) else macro)
        rows.append(row)
    return rows


def locked_2025_profit_component_rows(summary: dict) -> list[list[str]]:
    """生成ERCOT统一结算下所有完整三任务系统的收益分解。"""
    specs = [
        ("Transformer固定集成", "da_ensemble__rt_at_da_ensemble__rt_ensemble"),
        ("TimesFM-3 frozen", "da_timesfm3__rt_at_da_timesfm3__rt_timesfm3"),
        ("TimesFM-2.5 LoRA", "da_timesfm25__rt_at_da_timesfm25__rt_timesfm25"),
        ("pooled XGBoost", "da_pooled_xgboost__rt_at_da_pooled_xgboost__rt_pooled_xgboost"),
        ("逐节点XGBoost", "da_per_node_xgboost__rt_at_da_per_node_xgboost__rt_per_node_xgboost"),
        ("DLinear", "da_dlinear__rt_at_da_dlinear__rt_dlinear"),
        ("lag-168", "da_seasonal__rt_at_da_seasonal__rt_seasonal"),
    ]
    systems = summary["ercot_settlement"]["systems"]
    records = []
    for label, key in specs:
        system = systems[key]
        components = system["macro_mean_daily_revenue_components"]
        records.append({
            "label": label,
            "da": float(components["da_leg"]),
            "rt": float(components["rt_deviation_leg"]),
            "degradation": float(components["degradation_cost"]),
            "penalty": float(components["deviation_penalty"]),
            "throughput": float(system["macro_mean_daily_actual_throughput_mwh"]),
            "net": float(system["macro_mean_daily_revenue"]),
        })
    best_net = max(record["net"] for record in records)
    rows: list[list[str]] = [[
        "系统", "DA计划收益", "RT修正收益", "退化成本", "偏差罚金",
        "吞吐量(MWh)", "净收益",
    ]]
    for record in records:
        net = f"{record['net']:.2f}"
        rows.append([
            record["label"],
            f"{record['da']:.2f}",
            f"{record['rt']:.2f}",
            f"{record['degradation']:.2f}",
            f"{record['penalty']:.2f}",
            f"{record['throughput']:.2f}",
            B(net) if math.isclose(record["net"], best_net) else net,
        ])
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
            title="多市场无ID电价预测及其储能双结算价值评估",
            author="Electricity Price Forecasting Project",
            subject="跨市场电价预测、储能决策与经济价值评估",
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
    canvas.drawString(18 * mm, A4[1] - 10 * mm, "多市场无ID电价预测及其储能双结算价值评估")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"{page}")
    canvas.drawString(18 * mm, 9 * mm, "研究论文 - 2026-10-19")
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
    d.add(String(250, 34, "LOMO用于2023迁移证据；2025已按独立all-market冻结协议一次性评价",
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
    if locked_2025 is None:
        story.append(P(
            "为检验未见市场上的迁移能力，采用四折留一市场（Leave-One-Market-Out, LOMO）协议：每一折仅用三个市场训练，"
            "将第4个市场全部留作零样本测试。最终模型不读取节点或市场身份。2023年四个留出市场、三个任务共12项预测的平均绝对误差"
            "（mean absolute error, MAE）均优于168小时前同小时复制基线。盈利只在结算规则已完整实现的ERCOT上比较："
            "同训练数据比较中，当前Transformer在12项MAE中的9项优于pooled LOMO XGBoost，也在9项上优于源市场监督的TimesFM-2.5 LoRA；"
            "frozen TimesFM-3仅在四个滚动RT任务上占优。ERCOT 2023中，当前seed 0收益377.84，固定三seed集成407.17，"
            "仍未稳定超过同数据pooled XGBoost的418.06；2024冻结检查又落后于目标市场监督XGBoost。"
            "结果表明跨市场预测迁移已成立，但稳定盈利优势尚未成立。"
        ))
    else:
        final_profit = locked_2025_profit_evidence(locked_2025)
        story.append(P(
            "为检验迁移与部署能力，研究先采用四折留一市场（Leave-One-Market-Out, LOMO）协议，再在不读取2025的前提下冻结结构、"
            "损失、随机种子、集成方式和比较协议，使用四市场2020—2024数据训练最终部署模型，并对2025只评价一次。"
            "在四市场×三任务共12项预测中，固定三seed Transformer集成的MAE逐项低于TimesFM-3、TimesFM-2.5 LoRA、"
            "pooled XGBoost、逐节点XGBoost、DLinear和lag-168这六个完整三任务基线。"
            "只覆盖DA与RT-at-DA的LEAR在8个市场—任务单元中胜5项，Transformer胜3项，两任务宏平均总体接近。"
            f"但ERCOT 2025完整结算中，Transformer集成日均净收益为{final_profit['transformer']['macro_mean_daily_revenue']:.2f} USD/节点·日，"
            f"低于同数据pooled XGBoost的{final_profit['comparator']['macro_mean_daily_revenue']:.2f}；"
            f"逐日配对差为{final_profit['mean_difference']:.2f}，7日块Bootstrap 95%区间为"
            f"[{final_profit['ci'][0]:.2f}, {final_profit['ci'][1]:.2f}]。"
            "因此本文能够支持跨市场预测与最终年度预测精度优势，但不能支持经济收益优势；更低MAE没有稳定转化为更高利润。"
        ))
    story.append(P(
        "训练阶段出现MAE和Pseudo-Huber，是因为利润并非逐小时可直接监督的标签，而要经过跨时段电量状态、效率、偏差罚金和优化器才形成；"
        "因此价格误差用于稳定训练，净收益始终是最终系统评价指标。结构代号F0/F1、两类损失及其选择依据见第4.1—4.3节，"
        "随机种子与训练范围见第3.5和第4.4节。"
    ))
    story.append(P("关键词：电价预测；跨市场迁移；储能套利；日前—实时双结算；模型预测控制；LOMO", "small"))
    story.append(Spacer(1, 3 * mm))
    core_rows = [
        ["评价层", "评价范围", "自有系统", "主要基线", "结论"],
        ["预测", "四市场LOMO × 三任务", B("12/12优于lag-168"), "lag-168", "零样本迁移成立"],
    ]
    if locked_2025 is None:
        core_rows.extend([
            ["预测", "同数据零样本", B("9/12优于pooled XGBoost"), "TimesFM-2.5仅胜3/12", "需结合价差与收益"],
            ["盈利", "ERCOT 2023", B("377.84"), "292.29", "点估计+29.3%，区间跨0"],
            ["盈利", "ERCOT 2024冻结", "73.38", B("90.69"), "自有系统-19.1%，区间低于0"],
        ])
    else:
        core_rows.extend([
            ["预测", "2025四市场 × 三任务", B("12/12低于完整基线"), "6个三任务基线", "LEAR日级8项中胜5项"],
            ["盈利", "ERCOT 2025锁定", "28.04", B("pooled XGB 44.43"), "差−16.39，95%区间低于0"],
            ["风险", "去最高10日", "18.98", B("pooled XGB 31.06"), "盈利差并非仅由极少数尖峰造成"],
        ])
    story.append(styled_table(core_rows, [24 * mm, 36 * mm, 33 * mm, 33 * mm, 48 * mm]))
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
        ["预测能否转化为经济价值", "连续DA LP + 滚动RT MPC + 双结算账本", "ERCOT 2023/2024及锁定2025净收益与风险"],
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
        ["最终年度", "四市场2020-2024共同训练", "2025一次性锁定测试", "否", "预测、尾部、ERCOT净收益"],
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
        "用于判断结论是否依赖一次偶然初始化。项目在不同阶段使用种子的范围不同；最终all-market部署版已按预注册完成三seed，"
        "但早期LOMO四折仍以seed 0为正式迁移证据。"
    ))
    story.append(styled_table([
        ["实验阶段", "模型/配置范围", "重复设置", "用途"],
        ["早期单节点结构消融", "DA、RT-at-DA、滚动RT的F0/F1/F1b/F2候选", "每个候选3 seeds", "选择任务对应的融合结构"],
        ["早期单节点联合微调", "三模型组成的完整双结算系统", "3 seeds", "检查收益导向联合训练的稳定性"],
        ["历史多节点经济状态头", "冻结DA与RT-at-DA骨干，仅训练方向分类头", "3 seeds", "检查方向分类是否稳定"],
        ["无ID LOMO迁移系统", "4个留出市场 × 3个任务 = 12个checkpoint", "每折seed 0", "形成正式零样本迁移证据"],
        ["最终all-market部署系统", "3个任务 × 3个seed = 9个checkpoint", "seed 0/1/2固定等权集成", "2025一次性预测与ERCOT盈利评价"],
    ], [39 * mm, 69 * mm, 30 * mm, 36 * mm]))
    story.append(P(
        "因此，本报告把LOMO seed-0迁移证据与all-market三seed部署证据分开报告；二者承担不同主张，不能互相替代。"
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
        ["保存策略", "验证最优top-3", "LOMO采用seed 0；最终all-market采用预注册三seed固定等权集成"],
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
        ["cutoff-safe LEAR", "稀疏线性电价模型", "提供经典日级统计学习参照", "用于DA/RT-at-DA；不强行扩展滚动RT"],
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

    chapter(
        story,
        "11",
        "结论与后续工作",
        (
            "2025一次性锁定测试已经完成：预测精度优势成立，但预注册盈利主终点未胜出。"
            if locked_2025 is not None else
            "多节点阶段可以作为基础设施与验证协议收尾，但科学结论必须保留盈利稳定性和公平对照上的缺口。"
        ),
    )
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
    if locked_2025 is not None:
        story.append(bullets([
            "四市场2020—2024最终训练、三seed固定集成和2025一次性未污染测试。",
            "2025四市场三任务、11套预测系统、ERCOT 11套完整结算系统及受限先知诊断参照。",
            "129个预测/盈利档案和最终归档逐文件哈希审计；2025结果不再用于模型选择。",
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
            "下表由锁定汇总JSON自动生成。MAE为四个市场节点宏平均后的市场平均。ERCOT是当前唯一具有冻结结算合同的市场；"
            "收益单位为USD/节点·日。每列在该指标上有结果的方法中，最佳值加粗。受限先知参照只是在相同分层LP/MPC中把价格预测替换为真实价格，"
            "不是对所有联合DA/RT策略做全时域一次优化，因此不是严格数学上界。"
        ))
        story.append(styled_table(
            locked_2025_comparison_rows(locked_2025),
            [39 * mm, 25 * mm, 29 * mm, 29 * mm, 28 * mm, 24 * mm],
        ))
        story.append(P(
            "LEAR是cutoff-safe日级稀疏线性基线，只覆盖DA与RT-at-DA：8个市场—任务单元中LEAR胜5项、"
            "Transformer胜3项，两任务宏平均分别为12.83和12.80 USD/MWh。由于LEAR没有逐小时rolling RT，"
            "本报告不把它与其他模型临时拼成未预注册的三任务盈利系统。‘12/12’只表示Transformer相对六个覆盖"
            "全部三任务的基线逐项更低，不能扩写为胜过包括LEAR在内的所有基线。"
        ))
        final_derived = locked_2025["ercot_settlement_derived"]
        dlinear_2025 = final_derived["da_dlinear__rt_at_da_dlinear__rt_dlinear"]
        per_node_2025 = final_derived[
            "da_per_node_xgboost__rt_at_da_per_node_xgboost__rt_per_node_xgboost"
        ]
        timesfm25_2025 = final_derived[
            "da_timesfm25__rt_at_da_timesfm25__rt_timesfm25"
        ]
        story.append(P(
            f"完整盈利排序与MAE排序不同：DLinear为{dlinear_2025['macro_mean_daily_revenue']:.2f}，"
            f"逐节点XGBoost为{per_node_2025['macro_mean_daily_revenue']:.2f}，"
            f"TimesFM-2.5 LoRA为{timesfm25_2025['macro_mean_daily_revenue']:.2f}，"
            "均高于Transformer固定集成。TimesFM-3虽是大型frozen基础模型，但完整系统收益为负。"
            "这些结果说明模型规模与平均预测误差均不能替代同合同利润评价。"
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
            ["相对受限先知参照（非上界）", f"{100 * profit['transformer']['fraction_of_oracle']:.1f}%",
             f"{100 * profit['comparator']['fraction_of_oracle']:.1f}%"],
        ], [68 * mm, 53 * mm, 53 * mm]))
        story.append(callout(
            "为什么有系统会超过受限先知参照？",
            "该参照使用完美价格输入，但仍沿用分层DA计划和逐小时RT修正；其DA目标、RT修正和偏差罚金没有被一个全时域联合优化器同时最小化。"
            "因此它用于诊断预测误差，并不构成净收益上界。pooled XGBoost超过100%不表示超过理论最优。",
            "info",
        ))

        ensemble_tail = locked_2025["markets"]["ERCOT"]["systems"]["transformer_ensemble"]
        pooled_tail = locked_2025["markets"]["ERCOT"]["systems"]["pooled_xgboost"]
        story.append(H2("11.4 为什么MAE更低但盈利更低"))
        story.append(styled_table([
            ["2025 ERCOT诊断", "Transformer固定集成", "pooled XGBoost", "较优方向"],
            ["DA MAE", f"{ensemble_tail['da']['all']['mae']:.2f}",
             f"{pooled_tail['da']['all']['mae']:.2f}", "Transformer"],
            ["RT-at-DA MAE", f"{ensemble_tail['rt_at_da']['all']['mae']:.2f}",
             f"{pooled_tail['rt_at_da']['all']['mae']:.2f}", "Transformer"],
            ["滚动RT MAE", f"{ensemble_tail['rt']['all']['mae']:.2f}",
             f"{pooled_tail['rt']['all']['mae']:.2f}", "Transformer"],
            ["DA-RT方向准确率", f"{100 * ensemble_tail['da_rt_at_da_spread']['all']['direction_accuracy']:.1f}%",
             f"{100 * pooled_tail['da_rt_at_da_spread']['all']['direction_accuracy']:.1f}%", "Transformer"],
            ["极端1%价差幅度保留", f"{100 * ensemble_tail['da_rt_at_da_spread']['abs_spread_top_1pct']['magnitude_ratio']:.2f}%",
             f"{100 * pooled_tail['da_rt_at_da_spread']['abs_spread_top_1pct']['magnitude_ratio']:.2f}%", "均明显不足"],
            ["DA计划收益分量", f"{transformer_settlement['macro_mean_daily_revenue_components']['da_leg']:.2f}",
             f"{pooled_settlement['macro_mean_daily_revenue_components']['da_leg']:.2f}", "pooled XGBoost"],
            ["最终日均净收益", f"{profit['transformer']['macro_mean_daily_revenue']:.2f}",
             f"{profit['comparator']['macro_mean_daily_revenue']:.2f}", "pooled XGBoost"],
        ], [56 * mm, 42 * mm, 42 * mm, 34 * mm]))
        story.append(P(
            "Transformer降低了全样本平均价格误差，也略提高了总体价差方向准确率；但两套系统在极端1%价差上都只保留约1.3%的真实幅度。"
            "盈利差主要出现在DA计划收益分量：Transformer为51.64，pooled XGBoost为72.35 USD/节点·日。"
            "这说明均匀逐小时MAE没有充分衡量少数高价值时段的排序、幅度和动作后果。该解释只用于理解失败机制，不用于2025后回选模型。"
        ))
        stability = locked_2025["ercot_transformer_seed_stability"]
        story.append(styled_table([
            ["Transformer盈利稳定性", "seed 0", "seed 1", "seed 2", "均值±样本标准差", "固定集成"],
            ["ERCOT 2025", f"{stability['seed_values']['seed0']:.2f}",
             f"{stability['seed_values']['seed1']:.2f}", f"{stability['seed_values']['seed2']:.2f}",
             f"{stability['mean']:.2f} ± {stability['sample_std']:.2f}",
             f"{stability['fixed_ensemble']:.2f}"],
        ], [38 * mm, 22 * mm, 22 * mm, 22 * mm, 43 * mm, 27 * mm]))
        story.append(H2("11.5 最终停止规则"))
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
        ["最终2025统一汇总", "data/results/allmarket_locked_2025_summary_v2.json"],
        ["最终盈利主张审计", "data/results/allmarket_locked_2025_profit_claim_audit_v1.json"],
        ["历史经济状态头", "data/results/global_economic_regime_multiseed_continuous_mpc_v1.json"],
        ["最终口径文档", "docs/progress/54_2025一次性锁定测试最终结论.md"],
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
        ["模型稳定击败XGBoost", "不允许", (
            "2025固定集成比pooled XGBoost少16.39 USD/节点·日，配对区间整体低于0"
            if locked_2025 is not None else
            "2023点估计领先且区间跨0；2024冻结检查显著落后"
        )],
        ["经济状态头是最终无ID模型组成", "不允许", "它是历史带ID ERCOT上界实验"],
        ["三模型联合训练", "当前不允许", "当前三模型独立训练，只在LP/MPC决策层联合"],
        ["CAISO/NYISO/PJM盈利领先", "不允许", "三市场目前只有预测层零样本证据"],
        ["2025是最终测试", "允许" if locked_2025 is not None else "尚不允许",
         "已按冻结协议一次性评价且不得回选" if locked_2025 is not None else "2025仍锁定、未评价"],
        ["F1/F0已在四市场完整消融", "不允许", "完整消融来自早期单节点Huber阶段"],
        ["Transformer全面优于TimesFM", "不允许", (
            "2025的12项MAE逐项更低，但TimesFM-2.5盈利点估计更高；预测与经济结论须分开"
            if locked_2025 is not None else
            "多数DA/RT-at-DA占优；TimesFM-3在四个滚动RT任务上更准"
        )],
        ["受限先知参照是理论盈利上界", "不允许", "它复用分层LP/MPC，不是全时域联合最优；仅作诊断参照"],
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


def build_academic_story() -> list:
    """按学术论文逻辑组织现有证据；旧的逐阶段技术报告保留在build_story中供追溯。"""
    locked = load_json("data/results/allmarket_locked_2025_summary_v2.json")
    profit = locked_2025_profit_evidence(locked)

    lomo: dict[tuple[str, str], float] = {}
    for task_dir, task in (
        ("universal_lomo_da_v1", "DA"),
        ("universal_lomo_rt_at_da_v1", "RT-at-DA"),
        ("universal_lomo_rt_v1", "滚动RT"),
    ):
        for path in sorted((ROOT / "data" / "results" / task_dir).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert payload["uses_node_id_for_prediction"] is False
            assert payload["uses_market_id_for_prediction"] is False
            assert payload["normalization_mode"] == "context_robust"
            lomo[(payload["holdout_market"], task)] = float(payload["node_macro_mae"])

    seasonal = {
        ("CAISO", "DA"): 20.20, ("CAISO", "RT-at-DA"): 23.28, ("CAISO", "滚动RT"): 23.30,
        ("ERCOT", "DA"): 47.99, ("ERCOT", "RT-at-DA"): 50.60, ("ERCOT", "滚动RT"): 50.72,
        ("NYISO", "DA"): 12.28, ("NYISO", "RT-at-DA"): 16.17, ("NYISO", "滚动RT"): 16.17,
        ("PJM", "DA"): 10.10, ("PJM", "RT-at-DA"): 13.00, ("PJM", "滚动RT"): 12.99,
    }
    profit23 = load_json("data/results/universal_lomo_ercot_profit_continuous_mpc_v1.json")
    profit24 = load_json("data/results/universal_lomo_ercot_outer_frozen_continuous_mpc_v1.json")
    key_u = "da_universal__rt_at_da_universal__rt_universal"
    key_x = "da_xgboost_robust__rt_at_da_xgboost_robust__rt_xgboost_robust"
    u23, x23 = profit23["systems"][key_u], profit23["systems"][key_x]
    u24, x24 = profit24["systems"][key_u], profit24["systems"][key_x]
    final_derived = locked["ercot_settlement_derived"]
    transformer_label = "da_ensemble__rt_at_da_ensemble__rt_ensemble"
    pooled_label = "da_pooled_xgboost__rt_at_da_pooled_xgboost__rt_pooled_xgboost"
    transformer_settlement = locked["ercot_settlement"]["systems"][transformer_label]
    pooled_settlement = locked["ercot_settlement"]["systems"][pooled_label]

    story: list = []

    # Title page: intentionally restrained, matching an academic manuscript rather than a presentation.
    story.append(Spacer(1, 45 * mm))
    story.append(P("多市场无 ID 电价预测<br/>及其储能双结算价值评估", "title"))
    story.append(Spacer(1, 7 * mm))
    story.append(HRFlowable(width="48%", thickness=0.8, color=NAVY,
                            spaceBefore=0, spaceAfter=7 * mm, hAlign="CENTER"))
    story.append(P("跨市场迁移、预测-优化解耦与一次性时间外检验", "subtitle"))
    story.append(Spacer(1, 18 * mm))
    story.append(P("研究论文", "subtitle"))
    story.append(P("2026年10月19日", "subtitle"))
    story.append(PageBreak())

    story.append(H1("摘要"))
    story.append(P(
        "本文研究多市场电价预测能否在严格信息边界下迁移，并进一步转化为电池储能的日前-实时双结算收益。"
        "针对日前价格、日前时点实时价格和交付日滚动实时价格，构建三个互不共享参数的无身份Transformer。"
        "模型使用168小时多流历史与起报时已知日历，不读取node_id或market_id；价格预测随后进入日前线性规划和"
        "逐小时模型预测控制，真实价格仅用于事后结算。研究先以四折留一市场协议检验零样本迁移，再冻结结构、"
        "损失、随机种子、集成和比较协议，使用四市场2020-2024数据训练，并对2025只评价一次。"
    ))
    story.append(P(
        "结果显示，相对六个覆盖全部三任务的基线，固定三seed Transformer集成在2025四市场×三任务的12个预测"
        "单元中均取得更低MAE；但只覆盖DA与RT-at-DA的LEAR在8个日级市场-任务单元中胜5项，Transformer胜3项，"
        "两任务宏平均总体接近。预注册经济主终点未通过：ERCOT 2025中Transformer日均净收益为28.04 USD/节点·日，"
        "低于同数据pooled XGBoost的44.43；差值为-16.39，7日块Bootstrap 95%区间为[-23.19, -10.05]。"
        "进一步分析表明，平均价格误差的下降没有修复极端价差幅度收缩，DA计划收益不足是经济落后的主要来源。"
        "因此，本文支持无身份统一预测和跨市场迁移的有效性，但不支持其具有更高储能利润。"
    ))
    story.append(P(
        "关键词：电价预测；跨市场迁移；Transformer；时序基础模型；储能套利；模型预测控制；双结算",
        "small",
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(styled_table([
        ["研究层面", "主要问题", "核心证据", "结论"],
        ["迁移", "未见市场是否可预测", "四折LOMO，12项均优于lag-168", B("成立")],
        ["部署预测", "冻结后是否保持精度", "2025相对六个完整基线12/12更低MAE", B("成立")],
        ["日级强基线", "是否全面优于LEAR", "LEAR胜5/8，Transformer胜3/8", "总体接近"],
        ["经济价值", "是否优于公平XGBoost", "28.04 vs 44.43；区间低于0", B("不成立")],
    ], [30 * mm, 52 * mm, 62 * mm, 30 * mm]))
    story.append(P("表1  研究问题、证据与最终结论。加粗表示明确成立或不成立的预注册判断。", "caption"))

    story.append(PageBreak())
    story.append(H1("目录"))
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    story.append(toc)

    chapter(story, "1", "引言",
            "研究重点不是单独降低预测误差，而是检验跨市场预测能否在可执行约束下形成稳定经济价值。")
    story.append(H2("1.1 研究背景"))
    story.append(P(
        "日前与实时电价具有尖峰、负价、制度差异和强时变性，跨市场建模因而同时面临分布漂移与信息可用性约束[1,2]。"
        "对储能而言，预测误差并不等价于经济损失：同样的MAE可能发生在低价值普通小时，也可能改变尖峰时段的"
        "充放电方向。若不把预测放入统一的SOC、效率、退化与偏差罚金合同，无法判断模型是否真正改善业务目标。"
    ))
    story.append(H2("1.2 研究问题"))
    story.append(styled_table([
        ["编号", "研究问题", "验证设计"],
        ["RQ1", "无节点/市场身份输入的模型能否迁移到完全未见市场？", "四折Leave-One-Market-Out零样本评价"],
        ["RQ2", "共同训练后能否在冻结年份维持统一预测能力？", "四市场2020-2024训练，2025一次性评价"],
        ["RQ3", "更低预测误差能否转化为更高储能净收益？", "相同LP/MPC、双结算和配对收益检验"],
        ["RQ4", "若利润未改善，失败主要发生在哪个预测或决策环节？", "尾部价差、收益分量、跨年与多seed诊断"],
    ], [20 * mm, 82 * mm, 72 * mm]))
    story.append(P("表2  本文研究问题与对应验证设计。", "caption"))
    story.append(H2("1.3 主要贡献"))
    story.append(bullets([
        "建立严格区分DA、RT-at-DA与rolling RT信息时点的三任务预测协议，避免未来真实负荷、风光和价格泄漏。",
        "提出不读取node_id/market_id、只依赖样本历史窗口稳健归一化的多流Transformer，用LOMO检验未见市场迁移。",
        "同时纳入TimesFM、XGBoost、DLinear、LEAR和lag-168，明确区分同训练数据公平基线、目标市场监督强基线和部分任务基线。",
        "把所有完整三任务预测送入同一DA LP、rolling RT MPC和双结算账本，以净收益而非MAE作为经济主终点。",
        "执行预注册的一次性2025时间外检验，并保留经济目标未达成的负结果及其尾部机制解释。",
    ]))
    story.append(H2("1.4 论文结构"))
    story.append(P(
        "第2节综述相关研究；第3节给出问题定义；第4节说明数据与实验协议；第5节介绍方法；第6节定义比较系统；"
        "第7节报告主结果；第8节给出消融与稳健性；第9节讨论局限和含义；第10节总结。工程路径、历史路线和节点"
        "清单移至附录，避免打断主论证。"
    ))

    chapter(story, "2", "相关工作与研究定位",
            "本文将电价预测、跨市场迁移与储能决策放在同一个可审计实验框架中。")
    story.append(H2("2.1 电价预测与跨市场泛化"))
    story.append(P(
        "电价预测研究已形成以统计模型、树模型和深度神经网络为代表的技术体系[1,2]。LEAR提供经典稀疏线性日级"
        "参照，XGBoost通过梯度提升建模非线性交互[5]，Transformer则利用注意力处理长历史依赖[3]。本文关注的"
        "区别不是单一市场内拟合，而是模型在不读取市场身份、且目标市场完全不参与训练时的迁移能力。"
    ))
    story.append(H2("2.2 时序基础模型"))
    story.append(P(
        "预训练时序基础模型试图通过大规模序列学习获得通用预测能力[10]。本文同时评价frozen TimesFM-3和源市场"
        "监督适配的TimesFM-2.5 LoRA，并保持168小时上下文、预测范围和起报边界一致。该比较回答完整系统在统一"
        "协议下的表现，不把差异简单归因于模型规模或某个单一模块。"
    ))
    story.append(H2("2.3 预测与储能决策"))
    story.append(P(
        "储能套利必须同时考虑日前计划、实时修正、SOC、效率和循环退化[6-8]。早期项目也测试过零阶利润联合"
        "训练[9]，但在噪声、非光滑动作和跨时段信用分配下未获得稳定提升。故最终系统采用可审计的模块化"
        "预测-优化框架：神经网络学习价格，LP/MPC保证动作可行，利润仅作为最终评价。"
    ))
    story.append(callout(
        "研究定位",
        "本文不主张提出新的通用Transformer算子；贡献在于严格的无ID迁移协议、三任务信息合同、完整基线体系、"
        "可执行双结算评价，以及对‘预测更准但利润更低’这一负结果的机制分析。",
        "info",
    ))

    chapter(story, "3", "问题定义",
            "三个预测任务对应三个不同的信息集合，并在决策层而不是参数层联合。")
    story.append(H2("3.1 三个预测任务"))
    story.append(styled_table([
        ["任务", "起报时点", "预测输出", "合法附加信息", "决策用途"],
        ["DA", "交付日前一日", "目标日24小时DA价", "目标日8维日历", "生成日前计划"],
        ["RT-at-DA", "与DA完全相同", "目标日24小时RT预期", "目标日8维日历", "估计DA-RT价差"],
        ["rolling RT", "交付日每小时", "未来4小时RT价", "已公布DA价与日历", "逐小时修正动作"],
    ], [25 * mm, 34 * mm, 39 * mm, 38 * mm, 38 * mm]))
    story.append(P("表3  三个预测任务的信息集合与决策用途。", "caption"))
    story.append(H2("3.2 预测-决策-结算链"))
    story.append(system_architecture())
    story.append(P("图1  三个独立价格预测器与统一DA LP、RT MPC和双结算层。", "caption"))
    story.append(P(
        "三个Transformer不共享参数、优化器或checkpoint。DA与RT-at-DA的24小时预测共同进入日前LP；rolling RT"
        "每小时进入4小时MPC且只执行第一个动作。最终利润不会反向更新Transformer，因此本文模型不是端到端利润模型。"
    ))
    story.append(H2("3.3 电池与双结算目标"))
    story.append(equation(
        "R = Σ<sub>t</sub>[p<sup>DA</sup><sub>t</sub>u<sup>DA</sup><sub>t</sub> + "
        "p<sup>RT</sup><sub>t</sub>(u<sup>RT</sup><sub>t</sub>-u<sup>DA</sup><sub>t</sub>) "
        "- κ·throughput<sub>t</sub>] - P<sub>dev</sub>"
    ))
    story.append(equation(
        "P<sub>dev</sub> = Σ<sub>t</sub>2|p<sup>RT</sup><sub>t</sub>·max(|u<sup>RT</sup><sub>t</sub>-u<sup>DA</sup><sub>t</sub>| "
        "- 0.03|u<sup>DA</sup><sub>t</sub>|, 0)"
    ))
    story.append(styled_table([
        ["参数", "冻结值", "参数", "冻结值"],
        ["功率/容量", "1 MW / 4 MWh", "SOC范围", "0.4-3.6 MWh"],
        ["初始SOC", "2 MWh", "充/放效率", "0.95 / 0.95"],
        ["每日放电上限", "4 MWh", "吞吐成本κ", "5.7 USD/MWh"],
        ["偏差容忍带", "3%×|DA动作|", "超限罚金", "2×|RT价|×超限量"],
    ], [37 * mm, 50 * mm, 37 * mm, 50 * mm]))
    story.append(P("表4  ERCOT储能与结算合同。", "caption"))
    story.append(H2("3.4 评价假设"))
    story.append(bullets([
        "预测假设：更低MAE表示更准确的价格估计，但不预设其必然提高利润。",
        "迁移假设：若LOMO结果稳定优于lag-168，则模型学习到超越身份查找表的跨市场规律。",
        "经济假设：只有同日配对收益差为正且7日块Bootstrap 95%区间下界大于0，才称稳定盈利胜出。",
    ]))

    chapter(story, "4", "数据与实验协议",
            "训练、验证、冻结测试和结算评价在不同层级上回答不同研究问题。")
    story.append(H2("4.1 数据范围"))
    story.append(styled_table([
        ["市场", "节点数", "主要价格", "统一频率", "最终用途"],
        ["CAISO", "3", "DA / RT", "小时", "训练、LOMO、2025预测"],
        ["ERCOT", "15", "DA / RT", "小时", "训练、LOMO、2025预测与盈利"],
        ["NYISO", "11", "DA / RT", "小时", "训练、LOMO、2025预测"],
        ["PJM", "35", "DA / RT", "小时", "训练、LOMO、2025预测"],
        ["合计", B("64"), "价格+负荷+风光+日历", "小时", "市场-节点平衡采样"],
    ], [28 * mm, 23 * mm, 49 * mm, 28 * mm, 46 * mm]))
    story.append(P("表5  四市场数据范围。", "caption"))
    story.append(H2("4.2 输入与防泄漏"))
    story.append(P(
        "每个样本使用168小时历史DA、RT、负荷、风电、光伏和日历。目标日历包含小时、星期、月份及周期编码，"
        "它在起报时已知，不含未来价格或负荷。最终网络不读取node_id/market_id；归一化仅使用当前历史窗口的"
        "中位数和MAD。市场与节点ID只用于平衡采样、结果汇总、SOC隔离和审计。"
    ))
    story.append(H2("4.3 两级验证协议"))
    story.append(protocol_diagram())
    story.append(P("图2  LOMO迁移验证与all-market 2025锁定测试的关系。", "caption"))
    story.append(styled_table([
        ["协议", "训练数据", "评价数据", "回答问题", "是否允许选模"],
        ["四折LOMO", "3个源市场2020-2022", "第4市场2023", "未见市场迁移", "仅限开发期"],
        ["ERCOT冻结检查", "保持LOMO checkpoint", "ERCOT 2024", "跨年稳定性", "否"],
        ["all-market最终测试", "四市场2020-2024", "四市场2025", "最终部署能力", B("否，只评价一次")],
    ], [35 * mm, 45 * mm, 35 * mm, 36 * mm, 23 * mm]))
    story.append(P("表6  数据划分和选择边界。", "caption"))
    story.append(H2("4.4 指标与统计"))
    story.append(bullets([
        "预测：节点等权MAE、RMSE、偏差、负价识别、尖峰召回、价差方向与幅度。",
        "经济：节点等权日均净收益、DA/RT收益分量、退化成本、偏差罚金、最大回撤和最差节点。",
        "稳健性：三seed、不挑最高seed、去最高5/10日、同日配对7日块Bootstrap。",
        "主张边界：CAISO、NYISO、PJM未冻结完整结算合同，只报告预测和尾部指标。",
    ]))

    chapter(story, "5", "方法",
            "模型负责预测价格，优化器负责生成满足物理与市场约束的动作。")
    story.append(H2("5.1 无身份多流Transformer"))
    story.append(model_architecture("da"))
    story.append(P("图3  以DA为例的无ID多流Transformer；RT-at-DA和rolling RT复用骨干接口但参数独立。", "caption"))
    story.append(P(
        "价格、负荷、风光、系统量与日历先独立编码，连续时间流使用两层Transformer和RoPE[3,4]；融合后形成"
        "168小时Memory，再由24个或4个目标Query读取与目标小时相关的表示。模型宽度为128，注意力头数为4，"
        "前馈层宽度为512，dropout为0.1。"
    ))
    story.append(H2("5.2 任务专属融合与损失"))
    story.append(styled_table([
        ["任务", "融合结构", "损失", "选择理由"],
        ["DA", "F1来源注意力", "MAE", "跨来源关系对日级计划有用"],
        ["RT-at-DA", "F0残差MLP", "Pseudo-Huber", "结构简单且对尖峰梯度更稳健"],
        ["rolling RT", "F1来源注意力", "MAE", "滚动窗口需要动态来源交互"],
    ], [31 * mm, 42 * mm, 36 * mm, 65 * mm]))
    story.append(P("表7  最终三个预测器的冻结结构与训练损失。", "caption"))
    story.append(P(
        "F0先拼接同小时来源表示，再用残差MLP融合；F1用来源注意力学习来源关系。MAE是平均绝对误差；"
        "Pseudo-Huber在零点附近平滑、对大误差近似线性。二者是价格训练代理，不是利润函数。"
    ))
    story.append(H2("5.3 训练与集成"))
    story.append(styled_table([
        ["项目", "冻结设置", "项目", "冻结设置"],
        ["优化器", "AdamW", "初始学习率", "2×10^-4"],
        ["权重衰减", "0.01", "batch/epoch", "16 / 最多20"],
        ["早停", "patience=7", "调度器", "ReduceLROnPlateau"],
        ["选模", "节点MAE→市场宏平均", "随机种子", "0、1、2"],
        ["最终集成", "三seed逐点等权", "禁止项", "不按2025或利润挑seed"],
    ], [34 * mm, 53 * mm, 34 * mm, 53 * mm]))
    story.append(P("表8  最终训练和集成设置。", "caption"))
    story.append(H2("5.4 DA LP与rolling RT MPC"))
    story.append(P(
        "日前LP同时读取DA和RT-at-DA预测，在功率、SOC、效率与日循环约束下确定24小时计划；实时MPC每小时"
        "读取rolling RT预测、已锁定DA计划和当前SOC，求解4小时窗口并只执行首个动作。真实价格不参与动作生成。"
        "这种模块化设计便于审计和任务级诊断，也避免把不稳定的求解器梯度和非光滑门控强行传播至预测器。"
    ))

    chapter(story, "6", "比较方法",
            "基线按其训练数据、任务覆盖范围和是否能形成完整盈利系统分类。")
    story.append(H2("6.1 基线体系"))
    story.append(styled_table([
        ["方法", "类型", "训练边界", "任务覆盖", "在本文中的作用"],
        ["lag-168", "周季节复制", "无需训练", "三任务", "最简单周期底线"],
        ["DLinear", "简单深度模型", "四市场监督", "三任务", "检验复杂注意力必要性"],
        ["TimesFM-3", "frozen基础模型", "不使用项目标签", "三任务", "检验预训练迁移"],
        ["TimesFM-2.5", "LoRA基础模型", "源市场监督适配", "三任务", "检验监督适配"],
        ["pooled XGBoost", "共享树模型", "与Transformer同数据", "三任务", "主要公平对照"],
        ["逐节点XGBoost", "本地树模型组", "看过目标节点历史", "三任务", "实际监督强基线"],
        ["LEAR", "稀疏线性模型", "看过目标节点历史", "DA/RT-at-DA", "日级经典强基线"],
    ], [31 * mm, 31 * mm, 42 * mm, 30 * mm, 40 * mm]))
    story.append(P("表9  比较方法的角色与覆盖范围。", "caption"))
    story.append(H2("6.2 XGBoost系统的准确含义"))
    story.append(P(
        "逐节点XGBoost不是一个模型，而是由节点、任务和预测时距专属回归器构成的系统；每个回归器内部又包含180棵"
        "提升树。各节点只调用自己的模型组，再把全节点三任务预测送入同一LP/MPC。pooled XGBoost则在源市场合并"
        "训练，用于与无ID Transformer做同数据比较。两者不可混称。"
    ))
    story.append(H2("6.3 LEAR的比较边界"))
    story.append(callout(
        "部分任务基线",
        "LEAR已完成四市场DA与RT-at-DA共8项，但其标准定义不覆盖逐小时4小时rolling RT。因此LEAR进入日级"
        "预测表，不计入完整三任务的12项比较，也不与另一模型临时拼成未预注册盈利系统。",
        "risk",
    ))

    chapter(story, "7", "实验结果",
            "主结果依次回答零样本迁移、最终预测性能和预注册经济终点。")
    story.append(H2("7.1 四折LOMO零样本迁移"))
    lomo_rows = [["留出市场", "任务", "Transformer MAE", "lag-168 MAE", "相对改善"]]
    for market in ("CAISO", "ERCOT", "NYISO", "PJM"):
        for task in ("DA", "RT-at-DA", "滚动RT"):
            value, base = lomo[(market, task)], seasonal[(market, task)]
            gain = 100.0 * (base - value) / base
            lomo_rows.append([market, task, B(f"{value:.2f}"), f"{base:.2f}", f"{gain:+.1f}%"])
    story.append(styled_table(lomo_rows, [30 * mm, 32 * mm, 40 * mm, 36 * mm, 36 * mm],
                              alignments={2: "RIGHT", 3: "RIGHT", 4: "RIGHT"}))
    story.append(P(
        "表10  2023四折LOMO节点等权MAE（USD/MWh）。每一折完全留出一个市场，12项均优于lag-168。",
        "caption",
    ))
    story.append(P(
        "LOMO结果说明模型在不读取市场或节点身份时仍学到了可迁移的价格规律，但不能外推为任意市场普适，也不能"
        "由MAE直接推断利润。最终部署性能必须由独立all-market 2025协议回答。"
    ))
    story.append(H2("7.2 2025四市场预测与ERCOT盈利"))
    story.append(styled_table(
        locked_2025_comparison_rows(locked),
        [39 * mm, 25 * mm, 29 * mm, 29 * mm, 28 * mm, 24 * mm],
    ))
    story.append(P(
        "表11  2025四市场宏平均MAE与ERCOT日均净收益。每列在有结果的方法中加粗最优值；受限先知参照不是理论上界。",
        "caption",
    ))
    story.append(P(
        "相对六个覆盖全部三任务的基线，Transformer在12个市场-任务单元中均取得更低MAE。LEAR只覆盖两个日级"
        "任务：8项中LEAR胜5项、Transformer胜3项；二者两任务宏平均分别为12.83和12.80 USD/MWh，整体接近。"
        "完整盈利排序与MAE排序不同，DLinear和逐节点XGBoost的收益均高于Transformer。"
    ))
    story.append(H3("7.2.1 各市场预测误差完整比较"))
    story.append(P(
        "为避免四市场宏平均掩盖单一市场差异，下列三表给出所有已完成基线的逐市场节点等权MAE。"
        "每列最低误差加粗；LEAR只出现在其标准协议覆盖的DA与RT-at-DA任务。"
    ))
    for task, label, suffix in (
        ("da", "DA", "A"),
        ("rt_at_da", "RT-at-DA", "B"),
        ("rt", "rolling RT", "C"),
    ):
        story.append(styled_table(
            locked_2025_market_task_rows(locked, task),
            [43 * mm, 24 * mm, 24 * mm, 24 * mm, 24 * mm, 35 * mm],
            alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT"},
        ))
        story.append(P(
            f"表11{suffix}  2025四市场{label}节点等权MAE（USD/MWh）；加粗为各列最优值。",
            "caption",
        ))
    story.append(P(
        "完整逐市场结果表明，Transformer对六个覆盖三任务的基线在12个市场-任务单元中均保持最低MAE；"
        "但加入只覆盖日级任务的LEAR后，DA与RT-at-DA共8个单元中LEAR胜5项，因此不能表述为全面胜过所有基线。"
    ))
    story.append(H2("7.3 预注册经济主终点"))
    story.append(styled_table([
        ["主系统", "公平对照", "收益差", "7日块95%区间", "预注册判定"],
        [f"Transformer {profit['transformer']['macro_mean_daily_revenue']:.2f}",
         f"pooled XGBoost {profit['comparator']['macro_mean_daily_revenue']:.2f}",
         f"{profit['mean_difference']:+.2f}",
         f"[{profit['ci'][0]:.2f}, {profit['ci'][1]:.2f}]", B("未胜出")],
    ], [39 * mm, 39 * mm, 28 * mm, 40 * mm, 28 * mm]))
    story.append(P("表12  ERCOT 2025预注册主比较，收益单位为USD/节点·日。", "caption"))
    story.append(callout(
        "主结论",
        "固定三seed Transformer集成未在ERCOT 2025盈利上超过同数据pooled XGBoost。预测精度优势成立，"
        "经济收益优势不成立；两者必须在摘要、结果和结论中并列陈述。",
        "risk",
    ))
    story.append(H3("7.3.1 所有完整基线的ERCOT收益分解"))
    story.append(P(
        "下表中的Transformer与六个完整三任务基线全部进入同一个日前LP、滚动RT MPC和双结算账本。"
        "每个系统均计算5.7 USD/MWh吞吐退化成本、3%偏差容忍带及超限偏差罚金，不存在某一基线免除成本的情况。"
    ))
    story.append(styled_table(
        locked_2025_profit_component_rows(locked),
        [42 * mm, 24 * mm, 23 * mm, 21 * mm, 21 * mm, 23 * mm, 20 * mm],
        alignments={1: "RIGHT", 2: "RIGHT", 3: "RIGHT", 4: "RIGHT", 5: "RIGHT", 6: "RIGHT"},
    ))
    story.append(P(
        "表12A  2025 ERCOT所有完整三任务系统的日均收益分解，单位除吞吐量外均为USD/节点·日。"
        "净收益最高值加粗；成本与吞吐量是行为解释量，不能脱离收益单独定义胜者。LEAR缺少rolling RT，"
        "无法构成预注册完整结算系统，故不在此表。",
        "caption",
    ))
    dlinear_settlement = locked["ercot_settlement"]["systems"][
        "da_dlinear__rt_at_da_dlinear__rt_dlinear"
    ]
    dlinear_components = dlinear_settlement["macro_mean_daily_revenue_components"]
    story.append(P(
        f"DLinear的净收益为{dlinear_settlement['macro_mean_daily_revenue']:.2f}，并非因为免除了偏差罚金或退化成本。"
        f"它支付退化成本{dlinear_components['degradation_cost']:.2f}和偏差罚金{dlinear_components['deviation_penalty']:.2f}，"
        f"但DA计划收益达到{dlinear_components['da_leg']:.2f}，显著高于Transformer的"
        f"{transformer_settlement['macro_mean_daily_revenue_components']['da_leg']:.2f}。"
        "这进一步说明，价格MAE更低并不保证预测价差形成更有价值的日前仓位。"
    ))
    story.append(H2("7.4 为什么MAE更低但利润更低"))
    ensemble_tail = locked["markets"]["ERCOT"]["systems"]["transformer_ensemble"]
    pooled_tail = locked["markets"]["ERCOT"]["systems"]["pooled_xgboost"]
    story.append(styled_table([
        ["ERCOT 2025诊断", "Transformer", "pooled XGBoost", "较优/解释"],
        ["DA MAE", f"{ensemble_tail['da']['all']['mae']:.2f}", f"{pooled_tail['da']['all']['mae']:.2f}", "Transformer"],
        ["RT-at-DA MAE", f"{ensemble_tail['rt_at_da']['all']['mae']:.2f}", f"{pooled_tail['rt_at_da']['all']['mae']:.2f}", "Transformer"],
        ["rolling RT MAE", f"{ensemble_tail['rt']['all']['mae']:.2f}", f"{pooled_tail['rt']['all']['mae']:.2f}", "Transformer"],
        ["价差方向准确率", f"{100*ensemble_tail['da_rt_at_da_spread']['all']['direction_accuracy']:.1f}%",
         f"{100*pooled_tail['da_rt_at_da_spread']['all']['direction_accuracy']:.1f}%", "Transformer"],
        ["极端1%价差幅度保留", f"{100*ensemble_tail['da_rt_at_da_spread']['abs_spread_top_1pct']['magnitude_ratio']:.2f}%",
         f"{100*pooled_tail['da_rt_at_da_spread']['abs_spread_top_1pct']['magnitude_ratio']:.2f}%", "均严重收缩"],
        ["DA计划收益分量", f"{transformer_settlement['macro_mean_daily_revenue_components']['da_leg']:.2f}",
         f"{pooled_settlement['macro_mean_daily_revenue_components']['da_leg']:.2f}", "XGBoost"],
        ["最终净收益", f"{profit['transformer']['macro_mean_daily_revenue']:.2f}",
         f"{profit['comparator']['macro_mean_daily_revenue']:.2f}", "XGBoost"],
    ], [57 * mm, 38 * mm, 42 * mm, 37 * mm]))
    story.append(P("表13  平均误差、极端价差和收益分量的联合诊断。", "caption"))
    story.append(P(
        "Transformer改善了全样本平均误差和总体价差方向，但两套系统对最高1%价差都只恢复约1.3%的真实幅度。"
        "Transformer的DA计划收益分量低20.71 USD/节点·日，说明高价值小时的幅度与排序不足，比普通小时MAE更能"
        "解释最终利润落后。"
    ))

    chapter(story, "8", "消融与稳健性分析",
            "本节区分用于选择最终设计的开发期证据与只用于解释结果的冻结后诊断。")
    story.append(H2("8.1 结构消融"))
    story.append(styled_table([
        ["2023 ERCOT结构", "日均净收益", "与当前差异", "结论"],
        ["当前F1/F0/F1", "377.84", "基准", B("保留")],
        ["全F0", "291.24", "-86.60", "退化"],
        ["全F1", "344.28", "-33.56", "退化"],
        ["仅rolling RT改F0", B("378.81"), "+0.97", "区间跨0，不更换"],
    ], [48 * mm, 34 * mm, 34 * mm, 58 * mm]))
    story.append(P("表14  当前无ID协议下的受控结构替换；点估计不足以触发结构变更。", "caption"))
    story.append(H2("8.2 随机种子与固定集成"))
    stability = locked["ercot_transformer_seed_stability"]
    story.append(styled_table([
        ["ERCOT 2025", "seed 0", "seed 1", "seed 2", "均值±样本标准差", "固定集成"],
        ["日均净收益", f"{stability['seed_values']['seed0']:.2f}",
         f"{stability['seed_values']['seed1']:.2f}", f"{stability['seed_values']['seed2']:.2f}",
         f"{stability['mean']:.2f} ± {stability['sample_std']:.2f}", f"{stability['fixed_ensemble']:.2f}"],
    ], [34 * mm, 23 * mm, 23 * mm, 23 * mm, 45 * mm, 26 * mm]))
    story.append(P("表15  Transformer盈利的三seed稳定性；最终使用事前固定预测等权集成。", "caption"))
    story.append(H2("8.3 跨年收益与市场机会"))
    story.append(styled_table([
        ["年份/协议", "Transformer", "XGBoost", "差值", "统计解释"],
        ["2023 LOMO开发", f"{u23['macro_mean_daily_revenue']:.2f}", f"{x23['macro_mean_daily_revenue']:.2f}", "+85.55", "95%区间跨0"],
        ["2024冻结检查", f"{u24['macro_mean_daily_revenue']:.2f}", f"{x24['macro_mean_daily_revenue']:.2f}", "-17.32", "95%区间低于0"],
        ["2025 all-market最终", "28.04", "44.43（pooled）", "-16.39", "95%区间低于0"],
    ], [42 * mm, 34 * mm, 40 * mm, 27 * mm, 31 * mm]))
    story.append(P("表16  三个年度/协议下的ERCOT收益演进，单位USD/节点·日。", "caption"))
    story.append(P(
        "2024的DA日内价差和|DA-RT|相对2023分别缩小约58.5%和61.6%，高收益尖峰机会显著减少；但同年XGBoost"
        "仍领先，故不能把失败完全归因于市场变平。2025的独立结果进一步确认：预测平均误差优势并未形成经济优势。"
    ))
    story.append(H2("8.4 已排除的历史路线"))
    story.append(styled_table([
        ["历史路线", "目的", "结果", "与最终模型关系"],
        ["pre-encoder身份条件化", "利用已知节点/市场身份", "已知实体有效", "新市场无身份向量，移除"],
        ["经济状态方向头", "直接翻转DA-RT方向", "历史带ID点估计较高", "未迁移到无ID最终模型"],
        ["零阶利润联合训练", "让利润影响网络更新", "噪声大、验证不稳定", "最终不使用"],
        ["关键事件简单加权", "强调尖峰小时", "收益与回撤退化", "不保留"],
        ["显式价差回归", "直接预测DA-RT", "幅度仍收缩", "不保留"],
    ], [43 * mm, 48 * mm, 39 * mm, 44 * mm]))
    story.append(P("表17  主要历史负结果及其对最终设计的约束。", "caption"))

    chapter(story, "9", "讨论与局限",
            "结果的科学价值在于明确区分预测成功、经济失败和可推广边界。")
    story.append(H2("9.1 预测优势的含义"))
    story.append(P(
        "无身份模型在LOMO中超越周季节复制，并在all-market 2025中相对六个完整三任务基线取得更低MAE，"
        "说明共享骨干能够学习跨市场通用的短期价格规律。但LEAR在日级任务上胜5/8，表明复杂模型并未对所有"
        "预测场景形成绝对优势；日级稀疏线性结构仍是不可省略的强基线。"
    ))
    story.append(H2("9.2 预测精度与经济价值的错位"))
    story.append(P(
        "盈利由跨时段SOC、DA计划、RT修正、效率、退化和偏差罚金共同决定，不是逐小时误差的线性函数。均匀MAE"
        "偏重大量普通小时，而利润高度依赖少数高价差时段。本文的最终负结果说明，仅以平均误差选模可能得到"
        "价格更准但计划价值更低的系统。"
    ))
    story.append(H2("9.3 有效性威胁"))
    story.append(bullets([
        "外部有效性：四个美国市场不能代表所有电力市场，制度和发布时间差异仍需单独适配。",
        "经济有效性：只有ERCOT冻结了完整结算合同，其他三市场不能据此声称真实盈利。",
        "比较边界：LEAR缺少rolling RT，不能参加完整三任务盈利排名；逐节点XGBoost看过目标市场，不是纯架构对照。",
        "Oracle边界：受限先知参照复用分层LP/MPC，并非全时域联合最优，不构成理论盈利上界。",
        "尾部建模：当前-500 USD/MWh下界可能截断更低负价；极端价格与价差幅度仍是主要薄弱点。",
    ]))
    story.append(H2("9.4 后续研究"))
    story.append(P(
        "在不污染本次2025结论的前提下，后续工作可重新预注册尾部分位数预测、风险约束MPC、轻量在线校准或"
        "可微优化。其他三市场若完成结算合同冻结，可直接复用现有预测补算盈利，无需重训；这些属于扩展研究，"
        "不改变本文的停止规则。"
    ))

    chapter(story, "10", "结论",
            "本文完成了从跨市场预测到可执行储能结算的一次性时间外验证。")
    story.append(P(
        "本文构建了DA、RT-at-DA和rolling RT三个独立无ID Transformer，并以严格信息截止时点、LOMO迁移协议、"
        "多类强基线和统一LP/MPC结算评价其性能。模型在未见市场上表现出可迁移性，并在2025相对六个完整三任务"
        "基线取得统一MAE优势；然而LEAR在日级任务上仍具有竞争力，且ERCOT预注册盈利主终点明确未胜出。"
    ))
    story.append(P(
        "最重要的结论不是‘模型更准’或‘模型失败’的单一判断，而是：跨市场预测能力已经形成，但它尚未稳定转化为"
        "经济价值。极端价差幅度压缩和DA计划收益不足是下一阶段最值得研究的机制。2025结果仅用于确定主张强度，"
        "不再用于追加结构、损失、seed或阈值。"
    ))
    story.append(callout(
        "最终结论边界",
        "允许表述：预测迁移和相对完整三任务基线的精度优势成立；经济收益优势不成立。禁止表述：全面胜过LEAR、"
        "稳定击败XGBoost、四市场盈利领先或2025后继续选模。",
        "good",
    ))

    chapter(story, "附录A", "实现与复现信息")
    story.append(H2("A.1 关键代码与结果"))
    story.append(styled_table([
        ["内容", "路径"],
        ["模型实现", "src/decision_aware/model_da.py；model_rt_da.py；model_rt.py"],
        ["训练入口", "scripts/decision_aware/train_global_forecaster.py"],
        ["DA LP / RT MPC", "src/decision_aware/continuous_mpc.py"],
        ["2025统一汇总", "data/results/allmarket_locked_2025_summary_v2.json"],
        ["盈利审计", "data/results/allmarket_locked_2025_profit_claim_audit_v1.json"],
        ["最终结论", "docs/progress/54_2025一次性锁定测试最终结论.md"],
    ], [48 * mm, 126 * mm]))
    story.append(H2("A.2 完整性与停止规则"))
    story.append(bullets([
        "2025归档内327个文件逐项哈希一致；最终审计覆盖129个预测归档、11套ERCOT结算系统和12项市场任务。",
        "恢复只统一容差内重复真值，预测数组未改变。",
        "selection_allowed=false；除明确代码错误、文件损坏或新结算合同外，不重开当前实验。",
    ]))

    chapter(story, "附录B", "64节点与身份变量用途")
    story.append(P(
        "market_id稳定映射为CAISO=0、ERCOT=1、NYISO=2、PJM=3；node_id按市场顺序和市场内名称字典序"
        "编码为0-63。最终模型前向不读取这些ID，它们只用于采样、汇总、SOC隔离和审计。"
    ))
    story.append(styled_table([
        ["市场", "节点数", "node_id范围", "前向是否读取ID"],
        ["CAISO", "3", "0-2", B("否")],
        ["ERCOT", "15", "3-17", B("否")],
        ["NYISO", "11", "18-28", B("否")],
        ["PJM", "35", "29-63", B("否")],
    ], [40 * mm, 32 * mm, 48 * mm, 54 * mm]))

    chapter(story, "附录C", "结论边界速查")
    story.append(styled_table([
        ["表述", "是否允许", "严谨写法"],
        ["模型实现未见市场预测", "允许", "四折LOMO下12项均优于lag-168"],
        ["Transformer胜过所有基线", "不允许", "仅相对六个完整三任务基线12/12；LEAR胜5/8"],
        ["Transformer盈利优于XGBoost", "不允许", "2025相对pooled XGBoost差-16.39，区间低于0"],
        ["当前模型是端到端利润模型", "不允许", "三个价格模型独立训练，只在LP/MPC决策层联合"],
        ["四市场盈利均已验证", "不允许", "仅ERCOT冻结了完整结算合同"],
        ["受限先知是理论上界", "不允许", "它复用分层优化，只作诊断参照"],
    ], [62 * mm, 28 * mm, 84 * mm]))

    chapter(story, "参考文献", "")
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
    doc.multiBuild(build_academic_story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
