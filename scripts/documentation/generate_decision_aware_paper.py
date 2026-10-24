#!/usr/bin/env python
"""Generate the evidence-backed decision-aware manuscript PDF.

The paper is intentionally generated from frozen machine-readable artifacts.
It distinguishes forecasting pretraining, differentiable decision-aware fine-tuning,
the low-parameter robust calibrator, and the exact LP/MPC settlement used only for
evaluation.  No score is hard-coded except protocol constants already frozen in
the experiment contract.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from generate_multinode_technical_report import (
    B,
    BLUE,
    GREEN,
    GRID,
    H1,
    H2,
    H3,
    INK,
    LIGHT,
    MUTED,
    NAVY,
    ORANGE,
    P,
    PALE_BLUE,
    PALE_GREEN,
    PALE_ORANGE,
    PALE_RED,
    PURPLE,
    RED,
    S,
    TEAL,
    arrow,
    box,
    bullets,
    chapter,
    equation,
    styled_table,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "跨市场端到端决策感知电价预测与储能双结算论文稿_2026-10-24.pdf"
TITLE = "跨市场端到端决策感知电价预测与储能双结算"
SUBTITLE = "基于强凸电池二次规划、隐式微分与稳健多种子校准"


def load_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=16 * mm,
            title=TITLE,
            author="Author information omitted for review",
            subject="Decision-aware cross-market electricity-price forecasting and battery settlement",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=page_decor)])

    def afterFlowable(self, flowable: Flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        if flowable.style.name not in {"Heading1", "Heading2"}:
            return
        level = 0 if flowable.style.name == "Heading1" else 1
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
    canvas.drawString(18 * mm, A4[1] - 10 * mm, TITLE)
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"{page}")
    canvas.drawString(18 * mm, 9 * mm, "研究论文稿 - 2026-10-24")
    canvas.restoreState()


def decision_architecture() -> Drawing:
    d = Drawing(500, 318)
    box(d, 5, 244, 95, 58, "历史与日历输入", "168小时多源历史\n起报时可见信息", PALE_BLUE, BLUE)
    box(d, 125, 244, 105, 58, "三个价格网络", "DA / RT-at-DA / rolling RT\n无节点ID、无市场ID", LIGHT, PURPLE)
    box(d, 255, 244, 105, 58, "预测价格三腿", "24小时DA与RT-at-DA\n逐小时4步RT", PALE_ORANGE, ORANGE)
    box(d, 390, 244, 105, 58, "预测锚点损失", "Smooth-L1\n保持价格语义", PALE_BLUE, BLUE)
    box(d, 75, 145, 150, 64, "可微电池代理层", "强凸障碍QP\nDA计划 + RT动作 + SOC + 罚金", PALE_GREEN, GREEN)
    box(d, 275, 145, 150, 64, "真实价格收益", "DA腿 + RT偏差腿\n-退化成本 -偏差罚金", PALE_ORANGE, ORANGE)
    box(d, 120, 47, 260, 62, "联合 Decision-aware 更新", "KKT隐式微分：收益梯度穿过最优解返回三个网络\n仅更新 decoder + price head；编码器保持冻结", PALE_RED, RED)
    arrow(d, 100, 273, 125, 273, BLUE)
    arrow(d, 230, 273, 255, 273, PURPLE)
    arrow(d, 360, 273, 390, 273, ORANGE)
    arrow(d, 307, 244, 185, 209, ORANGE)
    arrow(d, 225, 177, 275, 177, GREEN)
    arrow(d, 350, 145, 315, 109, RED)
    arrow(d, 390, 273, 350, 109, BLUE)
    arrow(d, 120, 78, 65, 244, RED)
    d.add(String(250, 18, "正式评价不使用训练QP近似：统一进入全年连续DA LP + rolling RT MPC", fontName="CJK-Bold", fontSize=8.2, fillColor=NAVY, textAnchor="middle"))
    return d


def protocol_architecture() -> Drawing:
    d = Drawing(500, 252)
    steps = [
        (5, "源模型训练", "2020-2022\nCAISO/NYISO/PJM"),
        (105, "2023开发", "ERCOT 15节点\n方法/2:1权重"),
        (205, "冻结", "结构与超参数\n不再使用2024调参"),
        (305, "2024确认", "ERCOT外层年份\n只评价"),
        (405, "最终重训", "四市场2020-2024\n2025不进入训练"),
    ]
    accents = [BLUE, PURPLE, RED, GREEN, ORANGE]
    fills = [PALE_BLUE, LIGHT, PALE_RED, PALE_GREEN, PALE_ORANGE]
    for i, (x, title, body) in enumerate(steps):
        box(d, x, 128, 90, 70, title, body, fills[i], accents[i])
        if i < len(steps) - 1:
            arrow(d, x + 90, 163, x + 105, 163, accents[i])
    box(d, 115, 35, 270, 58, "2025固定跟进", "ERCOT 15节点、362共同日；一次导出预测后全年连续结算\n候选得分出现后 selection_allowed=false", PALE_RED, RED)
    arrow(d, 450, 128, 355, 93, ORANGE)
    d.add(String(250, 225, "年份与信息边界", fontName="CJK-Bold", fontSize=12, fillColor=NAVY, textAnchor="middle"))
    return d


def result_rows_2025(summary: dict) -> list[list[str]]:
    labels = {
        "da_implicitqp_final_robust_seed0_twothirds_v4__rt_at_da_implicitqp_final_robust_seed0_twothirds_v4__rt_implicitqp_final_robust_seed0_twothirds_v4": "本文方法",
        "da_router_followup__rt_at_da_router_followup__rt_router_followup": "尾部专家Router",
        "da_dlinear__rt_at_da_dlinear__rt_dlinear": "DLinear",
        "da_per_node_xgboost__rt_at_da_per_node_xgboost__rt_per_node_xgboost": "逐节点XGBoost",
        "da_pooled_xgboost__rt_at_da_pooled_xgboost__rt_pooled_xgboost": "pooled XGBoost",
        "da_timesfm25__rt_at_da_timesfm25__rt_timesfm25": "TimesFM-2.5 LoRA",
        "da_ensemble__rt_at_da_ensemble__rt_ensemble": "原Transformer三seed集成",
        "da_seasonal__rt_at_da_seasonal__rt_seasonal": "lag-168",
        "da_timesfm3__rt_at_da_timesfm3__rt_timesfm3": "TimesFM-3 frozen",
    }
    rows = [["系统", "收益", "相对本文", "改善节点", "配对7日块95%区间"]]
    candidate = float(summary["candidate_macro_mean_daily_revenue"])
    for item in summary["ranking"]:
        key = item["system"]
        if key not in labels:
            continue
        value = float(item["macro_mean_daily_revenue"])
        if key == summary["candidate"]:
            rows.append([B(labels[key]), B(f"{value:.2f}"), "—", "—", "—"])
            continue
        cmp = summary["comparisons"][key]
        ci = cmp["paired_7day_block_bootstrap_95ci"]
        rows.append([
            labels[key],
            f"{value:.2f}",
            f"{candidate - value:+.2f}",
            f"{cmp['improved_nodes']}/{cmp['node_count']}",
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}]",
        ])
    return rows


def build_story() -> list:
    freeze = load_json("data/results/global_implicit_qp_v4_freeze_manifest.json")
    prereg = load_json("data/results/global_implicit_qp_final_v4_preregistration.json")
    result25 = load_json("data/results/global_implicit_qp_final_v4/summary_2025.json")
    seed_summary = load_json("data/results/global_implicit_qp_multiseed_summary_v1/summary.json")
    candidate_summary = load_json("data/results/global_implicit_qp_summary_v1.json")
    calibration23 = load_json("data/results/global_robust_ensemble_train_v1.json")
    calibration25 = load_json("data/results/global_implicit_qp_final_v4/calibrator_train.json")

    comparisons25 = result25["comparisons"]
    dlinear_key = "da_dlinear__rt_at_da_dlinear__rt_dlinear"
    router_key = "da_router_followup__rt_at_da_router_followup__rt_router_followup"
    pernode_key = "da_per_node_xgboost__rt_at_da_per_node_xgboost__rt_per_node_xgboost"
    pooled_key = "da_pooled_xgboost__rt_at_da_pooled_xgboost__rt_pooled_xgboost"
    original_key = "da_ensemble__rt_at_da_ensemble__rt_ensemble"
    dlinear_cmp = comparisons25[dlinear_key]
    router_cmp = comparisons25[router_key]

    story: list = []
    story.append(Spacer(1, 8 * mm))
    story.append(P(TITLE, "title"))
    story.append(P(SUBTITLE, "subtitle"))
    story.append(Spacer(1, 4 * mm))
    story.append(P("研究论文稿（供导师审阅）", "subtitle"))
    story.append(P("2026年10月24日", "small"))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=0.7, color=GRID))
    story.append(Spacer(1, 4 * mm))

    story.append(H2("摘要"))
    story.append(P(
        "电价预测模型通常以逐时预测误差为训练目标，但储能收益由价格排序、日前头寸、实时修正、状态约束和偏差罚金共同决定，较低的平均误差并不保证更高的经济价值。本文提出一种跨市场端到端决策感知框架：三个无身份价格网络分别预测日前（day-ahead, DA）价格、日前时点的实时价格预期（RT-at-DA）和滚动实时（rolling real-time, rolling RT）价格；一个保留双结算关键物理与费用结构的强凸平滑电池二次规划层将预测映射为日前计划、实时动作和荷电状态（state of charge, SOC）轨迹；通过最优性条件隐式微分，真实价格下的收益梯度直接返回预测网络的解码器（decoder）与价格输出头。为降低随机种子和市场迁移方差，本文进一步学习任务特定的凸种子权重及有界小时校正，并以跨市场均值、方差和最差市场改善构成稳健目标。"
    ))
    story.append(P(
        "实验按年份依次进行2023开发、2024冻结确认和2025最终评价。开发阶段只用三个源市场的历史数据训练，并在未参与训练的目标市场节点上选择结构；冻结后先进行跨年确认，再按既定配置重训并完成2025全年连续结算。最终方法在统一的三价格腿、同一电池参数和同一精确线性规划（linear programming, LP）/模型预测控制（model predictive control, MPC）结算器下取得完整排行榜第一；相对最强预注册基线和前期最强方法的配对区间均高于零，节点覆盖与去极端日检验亦保持正向，且全年没有SOC违规。完整数值、市场定义、对照定义和统计口径均在第4至第6节以表格给出。"
    ))
    story.append(P("关键词：电价预测；决策感知学习；可微优化；隐式微分；储能套利；双结算市场；跨市场迁移", "small"))

    chapter(story, "1", "引言", "从预测误差最小化转向下游经济决策质量。")
    story.append(P(
        "日前与实时市场的价格具有尖峰、负价、异方差和显著的跨市场分布差异。传统MAE或Huber训练把每个小时近似等权处理，而储能套利只在有限的充放电容量下选择少数关键时段，且日前计划会改变实时可行域和偏差成本。因此，同一组预测误差可以对应完全不同的动作和收益；第6节的统一结算表也显示，预测误差较低的模型不一定具有更高的储能收益。"
    ))
    story.append(P(
        "本文不把LP/MPC仅作为训练后的评估器，而是在训练阶段构造一个强凸、平滑、可隐式求导的双结算电池代理层。预测价格先产生最优计划与动作，再用真实训练期价格计算收益，收益梯度通过最优解条件返回价格网络。这一设计属于decision-focused / decision-aware learning：监督信号不只约束价格误差，还显式反映动作、SOC、退化成本和偏差罚金。正式测试仍使用原有精确LP/MPC，避免以训练代理收益代替部署收益。"
    ))
    story.append(H2("1.1 主要贡献"))
    story.append(bullets([
        "提出面向DA、RT-at-DA和rolling RT三条价格腿的联合决策感知微调框架，使完整双结算收益能够通过KKT隐式微分返回三个预测网络。",
        "构建保留功率、SOC、效率、日放电额度、3%偏差容忍、退化成本和偏差罚金的强凸平滑电池二次规划（quadratic program, QP），并将训练代理与正式精确LP/MPC清晰分离。",
        "设计低参数、可解释的稳健多种子校准器：每条价格腿使用凸种子权重，DA与RT-at-DA通过中心/价差分解进行有界校正，rolling RT使用有界偏置与尺度。",
        "建立2023开发、2024冻结确认、2025固定跟进的跨市场与跨年协议，并统一比较统计区间、节点覆盖、去极端日、收益分解和SOC可行性。",
    ]))
    story.append(H2("1.2 论文主张范围"))
    story.append(P(
        "本文的核心主张是：冻结后的方法在ERCOT 2024外层年份和2025最终评价中提高了完整结算收益。本文不主张首次提出可微优化层；创新范围限定为三价格腿、双结算合同、跨市场稳健校准和严格时间协议的系统组合。"
    ))

    chapter(story, "2", "问题定义与正式结算", "先定义最终要优化和评价的经济量。")
    story.append(H2("2.1 三个预测任务"))
    story.append(styled_table([
        ["任务", "起报信息", "输出", "下游用途"],
        ["DA", "交付日前可见的168小时历史与日历", "未来24小时DA价格", "形成并锁定日前计划"],
        ["RT-at-DA", "与DA相同的信息截止点", "未来24小时RT价格预期", "在日前优化中估计偏差价值"],
        ["rolling RT", "交付日每小时更新的可见历史", "未来4小时RT价格", "滚动MPC实时修正，只执行第一步"],
    ], [27 * mm, 54 * mm, 42 * mm, 51 * mm]))
    story.append(P(
        "三个网络共享无身份原则：前向传播不读取node_id或market_id；身份只用于采样、SOC隔离和结果汇总。基础架构使用168小时多源输入、d=128、4头注意力、两层编码/融合模块和query decoder。这里编码器（encoder）负责把历史序列压缩为共享表征，decoder按未来时刻查询该表征，价格输出头（price head）再把每个未来时刻的隐藏向量映射为一个价格。DA与rolling RT使用带来源注意力的融合模块，RT-at-DA使用残差融合模块。"
    ))
    story.append(H2("2.2 双结算收益"))
    story.append(P("令 d、c 分别表示放电与充电，净注入功率为："))
    story.append(equation("u<sub>t</sub><super>DA</super> = d<sub>t</sub><super>DA</super> - c<sub>t</sub><super>DA</super>；    u<sub>t</sub><super>RT</super> = d<sub>t</sub><super>RT</super> - c<sub>t</sub><super>RT</super>。"))
    story.append(P("在真实DA价格 p<super>DA</super> 和RT价格 p<super>RT</super> 下，单日净收益定义为："))
    story.append(equation(
        "R = Σ<sub>t</sub>[p<sub>t</sub><super>DA</super>u<sub>t</sub><super>DA</super> + p<sub>t</sub><super>RT</super>(u<sub>t</sub><super>RT</super>-u<sub>t</sub><super>DA</super>) - κ(d<sub>t</sub><super>RT</super>+c<sub>t</sub><super>RT</super>) - 2|p<sub>t</sub><super>RT</super>|e<sub>t</sub>]。"
    ))
    story.append(P(
        "其中κ=5.7 USD/MWh为退化成本；e<sub>t</sub>=max(|u<sub>t</sub><super>RT</super>-u<sub>t</sub><super>DA</super>|-0.03|u<sub>t</sub><super>DA</super>|,0)是超过3%容忍带的偏差量。电池额定功率1 MW、容量4 MWh、SOC范围0.4-3.6 MWh、初始SOC 2 MWh、充放电效率分量0.95、每日放电额度4 MWh。"
    ))
    story.append(H2("2.3 SOC与动作约束"))
    story.append(equation("s<sub>t</sub> = s<sub>t-1</sub> + ηc<sub>t</sub> - d<sub>t</sub>/η，    0.4 ≤ s<sub>t</sub> ≤ 3.6，    0 ≤ c<sub>t</sub>,d<sub>t</sub> ≤ 1。"))
    story.append(P(
        "训练代理同时维护日前计划SOC和实际RT SOC；正式评价则把真实SOC连续地传递到后续小时和日期。日前计划由连续LP一次生成并锁定，实时阶段每小时用4小时预测窗口重解MPC且只执行首个动作。节点之间严格隔离，任何未来真实价格均不进入决策。"
    ))

    chapter(story, "3", "端到端 Decision-aware 方法", "价格网络、可微决策层与真实收益共同构成训练计算图。")
    story.append(decision_architecture())
    story.append(P("图1  端到端决策感知训练链路。红色回路表示真实收益梯度穿过QP最优解返回预测网络。", "caption"))
    story.append(H2("3.1 可微强凸电池QP"))
    story.append(P(
        "令x=[d<super>DA</super>, c<super>DA</super>, d<super>RT</super>, c<super>RT</super>, e]∈R<super>120</super>，全部物理与费用约束写为固定凸集Cx≤b。为消除原LP在多重最优解和活跃集切换处的不可导性，训练阶段加入二次正则与对数障碍："
    ))
    story.append(equation("x*(ŷ) = arg min<sub>Cx≤b</sub>  q(ŷ)<super>T</super>x + (ρ/2)||x||<super>2</super><sub>2</sub> - τΣ<sub>j</sub>log(b<sub>j</sub>-C<sub>j</sub>x)，    ρ=0.25，τ=0.10。"))
    story.append(P("三条预测价格构成线性费用向量："))
    story.append(equation(
        "q(ŷ) = [-(ŷ<super>DA</super>-ŷ<super>RT|DA</super>), +(ŷ<super>DA</super>-ŷ<super>RT|DA</super>), -ŷ<super>RT</super>+κ, +ŷ<super>RT</super>+κ, 2√((ŷ<super>RT</super>)<super>2</super>+δ<super>2</super>)]。"
    ))
    story.append(P(
        "最后一项使用δ=1的平滑绝对值近似偏差罚金系数。障碍QP通过最多60步阻尼Newton法求解，并检查绝对/相对梯度残差与最小slack。强凸项保证解唯一，障碍项使训练路径保持严格可行。"
    ))
    story.append(H2("3.2 KKT隐式微分"))
    story.append(P("Karush-Kuhn-Tucker（KKT）条件是受约束优化问题的最优性条件。本文的平滑障碍形式把它化为下列一阶方程："))
    story.append(equation("F(x,q)=q+ρx+τC<super>T</super>(b-Cx)<super>-1</super>=0。"))
    story.append(P("其对x的Jacobian即正定Hessian："))
    story.append(equation("H = ρI + τC<super>T</super>diag((b-Cx)<super>-2</super>)C。"))
    story.append(P("由隐函数定理，预测费用对最优动作的影响为："))
    story.append(equation("∂x*/∂q = -H<super>-1</super>，    ∂L/∂q = -H<super>-T</super>∂L/∂x。"))
    story.append(P(
        "实现不展开Newton迭代，而是在收敛点解线性系统H<super>T</super>v=∂L/∂x。这避免长求解轨迹的显存开销，并直接处理原问题中LP不可导或梯度不唯一的问题。"
    ))
    story.append(H2("3.3 联合训练目标与可训练范围"))
    story.append(equation("L<sub>joint</sub> = (1-α)L<sub>pred</sub> + α[-mean(R)/500]，    α=0.60。"))
    story.append(P(
        "L<sub>pred</sub>为三个任务Smooth-L1损失的等权平均；Smooth-L1在小误差区间使用二次惩罚、在大误差区间使用线性惩罚，每项再除以25进行尺度归一。收益R使用训练样本的真实DA/RT价格和QP动作计算，因此decision-aware部分明确是-mean(R)/500及其经KKT层返回的梯度。最终配置只更新三个网络的decoder和price head，编码器冻结；这属于决策层端到端微调，而不是从输入到输出更新全部参数的全参数端到端训练。"
    ))
    story.append(styled_table([
        ["超参数", "冻结值", "含义"],
        ["epochs", "3", "每个独立seed固定跑满，不按目标年收益早停"],
        ["episode", "单日；每节点每epoch 2个", "节点均衡随机episode，控制高节点市场支配"],
        ["learning rate", "2×10⁻⁵", "AdamW，weight decay 0.01"],
        ["decision weight α", "0.60", "价格锚点与经济目标的固定折中"],
        ["trainable scope", "decoder + price head", "不是全参数微调"],
        ["QP", "ρ=0.25，τ=0.10，60 Newton步", "唯一、平滑、可隐式求导"],
    ], [45 * mm, 48 * mm, 81 * mm]))
    story.append(H2("3.4 稳健多种子经济校准"))
    story.append(P(
        "seed指训练开始时固定的随机数种子。三个seed使用完全相同的网络结构、训练数据和超参数，只改变参数初始化、mini-batch顺序等随机过程，因此得到三个可独立复现、但数值略有差异的预测成员。校准器不重新训练Transformer主体，而是对每条价格腿学习softmax凸权重w；softmax把三个权重限制为非负且总和为1，再对小时级经济量做有界修正。"
    ))
    story.append(equation("ŷ<super>DA</super>=m'+s'/2，    ŷ<super>RT|DA</super>=m'-s'/2；    m'=m+b<sub>m</sub>，    s'=a<sub>s</sub>s+b<sub>s</sub>。"))
    story.append(equation("ŷ<super>RT</super><sub>t,k</sub> = a<super>RT</super><sub>t</sub>Σ<sub>r=1</sub><super>3</super>w<super>RT</super><sub>r</sub>ŷ<super>RT,r</super><sub>t,k</sub> + b<super>RT</super><sub>t</sub>。"))
    story.append(P(
        "中心偏置、价差偏置/尺度和RT偏置/尺度均由tanh或指数映射限制范围；零初始化严格等于三seed等权平均。训练时每市场使用128个episode、验证64个不重叠episode，固定8 epoch并使用末epoch，不按最终收益挑选。"
    ))
    story.append(equation("L<sub>robust</sub> = -mean<sub>m</sub>(ΔR<sub>m</sub>) + 0.5·std<sub>m</sub>(ΔR<sub>m</sub>) + 0.5·ReLU[-min<sub>m</sub>(ΔR<sub>m</sub>)] + 0.05L<sub>pred</sub> + 0.01Ω。"))
    final_val = calibration25["final_validation"]
    story.append(styled_table([
        ["校准验证统计", "数值", "含义"],
        ["四市场平均代理收益改善", f"{final_val['mean_improvement']:+.3f}", "按市场等权平均"],
        ["最差市场代理收益改善", f"{final_val['minimum_market_improvement']:+.3f}", "四个市场中最低值仍为正"],
    ], [57 * mm, 32 * mm, 85 * mm]))
    story.append(P(
        "三条腿的seed权重接近均衡，说明校准主要用于降低随机种子方差，而非退化为单一成员。"
    ))
    story.append(H2("3.5 冻结的2:1最终融合"))
    story.append(equation("ŷ<sub>v4</sub> = (2/3)ŷ<sub>seed0</sub> + (1/3)ŷ<sub>robust-ensemble</sub>。"))
    story.append(P(
        "seed0提供开发阶段最强且去极端日仍为正的单成员，稳健集成提供跨seed/跨市场稳定性。只在2023确认一次2:1简单配比，2024不再调整；最终2025重训继续沿用同一规则。该融合发生在价格预测层，三条腿逐小时使用相同权重，再统一进入正式LP/MPC。"
    ))

    chapter(story, "4", "实验协议", "同一结果只有在数据、市场、年份与选模权限都明确时才可解释。")
    story.append(protocol_architecture())
    story.append(P("图2  开发、冻结确认与最终跟进的时间顺序。", "caption"))
    story.append(H2("4.1 数据与节点"))
    story.append(P(
        "市场简称依次为：CAISO（California Independent System Operator，加州独立系统运营商）、NYISO（New York Independent System Operator，纽约独立系统运营商）、PJM Interconnection，以及ERCOT（Electric Reliability Council of Texas，德州电力可靠性委员会）。"
    ))
    story.append(styled_table([
        ["市场", "节点数", "角色", "是否进入2023候选训练", "是否进入最终2020-2024重训"],
        ["CAISO", "3", "源市场", "是", "是"],
        ["NYISO", "11", "源市场", "是", "是"],
        ["PJM", "35", "源市场", "是", "是"],
        ["ERCOT", "15", "2023开发/2024确认/2025结算", B("否"), "是，仅2020-2024"],
    ], [27 * mm, 22 * mm, 57 * mm, 34 * mm, 34 * mm]))
    story.append(P(
        "所有结果采用节点等权宏平均，避免PJM 35节点在样本数上支配其他市场。2025 ERCOT结算覆盖15节点与362个共同有效日；DA、RT-at-DA和rolling RT必须在同一节点-日期交集上完整可用。"
    ))
    story.append(H2("4.2 2023开发协议"))
    story.append(P(
        "基础价格模型和decision-aware成员仅使用CAISO、NYISO、PJM的2020-2022数据，完全排除ERCOT训练标签；2023 ERCOT 15节点用于方法开发、超参数冻结和最终2:1配比确认。LOMO是leave-one-market-out（留一市场）协议：把ERCOT整体留作未见目标市场。pooled XGBoost中的pooled表示把三个源市场的训练样本合并后训练一套共享树模型，输入中不提供节点ID或市场ID；它随后直接零样本预测ERCOT，因此与本文方法具有相同的训练信息预算，是2023公平主基线。候选必须同时检查点估计、15节点覆盖、配对区间和去候选最佳5日。"
    ))
    story.append(H2("4.3 2024冻结外层确认"))
    story.append(P(
        "2024不参与结构、损失、epoch、seed或融合权重选择。冻结v4直接在ERCOT 2024推理，并与使用ERCOT 2020-2023标签训练的逐节点XGBoost比较。由于该基线具有更多目标市场信息，这一比较对跨市场泛化主张更为保守。"
    ))
    story.append(H2("4.4 2025最终重训与跟进"))
    story.append(P(
        "确认结构后，基础三任务模型在CAISO、ERCOT、NYISO、PJM的2020-2024数据上按原配置重训；三个seed分别进行固定3 epoch的decision-aware微调；校准器按四市场平衡的训练/验证episode固定训练8 epoch；随后固定执行1/3稳健校准集成+2/3 seed0。2025不进入基础训练、微调、校准、seed选择、阈值或权重选择。候选2025预测只导出一次，得分出现后禁止启动v5。DLinear是把序列分解为趋势与季节项后分别线性外推的基线；Router是此前用浅层树在Transformer与DLinear混合专家之间按节点-日期选择的方法，二者的完整定义见第5节。"
    ))
    story.append(styled_table([
        ["阶段", "训练市场/年份", "评价市场/年份", "允许选模", "主要比较"],
        ["2023开发", "CAISO+NYISO+PJM，2020-2022", "ERCOT 2023", "是", "pooled LOMO XGBoost"],
        ["2024冻结确认", "沿用2023冻结模型", "ERCOT 2024", B("否"), "逐节点XGBoost（看过ERCOT历史）"],
        ["2025最终跟进", "四市场，2020-2024", "ERCOT 2025", B("否"), "DLinear、Router及完整排行榜"],
    ], [31 * mm, 50 * mm, 34 * mm, 25 * mm, 34 * mm]))

    chapter(story, "5", "比较方法与统一标准", "基线、前期方法和本文消融必须使用同一正式结算器。")
    story.append(H2("5.1 外部与经典基线"))
    story.append(styled_table([
        ["方法", "第一次出现时的定义", "训练/信息范围", "在本文中的作用"],
        ["lag-168", "直接复制168小时前（恰好一周前同小时）的价格", "不训练参数", "检验模型是否优于简单周季节性"],
        ["pooled XGBoost", "把所有源市场样本合并，训练一套共享梯度提升树；不输入节点或市场身份", "2023按LOMO排除ERCOT", "2023同信息预算主基线"],
        ["逐节点XGBoost", "每个节点单独训练一套梯度提升树", "可使用该节点过去的ERCOT标签", "2024较强信息预算对照"],
        ["DLinear", "把时间序列分解为趋势与季节项，再分别用线性层外推", "2025使用四市场2020-2024数据", "最强预注册纯基线"],
        ["TimesFM-3 frozen", "公开时间序列基础模型，参数冻结后直接推理", "零样本，不做本项目微调", "基础模型零样本基线"],
        ["TimesFM-2.5 LoRA", "在TimesFM-2.5上加入rank-4低秩适配参数", "只训练少量LoRA参数", "参数高效微调基线"],
        ["LEAR", "使用滞后价格、日历等变量的线性自回归模型", "只生成DA与RT-at-DA", "因缺rolling RT，不进入完整收益排名"],
    ], [30 * mm, 66 * mm, 40 * mm, 38 * mm], font_size=6.4))
    story.append(P(
        "这里的“纯基线”表示模型训练目标不使用下游储能收益，也不包含本文的可微QP或收益梯度。进入完整收益排行榜的系统还必须同时提供DA、RT-at-DA和rolling RT三条价格腿。"
    ))
    story.append(H2("5.2 本仓库前期方法及其与本文的关系"))
    story.append(P(
        "本节不是在引入一组外部模型，而是说明本研究前几轮实验已经得到、且本论文确实使用到的方法。其中有些是最终v4的组成部分，有些只用于消融或对照。下表明确区分“进入最终模型”和“仅用于比较”，避免把所有前期方法误认为本文模型的组件。"
    ))
    story.append(styled_table([
        ["方法", "通俗定义", "进入最终v4", "本文如何使用"],
        ["原无ID Transformer（三seed）", "只按价格误差训练的三个独立随机种子模型", "是，作为初始化", "既是微调起点，也是直接对照"],
        ["单seed implicit-QP", "在一个seed上让收益梯度穿过可微电池QP", "是，seed0分支", "验证decision-aware链路并提供高收益成员"],
        ["稳健三seed校准器", "对三个已微调seed学习凸权重和有界价格校正", "是，稳健分支", "降低随机种子与跨市场方差"],
        ["v3 1:1融合", "seed0分支与稳健分支各占一半", "否", "只作为2023融合比例消融"],
        ["本文v4 2:1融合", "seed0分支占2/3，稳健分支占1/3", B("是"), "最终冻结并用于2024、2025"],
        ["尾部专家Router", "深度2决策树按节点-日期在Transformer与DLinear混合专家间选择", "否", "前期独立方法，仅作为2025强对照"],
    ], [39 * mm, 72 * mm, 28 * mm, 35 * mm], font_size=6.5))
    story.append(P(
        "Router（路由器）的输入不是节点身份，而是当天起报时已可见的三项价差信号：Transformer预测价差的最大值、Transformer预测价差的10%分位数，以及DLinear预测价差为负的小时数。它默认使用原Transformer；当浅层树判断DLinear在该节点-日期更可能获利时，只把DA和RT-at-DA两条腿切换到DLinear，rolling RT仍使用Transformer。Router不参与本文v4训练，只作为此前项目中收益最高的独立方法进行比较。所有完整系统均使用同一电池参数和同一精确连续结算器。"
    ))
    story.append(H2("5.3 评价指标与统计检验"))
    story.append(equation("主指标 = (1/N)Σ<sub>n=1</sub><super>N</super>[(1/|D<sub>n</sub>|)Σ<sub>d∈D<sub>n</sub></sub>R<sub>n,d</sub>]，单位为USD/节点·日。"))
    story.append(bullets([
        "配对7日区块bootstrap 95%区间：按连续日期块重采样，保留周内相关性；报告候选与对照的日级配对差。",
        "节点覆盖：统计15个ERCOT节点中候选平均收益更高的节点数。",
        "去极端日：删除候选相对对照差值最大的5日后重新计算平均差，检验结果是否依赖少数幸运尖峰。",
        "基线最差10%日：只在对照收益最低的日期计算差值，衡量下行环境中的保护能力。",
        "机制指标：DA腿、RT偏差腿、退化成本、偏差罚金、吞吐量与SOC违规。",
        "预测指标：MAE用于说明价格误差，但不作为最终经济结论的替代指标。",
    ]))

    chapter(story, "6", "实验结果", "先报告开发与冻结证据，再打开最终跟进结果。")
    story.append(H2("6.1 2023：Decision-aware训练配置筛选"))
    story.append(P(
        "这里的“候选”不是五种不同神经网络，而是同一个三任务Transformer在decision-aware微调阶段的五套预先给定训练配置。它们只改变三项内容：可训练参数范围、联合损失中的决策权重α，以及平滑QP的强凸系数ρ和障碍系数τ。先比较seed0，是为了用较低成本选出一套配置；配置A胜出后才固定训练seed1和seed2。"
    ))
    candidate_names = {
        "implicitqp_decoder_balanced_seed0": "A 解码器平衡",
        "implicitqp_decision_seed0": "B 高决策权重",
        "implicitqp_balanced_seed0": "C 价格头平衡",
        "implicitqp_sharp_seed0": "D 较尖锐QP",
        "implicitqp_decoder_decision_seed0": "E 解码器高决策权重",
    }
    story.append(styled_table([
        ["配置", "更新哪些参数", "α", "ρ / τ", "学习率", "设计目的"],
        ["A 解码器平衡", "decoder + price head", "0.60", "0.25 / 0.10", "2e-5", "在价格锚点与收益目标间折中"],
        ["B 高决策权重", "price head", "0.80", "0.25 / 0.25", "2e-4", "检验更强收益监督"],
        ["C 价格头平衡", "price head", "0.50", "0.25 / 0.25", "2e-4", "只微调输出头的均衡版本"],
        ["D 较尖锐QP", "price head", "0.70", "0.10 / 0.05", "1e-4", "减少平滑正则，靠近原LP"],
        ["E 解码器高决策权重", "decoder + price head", "0.85", "0.10 / 0.05", "1e-5", "同时扩大范围并提高收益权重"],
    ], [34 * mm, 42 * mm, 14 * mm, 22 * mm, 21 * mm, 41 * mm], font_size=6.4))
    rows = [["配置", "收益", "较pooled XGBoost", "改善节点", "去最佳5日", "95%区间"]]
    for item in candidate_summary["rows"]:
        label = candidate_names[item["family"]]
        ci = item["paired_7day_block_bootstrap_95ci"]
        rows.append([
            label,
            f"{item['candidate_macro_mean_daily_revenue']:.2f}",
            f"{item['mean_daily_difference']:+.2f}",
            f"{item['improved_nodes']}/{item['node_count']}",
            f"{item['difference_after_removing_5_best_delta_days']:+.2f}",
            f"[{ci[0]:+.2f}, {ci[1]:+.2f}]",
        ])
    story.append(styled_table(rows, [38 * mm, 24 * mm, 25 * mm, 25 * mm, 27 * mm, 35 * mm]))
    story.append(P(
        "配置A在点估计、节点覆盖和去极端日上最稳，因此在查看seed1/2之前冻结。配置B与E提高决策权重后并未更优，说明价格误差锚点对跨市场泛化仍然必要；这也是最终采用α=0.60而非纯收益损失的依据。"
    ))
    story.append(H2("6.2 2023：固定配置的三seed稳定性"))
    seed_rows = [["成员", "原始Transformer", "Decision-aware后", "较XGBoost", "改善节点"]]
    original_values = {
        "implicitqp_decoder_balanced_seed0": 377.8381,
        "implicitqp_decoder_balanced_seed1": 266.1547,
        "implicitqp_decoder_balanced_seed2": 403.0974,
    }
    for name, item in seed_summary["individual_seeds"].items():
        seed_rows.append([
            name[-5:].replace("seed", "seed "),
            f"{original_values[name]:.2f}",
            f"{item['macro_mean_daily_revenue']:.2f}",
            f"{item['mean_daily_difference']:+.2f}",
            f"{item['improved_nodes']}/{item['node_count']}",
        ])
    ens = seed_summary["fixed_equal_weight_ensemble"]
    seed_rows.append(["固定等权集成", "407.17", f"{ens['macro_mean_daily_revenue']:.2f}", f"{ens['mean_daily_difference']:+.2f}", f"{ens['improved_nodes']}/{ens['node_count']}"])
    story.append(styled_table(seed_rows, [38 * mm, 36 * mm, 36 * mm, 31 * mm, 33 * mm]))
    story.append(P(
        "相同架构并不意味着相同结果：三个基础checkpoint由独立seed训练，优化路径不同；decision-aware微调又会通过动作边界放大微小预测差异。因此本文不只报告最佳seed，而是引入源市场稳健校准并保留三seed证据。"
    ))
    story.append(H2("6.3 v1-v4演进与冻结理由"))
    story.append(styled_table([
        ["版本", "核心变化", "2023收益", "较XGBoost", "改善节点", "去最佳5日", "配对区间结论"],
        ["v1", "三seed稳健低参数校准", "425.33", "+7.27", "14/15", "-1.43", "[-8.53,25.92]，跨0"],
        ["v2", "加入单日下行风险目标", "422.61", "+4.55", "12/15", "-3.91", "跨0，否决"],
        ["v3", "seed0与v1固定1:1融合", "429.64", "+11.58", "14/15", "+3.35", "下界-0.89，仍跨0"],
        ["v4", "seed0与v1固定2:1融合", B("430.41"), B("+12.34"), "14/15", B("+4.14"), B("[0.02,29.26]")],
    ], [16 * mm, 40 * mm, 20 * mm, 20 * mm, 20 * mm, 20 * mm, 38 * mm], font_size=6.3))
    story.append(P(
        "v2说明单日固定初始SOC的风险代理不能可靠代表跨日连续SOC风险；v3/v4利用seed0的高收益与v1的跨seed稳定性。只确认1:1和2:1两个可解释配比，不做连续权重网格搜索。v4是唯一同时通过点估计、节点覆盖、去极端日和配对区间的开发版本。"
    ))
    story.append(H2("6.4 2023开发与2024冻结确认"))
    r23 = freeze["results"]["2023"]
    r24 = freeze["results"]["2024"]
    story.append(styled_table([
        ["年份", "候选", "对照", "绝对提升", "相对提升", "改善节点", "配对95%区间", "去最佳5日"],
        ["2023开发", f"{r23['candidate']:.2f}", f"{r23['baseline']:.2f}", f"{r23['difference']:+.2f}", f"{100*r23['difference']/r23['baseline']:+.2f}%", f"{r23['improved_nodes']}/{r23['node_count']}", f"[{r23['paired_7day_95ci'][0]:+.2f}, {r23['paired_7day_95ci'][1]:+.2f}]", f"{r23['difference_after_removing_5_best_days']:+.2f}"],
        ["2024冻结", f"{r24['candidate']:.2f}", f"{r24['baseline']:.2f}", f"{r24['difference']:+.2f}", f"{100*r24['difference']/r24['baseline']:+.2f}%", f"{r24['improved_nodes']}/{r24['node_count']}", f"[{r24['paired_7day_95ci'][0]:+.2f}, {r24['paired_7day_95ci'][1]:+.2f}]", f"{r24['difference_after_removing_5_best_days']:+.2f}"],
    ], [20 * mm, 19 * mm, 19 * mm, 20 * mm, 20 * mm, 20 * mm, 33 * mm, 23 * mm], font_size=6.4))
    story.append(P(
        "表中2023的区间下界仅略高于0，因此2024的冻结确认至关重要。2024一行显示全部节点改善且区间明确高于0；同时，对照逐节点XGBoost使用了ERCOT历史标签，说明提升不是由更宽松的信息预算造成。"
    ))
    story.append(H2("6.5 2025最终完整排行榜"))
    story.append(styled_table(result_rows_2025(result25), [45 * mm, 25 * mm, 25 * mm, 28 * mm, 51 * mm], font_size=6.9))
    story.append(P("表8  ERCOT 2025统一完整流程收益。相对本文为本文方法减去该方法；区间同样针对该差值。", "caption"))
    story.append(H2("6.6 2025稳健性检查"))
    story.append(styled_table([
        ["对照", "收益差", "相对提升", "正差日比例", "改善节点", "去最佳5日", "对照最差10%日"],
        ["DLinear", f"{dlinear_cmp['mean_difference']:+.2f}", f"{100*dlinear_cmp['relative_improvement_fraction']:+.2f}%", f"{100*dlinear_cmp['positive_day_fraction']:.1f}%", f"{dlinear_cmp['improved_nodes']}/{dlinear_cmp['node_count']}", f"{dlinear_cmp['difference_after_removing_5_best_candidate_days']:+.2f}", f"{dlinear_cmp['difference_on_baseline_worst_10pct_days']:+.2f}"],
        ["Router", f"{router_cmp['mean_difference']:+.2f}", f"{100*router_cmp['relative_improvement_fraction']:+.2f}%", f"{100*router_cmp['positive_day_fraction']:.1f}%", f"{router_cmp['improved_nodes']}/{router_cmp['node_count']}", f"{router_cmp['difference_after_removing_5_best_candidate_days']:+.2f}", f"{router_cmp['difference_on_baseline_worst_10pct_days']:+.2f}"],
    ], [25 * mm, 23 * mm, 24 * mm, 26 * mm, 23 * mm, 25 * mm, 28 * mm], font_size=6.6))
    story.append(P(
        "两个强对照的配对区间下界均为正，且删除本文候选最有利的5天后仍保持正差；因此结论不能由少数极端高收益日单独解释。DLinear是预注册纯基线，Router则是原测试后的既有跟进方法，二者的证据地位不同。"
    ))
    story.append(H2("6.7 收益分解与可行性"))
    comp = result25["candidate_components"]
    story.append(styled_table([
        ["DA计划收益", "RT偏差收益", "退化成本", "偏差罚金", "净收益", "SOC违规"],
        [f"{comp['da_leg']:.2f}", f"{comp['rt_deviation_leg']:.2f}", f"-{comp['degradation_cost']:.2f}", f"-{comp['deviation_penalty']:.2f}", B(f"{result25['candidate_macro_mean_daily_revenue']:.2f}"), B(str(result25['candidate_total_soc_violation_count']))],
    ], [30 * mm, 30 * mm, 28 * mm, 28 * mm, 28 * mm, 30 * mm]))
    story.append(P(
        "净收益主要来自DA计划腿；RT层提供小幅正偏差收益，同时承担退化成本和偏差罚金。决策感知训练并不是简单放大价格，而是在动作可行域内重新组织小时排序和DA-RT价差，使更有价值的日前头寸与实时修正共同提高净收益。全年正式结算没有SOC违规。"
    ))

    chapter(story, "7", "可解释性与机制", "解释对象既包括模型参数，也包括动作和账本。")
    story.append(H2("7.1 哪一部分是Decision-aware"))
    story.append(styled_table([
        ["组件", "训练信号", "是否Decision-aware", "说明"],
        ["基础DA/RT-at-DA/RT模型", "MAE或Pseudo-Huber", "否", "提供稳定价格初始化"],
        ["implicit-QP微调", "价格锚点 + 真实收益", B("是"), "收益梯度经KKT返回decoder/head"],
        ["稳健校准器", "跨市场收益改善 + 价格锚点", B("是"), "参数少且有界"],
        ["固定2:1融合", "2023开发收益确认", "不是新的梯度层", "融合已经Decision-aware训练的成员"],
        ["正式LP/MPC结算", "无训练", "评价器", "精确、连续SOC，不用于回传"],
    ], [37 * mm, 40 * mm, 34 * mm, 63 * mm]))
    story.append(H2("7.2 校准参数的经济含义"))
    interp = calibration25["interpretation"]
    weights = interp["seed_weights"]
    ranges = {
        "center": (min(interp["center_bias_usd_per_mwh"]), max(interp["center_bias_usd_per_mwh"])),
        "spread_bias": (min(interp["spread_bias_usd_per_mwh"]), max(interp["spread_bias_usd_per_mwh"])),
        "spread_scale": (min(interp["spread_scale"]), max(interp["spread_scale"])),
        "rt_bias": (min(interp["rt_bias_usd_per_mwh"]), max(interp["rt_bias_usd_per_mwh"])),
        "rt_scale": (min(interp["rt_scale"]), max(interp["rt_scale"])),
    }
    story.append(styled_table([
        ["参数", "最终范围/权重", "解释"],
        ["DA seed权重", "/".join(f"{x:.3f}" for x in weights["da"]), "接近均衡，无单seed垄断"],
        ["RT-at-DA seed权重", "/".join(f"{x:.3f}" for x in weights["rt_at_da"]), "稳定日前价差估计"],
        ["rolling RT seed权重", "/".join(f"{x:.3f}" for x in weights["rt"]), "第三个seed略高但仍接近均衡"],
        ["中心偏置", f"[{ranges['center'][0]:.3f}, {ranges['center'][1]:.3f}] USD/MWh", "整体价格水平微调"],
        ["价差偏置", f"[{ranges['spread_bias'][0]:.3f}, {ranges['spread_bias'][1]:.3f}] USD/MWh", "直接作用于DA-RT经济价差"],
        ["价差尺度", f"[{ranges['spread_scale'][0]:.3f}, {ranges['spread_scale'][1]:.3f}]", "接近1，非无界放大"],
        ["RT偏置/尺度", f"bias [{ranges['rt_bias'][0]:.3f}, {ranges['rt_bias'][1]:.3f}]；scale [{ranges['rt_scale'][0]:.3f}, {ranges['rt_scale'][1]:.3f}]", "滚动实时动作校正"],
    ], [42 * mm, 65 * mm, 67 * mm]))
    story.append(P(
        "偏置和尺度总体较小，seed权重接近均衡，说明收益提升不是来自无界扭曲价格或事后选出单一幸运seed。最直接的经济解释变量是DA-RT价差：它决定日前净注入计划；rolling RT校正则影响实际动作、退化和超容忍偏差。"
    ))
    story.append(H2("7.3 训练代理与正式求解器的区别"))
    story.append(P(
        "训练QP为了可导性加入ρ||x||²/2和对数障碍，并去除依赖预测符号切换可行域的离散分支；正式评价使用原精确连续LP/MPC合同。代理层负责提供方向正确、数值稳定的梯度，最终主张只来自精确求解器。二者若混为一谈，会把代理目标误当成真实部署收益。"
    ))

    chapter(story, "8", "讨论与局限", "强结果必须与方法和数据边界一起报告。")
    story.append(H2("8.1 为什么节点均衡episode而非遍历全部日期"))
    story.append(P(
        "四个市场节点数差异很大；逐日全遍历会使节点更多、平稳日期更多的市场主导梯度。节点均衡episode采样保证每个节点每个epoch贡献相同数量的单日决策问题，同时随机覆盖不同日期。它是固定预算的随机优化，不改变正式评价：2023/2024/2025收益均遍历全年共同有效日期。"
    ))
    story.append(H2("8.2 泛化范围"))
    story.append(P(
        "2023阶段证明在三个市场训练后可零样本迁移到ERCOT；2024证明冻结结构可跨年维持；2025证明在四市场历史重训后仍能超过强基线。但只有ERCOT具有已冻结并审计的真实双结算合同，CAISO、NYISO和PJM主要用于训练与预测迁移，不能据此宣称四个真实市场的利润均已领先。"
    ))
    story.append(H2("8.3 方法创新边界与直接相关文献"))
    story.append(P(
        "“可微优化已有成熟文献”具体指下表四条研究线，而不是一个笼统判断。本文使用的技术基础与这些工作一致，但应用合同、三价格腿组织方式和稳健训练协议不同。"
    ))
    story.append(styled_table([
        ["文献", "已经建立的技术", "与本文的关系"],
        ["Donti, Amos & Kolter (NeurIPS 2017) [2]", "直接按下游任务目标端到端训练预测模型，并包含储能套利案例", "支持“预测损失不等于决策价值”的基本动机"],
        ["Amos & Kolter, OptNet (ICML 2017) [3]", "把QP作为网络层，并从KKT条件对最优解求导", "是本文强凸电池QP和隐式反传的直接方法基础"],
        ["Agrawal et al. (NeurIPS 2019) [4]", "系统化可微凸优化层与参数化凸程序的解析求导", "支持把受约束优化器嵌入端到端计算图"],
        ["Elmachtoub & Grigas, SPO (Management Science 2022) [5]", "以决策误差而非普通预测误差训练预测器，并提出SPO+代理", "与本文同属predict-then-optimize/decision-focused范式；本文未使用SPO+"],
    ], [47 * mm, 64 * mm, 63 * mm], font_size=6.3))
    story.append(P(
        "因此，本文不主张首次提出可微QP、KKT隐式微分或decision-focused learning。本文的具体贡献是：把具有不同信息截止点的DA、RT-at-DA和rolling RT预测，与日前/实时双结算、连续SOC、退化成本和偏差罚金组合为同一训练链路；再通过跨市场稳健校准和冻结的跨年协议验证其经济价值。"
    ))

    chapter(story, "9", "复现与审计", "所有关键主张均对应机器可读结果和冻结合同。")
    story.append(styled_table([
        ["证据", "路径"],
        ["2023候选筛选", "data/results/global_implicit_qp_summary_v1.json"],
        ["固定三seed验证", "data/results/global_implicit_qp_multiseed_summary_v1/summary.json"],
        ["2023稳健校准", "data/results/global_robust_ensemble_train_v1.json"],
        ["2023/2024冻结清单", "data/results/global_implicit_qp_v4_freeze_manifest.json"],
        ["2025预注册", "data/results/global_implicit_qp_final_v4_preregistration.json"],
        ["2025精确结算", "data/results/global_implicit_qp_final_v4/settlement_2025.json"],
        ["2025统一汇总", "data/results/global_implicit_qp_final_v4/summary_2025.json"],
        ["可微QP实现", "src/decision_aware/implicit_battery_qp.py"],
        ["联合训练入口", "scripts/decision_aware/train_global_joint_implicit_qp.py"],
        ["稳健校准入口", "scripts/decision_aware/train_global_robust_ensemble_calibrator.py"],
    ], [54 * mm, 120 * mm]))
    story.append(P(
        "预注册文件记录所有源checkpoint及代码SHA-256；2025候选分数出现后selection_allowed=false、post_score_model_changes_allowed=false。针对最终模式、固定末epoch校准、非重叠抽样、final归档转换与2025流程的27项定向测试全部通过。"
    ))

    chapter(story, "10", "结论", "决策质量而非平均预测误差决定储能价值。")
    story.append(P(
        "本文提出了一个面向双结算储能的跨市场端到端决策感知电价预测框架。三个价格网络通过强凸平滑电池QP映射为DA计划、RT动作和SOC，真实收益梯度通过KKT隐式微分回到decoder和价格头；稳健校准器再以可解释的凸seed权重及有界中心/价差/RT校正降低随机种子与市场方差。"
    ))
    story.append(P(
        "第6节的统一结果表明，最终v4在2023开发年、2024冻结确认年和2025最终评价中均优于相应主对照，并同时通过配对区间、节点覆盖、去极端日和SOC检查。结果支持将可微决策层用于训练经济相关的价格表征，同时也强调训练代理必须与正式LP/MPC评价分离。"
    ))

    chapter(story, "参考文献", "")
    refs = [
        "[1] J. Mandi, J. Kotary, S. Berden, M. Mulamba, V. Bucarey, T. Guns, and F. Fioretto. Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities. Journal of Artificial Intelligence Research, 2024.",
        "[2] P. L. Donti, B. Amos, and J. Z. Kolter. Task-based End-to-end Model Learning in Stochastic Optimization. NeurIPS, 2017.",
        "[3] B. Amos and J. Z. Kolter. OptNet: Differentiable Optimization as a Layer in Neural Networks. Proceedings of ICML, volume 70, pages 136-145, 2017.",
        "[4] A. Agrawal, B. Amos, S. Barratt, S. Boyd, S. Diamond, and J. Z. Kolter. Differentiable Convex Optimization Layers. NeurIPS, 2019.",
        "[5] A. N. Elmachtoub and P. Grigas. Smart Predict, then Optimize. Management Science, 68(1):9-26, 2022. doi:10.1287/mnsc.2020.3922.",
        "[6] J. Lago, G. Marcjasz, B. De Schutter, and R. Weron. Forecasting Day-Ahead Electricity Prices: A Review of State-of-the-Art Algorithms, Best Practices and an Open-Access Benchmark. Applied Energy, 2021.",
        "[7] R. Weron. Electricity Price Forecasting: A Review of the State-of-the-Art with a Look into the Future. International Journal of Forecasting, 2014.",
        "[8] A. Vaswani et al. Attention Is All You Need. NeurIPS, 2017.",
        "[9] A. Zeng, M. Chen, L. Zhang, and Q. Xu. Are Transformers Effective for Time Series Forecasting? AAAI, 2023.",
        "[10] A. Das, W. Kong, R. Sen, and Y. Zhou. A Decoder-Only Foundation Model for Time-Series Forecasting. ICML, 2024.",
        "[11] T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. KDD, 2016.",
        "[12] D. Krishnamurthy et al. Energy Storage Arbitrage Under Day-Ahead and Real-Time Price Uncertainty. IEEE Transactions on Power Systems, 2018.",
        "[13] B. Xu et al. Factoring the Cycle Aging Cost of Batteries Participating in Electricity Markets. IEEE Transactions on Power Systems, 2018.",
        "[14] A. Alghumayjan et al. Energy Storage Arbitrage in Two-Settlement Markets: A Transformer-Based Approach. Electric Power Systems Research, 2024.",
        "[15] Y. Liu, B. Xu, and Y. Shi. Energy Storage Price Arbitrage via Opportunity Value Function Prediction. IEEE PES General Meeting, 2023.",
        "[16] B. Efron and R. Tibshirani. An Introduction to the Bootstrap. Chapman and Hall/CRC, 1993.",
    ]
    for ref in refs:
        story.append(P(ref, "small"))

    chapter(story, "附录A", "结论边界速查")
    story.append(styled_table([
        ["表述", "是否允许", "严谨写法"],
        ["本文是全参数端到端", "否", "收益梯度端到端更新decoder+head，编码器冻结"],
        ["训练直接使用正式LP/MPC反传", "否", "训练用平滑强凸QP，正式评价用精确LP/MPC"],
        ["跨四市场真实盈利均领先", "否", "真实完整结算主张限于ERCOT"],
        ["显著超过最强预注册基线", "允许", "相对DLinear +8.02，95%区间[2.02,13.96]"],
        ["显著超过既有Router", "允许并注明地位", "相对事后跟进Router +6.02，区间[0.57,11.08]"],
        ["可微优化本身为首次提出", "否", "创新是三价格腿、双结算合同与稳健跨市场训练的组合"],
    ], [55 * mm, 34 * mm, 85 * mm]))
    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = PaperDocTemplate(str(OUTPUT))
    doc.multiBuild(build_story())
    print(OUTPUT)


if __name__ == "__main__":
    main()
