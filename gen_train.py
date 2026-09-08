#!/usr/bin/env python3
"""生成训练闭环图 — v2 清爽风格，中文。

数据源（全部核对过实现，勿凭印象改）：
  configs/decision_aware/formal_ercot_v8.yaml  ← 图中所有超参
  src/decision_aware/loss.py                   ← L_pred / L_proxy / total
  src/decision_aware/zero_order.py             ← 零阶梯度估计
  src/decision_aware/policy.py                 ← HardTopK / BESS 双结算 / LP Oracle

布局思路（对标 v2 参考图的纵向单列流）：
  - 纯纵向：输入 → 模型 → 预测 → 左列L_pred / 右列ZO → Total Loss → 回传
  - ZO 大虚线框包含 Policy+BESS+ZO估计 三步，体现从属关系
  - Oracle 从 BESS 侧连出，标注"同结构，用真实电价"
  - 连线尽量避免交叉：左列红色，右列紫色，纵向不跨列
"""
import os

cells = []
cid = 2

def esc(s):
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

def N(value, style, x, y, w, h):
    global cid
    cells.append(
        f'<mxCell id="{cid}" value="{value}" style="{style}" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />'
        f'</mxCell>')
    cid += 1; return cid - 1

# 连线锚点：显式指定出入边，避免 draw.io 自动选锚点导致绕框交叉。
ANCHOR_D = "exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;"
ANCHOR_R = "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;"


def E(src, tgt, label="", style="", anchor=""):
    """连线。label 渲染为带白底的小标签（图上每条边都标「输出：...」）。"""
    global cid
    if not style:
        style = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#666;"
    lbl = ""
    if label:
        lbl = ('value="&lt;div style=&quot;background-color:#FFFFFF;padding:1px 4px;'
               'border-radius:3px;font-size:9px;color:#666;&quot;&gt;'
               f'{esc(label)}&lt;/div&gt;" ')
    cells.append(
        f'<mxCell id="{cid}" {lbl}style="{style}{anchor}" edge="1" parent="1" source="{src}" target="{tgt}">'
        f'<mxGeometry relative="1" as="geometry" />'
        f'</mxCell>')
    cid += 1; return cid - 1

def box(title, sub="", tc="#333"):
    b = f'&lt;div style=&quot;font-weight:bold;font-size:12px;color:{tc};&quot;&gt;{esc(title)}&lt;/div&gt;'
    if sub:
        b += f'&lt;div style=&quot;font-size:10px;color:#777;margin-top:2px;&quot;&gt;{esc(sub)}&lt;/div&gt;'
    return b

def box3(title, sub="", detail="", tc="#333"):
    b = f'&lt;div style=&quot;font-weight:bold;font-size:12px;color:{tc};&quot;&gt;{esc(title)}&lt;/div&gt;'
    if sub:
        b += f'&lt;div style=&quot;font-size:10px;color:#777;margin-top:2px;&quot;&gt;{esc(sub)}&lt;/div&gt;'
    if detail:
        b += f'&lt;div style=&quot;font-size:9px;color:#aaa;margin-top:2px;&quot;&gt;{esc(detail)}&lt;/div&gt;'
    return b

# ═══ 样式 ═══
S_BLUE  = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1E88E5;strokeWidth=1.5;"
S_GREEN = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#66BB6A;strokeWidth=1.5;"
S_RED   = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFEBEE;strokeColor=#EF5350;strokeWidth=1.5;"
S_PURP  = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F3E5F5;strokeColor=#AB47BC;strokeWidth=1.5;"
S_ORAN  = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFF8E1;strokeColor=#FFA000;strokeWidth=1.5;"
S_YELL  = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFDE7;strokeColor=#FFD54F;strokeWidth=2;"
S_GREY  = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#ECEFF1;strokeColor=#78909C;strokeWidth=1.5;"
S_GREY_D= "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#ECEFF1;strokeColor=#78909C;strokeWidth=1.5;dashed=1;dashPattern=6 3;"
S_TXT   = "text;html=1;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;"
S_TXTC  = "text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;"

# 边颜色
ER = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#EF5350;"
EP = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#AB47BC;"
EG = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#66BB6A;"
EA = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#999;"
ED = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#66BB6A;dashed=1;dashPattern=6 3;"
EO = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#78909C;dashed=1;dashPattern=6 3;"

PW, PH = 1300, 1220

# 中轴
CX = 500

# ═══ TITLE ═══
N(f'&lt;div style=&quot;font-weight:bold;font-size:16px;color:#6A1B9A;&quot;&gt;{esc("正式实验版 (v8) — Decision-aware 训练回路 (双结算)")}&lt;/div&gt;'
  f'&lt;div style=&quot;font-size:10px;color:#888;margin-top:3px;&quot;&gt;{esc("零阶梯度 (ZO) + HardTopK 策略 (不可微) + BESS 双结算仿真 | alpha/beta 退火")}&lt;/div&gt;',
  S_TXTC, 100, 15, 1100, 50)

# ═══ ROW 1: 多模态输入 (y=85) ═══
inp = N(box("多模态输入 X_i", "历史电价/负荷 + 系统 + 日历 (5流)", "#1565C0"),
        S_BLUE, CX-175, 85, 350, 50)

# ═══ ROW 2: 模型 (y=175) ═══
model = N(box("Full Model f_theta", "多流 Encoder + Cross Attention + 并行 Query Decoder", "#1565C0"),
          S_BLUE, CX-200, 175, 400, 55)
E(inp, model, "输出: 5 流 ctx, 各 [B, 168, *]", anchor=ANCHOR_D)

# ═══ ROW 3: 预测输出 — 居中一个框 (y=275) ═══
pred = N(box3("预测输出",
              "p_DA [B,48]  p_RT|DA [B,48]  p_RT [B,24]  p_RT_windows [B,24,4]",
              "真实电价尺度 (反归一化 + clamp)", "#1565C0"),
         S_BLUE, CX-225, 275, 450, 55)
E(model, pred, "输出: rep [B, 48, 256] > 预测头", anchor=ANCHOR_D)

# ═══ ROW 4: 左右两列分叉 ═══
# 左列 x=80, 右列中心 x=750
LX = 80     # 左列起始
RX = 520    # 右列虚线框起始

# ── 左列: 预测损失 (y=390) ──
lpred = N(box3("预测损失 L_pred",
               "v8: Huber (delta=1.0) | 默认: half_se 0.5*MSE (fp32)",
               "LD = Huber(pDA) + Huber(pRT|DA); LR = Huber(pRT_windows); L_pred = LD + LR", "#C62828"),
          S_RED, LX, 400, 300, 60)

# 预测输出 → L_pred (红色，从左侧出)
E(pred, lpred, "输出: p_DA/p_RT|DA [B,48], p_RT_windows [B,24,4]", ER)

# 真实电价 targets (放在左列上方)
tgt = N(box("真实电价 targets", "price_da_tgt [B,48], price_rt_tgt [B,48] (结算用前 24h)", "#E65100"),
        S_ORAN, LX, 330, 300, 45)
E(tgt, lpred, "输出: 真值 [B, 48] / [B, 24, 4]", EA)

# ── 右列: 零阶梯度大虚线框 (y=370 ~ y=720) ──
ZO_BOX_Y = 370
ZO_BOX_H = 360
N("", f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#AB47BC;strokeWidth=2;dashed=1;dashPattern=8 4;",
  RX, ZO_BOX_Y, 720, ZO_BOX_H)
N(f'&lt;div style=&quot;font-weight:bold;font-size:11px;color:#AB47BC;&quot;&gt;{esc("零阶梯度路径 (不可微, 无 backward 穿过此区域)")}&lt;/div&gt;',
  S_TXT, RX+10, ZO_BOX_Y+5, 500, 16)

# 预测输出 → Policy (紫色，从右侧出)
# Step 1: Policy
policy = N(box3("1. HardTopK 策略 pi",
                "torch.topk 选 Top-K 放电 / Bot-K 充电 (不可微), K_c = K_d = 4",
                "价差门控: 仅保留 (c_dis - c_chg) > kappa/eta (v8 约 6.0) 的配对", "#7B1FA2"),
           S_PURP, RX+20, ZO_BOX_Y+30, 330, 60)

E(pred, policy, "输出: 价差 d = p_DA - p_RT|DA [B,48]; p_RT [B,24]", EP)

# Step 2: BESS 仿真
bess = N(box3("2. BESS 双结算仿真 (forward_dual, 结算前 24h)",
              "R = p_DA*u_DA + p_RT*(u_RT_act - u_DA) - kappa*(d_act + c_act) - P_dev",
              "偏差罚金 P_dev = 2|p_RT| * max(|delta_u| - 3%|u_DA|, 0)", "#2E7D32"),
         S_GREEN, RX+380, ZO_BOX_Y+35, 310, 55)

E(policy, bess, "输出: u_DA [B, 48], u_RT [B, 24] (取值 -1/0/+1)", EP)

# targets → BESS
E(tgt, bess, "输出: 真实 p_DA / p_RT [B, 24]", EA)

# Step 3: ZO 估计 (包在大框内, 在 Policy+BESS 下面)
zo = N(box3("3. 零阶梯度估计 (Nesterov-Spokoiny 双点)",
            "g = (1/K) * sum_k [F(p+eps*z) - F(p-eps*z)] / (2*eps) * z,  F = -R",
            "对 DA / RT|DA / RT 三个任务独立扰动, K=2, rho=0.05, eps_DA / eps_RT 分开", "#5E35B1"),
       S_PURP, RX+20, ZO_BOX_Y+115, 670, 55)

# 标注 ZO 内部循环关系：ZO 调用上面的 step1+step2
N(f'&lt;div style=&quot;font-size:9px;color:#AB47BC;font-style:italic;&quot;&gt;{esc("ZO 内部: 每个扰动样本 p+/p- 重复调用上面的 步骤1+步骤2 计算 R+/R-")}&lt;/div&gt;',
  S_TXT, RX+20, ZO_BOX_Y+95, 670, 16)

# Step 4: L_proxy
proxy = N(box3("4. 代理损失 L_proxy",
               "L_proxy = [dot(p_DA, g_DA) + dot(p_RT|DA, g_RT|DA) + mean(p_RT * g_RT)] / proxy_scale",
               "梯度注入: grad(L_proxy) = g_zo; DA / RT|DA 用点积 (不除 H), RT 用均值", "#5E35B1"),
          S_PURP, RX+20, ZO_BOX_Y+200, 670, 55)

E(zo, proxy, "输出: g_DA [B,48], g_RT|DA [B,48], g_RT [B,24] (detached)", EP)

# ── Oracle (在 BESS 右侧) ──
oracle = N(box3("Oracle 收益 R*_i (scipy linprog)",
                "用真实电价解 LP 求真上界 (无梯度)",
                "启用罚金时退化为单结算 LP (u_RT=u_DA 最优) | 仅日志, 不回传", "#546E7A"),
           S_GREY_D, RX+380, ZO_BOX_Y+280, 310, 55)
E(bess, oracle, "输出: R [B] — 与 R* 对比算 regret", EO)

# ═══ ROW 5: Total Loss (y=790) ═══
Y5 = ZO_BOX_Y + ZO_BOX_H + 20
total = N(box("Total Loss",
              "L = alpha * L_pred / pred_scale + beta * L_proxy / proxy_scale", "#C62828"),
          S_YELL, CX-200, Y5, 400, 50)
E(lpred, total, "输出: L_pred (标量) × alpha", ER)
E(proxy, total, "输出: L_proxy (标量) × beta", EP)

N(f'&lt;div style=&quot;font-size:9px;color:#999;&quot;&gt;{esc("退火: pretrain 8轮 (alpha=1, beta=0) > 线性退火 12轮 > alpha=0.5, beta=0.5 (v8, 共 20 epoch)")}&lt;/div&gt;',
  S_TXTC, CX-250, Y5+52, 500, 16)

# ═══ ROW 6: autograd (y=870) ═══
Y6 = Y5 + 75
optim = N(box("标准 autograd 回传",
              "AdamW + grad_clip + AMP autocast", "#37474F"),
          S_GREY, CX-150, Y6, 300, 50)
E(total, optim, "输出: L (标量) — .backward()", anchor=ANCHOR_D)

# 回传虚线箭头
E(optim, model, "输出: 参数更新 Δtheta", ED)

# ═══ 底部注释 ═══
Y7 = Y6 + 80
N(f'&lt;div style=&quot;font-size:10px;color:#999;line-height:1.6;&quot;&gt;'
  f'{esc("核心: 策略 pi 不可微 (HardTopK topk), 用零阶梯度 (Nesterov-Spokoiny 2017) 估计代理梯度, 通过 L_proxy 注入 autograd")}&lt;br/&gt;'
  f'{esc("BESS: 1MW / 4MWh, eta=0.95, kappa=5.7 $/MWh (v8 从 27 下调), SOC 0.4~3.6 MWh, E_cyc=4 MWh/日")}&lt;br/&gt;'
  f'{esc("启用偏差罚金时 u_RT 先跟随 u_DA (plan_track_override), 仅在 u_DA=0 的时段保留 TopK 套利动作")}'
  f'&lt;/div&gt;',
  S_TXTC, 100, Y7, 1100, 55)

# ═══ ASSEMBLE ═══
xml = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<mxfile host="km.sankuai.com" type="embed">\n'
    '<diagram name="\u8bad\u7ec3\u56de\u8def" id="train">\n'
    f'<mxGraphModel dx="{PW}" dy="{PH}" grid="1" gridSize="10" guides="1" tooltips="1" '
    f'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{PW}" pageHeight="{PH}" math="0" shadow="0">\n'
    '<root>\n'
    '<mxCell id="0" />\n'
    '<mxCell id="1" parent="0" />\n'
    + "\n".join(cells) + "\n"
    '</root>\n'
    '</mxGraphModel>\n'
    '</diagram>\n'
    '</mxfile>'
)

NAME = "training_loop.drawio"
out_path = os.path.join(os.path.dirname(__file__), "docs", NAME)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(xml)
print(f"Written {out_path} ({len(xml)} bytes, {len(cells)} cells)")
