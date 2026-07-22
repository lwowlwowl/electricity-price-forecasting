#!/usr/bin/env python3
"""V4: Updated based on w9 materials (md + PDF + dual-settlement PNGs).
Key changes from v3:
- Architecture: Output heads changed to Head_DA + Head_RT (dual settlement)
- Architecture: Input expanded with System + News (dashed = extensible)
- Architecture: Added task scope label, iteration roadmap, E2E training label
- Training loop: Dual settlement (DA path + RT path)
- Training loop: Surrogate gradient annotation, accurate Loss formula
"""
import os, zipfile

def esc(h):
    return h.replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')

def V(cid, html, style, x, y, w, h):
    val = esc(html) if html else ""
    return f'<mxCell id="{cid}" value="{val}" style="{style}" vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>'

def E(cid, src, tgt, label="", extra=""):
    val = esc(label) if label else ""
    base = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;"
    return f'<mxCell id="{cid}" value="{val}" style="{base}{extra}" edge="1" parent="1" source="{src}" target="{tgt}"><mxGeometry relative="1" as="geometry" /></mxCell>'

# Colors
BF,BS,BT = "#F0F7FF","#0066CC","#0055A4"
GF,GS,GT = "#F1F8E9","#689F38","#33691E"
OF,OS,OT = "#FFFBF0","#FF9900","#CC7A00"
RF,RS,RT_ = "#FFEBEE","#D32F2F","#C62828"
NF,NS,NT = "#F8F9FA","#B0BEC5","#455A64"
PF,PS = "#EDE7F6","#7E57C2"
YF,YS = "#FFF9C4","#F9A825"
CARD = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;strokeWidth=1.5;"
TXT = "text;html=1;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;"
CTXT = "text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;"

# ======================== PAGE 1: Architecture ========================
p1 = []
p1.append('<mxCell id="0" />')
p1.append('<mxCell id="1" parent="0" />')

# Layout
inp_w, inp_h = 170, 50
inp_gap = 20
Y0 = 100
total_cols = 6  # Price, Load, Weather, Calendar, System, News
total_w = total_cols * inp_w + (total_cols - 1) * inp_gap
inp_x0 = 80

# Title
p1.append(V("t1",
    '<b style="font-size:15px;color:#333;">Decision-aware Multi-modal TSFM 模型架构图</b>',
    CTXT, inp_x0, 10, total_w, 25))

# Task scope label (from w9 section 1)
p1.append(V("scope",
    '<div style="font-size:10px;color:#0066CC;line-height:1.5;"><b>适用任务: 日前预测 (Day-ahead)</b> — 从零训练, 不使用预训练权重<br/>日内预测 (Intra-day): 复用本架构, 换 Head 或微调 [待确认]</div>',
    TXT, inp_x0, 38, total_w, 30))

# --- ROW 1: Inputs (y=Y0) ---
p1.append(V("l1",'<div style="font-size:10px;color:#AAA;font-weight:bold;">1. 多模态输入</div>', TXT, 5, Y0+12, 75, 20))

# Core 4 inputs (solid)
core_inputs = [
    ("i1","历史电价 Price","7天历史 (168h)"),
    ("i2","历史负荷 Load","7天历史 (168h)"),
    ("i3","天气 Weather","未来天气预报(已知)"),
    ("i4","日历/节假日","离散类别变量"),
]
for idx, (cid, title, desc) in enumerate(core_inputs):
    x = inp_x0 + idx * (inp_w + inp_gap)
    p1.append(V(cid,
        f'<div style="font-size:11px;font-weight:bold;color:{BT};">{title}</div><div style="font-size:9px;color:#666;">{desc}</div>',
        f"{CARD}fillColor={BF};strokeColor={BS};", x, Y0, inp_w, inp_h))

# Extended inputs (dashed - from PDF section 2)
ext_inputs = [
    ("i5","系统变量 System","风光/备用/拥塞等"),
    ("i6","新闻 News","已发布的市场新闻"),
]
for idx, (cid, title, desc) in enumerate(ext_inputs):
    x = inp_x0 + (4 + idx) * (inp_w + inp_gap)
    p1.append(V(cid,
        f'<div style="font-size:11px;font-weight:bold;color:{BT};">{title}</div><div style="font-size:9px;color:#888;">{desc}</div>',
        f"{CARD}fillColor={BF};strokeColor={BS};dashed=1;dashPattern=5 5;", x, Y0, inp_w, inp_h))

# --- ROW 2: Encoders (y=200) ---
enc_y = 200
enc_h = 60
p1.append(V("l2",'<div style="font-size:10px;color:#AAA;font-weight:bold;">2. 多流编码器</div>', TXT, 5, enc_y+18, 75, 20))

encoders = [
    ("e1","Price Encoder","Transformer Layer","Self-Attn + FFN + RoPE"),
    ("e2","Load Encoder","Transformer Layer","Self-Attn + FFN + RoPE"),
    ("e3","Weather Encoder","Transformer/可选GRU","[待确认] 轻量建模"),
    ("e4","Event Encoder","Embedding","类别变量编码"),
    ("e5","System Encoder","Transformer/MLP","[可扩展]"),
    ("e6","News Encoder","Text Encoder","[可扩展]"),
]
for idx, (cid, title, l1, l2) in enumerate(encoders):
    x = inp_x0 + idx * (inp_w + inp_gap)
    dashed = "dashed=1;dashPattern=5 5;" if idx >= 4 else ""
    p1.append(V(cid,
        f'<div style="font-size:11px;font-weight:bold;color:{BT};">{title}</div><div style="font-size:9px;color:#666;">{l1}</div><div style="font-size:8px;color:#999;">{l2}</div>',
        f"{CARD}fillColor={BF};strokeColor={BS};{dashed}", x, enc_y, inp_w, enc_h))

# Edges: Input -> Encoder
for i in range(1,7):
    dashed = "dashed=1;dashPattern=5 5;" if i >= 5 else ""
    p1.append(E(f"ie{i}", f"i{i}", f"e{i}", "", f"strokeColor={BS};exitX=0.5;exitY=1;entryX=0.5;entryY=0;{dashed}"))

# --- ROW 3: Fusion (y=320) ---
fus_y = 320
fus_h = 110
p1.append(V("l3",'<div style="font-size:10px;color:#AAA;font-weight:bold;">3. 跨模态融合</div>', TXT, 5, fus_y+40, 75, 20))

p1.append(V("f0", "",
    f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;strokeWidth=1.5;fillColor={GF};strokeColor={GS};",
    inp_x0, fus_y, total_w, fus_h))
p1.append(V("ft",
    f'<div style="font-size:13px;font-weight:bold;color:{GT};">Cross-modal Fusion (Cross Attention)</div>',
    CTXT, inp_x0, fus_y+8, total_w, 20))
p1.append(V("fi",
    '<div style="font-size:10px;color:#555;">Weather->Load | Holiday->Load | Load->Price | Weather->Price | System->Price [可扩展]</div>',
    CTXT, inp_x0+10, fus_y+32, total_w-20, 20))
# Shared Memory
sm_y = fus_y + fus_h - 48
p1.append(V("sm",
    f'<div style="font-size:11px;font-weight:bold;color:{GT};">Shared Memory h_i = TSFM_θ(X_i)</div><div style="font-size:9px;color:#555;">融合后的共享表示, K/V 供 Decoder 查询</div>',
    f"{CARD}fillColor=#FFFFFF;strokeColor={GS};strokeWidth=2;", inp_x0+total_w//2-180, sm_y, 360, 40))

# Edges: Encoder -> Fusion
for idx in range(6):
    entry_x = round((idx + 0.5) / 6, 3)
    dashed = "dashed=1;dashPattern=5 5;" if idx >= 4 else ""
    p1.append(E(f"ef{idx+1}", f"e{idx+1}", "f0", "",
        f"strokeColor={GS};exitX=0.5;exitY=1;entryX={entry_x};entryY=0;{dashed}"))

# --- ROW 4: Decoder (y=480) ---
dec_x = inp_x0 + total_w//2 - 150
dec_w = 300
dec_y1 = 480
p1.append(V("l4",'<div style="font-size:10px;color:#AAA;font-weight:bold;">4. Decoder</div>', TXT, 5, dec_y1+60, 75, 20))

p1.append(V("d1",
    '<div style="font-size:11px;font-weight:bold;color:#5E35B1;">96 Learnable Queries</div><div style="font-size:9px;color:#666;">每个 Query 对应未来一个时间点 (非自回归)</div>',
    f"{CARD}fillColor={PF};strokeColor={PS};", dec_x, dec_y1, dec_w, 45))

dec_y2 = 550
p1.append(V("d2",
    '<div style="font-size:11px;font-weight:bold;color:#5E35B1;">Decoder Cross Attention</div><div style="font-size:9px;color:#666;">Q = Learnable Queries, K/V = Shared Memory</div>',
    f"{CARD}fillColor={PF};strokeColor={PS};strokeWidth=2;", dec_x, dec_y2, dec_w, 50))

dec_y3 = 630
p1.append(V("d3",
    '<div style="font-size:11px;font-weight:bold;color:#5E35B1;">Forecast Representation</div><div style="font-size:9px;color:#666;">96 x d_model, 一次 Forward 输出</div>',
    f"{CARD}fillColor={PF};strokeColor={PS};", dec_x, dec_y3, dec_w, 45))

# Edges: SM -> D2 (K,V), D1 -> D2 (Q), D2 -> D3
p1.append(E("sd", "sm", "d2",
    '<div style="background:#fff;padding:1px 4px;color:#689F38;border-radius:3px;font-size:10px;font-weight:bold;">K, V</div>',
    f"strokeColor={GS};strokeWidth=2;exitX=0.5;exitY=1;entryX=0.25;entryY=0;"))
p1.append(E("qd", "d1", "d2",
    '<div style="background:#fff;padding:1px 4px;color:#7E57C2;border-radius:3px;font-size:10px;font-weight:bold;">Q</div>',
    f"strokeColor={PS};exitX=0.5;exitY=1;entryX=0.75;entryY=0;"))
p1.append(E("dd", "d2", "d3", "", f"strokeColor={PS};exitX=0.5;exitY=1;entryX=0.5;entryY=0;"))

# --- ROW 5: Heads (y=730) - DA + RT + extensible (from w9 PDF) ---
head_y = 730
head_w = 200
head_gap = 50
heads_total = 3 * head_w + 2 * head_gap
head_x0 = inp_x0 + (total_w - heads_total) // 2

p1.append(V("l5",'<div style="font-size:10px;color:#AAA;font-weight:bold;">5. 预测头</div>', TXT, 5, head_y+8, 75, 20))

# Head DA
p1.append(V("h1",
    f'<div style="font-size:11px;font-weight:bold;color:{OT};">Head_DA (日前)</div><div style="font-size:9px;color:#666;">较长窗口电价预测 (H24/H48)</div>',
    f"{CARD}fillColor={OF};strokeColor={OS};", head_x0, head_y, head_w, 45))

# Head RT
p1.append(V("h2",
    f'<div style="font-size:11px;font-weight:bold;color:{OT};">Head_RT (实时)</div><div style="font-size:9px;color:#666;">较短窗口电价预测</div>',
    f"{CARD}fillColor={OF};strokeColor={OS};", head_x0 + head_w + head_gap, head_y, head_w, 45))

# Extensible head (dashed)
p1.append(V("h3",
    f'<div style="font-size:11px;font-weight:bold;color:{OT};">...可扩展</div><div style="font-size:9px;color:#888;">Load / BESS 等</div>',
    f"{CARD}fillColor={OF};strokeColor={OS};dashed=1;dashPattern=5 5;", head_x0 + 2*(head_w + head_gap), head_y, head_w, 45))

# Edges: Forecast Representation -> Heads (fan out)
for idx in range(3):
    exit_x = round((idx + 0.5) / 3, 3)
    dashed = "dashed=1;dashPattern=5 5;" if idx == 2 else ""
    p1.append(E(f"dh{idx+1}", "d3", f"h{idx+1}", "",
        f"strokeColor={OS};exitX={exit_x};exitY=1;entryX=0.5;entryY=0;{dashed}"))

# --- ROW 6: Outputs (y=825) ---
out_y = 825
out_w = 200

p1.append(V("o1",
    f'<div style="font-size:10px;font-weight:bold;color:{NT};">p\u0302_DA: 日前电价预测 (96点)</div>',
    f"{CARD}fillColor={NF};strokeColor={NS};", head_x0, out_y, out_w, 35))
p1.append(V("o2",
    f'<div style="font-size:10px;font-weight:bold;color:{NT};">p\u0302_RT: 实时电价预测</div>',
    f"{CARD}fillColor={NF};strokeColor={NS};", head_x0 + head_w + head_gap, out_y, out_w, 35))

p1.append(E("ho1", "h1", "o1", "", f"strokeColor={NS};exitX=0.5;exitY=1;entryX=0.5;entryY=0;"))
p1.append(E("ho2", "h2", "o2", "", f"strokeColor={NS};exitX=0.5;exitY=1;entryX=0.5;entryY=0;"))

# --- Foundation Backbone dashed frame ---
p1.append(V("bb",
    '<div style="font-size:9px;color:#BBB;font-style:italic;">Foundation Backbone (可复用, 端到端联合训练) — 换 Head 即可迁移任务</div>',
    "rounded=1;arcSize=6;whiteSpace=wrap;html=1;strokeWidth=1.5;dashed=1;dashPattern=8 4;fillColor=none;strokeColor=#DDDDDD;verticalAlign=bottom;",
    inp_x0-15, enc_y-15, total_w+30, dec_y3+45-enc_y+30))

# --- Bottom notes ---
p1.append(V("note1",
    '<div style="font-size:9px;color:#999;line-height:1.5;">参数量: 验证版~15M (d=256,2层) → 完整版~100M (d=512,4层) | Attention: MHA/可选MQA [待确认] | 可选GRU [待确认] | 训练: 2 epoch快速验证</div>',
    TXT, inp_x0, out_y+50, total_w, 20))

# Iteration roadmap (from w9 section 6)
p1.append(V("note2",
    '<div style="font-size:9px;color:#0066CC;line-height:1.5;"><b>迭代路径:</b> Step1 最小改动(单Encoder+简单拼接) → Step2 多流Encoder+CrossAttn → Step3 +Decision-aware训练 | 对比基线: TimesFM/Chronos/Moirai/Toto (同参数量)</div>',
    TXT, inp_x0, out_y+72, total_w, 20))


# ======================== PAGE 2: Training Loop (dual settlement) ========================
p2 = []
p2.append('<mxCell id="0" />')
p2.append('<mxCell id="1" parent="0" />')

CX, CW = 300, 280  # center
LX, LW = 30, 170   # left
DAX, DAW = 290, 180  # DA branch (center-left)
RTX, RTW = 530, 180  # RT branch (center-right)

# Title
p2.append(V("t2",
    '<b style="font-size:15px;color:#333;">Decision-aware 训练回路 (双结算)</b>',
    CTXT, 100, 10, 600, 28))

# Row 1: Input
p2.append(V("inp",
    f'<div style="font-size:12px;font-weight:bold;color:{BT};">多模态输入 X_i</div><div style="font-size:9px;color:#666;">历史电价/负荷 + 天气预报 + 日历 + ...</div>',
    f"{CARD}fillColor={BF};strokeColor={BS};", CX, 55, CW, 42))

# Row 2: Model
p2.append(V("fm",
    f'<div style="font-size:13px;font-weight:bold;color:{BT};">Foundation Model f_θ</div><div style="font-size:10px;color:#666;">多流 Encoder + Fusion + Decoder</div>',
    f"{CARD}fillColor={BF};strokeColor={BS};strokeWidth=2;", CX, 125, CW, 48))

# Row 3: Two outputs (DA + RT)
p2.append(V("pda",
    f'<div style="font-size:11px;font-weight:bold;color:{BT};">p\u0302_DA (日前电价预测)</div><div style="font-size:9px;color:#666;">较长窗口, 用于日前计划</div>',
    f"{CARD}fillColor={BF};strokeColor={BS};", DAX, 210, DAW, 42))
p2.append(V("prt",
    f'<div style="font-size:11px;font-weight:bold;color:{BT};">p\u0302_RT (实时电价预测)</div><div style="font-size:9px;color:#666;">较短窗口, 用于实时调整</div>',
    f"{CARD}fillColor={BF};strokeColor={BS};", RTX, 210, RTW, 42))

# Row 4: Two strategies
p2.append(V("pda_s",
    f'<div style="font-size:11px;font-weight:bold;color:{NT};">日前策略 π_DA</div><div style="font-size:9px;color:#666;">固定Greedy: 谷充峰放</div><div style="font-size:8px;color:#999;">不参与训练</div>',
    f"{CARD}fillColor={NF};strokeColor={NS};", DAX, 295, DAW, 55))
p2.append(V("prt_s",
    f'<div style="font-size:11px;font-weight:bold;color:{NT};">实时策略 π_RT</div><div style="font-size:9px;color:#666;">基于DA计划做偏差调整</div><div style="font-size:8px;color:#999;">不参与训练</div>',
    f"{CARD}fillColor={NF};strokeColor={NS};", RTX, 295, RTW, 55))

# Row 5: Actions merge -> Revenue
p2.append(V("acts",
    f'<div style="font-size:11px;font-weight:bold;color:{GT};">真实收益 R_i</div><div style="font-size:9px;color:#555;">u_DA x λ_DA + Δu_RT x λ_RT - 成本</div><div style="font-size:8px;color:#999;">用真实电价 p_i 结算</div>',
    f"{CARD}fillColor={GF};strokeColor={GS};", CX, 395, CW, 55))

# Left: Prediction Loss
p2.append(V("fl",
    f'<div style="font-size:11px;font-weight:bold;color:{RT_};">预测损失 L_pred</div><div style="font-size:9px;color:#666;">MAE / Huber / Quantile</div>',
    f"{CARD}fillColor={RF};strokeColor={RS};", LX, 310, LW, 48))

# Row 6: Oracle + Regret
p2.append(V("orc",
    f'<div style="font-size:10px;font-weight:bold;color:{NT};">Oracle 收益 R*_i</div><div style="font-size:9px;color:#666;">用真实电价跑同一策略</div>',
    f"{CARD}fillColor={NF};strokeColor={NS};", CX+CW+30, 395, 170, 42))

p2.append(V("bl",
    f'<div style="font-size:11px;font-weight:bold;color:{RT_};">业务损失 L_bus</div><div style="font-size:9px;color:#666;">= Regret = R*_i - R_i</div>',
    f"{CARD}fillColor={RF};strokeColor={RS};", CX, 490, CW, 42))

# Row 7: Total Loss
p2.append(V("tl",
    '<div style="font-size:13px;font-weight:bold;color:#333;">Total Loss</div><div style="font-size:11px;color:#666;">L = αL_pred + βL_bus</div>',
    f"{CARD}fillColor={YF};strokeColor={YS};strokeWidth=2;", CX, 570, CW, 45))

# Row 8: Backprop
p2.append(V("bp",
    f'<div style="font-size:11px;font-weight:bold;color:{BT};">代理梯度回传</div><div style="font-size:9px;color:#666;">g = 2Δt(u_ref - u*), 前向hard/反向surrogate</div>',
    f"rounded=1;arcSize=8;whiteSpace=wrap;html=1;strokeWidth=1.5;dashed=1;dashPattern=8 4;fillColor={BF};strokeColor={BS};",
    CX, 650, CW, 45))

# Insight note
p2.append(V("nt2",
    '<div style="font-size:10px;color:#555;line-height:1.6;"><b>核心:</b> 预测模型 f_θ 是训练对象, 储能策略 π 固定不学习。Forecast 是中间变量, Business Loss (Regret) 通过代理梯度回传, 让模型学习"哪些预测误差真正影响收益"。</div>',
    "text;html=1;align=left;verticalAlign=top;strokeColor=none;fillColor=none;",
    LX, 730, RTX+RTW-LX, 45))

# --- Edges ---
p2.append(E("te1","inp","fm","","exitX=0.5;exitY=1;entryX=0.5;entryY=0;"))
# Model -> two predictions (fan out)
p2.append(E("te2a","fm","pda","","exitX=0.35;exitY=1;entryX=0.5;entryY=0;"))
p2.append(E("te2b","fm","prt","","exitX=0.65;exitY=1;entryX=0.5;entryY=0;"))
# Predictions -> Strategies
p2.append(E("te3a","pda","pda_s","","exitX=0.5;exitY=1;entryX=0.5;entryY=0;"))
p2.append(E("te3b","prt","prt_s","","exitX=0.5;exitY=1;entryX=0.5;entryY=0;"))
# Strategies -> Revenue (converge)
p2.append(E("te4a","pda_s","acts","",f"strokeColor={GS};exitX=0.5;exitY=1;entryX=0.35;entryY=0;"))
p2.append(E("te4b","prt_s","acts","",f"strokeColor={GS};exitX=0.5;exitY=1;entryX=0.65;entryY=0;"))
# Prediction -> Pred Loss (from pda left side)
p2.append(E("te5","pda","fl","",f"strokeColor={RS};exitX=0;exitY=0.5;entryX=0.5;entryY=0;"))
# Oracle
p2.append(E("te6","acts","bl",
    '<div style="background:#fff;padding:1px 3px;color:#666;border-radius:3px;font-size:9px;">R_i</div>',
    f"strokeColor={RS};exitX=0.5;exitY=1;entryX=0.5;entryY=0;"))
p2.append(E("te6b","orc","bl",
    '<div style="background:#fff;padding:1px 3px;color:#666;border-radius:3px;font-size:9px;">R*_i</div>',
    f"strokeColor={RS};exitX=0.5;exitY=1;entryX=1;entryY=0.5;"))
# Pred Loss -> Total
p2.append(E("te7","fl","tl",
    '<div style="background:#fff;padding:1px 3px;color:#999;border-radius:3px;font-size:10px;">α</div>',
    f"strokeColor={RS};exitX=0.5;exitY=1;entryX=0;entryY=0.5;"))
# Bus Loss -> Total
p2.append(E("te8","bl","tl",
    '<div style="background:#fff;padding:1px 3px;color:#999;border-radius:3px;font-size:10px;">β</div>',
    f"strokeColor={RS};exitX=0.5;exitY=1;entryX=0.5;entryY=0;"))
# Total -> Backprop
p2.append(E("te9","tl","bp","","exitX=0.5;exitY=1;entryX=0.5;entryY=0;"))
# Backprop -> Model (left side loop)
p2.append(E("te10","bp","fm",
    '<div style="background:#fff;padding:2px 5px;color:#0066CC;border-radius:3px;font-size:10px;font-weight:bold;">更新 f_θ</div>',
    f"strokeColor={BS};dashed=1;dashPattern=6 4;strokeWidth=2;exitX=0;exitY=0.5;entryX=0;entryY=0.5;"))


# ======================== Assemble ========================
def page(name, pid, cells, w, h):
    x = f'<diagram name="{name}" id="{pid}"><mxGraphModel dx="1000" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{w}" pageHeight="{h}" math="0" shadow="0"><root>'
    for c in cells:
        x += c
    x += '</root></mxGraphModel></diagram>'
    return x

xml = '<?xml version="1.0" encoding="UTF-8"?>'
xml += '<mxfile host="km.sankuai.com" type="embed">'
xml += page("模型架构图", "arch", p1, 1300, 950)
xml += page("训练回路", "train", p2, 850, 810)
xml += '</mxfile>'

out = "/Users/wanghaochen/school/docs/model_architecture.drawio"
with open(out, 'w', encoding='utf-8') as f:
    f.write(xml)
print(f"Written: {out} ({os.path.getsize(out)} bytes)")

zp = "/Users/wanghaochen/school/docs/model_architecture.zip"
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
    z.write(out, "model_architecture.drawio")
print(f"Zip: {zp} ({os.path.getsize(zp)} bytes)")
