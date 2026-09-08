#!/usr/bin/env python3
"""生成模型架构图 — v2 清爽风格，中文，精简。

数据源（全部核对过实现，勿凭印象改）：
  configs/decision_aware/formal_ercot_v8.yaml  ← 图中所有超参
  src/decision_aware/model.py                  ← 图中所有结构

风格对标 v2 参考图：
  - 每个框只有粗体标题 + 一行说明，不堆代码
  - 只展开 enc_price_da 的内部结构（代表所有 Transformer 编码器）
  - 分层编号：1.多模态输入 → 2.多流编码器 → 3.跨模态融合 → 4.解码器 → 5.预测头
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

# 连线锚点：显式指定出入边，避免 draw.io 自动选锚点导致绕框/箭头打架。
# D=下出上入（竖向流），R=右出左入（横向流）。
ANCHOR_D = "exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;"
ANCHOR_R = "exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;"


def E(src, tgt, label="", style="", anchor=""):
    """连线。label 会渲染成带白底的小标签（图上每条边都标「输出：形状」）。"""
    global cid
    if not style:
        style = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#666;"
    lbl = ""
    if label:
        # 白底 + 圆角，避免标签压在连线上看不清
        lbl = ('value="&lt;div style=&quot;background-color:#FFFFFF;padding:1px 4px;'
               'border-radius:3px;font-size:9px;color:#666;&quot;&gt;'
               f'{esc(label)}&lt;/div&gt;" ')
    cells.append(
        f'<mxCell id="{cid}" {lbl}style="{style}{anchor}" edge="1" parent="1" source="{src}" target="{tgt}">'
        f'<mxGeometry relative="1" as="geometry" />'
        f'</mxCell>')
    cid += 1; return cid - 1

# ═══ 辅助：简洁 HTML（v2 风格：粗标题 + 一行说明）═══
def box(title, sub="", tc="#333"):
    b = f'&lt;div style=&quot;font-weight:bold;font-size:12px;color:{tc};&quot;&gt;{esc(title)}&lt;/div&gt;'
    if sub:
        b += f'&lt;div style=&quot;font-size:10px;color:#777;margin-top:2px;&quot;&gt;{esc(sub)}&lt;/div&gt;'
    return b

def box_sm(title, sub="", tc="#333"):
    b = f'&lt;div style=&quot;font-weight:bold;font-size:11px;color:{tc};&quot;&gt;{esc(title)}&lt;/div&gt;'
    if sub:
        b += f'&lt;div style=&quot;font-size:9px;color:#888;margin-top:2px;&quot;&gt;{esc(sub)}&lt;/div&gt;'
    return b

# ═══ 样式 ═══
S_INPUT = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#90CAF9;strokeWidth=1.5;"
S_ENC   = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1E88E5;strokeWidth=1.5;"
S_ENC_MLP = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFF8E1;strokeColor=#FFA000;strokeWidth=1.5;"
S_FUSE  = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#E8F5E9;strokeColor=#66BB6A;strokeWidth=2;"
S_DEC   = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#F3E5F5;strokeColor=#AB47BC;strokeWidth=1.5;"
S_HEAD  = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFF3E0;strokeColor=#FF9800;strokeWidth=1.5;"
S_OUT   = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#ECEFF1;strokeColor=#78909C;strokeWidth=1.5;"
S_TB    = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#42A5F5;strokeWidth=1;"
S_GRP   = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=none;strokeWidth=2;dashed=1;dashPattern=8 4;"
# 内层 TransformerBlock 分组：显式浅蓝色、细短虚线；与外层黑色总编码器框区分。
S_TB_GRP = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#90CAF9;strokeWidth=1;dashed=1;dashPattern=3 3;"
S_TXT   = "text;html=1;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;"
S_TXTC  = "text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;"

EA = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#666;"
ED = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#999;dashed=1;dashPattern=6 3;"

PW, PH = 1400, 1180

# ═══ TITLE ═══
N(f'&lt;div style=&quot;font-weight:bold;font-size:16px;color:#1565C0;&quot;&gt;{esc("正式实验版 (v8) — Decision-aware Multi-modal TSFM 模型架构图")}&lt;/div&gt;'
  f'&lt;div style=&quot;font-size:10px;color:#888;margin-top:4px;&quot;&gt;{esc("当前配置：dual_split=True | d_model=256 | 编码器/融合层均为 4 heads | FFN=1024 | 编码器/融合层均为 2 层 | 约 10.7M 参数；这些是当前选择，尚非最优性结论")}&lt;/div&gt;',
  S_TXTC, 50, 15, 1300, 50)

# ═══ 1. 多模态输入 (y=90) ═══
Y1 = 90
IW, IH = 185, 50
# 每个输入框与其下方的编码器列对齐（展开框取内部列中心）
inputs_x = [148, 470, 665, 860, 1113]
input_data = [
    ("历史电价 DA", "[B, 168, 1]"),
    ("历史电价 RT", "[B, 168, 1]"),
    ("历史负荷 Load", "[B, 168, 1]"),
    ("系统变量 System", "[B, 168, 2] 风电+光伏实际出力"),
    ("日历 Calendar", "[B, 168, 8] 本地 hour/dow/month sin/cos + weekend/holiday"),
]
N(f'&lt;div style=&quot;font-weight:bold;font-size:10px;color:#999;&quot;&gt;{esc("1. 多模态输入")}&lt;/div&gt;',
  S_TXT, 10, Y1, 70, 20)
inp_ids = []
for i, (name, desc) in enumerate(input_data):
    nid = N(box_sm(name, desc, "#1565C0"), S_INPUT, inputs_x[i], Y1, IW, IH)
    inp_ids.append(nid)

# ═══ 2. 多流编码器 (y=210) ═══
Y2 = 210
N(f'&lt;div style=&quot;font-weight:bold;font-size:10px;color:#999;&quot;&gt;{esc("2. 多流编码器")}&lt;/div&gt;',
  S_TXT, 10, Y2, 70, 20)

# 2a. enc_price_da 展开框（其余 Transformer 编码器以折叠框表示）
BW = 300
EXP_X, EXP_Y = 60, Y2 + 5
EXP_W, EXP_H = 360, 430
EXP_CX = EXP_X + (EXP_W - BW) // 2
N("", S_GRP.replace("strokeColor=", "strokeColor=#42A5F5;"), EXP_X, EXP_Y, EXP_W, EXP_H)
N(f'&lt;div style=&quot;font-weight:bold;font-size:10px;color:#1E88E5;&quot;&gt;{esc("enc_price_da — StreamEncoder (Transformer)")}&lt;/div&gt;'
  f'&lt;div style=&quot;font-size:9px;color:#999;&quot;&gt;{esc("展开视图；enc_price_rt、enc_load、enc_system 结构相同")}&lt;/div&gt;',
  S_TXT, EXP_X+5, EXP_Y+2, EXP_W-10, 30)
proj = N(box_sm("Linear Projection", "Linear(1, 256)", "#333"), S_TB, EXP_CX, EXP_Y+42, BW, 34)
tb1_y = EXP_Y + 106
# 小分组圈：一个圈对应一个完整 TransformerBlock（Attention + FFN）
N("", S_TB_GRP, EXP_X+12, tb1_y-24, EXP_W-24, 126)
N(f'&lt;div style=&quot;font-size:9px;color:#999;&quot;&gt;{esc("TransformerBlock Layer 1（pre-LN）")}&lt;/div&gt;', S_TXT, EXP_CX, tb1_y-14, 270, 12)
sa1 = N(box_sm("Self-Attention（4 heads + RoPE；pre-LN 残差）", "跨时间取信息：x = x + RotaryMHA(LN(x))", "#333"), S_TB, EXP_CX, tb1_y, BW, 38)
ff1 = N(box_sm("FFN（逐时间点非线性加工；pre-LN 残差）", "256 → 1024 → GELU → 256；x = x + FFN(LN(x))", "#333"), S_TB, EXP_CX, tb1_y+58, BW, 38)
tb2_y = EXP_Y + 232
N("", S_TB_GRP, EXP_X+12, tb2_y-24, EXP_W-24, 120)
N(f'&lt;div style=&quot;font-size:9px;color:#999;&quot;&gt;{esc("TransformerBlock Layer 2（pre-LN）")}&lt;/div&gt;', S_TXT, EXP_CX, tb2_y-14, 270, 12)
sa2 = N(box_sm("Self-Attention（4 heads + RoPE；pre-LN 残差）", "同 Layer 1：在已更新表示上再次跨时间取信息", "#333"), S_TB, EXP_CX, tb2_y, BW, 34)
ff2 = N(box_sm("FFN（逐时间点非线性加工；pre-LN 残差）", "同 Layer 1：不是跨时间交流", "#333"), S_TB, EXP_CX, tb2_y+52, BW, 34)
fn = N(box_sm("final_norm", "LayerNorm(256)", "#333"), S_TB, EXP_CX, EXP_Y+352, BW, 32)
SHAPE_TOK = "输出: [B, 168, 256]"
E(proj, sa1, SHAPE_TOK, anchor=ANCHOR_D)
E(sa1, ff1, SHAPE_TOK, anchor=ANCHOR_D)
E(ff1, sa2, SHAPE_TOK, anchor=ANCHOR_D)
E(sa2, ff2, SHAPE_TOK, anchor=ANCHOR_D)
E(ff2, fn, SHAPE_TOK, anchor=ANCHOR_D)
N(f'&lt;div style=&quot;font-size:9px;color:#1E88E5;&quot;&gt;{esc("输出: [B, 168, 256]")}&lt;/div&gt;', S_TXT, EXP_CX, EXP_Y+390, 200, 14)

# 2b. 其余 Transformer 编码器保留折叠表示（与左侧展开结构相同）
other_x = [470, 665, 860]
other_data = [
    ("enc_price_rt", "Linear(1, 256) + TransformerBlock x2（结构同左）+ final_norm"),
    ("enc_load", "Linear(1, 256) + TransformerBlock x2（结构同左）+ final_norm"),
    ("enc_system", "Linear(2, 256) + TransformerBlock x2（结构同左）+ final_norm"),
]
enc_ids = []
for i, (name, desc) in enumerate(other_data):
    enc_ids.append(N(box_sm(name, f"Transformer: {desc}", "#1565C0"), S_ENC, other_x[i], Y2+70, 185, 80))

# 2c. enc_calendar 展开框（唯一的 MLP 流，结构与 Transformer 不同，故单独展开）
# 对应 StreamEncoder(kind="mlp"): proj → x + mlp(x) → final_norm
# mlp = Linear(256,1024) > GELU > Dropout > Linear(1024,256) > Dropout
CAL_X, CAL_W = 1060, 290
CAL_BW = 250
CAL_CX = CAL_X + (CAL_W - CAL_BW) // 2
N("", S_GRP.replace("strokeColor=", "strokeColor=#FFA000;"), CAL_X, EXP_Y, CAL_W, EXP_H)
N(f'&lt;div style=&quot;font-weight:bold;font-size:10px;color:#E65100;&quot;&gt;{esc("enc_calendar — StreamEncoder (MLP)")}&lt;/div&gt;'
  f'&lt;div style=&quot;font-size:9px;color:#999;&quot;&gt;{esc("展开视图；日历流无 Self-Attn（逐时间点编码）")}&lt;/div&gt;',
  S_TXT, CAL_X+5, EXP_Y+2, CAL_W-10, 30)

cal_proj = N(box_sm("Linear Projection", "Linear(8, 256)", "#333"),
             S_ENC_MLP, CAL_CX, EXP_Y+42, CAL_BW, 34)
cal_fc1 = N(box_sm("Linear + GELU", "Linear(256, 1024) > GELU > Dropout", "#333"),
            S_ENC_MLP, CAL_CX, EXP_Y+106, CAL_BW, 38)
cal_fc2 = N(box_sm("Linear", "Linear(1024, 256) > Dropout", "#333"),
            S_ENC_MLP, CAL_CX, EXP_Y+164, CAL_BW, 38)
cal_add = N(box_sm("残差相加", "x = x + mlp(x)", "#333"),
            S_ENC_MLP, CAL_CX, EXP_Y+232, CAL_BW, 34)
cal_fn = N(box_sm("final_norm", "LayerNorm(256)", "#333"),
           S_ENC_MLP, CAL_CX, EXP_Y+352, CAL_BW, 32)

E(cal_proj, cal_fc1, "输出: [B, 168, 256]", anchor=ANCHOR_D)
E(cal_fc1, cal_fc2, "输出: [B, 168, 1024]", anchor=ANCHOR_D)
E(cal_fc2, cal_add, "输出: [B, 168, 256]", anchor=ANCHOR_D)
E(cal_add, cal_fn, "输出: [B, 168, 256]", anchor=ANCHOR_D)
# 残差旁路：proj 的输出绕过 mlp 直接汇入相加节点
E(cal_proj, cal_add, "残差旁路 x", ED,
  "exitX=0;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;")
N(f'&lt;div style=&quot;font-size:9px;color:#E65100;&quot;&gt;{esc("输出: [B, 168, 256]")}&lt;/div&gt;',
  S_TXT, CAL_CX, EXP_Y+390, 200, 14)

# 输入 → 编码器连线
E(inp_ids[0], proj, "[B, 168, 1]", anchor=ANCHOR_D)
for i in range(3):
    E(inp_ids[i+1], enc_ids[i], f"[B, 168, {1 if i < 2 else 2}]", anchor=ANCHOR_D)
E(inp_ids[4], cal_proj, "[B, 168, 8]", anchor=ANCHOR_D)

# ═══ modality_emb 标注 ═══
N(f'&lt;div style=&quot;font-size:9px;color:#999;font-style:italic;&quot;&gt;{esc("+ modality_emb [5, 256] 每流可学习模态标记")}&lt;/div&gt;',
  S_TXTC, 300, Y2+EXP_H+20, 800, 16)

# ═══ 3. 跨模态融合 (y=600) ═══
Y3 = Y2 + EXP_H + 45
N(f'&lt;div style=&quot;font-weight:bold;font-size:10px;color:#999;&quot;&gt;{esc("3. 跨模态融合")}&lt;/div&gt;',
  S_TXT, 10, Y3, 70, 20)

fuse = N(box("全局多模态 Self-Attention 融合", "concat 5流 (5x168 = 840 tokens) > 全局 Self-Attn + FFN x2 (RoPE=OFF) > final_norm；非显式 Cross-Attention", "#2E7D32"),
         S_FUSE, 200, Y3, 1000, 55)

fuse_mem = N(box_sm("Shared Memory h_i", "融合后的共享表示, K/V 供 Decoder 查询", "#2E7D32"),
             S_FUSE, 400, Y3+70, 600, 40)
E(fuse, fuse_mem, "输出: [B, 840, 256]", anchor=ANCHOR_D)

# 编码器 → 融合 连线（每条流出口都是 [B, 168, 256]，concat 后才变 840）
SHAPE_STREAM = "输出: [B, 168, 256]"
E(fn, fuse, SHAPE_STREAM, anchor=ANCHOR_D)
for enc_id in enc_ids:
    E(enc_id, fuse, SHAPE_STREAM, anchor=ANCHOR_D)
E(cal_fn, fuse, SHAPE_STREAM, anchor=ANCHOR_D)

# ═══ 4. 解码器 (y=770) ═══
Y4 = Y3 + 140
N(f'&lt;div style=&quot;font-weight:bold;font-size:10px;color:#999;&quot;&gt;{esc("4. 并行解码器")}&lt;/div&gt;',
  S_TXT, 10, Y4, 70, 20)

da_dec = N(box("DA 解码器 (48 Queries)", "query+pos+ctx > Self-Attn (RoPE) > Cross-Attn > FFN > final_norm", "#7B1FA2"),
           S_DEC, 150, Y4, 440, 55)

rt_dec = N(box("RT 解码器 (24 Queries)", "独立解码器, 结构同 DA, 对应 24 个滚动窗口", "#7B1FA2"),
           S_DEC, 800, Y4, 440, 55)

E(fuse_mem, da_dec, "K, V: [B, 840, 256]")
E(fuse_mem, rt_dec, "K, V: [B, 840, 256]")

# ═══ 5. 预测头 + 反归一化 (y=900) ═══
Y5 = Y4 + 90
N(f'&lt;div style=&quot;font-weight:bold;font-size:10px;color:#999;&quot;&gt;{esc("5. 预测头")}&lt;/div&gt;',
  S_TXT, 10, Y5, 70, 20)

head_da = N(box_sm("head_da", "Linear(256, 2) > 拆成 pDA + pRT|DA", "#E65100"),
            S_HEAD, 220, Y5, 300, 45)
head_rt_act = N(box_sm("head_rt_action", "Linear(256, 1) > 每窗口动作信号", "#E65100"),
                S_HEAD, 750, Y5, 200, 45)
head_rt_win = N(box_sm("head_rt_windows", "Linear(256, 4) > 完整窗口 H_rt=4", "#E65100"),
                S_HEAD, 1010, Y5, 200, 45)

E(da_dec, head_da, "输出: [B, 48, 256]", anchor=ANCHOR_D)
E(rt_dec, head_rt_act, "输出: [B, 24, 256]")
E(rt_dec, head_rt_win, "输出: [B, 24, 256]")

# 输出
Y6 = Y5 + 75

out_da = N(box_sm("p_DA: 日前电价预测", "[B, 48] 反归一化 + clamp", "#37474F"),
           S_OUT, 150, Y6, 200, 45)
out_rtda = N(box_sm("p_RT|DA: 条件RT预测", "[B, 48] 反归一化 + clamp", "#37474F"),
             S_OUT, 380, Y6, 200, 45)
out_rt = N(box_sm("p_RT: RT动作信号", "[B, 24] 反归一化 + clamp", "#37474F"),
           S_OUT, 750, Y6, 200, 45)
out_rtw = N(box_sm("p_RT_windows: RT窗口", "[B, 24, 4] 反归一化 + clamp", "#37474F"),
            S_OUT, 1000, Y6, 220, 45)
# 注：head_rt_windows 与 out_rtw 中心对齐（均为 1110）

# head_da 的 [B,48,2] 沿最后一维拆成两条 48h 曲线
E(head_da, out_da, "输出: [B, 48] (out[..., 0])")
E(head_da, out_rtda, "输出: [B, 48] (out[..., 1])")
E(head_rt_act, out_rt, "输出: [B, 24]", anchor=ANCHOR_D)
E(head_rt_win, out_rtw, "输出: [B, 24, 4]", anchor=ANCHOR_D)

# ═══ 底部注释 ═══
Y7 = Y6 + 70
N(f'&lt;div style=&quot;font-size:10px;color:#999;line-height:1.6;&quot;&gt;'
  f'{esc("TransformerBlock (pre-LN): x = x + Attn(LN(x)); x = x + FFN(LN(x)), FFN = Linear > GELU > Linear")}&lt;br/&gt;'
  f'{esc("融合层: 关闭 RoPE (A1), 2 层, 使用 n_heads_fusion=4; 解码器同样用 n_heads_fusion")}&lt;br/&gt;'
  f'{esc("解码器: Query Self-Attn 用 RotaryMHA, Cross-Attn 用 nn.MultiheadAttention (PyTorch 原生)")}&lt;br/&gt;'
  f'{esc("反归一化: pDA 用 DA 的 mean/std, pRT|DA / pRT / pRT_windows 用 RT 的 mean/std, 再 clamp")}'
  f'&lt;/div&gt;',
  S_TXTC, 100, Y7, 1200, 70)

# ═══ ASSEMBLE ═══
xml = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<mxfile host="km.sankuai.com" type="embed">\n'
    f'<diagram name="\u6a21\u578b\u67b6\u6784" id="arch">\n'
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

NAME = "model_architecture.drawio"
out_path = os.path.join(os.path.dirname(__file__), "docs", NAME)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(xml)
print(f"Written {out_path} ({len(xml)} bytes, {len(cells)} cells)")
