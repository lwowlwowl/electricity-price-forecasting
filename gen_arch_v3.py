#!/usr/bin/env python3
"""生成 v3 模型架构图 — v2 清爽风格，中文，精简。

风格对标 v2 参考图：
  - 每个框只有粗体标题 + 一行说明，不堆代码
  - 只展开 enc_price_da 的内部结构（代表所有 Transformer 编码器）
  - 分层编号：1.多模态输入 → 2.多流编码器 → 3.跨模态融合 → 4.解码器 → 5.预测头
"""
import zipfile, os

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

def E(src, tgt, label="", style=""):
    global cid
    if not style:
        style = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#666;"
    lbl = ""
    if label:
        lbl = (f'value="&lt;span style=&quot;font-size:9px;color:#666;&quot;&gt;{esc(label)}&lt;/span&gt;" ')
    cells.append(
        f'<mxCell id="{cid}" {lbl}style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">'
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
S_TXT   = "text;html=1;align=left;verticalAlign=middle;strokeColor=none;fillColor=none;"
S_TXTC  = "text;html=1;align=center;verticalAlign=middle;strokeColor=none;fillColor=none;"

EA = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#666;"
ED = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=1.5;strokeColor=#999;dashed=1;dashPattern=6 3;"

PW, PH = 1400, 1350

# ═══ TITLE ═══
N(f'&lt;div style=&quot;font-weight:bold;font-size:16px;color:#1565C0;&quot;&gt;{esc("正式实验版 (Full) — Decision-aware Multi-modal TSFM v3 模型架构图")}&lt;/div&gt;'
  f'&lt;div style=&quot;font-size:10px;color:#888;margin-top:4px;&quot;&gt;{esc("dual_split=True | d_model=256 | n_heads_enc=4 | n_heads_fusion=4 | dim_ff=1024 | n_layers_enc=2 | 5流编码器 + 跨模态融合 + 并行解码器")}&lt;/div&gt;',
  S_TXTC, 50, 15, 1300, 50)

# ═══ 1. 多模态输入 (y=90) ═══
Y1 = 90
IW, IH = 200, 50
GAP = 40
inputs_x = [80, 80+IW+GAP, 80+2*(IW+GAP), 80+3*(IW+GAP), 80+4*(IW+GAP)]
input_data = [
    ("历史电价 DA", "[B, 168, 1]"),
    ("历史电价 RT", "[B, 168, 1]"),
    ("历史负荷 Load", "[B, 168, 1]"),
    ("系统变量 System", "[B, 168, 2] 风光+光伏"),
    ("日历 Calendar", "[B, 168, 6] sin/cos+flags"),
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

# 2a. enc_price_da 展开框 (代表 Transformer 编码器结构)
EXP_X, EXP_Y = 60, Y2 + 5
EXP_W, EXP_H = 440, 340
N("", S_GRP.replace("strokeColor=", "strokeColor=#42A5F5;"), EXP_X, EXP_Y, EXP_W, EXP_H)
N(f'&lt;div style=&quot;font-weight:bold;font-size:10px;color:#1E88E5;&quot;&gt;{esc("enc_price_da — StreamEncoder (Transformer)")}&lt;/div&gt;'
  f'&lt;div style=&quot;font-size:9px;color:#999;&quot;&gt;{esc("展开视图；enc_price_rt, enc_load, enc_system 结构相同")}&lt;/div&gt;',
  S_TXT, EXP_X+5, EXP_Y+2, EXP_W-10, 30)

# 内部节点
proj = N(box_sm("Linear Projection", "Linear(in_dim, 256)", "#333"), S_TB, EXP_X+20, EXP_Y+40, 400, 38)

# TB Layer 1
tb1_y = EXP_Y + 95
sa1 = N(box_sm("Self-Attention (残差)", "x = x + RotaryMHA(LN(x))", "#333"), S_TB, EXP_X+20, tb1_y, 185, 45)
ff1 = N(box_sm("FFN (残差)", "x = x + FFN(LN(x))", "#333"), S_TB, EXP_X+220, tb1_y, 185, 45)
E(proj, sa1)
E(sa1, ff1)
N(f'&lt;div style=&quot;font-size:9px;color:#999;&quot;&gt;{esc("TransformerBlock Layer 1 (pre-LN)")}&lt;/div&gt;',
  S_TXT, EXP_X+20, tb1_y-15, 200, 14)

# TB Layer 2
tb2_y = EXP_Y + 160
sa2 = N(box_sm("Self-Attention (残差)", "同 Layer 1 结构", "#333"), S_TB, EXP_X+20, tb2_y, 185, 40)
ff2 = N(box_sm("FFN (残差)", "同 Layer 1 结构", "#333"), S_TB, EXP_X+220, tb2_y, 185, 40)
E(ff1, sa2)
E(sa2, ff2)
N(f'&lt;div style=&quot;font-size:9px;color:#999;&quot;&gt;{esc("TransformerBlock Layer 2 (pre-LN)")}&lt;/div&gt;',
  S_TXT, EXP_X+20, tb2_y-15, 200, 14)

# final_norm
fn = N(box_sm("final_norm", "LayerNorm(256)", "#333"), S_TB, EXP_X+100, EXP_Y+270, 220, 35)
E(ff2, fn)
N(f'&lt;div style=&quot;font-size:9px;color:#999;&quot;&gt;{esc("输出: [B, 168, 256]")}&lt;/div&gt;',
  S_TXT, EXP_X+100, EXP_Y+308, 200, 14)

# 2b. 其他编码器（折叠框）
other_x = [540, 720, 900, 1100]
other_data = [
    ("enc_price_rt", "Transformer", "Linear(1, 256) + TB x2 + final_norm", S_ENC),
    ("enc_load", "Transformer", "Linear(1, 256) + TB x2 + final_norm", S_ENC),
    ("enc_system", "Transformer", "Linear(2, 256) + TB x2 + final_norm", S_ENC),
    ("enc_calendar", "MLP", "Linear(6, 256) + GELU + final_norm", S_ENC_MLP),
]
enc_ids = []
for i, (name, kind, desc, sty) in enumerate(other_data):
    nid = N(box_sm(name, f"{kind}: {desc}", "#1565C0" if "Transformer" in kind else "#E65100"),
            sty, other_x[i], Y2+70, 170, 70)
    enc_ids.append(nid)

# 输入 → 编码器 连线
E(inp_ids[0], proj)    # DA → expanded enc
for i in range(4):
    E(inp_ids[i+1], enc_ids[i])

# ═══ modality_emb 标注 ═══
N(f'&lt;div style=&quot;font-size:9px;color:#999;font-style:italic;&quot;&gt;{esc("+ modality_emb [5, 256] 每流可学习模态标记")}&lt;/div&gt;',
  S_TXTC, 300, Y2+EXP_H+20, 800, 16)

# ═══ 3. 跨模态融合 (y=600) ═══
Y3 = Y2 + EXP_H + 45
N(f'&lt;div style=&quot;font-weight:bold;font-size:10px;color:#999;&quot;&gt;{esc("3. 跨模态融合")}&lt;/div&gt;',
  S_TXT, 10, Y3, 70, 20)

fuse = N(box("Cross-modal Fusion (跨模态注意力)", "concat 5流 (840 tokens) > TransformerBlock (RoPE=OFF) > final_norm", "#2E7D32"),
         S_FUSE, 200, Y3, 1000, 55)

fuse_mem = N(box_sm("Shared Memory h_i", "融合后的共享表示, K/V 供 Decoder 查询", "#2E7D32"),
             S_FUSE, 400, Y3+70, 600, 40)
E(fuse, fuse_mem)

# 编码器 → 融合 连线
E(fn, fuse)
for enc_id in enc_ids:
    E(enc_id, fuse)

# ═══ 4. 解码器 (y=770) ═══
Y4 = Y3 + 140
N(f'&lt;div style=&quot;font-weight:bold;font-size:10px;color:#999;&quot;&gt;{esc("4. 并行解码器")}&lt;/div&gt;',
  S_TXT, 10, Y4, 70, 20)

da_dec = N(box("DA 解码器 (48 Queries)", "Query Self-Attn (RoPE) + Cross-Attn + FFN + final_norm", "#7B1FA2"),
           S_DEC, 150, Y4, 440, 55)

rt_dec = N(box("RT 解码器 (24 Queries)", "独立解码器, 结构同 DA, 24 个滚动窗口", "#7B1FA2"),
           S_DEC, 800, Y4, 440, 55)

E(fuse_mem, da_dec, "K, V")
E(fuse_mem, rt_dec, "K, V")

# ═══ 5. 预测头 + 反归一化 (y=900) ═══
Y5 = Y4 + 90
N(f'&lt;div style=&quot;font-weight:bold;font-size:10px;color:#999;&quot;&gt;{esc("5. 预测头")}&lt;/div&gt;',
  S_TXT, 10, Y5, 70, 20)

head_da = N(box_sm("head_da", "Linear(256, 2) > 输出 pDA + pRT|DA", "#E65100"),
            S_HEAD, 200, Y5, 300, 45)
head_rt_act = N(box_sm("head_rt_action", "Linear(256, 1) > 每窗口第一步", "#E65100"),
                S_HEAD, 850, Y5, 200, 45)
head_rt_win = N(box_sm("head_rt_windows", "Linear(256, 4) > 完整窗口 H=4", "#E65100"),
                S_HEAD, 1080, Y5, 200, 45)

E(da_dec, head_da)
E(rt_dec, head_rt_act)
E(rt_dec, head_rt_win)

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

E(head_da, out_da)
E(head_da, out_rtda)
E(head_rt_act, out_rt)
E(head_rt_win, out_rtw)

# ═══ 底部注释 ═══
Y7 = Y6 + 70
N(f'&lt;div style=&quot;font-size:10px;color:#999;line-height:1.6;&quot;&gt;'
  f'{esc("TransformerBlock (pre-LN): Self-Attn 使用 RotaryMHA (RoPE + SDPA), FFN = Linear > GELU > Linear")}&lt;br/&gt;'
  f'{esc("融合层: 关闭 RoPE (A1), 使用 n_heads_fusion=4 (非 n_heads_enc)")}&lt;br/&gt;'
  f'{esc("解码器: Self-Attn 用 RotaryMHA, Cross-Attn 用 nn.MultiheadAttention (PyTorch 原生)")}'
  f'&lt;/div&gt;',
  S_TXTC, 100, Y7, 1200, 55)

# ═══ ASSEMBLE ═══
xml = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<mxfile host="km.sankuai.com" type="embed">\n'
    f'<diagram name="\u6a21\u578b\u67b6\u6784 v3" id="arch-v3">\n'
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

out_path = os.path.join(os.path.dirname(__file__), "docs", "model_architecture_v3.drawio")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(xml)
print(f"Written {out_path} ({len(xml)} bytes, {len(cells)} cells)")

zp = out_path.replace(".drawio", ".zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(out_path, "model_architecture_v3.drawio")
print(f"Zipped {zp}")
