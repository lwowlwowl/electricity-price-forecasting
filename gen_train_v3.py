#!/usr/bin/env python3
"""生成 v3 训练闭环图 — v2 清爽风格，中文。

布局思路（对标 v2 参考图的纵向单列流）：
  - 纯纵向：输入 → 模型 → 预测 → 左列L_pred / 右列ZO → Total Loss → 回传
  - ZO 大虚线框包含 Policy+BESS+ZO估计 三步，体现从属关系
  - Oracle 从 BESS 侧连出，标注"同结构，用真实电价"
  - 连线尽量避免交叉：左列红色，右列紫色，纵向不跨列
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

PW, PH = 1300, 1200

# 中轴
CX = 500

# ═══ TITLE ═══
N(f'&lt;div style=&quot;font-weight:bold;font-size:16px;color:#6A1B9A;&quot;&gt;{esc("正式实验版 (Full) — Decision-aware 训练回路 (双结算)")}&lt;/div&gt;'
  f'&lt;div style=&quot;font-size:10px;color:#888;margin-top:3px;&quot;&gt;{esc("零阶梯度 (ZO) + HardTopK 策略 (不可微) + BESS 双结算仿真 | alpha/beta 退火")}&lt;/div&gt;',
  S_TXTC, 100, 15, 1100, 50)

# ═══ ROW 1: 多模态输入 (y=85) ═══
inp = N(box("多模态输入 X_i", "历史电价/负荷 + 系统 + 日历 (5流)", "#1565C0"),
        S_BLUE, CX-175, 85, 350, 50)

# ═══ ROW 2: 模型 (y=175) ═══
model = N(box("Full Model f_theta", "多流 Encoder + Cross Attention + 并行 Query Decoder", "#1565C0"),
          S_BLUE, CX-200, 175, 400, 55)
E(inp, model)

# ═══ ROW 3: 预测输出 — 居中一个框 (y=275) ═══
pred = N(box3("预测输出",
              "p_DA [B,48]  p_RT|DA [B,48]  p_RT [B,24]  p_RT_windows [B,24,4]",
              "真实电价尺度 (反归一化 + clamp)", "#1565C0"),
         S_BLUE, CX-225, 275, 450, 55)
E(model, pred)

# ═══ ROW 4: 左右两列分叉 ═══
# 左列 x=80, 右列中心 x=750
LX = 80     # 左列起始
RX = 520    # 右列虚线框起始

# ── 左列: 预测损失 (y=390) ──
lpred = N(box3("预测损失 L_pred",
               "v8: Huber (delta=1.0) | 默认: half_se 0.5*MSE (fp32)",
               "L_DA + L_RT|DA + L_RT (由 use_mse_loss 切换)", "#C62828"),
          S_RED, LX, 400, 300, 60)

# 预测输出 → L_pred (红色，从左侧出)
E(pred, lpred, "预测值", ER)

# 真实电价 targets (放在左列上方)
tgt = N(box("真实电价 targets", "price_da_tgt, price_rt_tgt", "#E65100"),
        S_ORAN, LX, 330, 300, 45)
E(tgt, lpred, "", EA)

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
                "torch.topk 选 Top-K 放电 / Bot-K 充电 (不可微)",
                "价差门控: 仅保留 (c_dis - c_chg) > kappa/eta (约 28.4) 的配对", "#7B1FA2"),
           S_PURP, RX+20, ZO_BOX_Y+30, 330, 60)

E(pred, policy, "d = p_DA - p_RT|DA (定 u_DA); p_RT (定 u_RT)", EP)

# Step 2: BESS 仿真
bess = N(box3("2. BESS 双结算仿真",
              "R = DA_leg + RT_leg - degradation - deviation_penalty",
              "退化 = kappa * (d_act + c_act); 偏差罚金 (若启用) = 2|p_RT|*超3%部分", "#2E7D32"),
         S_GREEN, RX+380, ZO_BOX_Y+35, 310, 55)

E(policy, bess, "u_DA [B,48], u_RT [B,24]", EP)

# targets → BESS
E(tgt, bess, "真实电价", EA)

# Step 3: ZO 估计 (包在大框内, 在 Policy+BESS 下面)
zo = N(box3("3. 零阶梯度估计",
            "双点高斯平滑: 扰动预测 > 策略 > 仿真 > 收益差",
            "对 DA / RT|DA / RT 三个任务独立扰动, K=2 对, rho=0.05", "#5E35B1"),
       S_PURP, RX+20, ZO_BOX_Y+115, 670, 55)

# 标注 ZO 内部循环关系：ZO 调用上面的 step1+step2
N(f'&lt;div style=&quot;font-size:9px;color:#AB47BC;font-style:italic;&quot;&gt;{esc("ZO 内部: 每个扰动样本 p+/p- 重复调用上面的 步骤1+步骤2 计算 R+/R-")}&lt;/div&gt;',
  S_TXT, RX+20, ZO_BOX_Y+95, 670, 16)

# Step 4: L_proxy
proxy = N(box3("4. 代理损失 L_proxy",
               "L_proxy = p_hat * stopgrad(g_zo)",
               "梯度注入: grad(L_proxy) = g_zo, 链式法则传回模型", "#5E35B1"),
          S_PURP, RX+20, ZO_BOX_Y+200, 670, 55)

E(zo, proxy, "g_DA, g_RT|DA, g_RT", EP)

# ── Oracle (在 BESS 右侧) ──
oracle = N(box3("Oracle 收益 R*_i",
                "用真实电价跑同一 LP 结构 (无梯度)",
                "regret = R* - R_model | 仅日志, 不参与梯度回传 (--no-oracle-train 时训练跳过)", "#546E7A"),
           S_GREY_D, RX+380, ZO_BOX_Y+280, 310, 55)
E(bess, oracle, "同结构", EO)

# ═══ ROW 5: Total Loss (y=790) ═══
Y5 = ZO_BOX_Y + ZO_BOX_H + 20
total = N(box("Total Loss",
              "L = alpha * L_pred / pred_scale + beta * L_proxy / proxy_scale", "#C62828"),
          S_YELL, CX-200, Y5, 400, 50)
E(lpred, total, "alpha", ER)
E(proxy, total, "beta", EP)

N(f'&lt;div style=&quot;font-size:9px;color:#999;&quot;&gt;{esc("退火: pretrain 8轮 (alpha=1, beta=0) > 线性退火 12轮 > alpha=0.5, beta=0.5 (v8)")}&lt;/div&gt;',
  S_TXTC, CX-200, Y5+52, 400, 16)

# ═══ ROW 6: autograd (y=870) ═══
Y6 = Y5 + 75
optim = N(box("标准 autograd 回传",
              "AdamW + grad_clip + AMP autocast", "#37474F"),
          S_GREY, CX-150, Y6, 300, 50)
E(total, optim)

# 回传虚线箭头
E(optim, model, "更新 f_theta", ED)

# ═══ 底部注释 ═══
Y7 = Y6 + 80
N(f'&lt;div style=&quot;font-size:10px;color:#999;line-height:1.6;&quot;&gt;'
  f'{esc("核心: 策略 pi 不可微 (HardTopK argsort), 用零阶梯度 (Nesterov-Spokoiny 2017) 估计代理梯度, 通过 L_proxy 注入 autograd")}&lt;br/&gt;'
  f'{esc("DA/RT|DA 用 per_sample_sum (点积), RT 用 .mean() (1/24 平均) | BESS: 1MW/4MWh, eta=0.95, kappa=27$/MWh")}'
  f'&lt;/div&gt;',
  S_TXTC, 100, Y7, 1100, 40)

# ═══ ASSEMBLE ═══
xml = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<mxfile host="km.sankuai.com" type="embed">\n'
    '<diagram name="\u8bad\u7ec3\u56de\u8def v3" id="train-v3">\n'
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

out_path = os.path.join(os.path.dirname(__file__), "docs", "training_loop_v3.drawio")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(xml)
print(f"Written {out_path} ({len(xml)} bytes, {len(cells)} cells)")

zp = out_path.replace(".drawio", ".zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(out_path, "training_loop_v3.drawio")
print(f"Zipped {zp}")
