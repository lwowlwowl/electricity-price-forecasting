# 公开EXAA数据

- `EXAA_AT_2025.csv`：Salzburg AG公开的2025年EXAA历史CSV，原始链接：
  <https://www.salzburg-ag.at/content/dam/web18/dokumente/strom/flexspot/EXAA_AT_2025.csv>
- `EXAA_AT_2025_2026_hourly.csv`：由
  `scripts/decision_aware/download_austria_exaa.py`生成。2025年读取上述官方CSV；
  2026年读取Salzburg AG FlexSpot官方网页使用的公开接口，并按网页公开
  税费公式反推EXAA原价。公开API key只在运行时从网页读取，不持久化。

SHA-256：

| 文件 | SHA-256 |
|---|---|
| `EXAA_AT_2025.csv` | `a6fcb86eef6f70349447660e7f9166f1804355ff9a1cf8e4dd4a920bb359a656` |
| `EXAA_AT_2025_2026_hourly.csv` | `d6f142cec9e79bb361bb81552c87672e465ed8ac0cb56584ef699ad089bd1c35` |

合并后数据为连续UTC小时轴，单位`EUR/MWh`；春季和秋季夏令时切换先按
`Europe/Vienna`本地时间解析，再转UTC，不删除或伪造交易小时。

## 因果可用时点

EXAA官方说明其独立日前拍卖在10:15完成，并将该结果称为随后交易日的首个
价格信号；欧洲市场耦合拍卖在12:00进行。因此本项目只把10:15 EXAA价格作为
随后奥地利DA预测的`past_future`输入：

- <https://www.exaa.at/en/energytrading/handel-mit-exaa/>
- <https://www.exaa.at/site/assets/files/1/3_04_trading_rules_spot_market_products_electric_power.pdf>
- <https://www.e-control.at/de/strom/auffangversorgung>（明确说明奥地利EXAA价格为10:15拍卖）
