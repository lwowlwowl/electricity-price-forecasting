# 论文补实验回收审计

- 总状态：通过
- 错误数：0
- 2025：保持锁定（仅当下表和错误列表没有相反证据）

## 预测产物

| 分组 | 市场/结构 | 任务 | seed | 节点 | MAE | SHA-256 |
|---|---|---|---:|---:|---:|---|
| 多 seed | ERCOT | da | 1 | 15 | 36.7035 | 790ee2c519e92f14bc4d43bc655c883259dd57d1bdf343e24b7fcb84bd6af270 |
| 多 seed | ERCOT | da | 2 | 15 | 37.3364 | 54f58a2637008201a84665b3bbc906b041aa35bb41c9bdb0ac3ca047bf0cfd33 |
| 多 seed | ERCOT | rt_at_da | 1 | 15 | 34.7720 | 88d0e3115a54a612efab169982b09dfe16d7842bdb4e636176614f15d5bdc302 |
| 多 seed | ERCOT | rt_at_da | 2 | 15 | 34.8882 | 84e5f9c3bdb9208103ac4170c7523a39de44276d98a02e848d6d16199e18192e |
| 多 seed | ERCOT | rt | 1 | 15 | 28.9861 | e67b4a5604e72bf7ab66b5761f38850cc1407aed7f11f0336feddf228b3d67e8 |
| 多 seed | ERCOT | rt | 2 | 15 | 29.0340 | ab2fb9258e168bc85f30b2f39f3c42ed5a68311f827a05a03409f0a6a0503629 |
| pooled XGBoost | CAISO | da | 0 | 3 | 19.9531 | 2fcce6bd5d944fd13a73411f72f223fcdd495087b5b8e15414d018416f0955cb |
| pooled XGBoost | CAISO | rt_at_da | 0 | 3 | 23.7239 | e140cd8179d60701745ea0fce5ee060302f41054864a02fcb9e53db4b92089c0 |
| pooled XGBoost | CAISO | rt | 0 | 3 | 9.8656 | f9e7d3e167b9fb5793db9a346f5bf6ff0f19d6d1c8f57ba3756159ff0179d725 |
| pooled XGBoost | ERCOT | da | 0 | 15 | 35.9000 | 45b33ac53481d10e5df011065def3b78adf0486dbbb090298d20d8e49a806268 |
| pooled XGBoost | ERCOT | rt_at_da | 0 | 15 | 35.9945 | ecce98e6bf621b55f3b754212a4c13b0f3a0cff50005488dfaa1d941703a3016 |
| pooled XGBoost | ERCOT | rt | 0 | 15 | 29.1097 | 7ffec90d88468a96056547316580f26a07ec328825fd7a320b4a7a5cfb208d2c |
| pooled XGBoost | NYISO | da | 0 | 11 | 10.8258 | 2cc39420d9e5320905834a9c294a0dba5eec98eb2f8782f7e71d53c0b7f8c075 |
| pooled XGBoost | NYISO | rt_at_da | 0 | 11 | 14.7366 | 40be260cbac5328dbc783abc8cb1308afdae74d08294782011ab4eeef148110b |
| pooled XGBoost | NYISO | rt | 0 | 11 | 6.7606 | 62ddd8540d813b3cbf668236c323556e8c3126c093001b58dc476afa3a41e496 |
| pooled XGBoost | PJM | da | 0 | 35 | 7.2549 | 23b34b80999bf70308ea88038082176acc6a6efd8b720d70d4900433c93bc438 |
| pooled XGBoost | PJM | rt_at_da | 0 | 35 | 10.2113 | 806f62b4d557328e7e08049fd5ce77a6ba0dcbbf3b836f6fec5445ceda102f7b |
| pooled XGBoost | PJM | rt | 0 | 35 | 6.4912 | 39dfe3c8e7a22daef7859297ddb50b8b0065fed5c55cf41cef84f5d19bc8961b |
| 结构消融 | all_f0 | da | 0 | 15 | 37.2103 | ba3a68a6390c96e1692d6089a015648cd311501d80ae1184c14261e9f10375ec |
| 结构消融 | all_f0 | rt | 0 | 15 | 29.3207 | 34ec133c420b69f132d3398121d8e0ca32d9649fb1028d4635f6aa6e7dc1d5b6 |
| 结构消融 | all_f1 | rt_at_da | 0 | 15 | 35.1351 | 197ee1e6967c3762beaeff82056497dbd2b0b57c8c41fd813a934e17e632f7d8 |

## ERCOT 同合同净收益

### multiseed

| 系统 | 日均净收益 | 7日块95%区间 |
|---|---:|---|
| seed0 | 377.8381252155746 | [151.12783273330467, 657.7054872329755] |
| seed1 | 266.1546694302914 | [117.85154135408847, 452.897553448223] |
| seed2 | 403.0974114333497 | [176.4763056125236, 681.9443137899766] |
| ensemble | 407.166707634525 | [174.85922669313166, 693.1729548319751] |

### ablation

| 系统 | 日均净收益 | 7日块95%区间 |
|---|---:|---|
| current | 377.8381252155746 | [151.12783273330467, 657.7054872329755] |
| all_f0 | 291.2374316687672 | [122.80727637009203, 493.94690810091896] |
| all_f1 | 344.2810774952493 | [132.13356010986996, 608.0569340344562] |
