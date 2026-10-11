# 示例策略

6 个日线 `matrix_native` 策略模板，覆盖均线多头发散、放量突破、金叉共振、均线粘合突破、RPS 强度、强势回踩。

使用：复制到 `data/strategies/custom/` 后在策略页点「重载」（或调用 `POST /api/strategies/reload`），即出现在策略池；参数可在策略设置中调整。

| 文件 | 策略 | 核心条件 |
|---|---|---|
| `custom_ma_bull_divergence.py` | 均线多头发散 | MA5>10>20>60，MA5-MA20 间距扩大，MA20 上行 |
| `custom_volume_breakout.py` | 放量突破 | 收盘创 60 日新高，量比≥2，涨幅 3%~9.5%，收在高位 |
| `custom_golden_cross.py` | 金叉共振 | MA5 上穿 MA20 + 近 3 日 MACD 金叉，收盘在 MA60 上 |
| `custom_ma_convergence.py` | 均线粘合突破 | 四线粘合 5 日后放量阳线站上四线 |
| `custom_rps_strength.py` | RPS 强度 | RPS120≥90，行业内 RPS60≥80，站上 MA60/MA120 |
| `custom_rps_pullback.py` | 强势回踩 | RPS120≥80 但 RPS20≤50，回踩 MA20 收阳 |

注意：这些是框架模板，未做参数优化。2024-10 ~ 2026-09 回测（10 只持仓、次日开盘成交）均跑输上证指数，
同期因子检验显示 RPS/动量在 A 股偏反转。实盘前请自行回测与调参；行业内 RPS 在行业归属存档开始前使用当前快照，存在未来函数偏差。
