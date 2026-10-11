# Tushare 数据源插件设计

## 目标
新增内置插件 `tushare`，按 token 实际权限声明数据集，无权限的数据集不声明（自动回退 TickFlow）。不引入 tushare SDK，直接 HTTP。

## 实测权限（2026-10-11，用户 token）
| 接口 | 结果 | 用途 |
|---|---|---|
| stock_basic / trade_cal | ✅ | instruments / 交易日 |
| daily / adj_factor / daily_basic | ✅ | daily / adj_factor / 换手率 |
| index_daily / fund_daily / fund_adj | ✅ | 指数、ETF 日K与因子 |
| income / balancesheet / cashflow / fina_indicator | ✅ | financial |
| stk_mins | ✅ | minute |
| share_float | ✅ | （暂不接入，股本表无可靠全量来源时留空） |
| rt_k | ❌ 40203 无权限 | realtime 不声明 |

## 结构
```
backend/app/plugins/tushare/
├── plugin.yaml   # name: tushare, runtime: none, api_key_env: TUSHARE_TOKEN,
│                 # datasets: [daily, adj_factor, minute, financial, realtime]（全集，运行时裁剪）
├── client.py     # TushareClient: POST http://api.tushare.pro {api_name, token, params, fields}
└── provider.py   # get_api_key / availability / probe_api_key / TushareProvider
```

## client.py
- `query(api_name, params, fields="") -> pl.DataFrame`：解包 `{code, msg, data:{fields, items}}`。
- 错误分类：`code==40203` → `TusharePermissionError`；`40101/token 无效` → `TushareAuthError`；限频（msg 含"每分钟最多访问"）→ 退避重试（最多 3 次）；其余 → `TushareError`。
- 限频：进程内按 api_name 的令牌桶，默认 200 次/分钟，可由 `TUSHARE_RPM` 环境变量覆盖。
- 单次返回行数有上限（daily 6000 行）：按日期区间分片 + 按 `offset` 分页，带页数上限。
- 日志与异常中不得包含 token。

## 权限探测
- `TushareProvider.__init__` 不发请求。首次访问 `config.datasets` 前（或 loader 注册时调用 `refresh_capabilities()`）对每个数据集的代表接口做 1 行试拉：
  daily→`daily`，adj_factor→`adj_factor`，minute→`stk_mins`，financial→`income`，realtime→`rt_k`。
- 仅 `TusharePermissionError` 视为"无权限"，网络错误视为"未知"并保留声明（避免网络抖动导致能力永久消失），下次探测再判定。
- 结果写 `data/cache/tushare/capabilities.json`（含 token 指纹 sha256 前 8 位 + 时间），TTL 24h；token 变更则失效。
- `probe_api_key(key)`：调用 `trade_cal` 1 行验证 token 有效。

## 数据集映射（遵守 docs/plugin-development.md 契约）
| 数据集 | 方法 | 来源 | 转换 |
|---|---|---|---|
| daily | `get_daily` / `iter_daily` | 股票 `daily`，ETF `fund_daily`，指数 `index_daily` | `ts_code`→symbol（已带后缀，BJ 同）；`trade_date` %Y%m%d→Date；`vol` 已是手；`amount` 千元 ×1000→元；输出走 `normalize_daily`。全市场窗口优先按 `trade_date` 逐日拉（每日一请求覆盖全市场），单标的长窗口按 `ts_code`+区间拉 |
| adj_factor | `get_adj_factors` | `adj_factor` / `fund_adj` | Tushare 为累积后复权因子；转换为项目单事件比值：`ex_factor = adj(t)/adj(t-1)`，仅输出比值≠1 的事件日。方向与日期对齐必须用测试对照 `pipeline._apply_adj_factor` 验证（前复权后最新价不变、除权日前后价格连续） |
| minute | `get_minute` | `stk_mins` freq=1min | `trade_time` 已为北京墙钟 naive；`vol` 为股 → //100 手；`amount` 元。声明 `minute_history_days` 不设（深历史） |
| financial | `get_financials(table, symbols, latest_only)` | income/balancesheet/cashflow/fina_indicator | 列名映射到 TickFlow canonical（参照 fuyao `_financial_statements`/`_financial_metrics` 的目标列）；只取 `report_type=1`（合并报表）；百分数指标 /100 与 fuyao 口径一致；`shares` 表返回空 DataFrame |
| instruments | `get_instruments` | `stock_basic`（L/D/P）+ `daily_basic` 最新日股本 | 产出与 TickFlow Instrument 同形的行（参照 instrument_sync flatten 期望字段） |
| realtime | — | rt_k 无权限 | 不实现；探测通过才声明（预留，第一期不实现方法，探测结果为可用时也不声明，直至实现） |

`test_dataset(dataset, symbols)` 对每个已声明数据集返回 `{provider, dataset, rows, columns, preview}`。

## 设置页
无需前端改动：`api_key_env` 声明后已有 Key 输入框（先探后存，写 `secrets.json` 的 `tushare_api_key`）。`description` 写明"数据集按 token 权限自动裁剪，实时行情需 rt_k 权限"。

## 测试 `backend/tests/test_tushare_provider.py`
以 `test_fuyao_provider.py` 为范本，假 client 注入，不依赖网络：
1. daily 单位：amount 千元→元、日期解析、ETF/指数路由到对应接口
2. adj_factor 累积→单事件转换：无事件输出空；有事件时 `_apply_adj_factor` 结果最新价不变、除权前后连续
3. minute：vol 股→手、datetime naive
4. financial：列映射、只取合并报表、shares 空
5. 权限裁剪：40203 → 数据集不声明；网络错误 → 保留；缓存 TTL 与 token 指纹失效
6. client：错误码分类、限频重试、分页终止与页数上限、异常信息不含 token
7. availability 两态、probe_api_key、loader 注册

## 文档
`docs/plugin-development.md` "现有插件参考"补 tushare 条目。
