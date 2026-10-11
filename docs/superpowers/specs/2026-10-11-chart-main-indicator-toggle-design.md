# 个股日K 主图指标逐线开关 + 布局持久化

## 目标
主图叠加线可逐条开关（MA5/10/20/30/60、BOLL），主图与副图选择全局持久化到浏览器 localStorage，下次打开任意个股保持上次选择。结构预留给自定义指标（知行等）复用。

## 实现
- `frontend/src/lib/chartLayout.ts`：`ChartLayout {version:1, main:[{id, outputs[]}], sub[]}`；`MA_OUTPUTS`（均线 key/标签/颜色单一来源）、`DEFAULT_CHART_LAYOUT`（MA5/10/20/60 + 成交量，与改造前一致）、`sanitizeChartLayout`、`toggleMainOutput`、`toggleSub`、`load/saveChartLayout`。
- `storage.stockChartLayout` → localStorage 键 `stock_chart_layout`。
- 清洗策略：结构/版本不符回落默认；剔除非法条目、重复项、未知副图。**主图不按当前已知指标过滤**——自定义指标列表异步加载，过早过滤会抹掉用户选择；未知 id 渲染时不出线，由指标删除流程负责清理。
- `EChartsCandlestick`：新增 `maLines` prop（`showMA` 仍为总开关，`Indices` 页等调用方不受影响），均线 series 与信息栏按 `maLines` 渲染；新增 MA30。
- `StockDailyKChart`：工具栏首位新增「主图 ▾」下拉（逐线复选，点外部/Esc 关闭），BOLL 从胶囊移入下拉；副图胶囊改读写 `layout.sub`。

## 范围外
- `stock-analysis/AnalysisKChart.tsx`（个股分析页价位图）为独立组件，本期不接入。
- 同页多个日K实例不实时同步，重新挂载时读取最新布局。

## 验证
`vitest run src/lib/chartLayout.test.ts`（默认值、保存/重载、取消至空、脏数据），`pnpm build`。
