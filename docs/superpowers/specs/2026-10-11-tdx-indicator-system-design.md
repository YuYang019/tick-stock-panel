# 通达信指标系统（自定义主图/副图指标）设计

## 定位
"指标" = 单只股票逐 bar 的时间序列，用通达信公式定义。可上图（主图叠加/副图），其输出线可在自定义信号、策略评分、回测中引用。与因子 DSL（截面因子）并存、分工明确；共用 Polars 计算与因子注册表物化路径，不形成第二套计算口径。

## 后端

### 1. 编译器 `backend/app/indicators/tdx/`
- `lexer.py`：NUMBER / STRING / IDENT（含中文，统一大写）/ 运算符 `+ - * / > < >= <= = <> != AND OR NOT && ||` / `( ) , ; : :=`。`{...}` 注释跳过；全角 `；：，（）` 归一；未知字符报错（带行列），不静默丢弃。
- `parser.py`（递归下降）：语句 = `NAME : expr [, modifier]* ;` | `NAME := expr ;` | `expr [, modifier]* ;`（匿名输出，名称自动 `OUT1..`）。优先级 OR < AND < 比较 < 加减 < 乘除 < 一元(-, NOT) < 调用/原子。修饰符：`COLORxxx`（具名色 RED/GREEN/BLUE/YELLOW/WHITE/BLACK/MAGENTA/CYAN/GRAY/LIGRAY/LIRED/LIGREEN/LIBLUE + `COLOR` 后接 6 位 BBGGRR 十六进制，按通达信 BGR 顺序转 #RRGGBB）、`LINETHICK0-9`、`NODRAW`、`DOTLINE`、`STICK`、`COLORSTICK`、`VOLSTICK`、`CIRCLEDOT`、`POINTDOT`。未知修饰符→警告忽略。
- `compiler.py`：AST → `pl.Expr`，时序函数一律 `.over("symbol")`（帧按 symbol,date 排序）。
  - 变量：`O/OPEN H/HIGH L/LOW C/CLOSE V/VOL/VOLUME AMOUNT`→`open/high/low/close/volume/amount`（enriched 前复权价，与图表一致）。
  - 函数（参数 N 必须是常数整数 1..512，否则编译错误）：
    `MA, EMA(ewm_mean span=N adjust=False), SMA(X,N,M)(ewm_mean alpha=M/N adjust=False), WMA, REF, HHV, LLV, SUM(N=0 为累加), STD(样本), AVEDEV, COUNT, EVERY, EXIST, CROSS(A,B)=A>B & REF(A,1)<=REF(B,1), LONGCROSS 不支持, IF/IFF, MAX, MIN, ABS, ROUND, SQRT, POW, LOG, BARSLAST(cum 计数实现), BARSCOUNT, BETWEEN, FILTER 不支持→错误`。
  - 不可表达（`DRAWTEXT, DRAWTEXT_FIX, DRAWICON, DRAWNUMBER, STRCAT, VAR2STR, HYBLOCK, DYBLOCK, GNBLOCK, ISLASTBAR, BACKSET` 等）：所在语句**跳过并给警告**，被跳过语句定义的变量若被后续引用则报错。`STICKLINE(COND,P1,P2,W,EMPTY)` 输出 `kind=stick`（两列 p1/p2 + cond）——第一期支持。
  - 中间变量 `:=` 以 CTE 方式内联为表达式（不物化列）；表达式深度 ≤ 32，语句数 ≤ 64，公式长度 ≤ 8000。
- 结果：
  ```python
  @dataclass(frozen=True)
  class IndicatorOutput:
      name: str            # 原名, 如 "知行短期趋势线"
      key: str             # 列安全键: 小写 ascii 或 o{序号}, 用于列名
      expr: pl.Expr | None # stick 时为 None, 见 stick 字段
      kind: Literal["line", "stick", "colorstick", "volstick", "dot"]
      color: str | None; thick: int; dashed: bool; visible: bool   # NODRAW → visible=False (仍可被引用)
      stick: tuple[pl.Expr, pl.Expr, pl.Expr, float, bool] | None   # cond,p1,p2,width,empty
  @dataclass(frozen=True)
  class CompiledIndicator:
      ok: bool; errors: list[TdxError]; warnings: list[TdxError]   # TdxError(code, message, line, col)
      outputs: list[IndicatorOutput]; warmup_bars: int; dependencies: frozenset[str]
  def compile_indicator(text: str) -> CompiledIndicator          # 不抛异常
  def compile_indicator_cached(text: str) -> CompiledIndicator   # lru_cache
  def evaluate(compiled, frame: pl.DataFrame) -> pl.DataFrame    # 追加各输出列 __ind_{key}
  ```
  `warmup_bars` = 依赖链窗口长度之和的保守估计（EMA/SMA 按 4×N 计以收敛）。

### 2. 存储 `backend/app/indicators/tdx/store.py`
- `data/user_data/indicators/{id}.json`：`{id, name, formula, pane: "main"|"sub", description, builtin, created_at, updated_at}`。id `^[a-z0-9_]{1,40}$`；保存前编译，`ok=False` 拒绝保存；路径校验防穿越（参照 `factors/store.py`）。
- 预设种子（`seed_if_empty`，启动时调用，参照 `custom_signals.seed_if_empty`）：
  - `zhixing` 知行（main）：用户原文（含 `DRAWTEXT_FIX`/`STRCAT` 行，验证可容错）：
    ```
    知行短期趋势线:EMA(EMA(C,10),10),COLORWHITE,LINETHICK1;
    知行多空线:(MA(CLOSE,14)+MA(CLOSE,28)+MA(CLOSE,57)+MA(CLOSE,114))/4,COLORYELLOW;
    ```
  - 预设 `builtin=True`：可编辑（另存为新 id）不可删除。

### 3. 注册为因子（打通信号/策略/回测）
- 新增 `FactorSpec.kind = "indicator"`。每个已保存指标的每个 line 类输出注册一个因子：id `ind_{indicator_id}_{output.key}`，label `{指标名}.{输出名}`，group `"自定义指标"`，`formula_text` 存指标 id+输出 key（或完整公式），`warmup_bars` 取编译结果，`scale_free=False`，`incremental_safe=False`。
- `strategy/scoring.py:materialize_scoring_columns` 增加 `kind=="indicator"` 分支：按指标 id 分组，一次 `evaluate` 产出该指标全部输出，再 alias 成因子列。`scoring.py:63`、`backtest/factor.py:693` 中按 `("custom","composite")` 判定的地方同步纳入 `"indicator"`（逐处读代码确认语义后再改）。
- 保存/删除指标时注册/注销因子，并调用 `invalidate_custom_signals()` 等现有缓存失效函数。删除被信号引用的指标：参照删除被引用因子的现有处理（若无则拒绝删除并返回引用方列表）。
- 盘中当日路径：沿用现有"因子依赖信号在单日快照缺列告警跳过"的降级语义（`custom_signals.allowed_fields` docstring），不做额外实现；前端信号编辑器选到指标字段时提示"盘中实时预警不支持，收盘后计算"。
- `strategy/prompts/strategy-guide.md` 补一小节：策略中通过 `REQUIRED_FEATURES`/评分字段引用 `ind_*` 列。

### 4. API `backend/app/api/indicators.py`，prefix `/api/indicators`
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `` | 列表：`[{id,name,pane,builtin,description,formula,outputs:[{name,key,kind,color,thick,dashed,visible}],warnings}]` |
| GET | `/functions` | 函数/变量/修饰符清单 `{functions:[{name,signature,desc}],variables:[...],modifiers:[...]}`（编辑器面板） |
| POST | `/validate` | body `{formula}` → `{ok,errors,warnings,outputs,warmup_bars}` |
| POST | `` | 新建 body `{id?,name,formula,pane,description?}`（无 id 时由 name 生成 slug 冲突加后缀） |
| PUT | `/{id}` | 更新（builtin 返回 400 提示另存） |
| DELETE | `/{id}` | builtin 400；被引用 409 + 引用方 |
| GET | `/series` | `symbol, ids(逗号分隔), start_date?, end_date?` → `{symbol, dates:[YYYY-MM-DD...], indicators:{id:{outputs:[{key,name,kind,color,thick,dashed,visible,values:[float|null...]} | stick:{p1:[],p2:[],cond:[bool]},width,empty]}}, errors:{id:msg}}` |
| POST | `/preview` | body `{formula, symbol, start_date?, end_date?}`，返回同 series 单指标结构（编辑器预览未保存公式） |
- `series`/`preview` 数据：与 `/api/kline/daily` 同一读取路径（`repo.get_daily_asset`；起点向前扩 `warmup_bars` 个交易日 + 余量），算完裁回区间；含当日 live candle（复用 `api/kline.py` 的合并逻辑，抽成共享函数而不是复制）。`dates` 与 `/api/kline/daily` rows 的日期一一对齐。NaN/inf → null。单次 ids ≤ 10。
- 路由在 `main.py` 注册；API 层保持薄，计算在 `indicators/tdx/service.py`。

### 5. 测试
- `tests/test_tdx_lexer_parser.py`：语句类型、修饰符、全角、注释、中文名、错误行列、一元负号。
- `tests/test_tdx_functions.py`：每个函数固定样本数值断言；EMA/SMA 与手算递推一致（首值 = 首个 X）；CROSS/BARSLAST/COUNT 边界；多 symbol 帧不串号。
- `tests/test_tdx_zhixing.py`：知行原文编译 ok、2 个输出、警告含被跳过的 DRAWTEXT_FIX；数值与 `EMA(EMA(C,10),10)`、四均线均值的独立 numpy 实现一致。
- `tests/test_indicators_api.py`：CRUD、builtin 保护、validate、series 对齐 kline 日期与 warmup 裁剪、ids 上限、空数据。
- `tests/test_custom_signals_indicator_conditions.py`：参照 `test_custom_signals_factor_conditions.py`，信号 `ind_zhixing_x 上穿/大于 ind_zhixing_y` 在历史路径计算正确，盘中路径优雅跳过。

## 前端
- `lib/api.ts` 类型与请求：`listIndicators/getIndicatorFunctions/validateIndicator/createIndicator/updateIndicator/deleteIndicator/indicatorSeries/previewIndicator`；`lib/queryKeys.ts` 新增 `indicators.list`、`indicators.series(symbol, ids, start, end)`。
- K 线（`StockDailyKChart` + `EChartsCandlestick`）：
  - 「主图 ▾」下拉在内置组之后列出 `pane=main` 的自定义指标，逐输出线复选，选择写入 `lib/chartLayout.ts` 的 `layout.main`（已持久化）；`visible=false`(NODRAW) 输出不出现在下拉。
  - 副图：`pane=sub` 的指标作为副图胶囊出现，key `ind:{id}`，写入 `layout.sub`（`SUB_KEYS` 校验需纳入动态 key —— 对 `ind:` 前缀放行）。
  - 只为布局中可见的指标发 `/series` 请求；EChartsCandlestick 新增 `overlays` prop（主图叠加 line/stick 系列，按 `dates` 对齐 `data`），副图指标以动态 `SubChartDef` 渲染；信息栏显示可见输出当前值。
  - 指标被删除后，下拉不再列出，布局中残留 id 渲染时忽略；删除成功时从布局中清除该 id。
- 指标库页 `pages/indicators/`（路由 + 导航入口，参照 factors 页的注册方式）：列表（名称、主/副图、输出线、预设标记、操作）+ 编辑器（textarea 公式、函数面板点击插入、`/validate` 防抖校验与错误行定位、警告列表、选择股票后 `/preview` 实时预览小 K 线、保存/另存为）。
- 自定义信号编辑器字段下拉：新增"自定义指标"分组（来自因子注册表 group），选中时显示盘中降级提示。
- vitest：chartLayout 动态副图 key、series→ECharts 系列映射纯函数；`pnpm build`。

## 不做（第一期）
分钟/周月线指标、跨周期引用、DRAWTEXT 文本绘制、参数化指标（N 可调）、导入 .tn6 文件。
