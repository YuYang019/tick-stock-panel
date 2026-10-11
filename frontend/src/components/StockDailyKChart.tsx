import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api, type IndicatorItem, type KlineRow } from '@/lib/api'
import { klineDailyQueryOptions } from '@/lib/kline'
import { QK } from '@/lib/queryKeys'
import { storage } from '@/lib/storage'
import {
  BUILTIN_MAIN_INDICATORS,
  indicatorSubKey,
  layoutIndicatorIds,
  loadChartLayout,
  saveChartLayout,
  toggleMainOutput,
  toggleSub,
  visibleOutputs,
  type ChartLayout,
  type MaKey,
} from '@/lib/chartLayout'
import {
  EChartsCandlestick,
  INDICATOR_SUB_HEIGHT,
  SUB_CHARTS,
  resolveSubChartDefs,
  subChartsTotalHeight,
  type ChartMarker,
  type IndicatorOverlay,
  type SubChartDef,
  type ChartPriceLine,
  type ChartRange,
  type OHLC,
  type VolumeCompareConfig,
} from '@/components/EChartsCandlestick'

const DEFAULT_VOLUME_COMPARE: VolumeCompareConfig = { enabled: true, days: 1 }
const SUB_KEYS = SUB_CHARTS.map(s => s.key)
/** 后端 /series 单次 ids 上限 */
const MAX_SERIES_IDS = 10

interface MenuGroup {
  id: string
  label: string
  outputs: readonly { key: string; label: string; color?: string | null }[]
}

/** 自定义主图指标 → 下拉分组; NODRAW (visible=false) 输出不出现 */
function customMainGroups(indicators: readonly IndicatorItem[]): MenuGroup[] {
  return indicators
    .filter(ind => ind.pane === 'main')
    .map(ind => ({
      id: ind.id,
      label: ind.builtin ? `${ind.name} · 预设` : ind.name,
      outputs: ind.outputs.filter(o => o.visible).map(o => ({ key: o.key, label: o.name, color: o.color })),
    }))
    .filter(g => g.outputs.length > 0)
}

function normalizeVolumeCompare(config: VolumeCompareConfig): VolumeCompareConfig {
  return {
    enabled: config.enabled !== false,
    days: Math.max(1, Math.min(20, Math.round(Number(config.days) || 1))),
  }
}

interface Props {
  symbol: string
  height?: number
  className?: string
  dateRange?: { start: string; end: string }
  markers?: ChartMarker[]
  ranges?: ChartRange[]
  priceLines?: ChartPriceLine[]
  showLimitMarkers?: boolean
  showIndicatorControls?: boolean
  showMarkerToggle?: boolean
  showMA?: boolean
  showInfoBar?: boolean
  /** 初始可见蜡烛根数; 'all' = 适配显示全部数据 */
  visibleBars?: number | 'all'
  linkedPrice?: number | null
  onDateClick?: (date: string) => void
  onPriceDoubleClick?: (price: number, currentPrice: number) => void
  /** 扩展数据列参数（逗号分隔 config_id.field_name），透传给 klineDaily 接口 */
  extColumns?: string
  /** 加入自选日 (北京时间 YYYY-MM-DD); 有值时日K主图绘制「自选」竖虚线 */
  addedDate?: string | null
}

function isValidRow(r: any): boolean {
  return r && r.date != null && r.open != null && r.close != null
}

export function toOHLC(rows: KlineRow[]): OHLC[] {
  return rows
    .filter(isValidRow)
    .map(r => ({
      date: typeof r.date === 'string' ? r.date.slice(0, 10) : String(r.date),
      open: Number(r.open),
      high: Number(r.high),
      low: Number(r.low),
      close: Number(r.close),
      volume: Number(r.volume ?? 0),
      ma5: r.ma5 != null ? Number(r.ma5) : null,
      ma10: r.ma10 != null ? Number(r.ma10) : null,
      ma20: r.ma20 != null ? Number(r.ma20) : null,
      ma30: r.ma30 != null ? Number(r.ma30) : null,
      ma60: r.ma60 != null ? Number(r.ma60) : null,
      ma120: r.ma120 != null ? Number(r.ma120) : null,
      macd_dif: r.macd_dif != null ? Number(r.macd_dif) : null,
      macd_dea: r.macd_dea != null ? Number(r.macd_dea) : null,
      macd_hist: r.macd_hist != null ? Number(r.macd_hist) : null,
      rsi_6: r.rsi_6 != null ? Number(r.rsi_6) : null,
      rsi_14: r.rsi_14 != null ? Number(r.rsi_14) : null,
      rsi_24: r.rsi_24 != null ? Number(r.rsi_24) : null,
      kdj_k: r.kdj_k != null ? Number(r.kdj_k) : null,
      kdj_d: r.kdj_d != null ? Number(r.kdj_d) : null,
      kdj_j: r.kdj_j != null ? Number(r.kdj_j) : null,
      boll_upper: r.boll_upper != null ? Number(r.boll_upper) : null,
      boll_lower: r.boll_lower != null ? Number(r.boll_lower) : null,
    }))
}

function buildLimitUpMarkers(rows: KlineRow[]): ChartMarker[] {
  const markers: ChartMarker[] = []
  for (const r of rows) {
    const date = typeof r.date === 'string' ? r.date.slice(0, 10) : String(r.date)
    if (r.signal_broken_limit_up) {
      markers.push({ date, kind: 'neutral', above: true, color: '#8B5CF6', label: '炸' })
    } else if (r.signal_limit_up) {
      const boards: number = r.consecutive_limit_ups ?? 1
      markers.push({ date, kind: 'buy', above: true, color: '#FACC15', label: boards <= 1 ? '板' : String(boards) })
    }
  }
  return markers
}

export function getDefaultRange(): { start: string; end: string } {
  const now = new Date()
  const end = now.toISOString().slice(0, 10)
  const s = new Date(now)
  s.setMonth(s.getMonth() - 6)
  const start = s.toISOString().slice(0, 10)
  return { start, end }
}

/** 工具栏小胶囊开关（指标与标注开关共用一套尺寸/关闭态样式，激活色由调用方给字面量）。 */
function ChartPill({
  active, label, activeClass, title, onClick, buttonRef, ariaProps,
}: {
  active: boolean
  label: string
  activeClass: string
  title?: string
  onClick: () => void
  buttonRef?: React.Ref<HTMLButtonElement>
  /** 覆盖默认 aria-pressed (如下拉触发器用 aria-haspopup/aria-expanded) */
  ariaProps?: React.AriaAttributes
}) {
  return (
    <button
      ref={buttonRef}
      type="button"
      onClick={onClick}
      title={title}
      {...(ariaProps ?? { 'aria-pressed': active })}
      className={`inline-flex h-5 items-center px-2 rounded text-[10px] leading-none font-mono cursor-pointer transition-colors ${
        active ? activeClass : 'bg-elevated text-muted hover:text-secondary'
      }`}
    >
      {label}
    </button>
  )
}

/** 主图叠加指标下拉: 逐条输出线勾选 (内置 MA/BOLL 在前, pane=main 的自定义指标在后) */
function MainIndicatorMenu({
  layout, showMA, customGroups, customLoading, onToggle,
}: {
  layout: ChartLayout
  showMA: boolean
  customGroups: readonly MenuGroup[]
  customLoading: boolean
  onToggle: (id: string, output: string) => void
}) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    // 打开即把焦点移到首个复选项, 键盘可直接操作
    menuRef.current?.querySelector<HTMLInputElement>('input[type="checkbox"]')?.focus()
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false)
        triggerRef.current?.focus()
      }
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  const builtinGroups: MenuGroup[] = BUILTIN_MAIN_INDICATORS.filter(g => showMA || g.id !== 'ma')
  const groups = [...builtinGroups, ...customGroups]
  // 计数只算当前能渲染的线 (已删除指标的残留选择不计)
  const activeCount = groups.reduce(
    (n, g) => n + visibleOutputs(layout, g.id).filter(k => g.outputs.some(o => o.key === k)).length,
    0,
  )

  // 上下方向键在复选项间移动焦点 (role=menu 的键盘约定)
  const onMenuKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp' && e.key !== 'Home' && e.key !== 'End') return
    const items = Array.from(menuRef.current?.querySelectorAll<HTMLInputElement>('input[type="checkbox"]') ?? [])
    if (items.length === 0) return
    e.preventDefault()
    const cur = items.indexOf(document.activeElement as HTMLInputElement)
    let next = 0
    if (e.key === 'End') next = items.length - 1
    else if (e.key === 'ArrowDown') next = cur < 0 ? 0 : (cur + 1) % items.length
    else if (e.key === 'ArrowUp') next = cur <= 0 ? items.length - 1 : cur - 1
    items[next].focus()
  }

  // flex 包裹: 块级 div 内的行内按钮会带行框基线留白, 比同行胶囊高出一截
  return (
    <div ref={rootRef} className="relative flex items-center">
      <ChartPill
        buttonRef={triggerRef}
        active={activeCount > 0}
        label={`主图${activeCount > 0 ? ` ${activeCount}` : ''} ▾`}
        activeClass="bg-accent/20 text-accent"
        title="选择主图叠加指标"
        ariaProps={{ 'aria-haspopup': 'menu', 'aria-expanded': open }}
        onClick={() => setOpen(v => !v)}
      />
      {open && (
        <div
          ref={menuRef}
          role="menu"
          aria-label="主图指标"
          onKeyDown={onMenuKeyDown}
          className="absolute left-0 top-full z-20 mt-1 max-h-[60vh] min-w-[160px] overflow-y-auto rounded border border-border bg-base p-1.5 shadow-lg"
        >
          {groups.map((group, gi) => {
            const visible = visibleOutputs(layout, group.id)
            return (
              <div
                key={group.id}
                role="group"
                aria-label={group.label}
                className={`py-0.5 ${gi === builtinGroups.length && gi > 0 ? 'mt-0.5 border-t border-border/70 pt-1' : ''}`}
              >
                <div className="px-1 pb-0.5 text-[10px] text-muted" aria-hidden="true">{group.label}</div>
                {group.outputs.map(out => (
                  <label
                    key={out.key}
                    className="flex cursor-pointer items-center gap-1.5 rounded px-1 py-0.5 text-[11px] font-mono text-secondary hover:bg-elevated"
                  >
                    <input
                      type="checkbox"
                      role="menuitemcheckbox"
                      aria-checked={visible.includes(out.key)}
                      checked={visible.includes(out.key)}
                      onChange={() => onToggle(group.id, out.key)}
                      className="h-3 w-3 accent-current"
                    />
                    <span
                      className="inline-block h-0.5 w-3 shrink-0 rounded"
                      style={{ backgroundColor: out.color ?? 'currentColor' }}
                      aria-hidden="true"
                    />
                    <span className="truncate">{out.label}</span>
                  </label>
                ))}
              </div>
            )
          })}
          {customLoading && customGroups.length === 0 && (
            <div className="px-1 pt-1 text-[10px] text-muted" role="status">自定义指标加载中…</div>
          )}
        </div>
      )}
    </div>
  )
}

export function StockDailyKChart({
  symbol,
  height = 520,
  className,
  dateRange: externalDateRange,
  markers,
  ranges,
  priceLines,
  showLimitMarkers = true,
  showIndicatorControls = true,
  showMarkerToggle = true,
  showMA = true,
  showInfoBar = true,
  visibleBars = 60,
  linkedPrice,
  onDateClick,
  onPriceDoubleClick,
  extColumns,
  addedDate,
}: Props) {
  // 指标布局全局持久化 (localStorage): 下次打开任意个股仍按上次的主图/副图选择展示
  const [layout, setLayout] = useState<ChartLayout>(() => loadChartLayout(SUB_KEYS))
  const updateLayout = useCallback((fn: (prev: ChartLayout) => ChartLayout) => {
    setLayout(prev => {
      const next = fn(prev)
      saveChartLayout(next)
      return next
    })
  }, [])
  const maLines = useMemo(
    () => visibleOutputs(layout, 'ma') as MaKey[],
    [layout],
  )
  // EChartsCandlestick 以 activeIndicators 承载副图顺序 + BOLL 叠加开关
  const activeIndicators = useMemo(
    () => (visibleOutputs(layout, 'boll').length > 0 ? [...layout.sub, 'boll'] : layout.sub),
    [layout],
  )
  const [showMarkers, setShowMarkers] = useState(true)
  // 加入自选日标注（与「异动」标记相互独立）
  const [showAddedMark, setShowAddedMark] = useState(true)
  const [volumeCompare, setVolumeCompare] = useState<VolumeCompareConfig>(() =>
    normalizeVolumeCompare(storage.stockVolumeCompare.get(DEFAULT_VOLUME_COMPARE)),
  )
  const dateRange = externalDateRange ?? getDefaultRange()

  // 查询配置统一来自 klineDailyQueryOptions, 与 StockPanel 信息条/邻近预取共享同一 cache key (只发一次请求)
  const kline = useQuery({ ...klineDailyQueryOptions(symbol, dateRange, extColumns), enabled: !!symbol })

  const rows = useMemo(() => toOHLC(kline.data?.rows ?? []), [kline.data?.rows])
  const stockInfo = kline.data?.stock_info
  const limitMarkers = useMemo(() => buildLimitUpMarkers(kline.data?.rows ?? []), [kline.data?.rows])
  const allMarkers = useMemo(() => [
    ...(markers ?? []),
    ...(showLimitMarkers ? limitMarkers : []),
  ], [limitMarkers, markers, showLimitMarkers])

  // ===== 通达信自定义指标 =====
  // 列表只在有工具栏或布局里有自定义指标时需要; 布局里残留的已删除 id 由此过滤掉
  const layoutIds = useMemo(() => layoutIndicatorIds(layout), [layout])
  const indicatorList = useQuery({
    queryKey: QK.indicators.list,
    queryFn: api.listIndicators,
    enabled: !!symbol && (showIndicatorControls || layoutIds.length > 0),
    staleTime: 60_000,
  })
  const indicators = useMemo(() => indicatorList.data ?? [], [indicatorList.data])
  const indicatorById = useMemo(() => new Map(indicators.map(i => [i.id, i])), [indicators])
  const customGroups = useMemo(() => customMainGroups(indicators), [indicators])
  const subIndicators = useMemo(() => indicators.filter(i => i.pane === 'sub'), [indicators])

  // 只为"布局中可见 + 仍存在"的指标取数: 主图需至少一条勾选线, 副图需胶囊开启
  const visibleIds = useMemo(() => {
    const ids = layoutIds.filter(id => {
      const ind = indicatorById.get(id)
      if (!ind) return false
      if (ind.pane === 'sub') return layout.sub.includes(indicatorSubKey(id))
      const sel = visibleOutputs(layout, id)
      return ind.outputs.some(o => o.visible && sel.includes(o.key))
    })
    return ids.slice(0, MAX_SERIES_IDS)
  }, [layoutIds, indicatorById, layout])

  const series = useQuery({
    queryKey: QK.indicators.series(symbol, visibleIds, dateRange.start, dateRange.end),
    queryFn: () => api.indicatorSeries(symbol, visibleIds, dateRange),
    enabled: !!symbol && visibleIds.length > 0,
    // 勾选变化/刷新时保留上一份 (同 symbol) 叠加线, 避免闪烁; 切股不透传上一只的线
    placeholderData: (prev, prevQuery) => (prevQuery?.queryKey[2] === symbol ? prev : undefined),
    staleTime: 30_000,
    retry: 1,
  })

  const overlays = useMemo<IndicatorOverlay[]>(() => {
    const res = series.data
    if (!res || res.symbol !== symbol) return []
    const out: IndicatorOverlay[] = []
    for (const id of visibleIds) {
      const meta = indicatorById.get(id)
      const entry = res.indicators?.[id]
      if (!meta || !entry) continue
      const selected = meta.pane === 'main' ? visibleOutputs(layout, id) : null
      const outputs = entry.outputs.filter(o => o.visible && (selected == null || selected.includes(o.key)))
      if (outputs.length === 0) continue
      out.push({ id, name: meta.name, pane: meta.pane, dates: res.dates, outputs })
    }
    return out
  }, [series.data, symbol, visibleIds, indicatorById, layout])

  const indicatorErrors = useMemo(() => {
    const errs = series.data?.symbol === symbol ? series.data?.errors ?? {} : {}
    return Object.entries(errs)
      .filter(([id]) => visibleIds.includes(id))
      .map(([id, msg]) => `${indicatorById.get(id)?.name ?? id}: ${msg}`)
  }, [series.data, symbol, visibleIds, indicatorById])

  const toggleIndicator = useCallback((key: string) => {
    updateLayout(prev => toggleSub(prev, key))
  }, [updateLayout])
  const toggleMain = useCallback((id: string, output: string) => {
    updateLayout(prev => toggleMainOutput(prev, id, output))
  }, [updateLayout])

  const updateVolumeCompare = useCallback((patch: Partial<VolumeCompareConfig>) => {
    setVolumeCompare(prev => {
      const next = normalizeVolumeCompare({ ...prev, ...patch })
      storage.stockVolumeCompare.set(next)
      return next
    })
  }, [])

  // 与 EChartsCandlestick 同一套副图解析 (仅高度用, 系列由图表自建); 指标副图数据到达后才占位
  const subPlaceholders = useMemo<SubChartDef[]>(
    () => overlays.filter(o => o.pane === 'sub').map(o => ({
      key: indicatorSubKey(o.id), label: o.name, height: INDICATOR_SUB_HEIGHT,
      buildSeries: () => [], buildInfo: () => [],
    })),
    [overlays],
  )
  const activeSubDefs = resolveSubChartDefs(activeIndicators, subPlaceholders)
  const subExtraH = activeSubDefs.length > 0 ? subChartsTotalHeight(activeSubDefs) + 14 : 0
  const chartHeight = height + subExtraH

  if (!symbol) return null

  return (
    <div className={className} style={{ minHeight: chartHeight }}>
      {showIndicatorControls && rows.length > 0 && (
        <div className="flex items-center gap-1.5 px-1 pb-0.5">
          <MainIndicatorMenu
            layout={layout}
            showMA={showMA}
            customGroups={customGroups}
            customLoading={indicatorList.isLoading}
            onToggle={toggleMain}
          />
          {SUB_CHARTS.map(ind => (
            <ChartPill
              key={ind.key}
              active={layout.sub.includes(ind.key)}
              label={ind.label}
              activeClass="bg-accent/20 text-accent"
              onClick={() => toggleIndicator(ind.key)}
            />
          ))}
          {subIndicators.map(ind => {
            const key = indicatorSubKey(ind.id)
            return (
              <ChartPill
                key={key}
                active={layout.sub.includes(key)}
                label={ind.name}
                activeClass="bg-accent/20 text-accent"
                title={`副图指标 · ${ind.name}${ind.builtin ? ' (预设)' : ''}`}
                onClick={() => toggleIndicator(key)}
              />
            )
          })}
          {visibleIds.length > 0 && series.isFetching && (
            <span className="text-[10px] text-muted" role="status" aria-live="polite">指标计算中…</span>
          )}
          {layout.sub.includes('vol') && (
            <div className="ml-0.5 flex h-5 items-center gap-1.5 border-l border-border/70 pl-2">
              <span className="text-[10px] text-muted">量比</span>
              <button
                type="button"
                role="switch"
                aria-checked={volumeCompare.enabled}
                aria-label="开启量能对比"
                title={volumeCompare.enabled ? '关闭量能对比' : '开启量能对比'}
                onClick={() => updateVolumeCompare({ enabled: !volumeCompare.enabled })}
                className={`relative h-3.5 w-6 shrink-0 rounded-full transition-colors ${
                  volumeCompare.enabled ? 'bg-accent' : 'bg-elevated'
                }`}
              >
                <span className={`absolute left-0 top-0.5 h-2.5 w-2.5 rounded-full bg-white transition-transform ${
                  volumeCompare.enabled ? 'translate-x-3' : 'translate-x-0.5'
                }`} />
              </button>
              <select
                aria-label="量能对比周期"
                value={volumeCompare.days}
                disabled={!volumeCompare.enabled}
                onChange={event => updateVolumeCompare({ days: Number(event.target.value) })}
                className="h-5 rounded border border-border bg-base px-1 text-[10px] text-secondary outline-none disabled:opacity-40"
              >
                {Array.from({ length: 20 }, (_, index) => index + 1).map(days => (
                  <option key={days} value={days}>前{days}日均量</option>
                ))}
              </select>
            </div>
          )}
          {/* 两个标注开关统一右对齐 */}
          <div className="ml-auto flex items-center gap-1.5">
            {showMarkerToggle && showLimitMarkers && (
              <ChartPill
                active={showMarkers}
                label="异动"
                activeClass="text-[#FACC15] bg-[#FACC15]/10"
                onClick={() => setShowMarkers(v => !v)}
              />
            )}
            {/* 激活色与图表里的 ADDED_DATE_COLOR 一致; Tailwind 只认字面量类名, 故写死色值 */}
            {showMarkerToggle && addedDate && (
              <ChartPill
                active={showAddedMark}
                label="自选"
                activeClass="text-[#3B82F6] bg-[#3B82F6]/10"
                title={showAddedMark ? '隐藏K线上的加入自选日标注' : '显示K线上的加入自选日标注'}
                onClick={() => setShowAddedMark(v => !v)}
              />
            )}
          </div>
        </div>
      )}
      {/* 指标失败不影响 K 线: 图照常渲染 (无叠加), 这里只给一行小提示 */}
      {rows.length > 0 && visibleIds.length > 0 && (series.isError || indicatorErrors.length > 0) && (
        <div className="px-1 pb-0.5 text-[10px] text-danger" role="alert">
          {series.isError
            ? `指标加载失败${series.data ? '（显示上次结果）' : ''}: ${(series.error as Error)?.message ?? ''}`
            : `部分指标计算失败 · ${indicatorErrors.join('；')}`}
        </div>
      )}
      {kline.isLoading && <div className="text-sm text-muted py-4">加载中…</div>}
      {kline.isError && <div className="text-sm text-danger py-2">日K加载失败</div>}
      {!kline.isLoading && !kline.isError && (kline.data?.rows?.length ?? 0) > 0 && rows.length === 0 && (
        <div className="text-sm text-danger py-2">数据格式异常，请刷新页面</div>
      )}
      {rows.length > 0 && (
        <EChartsCandlestick
          data={rows}
          markers={allMarkers}
          ranges={ranges}
          priceLines={priceLines}
          height={chartHeight - 22}
          showMA={showMA}
          maLines={maLines}
          showInfoBar={showInfoBar}
          showMarkers={showMarkers}
          stockInfo={stockInfo}
          symbol={symbol}
          linkedPrice={linkedPrice}
          onDateClick={onDateClick}
          onPriceDoubleClick={onPriceDoubleClick}
          visibleBars={visibleBars}
          activeIndicators={activeIndicators}
          volumeCompare={volumeCompare}
          addedDate={showAddedMark ? addedDate : null}
          overlays={overlays}
        />
      )}
    </div>
  )
}
