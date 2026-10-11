/**
 * 个股日K指标布局 (主图叠加线 + 副图顺序), 全局一份持久化到 localStorage。
 *
 * 主图按「指标 id + 可见输出线」记录, 内置 MA/BOLL 与后续自定义指标走同一结构,
 * 这样只勾"知行.趋势线"时也能精确还原到单条线。
 */
import { storage } from '@/lib/storage'

export interface MainIndicatorSelection {
  id: string
  outputs: string[]
}

export interface ChartLayout {
  version: 1
  main: MainIndicatorSelection[]
  sub: string[]
}

export type MaKey = 'ma5' | 'ma10' | 'ma20' | 'ma30' | 'ma60' | 'ma120'

/** 均线定义 (key 与 /api/kline/daily 行字段同名); 图表线色与信息栏共用 */
export const MA_OUTPUTS: readonly { key: MaKey; label: string; color: string }[] = [
  { key: 'ma5', label: 'MA5', color: '#A1A1AA' },
  { key: 'ma10', label: 'MA10', color: '#3B82F6' },
  { key: 'ma20', label: 'MA20', color: '#F97316' },
  { key: 'ma30', label: 'MA30', color: '#10B981' },
  { key: 'ma60', label: 'MA60', color: '#8B5CF6' },
  { key: 'ma120', label: 'MA120', color: '#EC4899' },
]

// 改造前固定画这四条; 新增的 MA30 默认不打扰, 由用户按需勾选
export const DEFAULT_MA_LINES: readonly MaKey[] = ['ma5', 'ma10', 'ma20', 'ma60']

export const BUILTIN_MAIN_INDICATORS: readonly { id: string; label: string; outputs: readonly { key: string; label: string; color: string }[] }[] = [
  { id: 'ma', label: '均线', outputs: MA_OUTPUTS },
  { id: 'boll', label: '布林带', outputs: [{ key: 'boll', label: 'BOLL', color: '#E879F9' }] },
]

export const DEFAULT_CHART_LAYOUT: ChartLayout = {
  version: 1,
  main: [{ id: 'ma', outputs: [...DEFAULT_MA_LINES] }],
  sub: ['vol'],
}

/** 自定义副图指标的 layout.sub key 前缀: `ind:{indicatorId}` */
export const IND_SUB_PREFIX = 'ind:'
/** 与后端指标 id 规则一致 (^[a-z0-9_]{1,40}$) */
const INDICATOR_ID_RE = /^[a-z0-9_]{1,40}$/

export function indicatorSubKey(id: string): string {
  return `${IND_SUB_PREFIX}${id}`
}

/** `ind:{id}` → id; 非自定义副图 key 返回 null */
export function parseIndicatorSubKey(key: string): string | null {
  if (!key.startsWith(IND_SUB_PREFIX)) return null
  const id = key.slice(IND_SUB_PREFIX.length)
  return INDICATOR_ID_RE.test(id) ? id : null
}

function isValidSubKey(key: string, validSub: readonly string[]): boolean {
  return validSub.includes(key) || parseIndicatorSubKey(key) != null
}

/**
 * 清洗持久化数据: 结构不符回落默认, 剔除非法条目与未知副图。
 * 副图 `ind:{id}` 动态 key 与主图同理只校验格式, 不按已加载指标过滤。
 * 主图不按"当前已知指标"过滤 —— 自定义指标列表异步加载, 过早过滤会把用户的选择抹掉;
 * 未知 id 在渲染时自然不出线, 指标被删除时由删除流程清理。
 */
export function sanitizeChartLayout(raw: unknown, validSub: readonly string[]): ChartLayout {
  if (!raw || typeof raw !== 'object') return DEFAULT_CHART_LAYOUT
  const r = raw as Partial<ChartLayout>
  if (r.version !== 1 || !Array.isArray(r.main) || !Array.isArray(r.sub)) return DEFAULT_CHART_LAYOUT

  const main: MainIndicatorSelection[] = []
  const seen = new Set<string>()
  for (const item of r.main) {
    if (!item || typeof item.id !== 'string' || !Array.isArray(item.outputs) || seen.has(item.id)) continue
    const outputs = [...new Set(item.outputs.filter((o): o is string => typeof o === 'string'))]
    if (outputs.length === 0) continue
    seen.add(item.id)
    main.push({ id: item.id, outputs })
  }
  const sub = [...new Set(r.sub.filter((k): k is string => typeof k === 'string' && isValidSubKey(k, validSub)))]
  return { version: 1, main, sub }
}

export function visibleOutputs(layout: ChartLayout, id: string): string[] {
  return layout.main.find(m => m.id === id)?.outputs ?? []
}

/** 切换单条输出线; 指标最后一条线被取消时整项移除 */
export function toggleMainOutput(layout: ChartLayout, id: string, output: string): ChartLayout {
  const idx = layout.main.findIndex(m => m.id === id)
  if (idx < 0) return { ...layout, main: [...layout.main, { id, outputs: [output] }] }
  const cur = layout.main[idx]
  const outputs = cur.outputs.includes(output) ? cur.outputs.filter(o => o !== output) : [...cur.outputs, output]
  const main = [...layout.main]
  if (outputs.length === 0) main.splice(idx, 1)
  else main[idx] = { id, outputs }
  return { ...layout, main }
}

export function toggleSub(layout: ChartLayout, key: string): ChartLayout {
  const sub = layout.sub.includes(key) ? layout.sub.filter(k => k !== key) : [...layout.sub, key]
  return { ...layout, sub }
}

/** 布局中引用到的自定义指标 id (主图有可见线 + 副图胶囊), 去重排序; 内置主图组不计 */
export function layoutIndicatorIds(layout: ChartLayout, builtinIds: readonly string[] = BUILTIN_MAIN_INDICATORS.map(b => b.id)): string[] {
  const ids = new Set<string>()
  for (const m of layout.main) if (!builtinIds.includes(m.id)) ids.add(m.id)
  for (const k of layout.sub) {
    const id = parseIndicatorSubKey(k)
    if (id) ids.add(id)
  }
  return [...ids].sort()
}

/** 指标被删除: 从主图与副图中一并清除 */
export function removeIndicatorFromLayout(layout: ChartLayout, id: string): ChartLayout {
  const subKey = indicatorSubKey(id)
  return {
    ...layout,
    main: layout.main.filter(m => m.id !== id),
    sub: layout.sub.filter(k => k !== subKey),
  }
}

/** 删除流程直接改持久化布局 (指标库页没有挂载图表, 下次打开图表即读到清理后的布局) */
export function removeIndicatorFromStoredLayout(id: string, validSub: readonly string[]): void {
  const layout = loadChartLayout(validSub)
  const next = removeIndicatorFromLayout(layout, id)
  if (next.main.length !== layout.main.length || next.sub.length !== layout.sub.length) saveChartLayout(next)
}

export function loadChartLayout(validSub: readonly string[]): ChartLayout {
  return sanitizeChartLayout(storage.stockChartLayout.get(DEFAULT_CHART_LAYOUT), validSub)
}

export function saveChartLayout(layout: ChartLayout): void {
  storage.stockChartLayout.set(layout)
}
