/**
 * 通达信指标 /series 输出 → K 线图 (EChartsCandlestick) 系列的纯映射。
 *
 * - 对齐: 后端 dates 与 /api/kline/daily rows 理论上一一对应, 但两者是独立请求
 *   (切区间/盘中 live candle 先后到达), 这里一律按日期对齐到图表 data, 缺失补 null,
 *   多余日期丢弃 —— 不按下标硬套, 避免错位一根。
 * - 颜色: 输出自带 color (公式 COLORxxx) 优先, 否则按调色板分配; 再做主题可读性修正
 *   (见 resolveOverlayColor)。
 * - 本文件不依赖 echarts 运行时, 便于 vitest 直接测试。
 */
import type { IndicatorOutputKind, IndicatorPane, IndicatorSeriesOutput } from '@/lib/api'
import type { Theme } from '@/lib/theme'

export interface IndicatorOverlay {
  id: string
  name: string
  pane: IndicatorPane
  /** /series 返回的日期轴 (YYYY-MM-DD) */
  dates: string[]
  /** 需要绘制的输出 (主图 = 用户勾选的可见线; 副图 = 全部 visible 输出) */
  outputs: IndicatorSeriesOutput[]
}

export interface AlignedStick {
  p1: (number | null)[]
  p2: (number | null)[]
  cond: boolean[]
  width: number
  empty: boolean
}

/** 已对齐到图表 data 的单条输出 */
export interface AlignedOutput {
  overlayId: string
  overlayName: string
  key: string
  name: string
  kind: IndicatorOutputKind
  color: string
  thick: number
  dashed: boolean
  values: (number | null)[]
  stick?: AlignedStick
}

/** 与 MA/BOLL 内置色错开的 8 色调色板 (两种主题下都有足够对比度) */
export const OVERLAY_PALETTE = [
  '#F59E0B', '#06B6D4', '#EC4899', '#84CC16',
  '#6366F1', '#EF4444', '#14B8A6', '#A855F7',
] as const

const BULL = '#C74040'
const BEAR = '#2D9B65'

function toNum(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

/** 把 srcDates 上的值按日期映射到 targetDates; 缺失日期取 fill */
export function alignToDates<T>(
  srcDates: readonly string[],
  values: readonly T[] | undefined,
  targetDates: readonly string[],
  fill: T,
): T[] {
  const byDate = new Map<string, T>()
  if (values) {
    const n = Math.min(srcDates.length, values.length)
    for (let i = 0; i < n; i++) byDate.set(String(srcDates[i]).slice(0, 10), values[i])
  }
  return targetDates.map(d => (byDate.has(d) ? (byDate.get(d) as T) : fill))
}

function parseHex(hex: string): [number, number, number] | null {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim())
  if (!m) return null
  const n = parseInt(m[1], 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

function toHex([r, g, b]: [number, number, number]): string {
  return '#' + [r, g, b].map(c => Math.round(Math.max(0, Math.min(255, c))).toString(16).padStart(2, '0')).join('').toUpperCase()
}

function luminance([r, g, b]: [number, number, number]): number {
  const lin = (c: number) => {
    const s = c / 255
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
}

function mix(rgb: [number, number, number], target: number, ratio: number): [number, number, number] {
  return rgb.map(c => c + (target - c) * ratio) as [number, number, number]
}

/**
 * 输出线颜色: 公式色优先, 缺省走调色板; 再按主题修正可读性。
 *
 * 通达信默认黑底, 公式里常见 COLORWHITE / COLORYELLOW; 本应用有亮/暗两套主题:
 * - 暗色主题 + 近黑色 (COLORBLACK 等): 无彩色 → 回落主题文字色; 有彩色 (深蓝等) → 向白色提亮。
 * - 亮色主题 + 近白色 (COLORWHITE / LIGRAY 等): 无彩色 → 回落主题文字色; 高亮彩色 (COLORYELLOW 等) → 向黑色压暗。
 * 这样白线在亮底上变成文字灰、黄线变成可读的暗黄, 仍与其他线区分。
 */
export function resolveOverlayColor(
  color: string | null | undefined,
  theme: Theme,
  paletteIndex: number,
  themeTextColor: string,
): string {
  const base = color && parseHex(color) ? color : OVERLAY_PALETTE[((paletteIndex % OVERLAY_PALETTE.length) + OVERLAY_PALETTE.length) % OVERLAY_PALETTE.length]
  const rgb = parseHex(base)
  if (!rgb) return themeTextColor
  const lum = luminance(rgb)
  const chroma = Math.max(...rgb) - Math.min(...rgb)
  const neutral = chroma < 40
  if (theme === 'dark' && lum < 0.06) return neutral ? themeTextColor : toHex(mix(rgb, 255, 0.5))
  if (theme === 'light' && lum > 0.7) return neutral ? themeTextColor : toHex(mix(rgb, 0, 0.45))
  return toHex(rgb)
}

/**
 * 把指标覆盖层对齐到图表日期轴并解析颜色。
 * paletteStart: 本覆盖层第一条输出在调色板中的起始位置 (多个指标共用调色板时错开)。
 */
export function alignOverlay(
  overlay: IndicatorOverlay,
  targetDates: readonly string[],
  theme: Theme,
  themeTextColor: string,
  paletteStart = 0,
): AlignedOutput[] {
  return overlay.outputs.map((out, i) => {
    const color = resolveOverlayColor(out.color, theme, paletteStart + i, themeTextColor)
    const base: AlignedOutput = {
      overlayId: overlay.id,
      overlayName: overlay.name,
      key: out.key,
      name: out.name,
      kind: out.kind,
      color,
      thick: out.thick,
      dashed: out.dashed,
      values: alignToDates(overlay.dates, (out.values ?? []).map(toNum), targetDates, null),
    }
    if (out.stick) {
      const width = out.stick.width ?? out.width
      base.stick = {
        p1: alignToDates(overlay.dates, out.stick.p1.map(toNum), targetDates, null),
        p2: alignToDates(overlay.dates, out.stick.p2.map(toNum), targetDates, null),
        cond: alignToDates(overlay.dates, out.stick.cond.map(c => c === true), targetDates, false),
        width: Number.isFinite(width) ? Number(width) : 8,
        empty: !!(out.stick.empty ?? out.empty),
      }
    }
    return base
  })
}

/** 对齐全部覆盖层, 调色板在所有输出间连续分配 */
export function alignOverlays(
  overlays: readonly IndicatorOverlay[],
  targetDates: readonly string[],
  theme: Theme,
  themeTextColor: string,
): Map<string, AlignedOutput[]> {
  const result = new Map<string, AlignedOutput[]>()
  let palette = 0
  for (const ov of overlays) {
    result.set(ov.id, alignOverlay(ov, targetDates, theme, themeTextColor, palette))
    palette += ov.outputs.length
  }
  return result
}

/** LINETHICK0-9 → 像素线宽 (0 视作 1; 封顶 4 避免盖住 K 线) */
export function lineWidth(thick: number | null | undefined): number {
  const t = Number(thick)
  if (!Number.isFinite(t) || t <= 1) return 1
  return Math.min(4, t)
}

/** STICKLINE 宽度 → 占类目宽度的比例 (通达信常用 0-12; 负数 = 1px 细线) */
export function stickWidthPx(width: number, bandWidth: number): number {
  if (width < 0) return 1
  const ratio = Math.min(width, 12) / 12
  return Math.max(1, bandWidth * 0.9 * (ratio || 0.5))
}

interface AxisRef { xAxisIndex?: number; yAxisIndex?: number }

/**
 * 对齐后的输出 → ECharts series 配置 (纯函数)。
 * data 仅用于 VOLSTICK 着色 (按 K 线涨跌), 长度须与 values 一致。
 */
export function overlayToEChartsSeries(
  outputs: readonly AlignedOutput[],
  data: readonly { open: number; close: number }[],
  axis: AxisRef = {},
): any[] {
  const series: any[] = []
  for (const o of outputs) {
    const name = `${o.overlayName}.${o.name}`
    const common = { name, animation: false, silent: true, ...axis }
    if (o.kind === 'stick' && o.stick) {
      const { p1, p2, cond, width, empty } = o.stick
      const points: [number, number, number][] = []
      for (let i = 0; i < cond.length; i++) {
        const a = p1[i]
        const b = p2[i]
        if (cond[i] && a != null && b != null) points.push([i, a, b])
      }
      const color = o.color
      series.push({
        ...common,
        type: 'custom',
        data: points,
        encode: { x: 0, y: [1, 2] },
        renderItem: (_params: any, api: any) => {
          const x = api.value(0)
          const top = api.coord([x, api.value(1)])
          const bottom = api.coord([x, api.value(2)])
          const w = stickWidthPx(width, api.size([1, 0])[0])
          return {
            type: 'rect',
            shape: {
              x: top[0] - w / 2,
              y: Math.min(top[1], bottom[1]),
              width: w,
              height: Math.max(1, Math.abs(top[1] - bottom[1])),
            },
            style: empty ? { fill: 'none', stroke: color, lineWidth: 1 } : { fill: color },
          }
        },
      })
      continue
    }
    if (o.kind === 'colorstick') {
      series.push({
        ...common,
        type: 'bar',
        barWidth: 1,
        data: o.values.map(v => (v == null ? '-' : { value: v, itemStyle: { color: v >= 0 ? BULL : BEAR } })),
      })
      continue
    }
    if (o.kind === 'volstick') {
      series.push({
        ...common,
        type: 'bar',
        barWidth: '60%',
        data: o.values.map((v, i) => {
          if (v == null) return '-'
          const d = data[i]
          return { value: v, itemStyle: { color: d && d.close >= d.open ? BULL : BEAR } }
        }),
      })
      continue
    }
    const isDot = o.kind === 'dot'
    series.push({
      ...common,
      type: 'line',
      data: o.values.map(v => (v == null ? '-' : v)),
      smooth: false,
      connectNulls: false,
      symbol: isDot ? 'circle' : 'none',
      symbolSize: isDot ? 2 + lineWidth(o.thick) : undefined,
      showAllSymbol: isDot ? true : undefined,
      lineStyle: {
        width: lineWidth(o.thick),
        color: o.color,
        type: o.dashed ? 'dashed' : 'solid',
        opacity: isDot ? 0 : 1,
      },
      itemStyle: { color: o.color },
    })
  }
  return series
}

/** 信息栏数值: 大数按 万/亿 缩写 (成交量类指标) */
export function formatIndicatorValue(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return '—'
  const abs = Math.abs(v)
  if (abs >= 1e8) return (v / 1e8).toFixed(2) + '亿'
  if (abs >= 1e5) return (v / 1e4).toFixed(1) + '万'
  if (abs >= 1000) return v.toFixed(1)
  return v.toFixed(abs < 10 ? 3 : 2)
}

/** 某根 K 线上各输出的当前值 (STICKLINE 无单值, 跳过) */
export function overlayInfoItems(
  outputs: readonly AlignedOutput[],
  idx: number,
): { label: string; color: string; value: string }[] {
  return outputs
    .filter(o => o.kind !== 'stick')
    .map(o => ({ label: o.name, color: o.color, value: formatIndicatorValue(o.values[idx]) }))
}

/** 注入 innerHTML 前转义 (输出名来自用户公式) */
export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c] as string))
}
