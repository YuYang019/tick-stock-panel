import { describe, expect, it } from 'vitest'
import type { IndicatorSeriesOutput } from '@/lib/api'
import {
  OVERLAY_PALETTE,
  alignOverlay,
  alignOverlays,
  alignToDates,
  escapeHtml,
  formatIndicatorValue,
  overlayInfoItems,
  overlayToEChartsSeries,
  resolveOverlayColor,
  type IndicatorOverlay,
} from './indicatorOverlay'

const TEXT_DARK = '#A1A1AA'
const TEXT_LIGHT = '#71717A'

function out(partial: Partial<IndicatorSeriesOutput> & { key: string }): IndicatorSeriesOutput {
  return {
    name: partial.key.toUpperCase(),
    kind: 'line',
    color: null,
    thick: 1,
    dashed: false,
    visible: true,
    ...partial,
  }
}

const chartDates = ['2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08']
const candles = [
  { open: 10, close: 11 },
  { open: 11, close: 10 },
  { open: 10, close: 10.5 },
  { open: 10.5, close: 10.2 },
]

describe('alignToDates', () => {
  it('按日期对齐: 缺失补 fill, 多余丢弃, 截断到 YYYY-MM-DD', () => {
    const src = ['2026-01-02', '2026-01-06T00:00:00', '2026-01-08']
    expect(alignToDates(src, [1, 2, 3], chartDates, null)).toEqual([null, 2, null, 3])
  })

  it('values 比 dates 短时不越界', () => {
    expect(alignToDates(chartDates, [1], chartDates, null)).toEqual([1, null, null, null])
  })
})

describe('alignOverlay', () => {
  const overlay: IndicatorOverlay = {
    id: 'zhixing',
    name: '知行',
    pane: 'main',
    // 后端多一天 (live candle 已到) 且错开一根 (缺 01-05)
    dates: ['2026-01-06', '2026-01-07', '2026-01-08', '2026-01-09'],
    outputs: [
      out({ key: 'zxdq', name: '知行短期趋势线', color: '#FFFFFF', values: [1, null, 3, 4] }),
      out({ key: 'zxdk', name: '知行多空线', color: '#FFFF00', values: [NaN as unknown as number, 2, Infinity as unknown as number, 5] }),
    ],
  }

  it('日期错位 + null/NaN/inf → null', () => {
    const [a, b] = alignOverlay(overlay, chartDates, 'dark', TEXT_DARK)
    expect(a.values).toEqual([null, 1, null, 3])
    expect(b.values).toEqual([null, null, 2, null])
  })

  it('STICKLINE: cond 缺失视为 false, p1/p2 对齐', () => {
    const stickOverlay: IndicatorOverlay = {
      ...overlay,
      outputs: [out({
        key: 'o1', kind: 'stick', width: 6, empty: true,
        stick: { p1: [5, 6, 7], p2: [1, 2, null], cond: [true, null, true] },
      })],
      dates: ['2026-01-05', '2026-01-06', '2026-01-07'],
    }
    const [s] = alignOverlay(stickOverlay, chartDates, 'dark', TEXT_DARK)
    expect(s.stick).toEqual({ p1: [5, 6, 7, null], p2: [1, 2, null, null], cond: [true, false, true, false], width: 6, empty: true })
  })
})

describe('resolveOverlayColor', () => {
  it('无色 → 调色板 (循环)', () => {
    expect(resolveOverlayColor(null, 'dark', 0, TEXT_DARK)).toBe(OVERLAY_PALETTE[0])
    expect(resolveOverlayColor(undefined, 'dark', OVERLAY_PALETTE.length + 1, TEXT_DARK)).toBe(OVERLAY_PALETTE[1])
  })

  it('亮色主题: COLORWHITE → 主题文字色; COLORYELLOW → 压暗', () => {
    expect(resolveOverlayColor('#FFFFFF', 'light', 0, TEXT_LIGHT)).toBe(TEXT_LIGHT)
    const yellow = resolveOverlayColor('#FFFF00', 'light', 0, TEXT_LIGHT)
    expect(yellow).not.toBe('#FFFF00')
    expect(yellow).not.toBe(TEXT_LIGHT)
  })

  it('暗色主题: COLORBLACK → 主题文字色; 白/黄保持', () => {
    expect(resolveOverlayColor('#000000', 'dark', 0, TEXT_DARK)).toBe(TEXT_DARK)
    expect(resolveOverlayColor('#ffffff', 'dark', 0, TEXT_DARK)).toBe('#FFFFFF')
    expect(resolveOverlayColor('#FFFF00', 'dark', 0, TEXT_DARK)).toBe('#FFFF00')
  })

  it('非法色值回落调色板', () => {
    expect(resolveOverlayColor('red', 'dark', 2, TEXT_DARK)).toBe(OVERLAY_PALETTE[2])
  })
})

describe('alignOverlays', () => {
  it('多个指标间调色板连续分配', () => {
    const mk = (id: string, n: number): IndicatorOverlay => ({
      id, name: id, pane: 'main', dates: chartDates,
      outputs: Array.from({ length: n }, (_, i) => out({ key: `o${i}`, values: [1, 2, 3, 4] })),
    })
    const m = alignOverlays([mk('a', 2), mk('b', 1)], chartDates, 'dark', TEXT_DARK)
    expect(m.get('a')!.map(o => o.color)).toEqual([OVERLAY_PALETTE[0], OVERLAY_PALETTE[1]])
    expect(m.get('b')![0].color).toBe(OVERLAY_PALETTE[2])
  })
})

describe('overlayToEChartsSeries', () => {
  const overlay: IndicatorOverlay = {
    id: 'x', name: '测试', pane: 'main', dates: chartDates,
    outputs: [
      out({ key: 'l', name: 'L', color: '#3B82F6', thick: 3, dashed: true, values: [1, null, 3, 4] }),
      out({ key: 'd', name: 'D', kind: 'dot', values: [1, 2, 3, 4] }),
      out({ key: 'c', name: 'C', kind: 'colorstick', values: [1, -1, null, 0] }),
      out({ key: 'v', name: 'V', kind: 'volstick', values: [100, 200, 300, 400] }),
      out({ key: 's', name: 'S', kind: 'stick', width: 4, empty: false, stick: { p1: [11, 10, 10.5, 9], p2: [10, 11, null, 8], cond: [true, false, true, true] } }),
    ],
  }
  const aligned = alignOverlay(overlay, chartDates, 'dark', TEXT_DARK)
  const series = overlayToEChartsSeries(aligned, candles, { xAxisIndex: 1, yAxisIndex: 1 })

  it('line: null → "-" 断线, 线宽/虚线/颜色/轴索引', () => {
    const s = series[0]
    expect(s).toMatchObject({ type: 'line', name: '测试.L', xAxisIndex: 1, yAxisIndex: 1, symbol: 'none' })
    expect(s.data).toEqual([1, '-', 3, 4])
    expect(s.lineStyle).toMatchObject({ width: 3, type: 'dashed', color: '#3B82F6' })
  })

  it('dot: 只画点', () => {
    expect(series[1]).toMatchObject({ type: 'line', symbol: 'circle' })
    expect(series[1].lineStyle.opacity).toBe(0)
  })

  it('colorstick: 正红负绿, null → "-"', () => {
    const data = series[2].data
    expect(data[0].itemStyle.color).toBe('#C74040')
    expect(data[1].itemStyle.color).toBe('#2D9B65')
    expect(data[2]).toBe('-')
    expect(data[3].itemStyle.color).toBe('#C74040')
  })

  it('volstick: 按 K 线涨跌着色', () => {
    const data = series[3].data
    expect(data.map((d: any) => d.itemStyle.color)).toEqual(['#C74040', '#2D9B65', '#C74040', '#2D9B65'])
  })

  it('stick: 只保留 cond 为真且 p1/p2 都有值的点', () => {
    expect(series[4]).toMatchObject({ type: 'custom', encode: { x: 0, y: [1, 2] } })
    expect(series[4].data).toEqual([[0, 11, 10], [3, 9, 8]])
    const shape = series[4].renderItem({}, {
      value: (i: number) => [3, 9, 8][i],
      coord: ([x, y]: [number, number]) => [x * 10, 100 - y],
      size: () => [10, 0],
    })
    expect(shape.type).toBe('rect')
    expect(shape.shape.height).toBeCloseTo(1)
  })

  it('信息栏数值: 跳过 stick, 缺值显示 —', () => {
    const items = overlayInfoItems(aligned, 1)
    expect(items.map(i => i.label)).toEqual(['L', 'D', 'C', 'V'])
    expect(items[0].value).toBe('—')
    expect(items[1].value).toBe('2.000')
  })
})

describe('formatIndicatorValue / escapeHtml', () => {
  it('format', () => {
    expect(formatIndicatorValue(null)).toBe('—')
    expect(formatIndicatorValue(12.345)).toBe('12.35')
    expect(formatIndicatorValue(250_000)).toBe('25.0万')
    expect(formatIndicatorValue(3e8)).toBe('3.00亿')
  })
  it('escape', () => {
    expect(escapeHtml('<b>"x"&\'')).toBe('&lt;b&gt;&quot;x&quot;&amp;&#39;')
  })
})
