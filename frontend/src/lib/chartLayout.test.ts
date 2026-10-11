// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from 'vitest'
import {
  DEFAULT_CHART_LAYOUT,
  indicatorSubKey,
  layoutIndicatorIds,
  loadChartLayout,
  parseIndicatorSubKey,
  removeIndicatorFromLayout,
  removeIndicatorFromStoredLayout,
  sanitizeChartLayout,
  saveChartLayout,
  toggleMainOutput,
  toggleSub,
  visibleOutputs,
} from './chartLayout'

const SUB = ['vol', 'macd', 'rsi', 'kdj']

describe('chartLayout', () => {
  beforeEach(() => localStorage.clear())

  it('首次打开回落默认: MA5/10/20/60 + 成交量', () => {
    const layout = loadChartLayout(SUB)
    expect(layout).toEqual(DEFAULT_CHART_LAYOUT)
    expect(visibleOutputs(layout, 'ma')).toEqual(['ma5', 'ma10', 'ma20', 'ma60'])
  })

  it('保存后重新加载保持上次选择 (含尚未加载的自定义指标)', () => {
    let layout = loadChartLayout(SUB)
    layout = toggleMainOutput(layout, 'ma', 'ma10')
    layout = toggleMainOutput(layout, 'zhixing', '趋势线')
    layout = toggleSub(layout, 'macd')
    saveChartLayout(layout)

    const reloaded = loadChartLayout(SUB)
    expect(visibleOutputs(reloaded, 'ma')).toEqual(['ma5', 'ma20', 'ma60'])
    expect(visibleOutputs(reloaded, 'zhixing')).toEqual(['趋势线'])
    expect(reloaded.sub).toEqual(['vol', 'macd'])
  })

  it('取消指标最后一条线时整项移除, 全部取消后不回落默认', () => {
    let layout = { ...DEFAULT_CHART_LAYOUT, main: [{ id: 'ma', outputs: ['ma5'] }] }
    layout = toggleMainOutput(layout, 'ma', 'ma5')
    expect(layout.main).toEqual([])
    saveChartLayout(layout)
    expect(loadChartLayout(SUB).main).toEqual([])
  })

  it('脏数据: 结构/版本不符回落默认', () => {
    expect(sanitizeChartLayout(null, SUB)).toEqual(DEFAULT_CHART_LAYOUT)
    expect(sanitizeChartLayout('x', SUB)).toEqual(DEFAULT_CHART_LAYOUT)
    expect(sanitizeChartLayout({ version: 2, main: [], sub: [] }, SUB)).toEqual(DEFAULT_CHART_LAYOUT)
    localStorage.setItem('stock_chart_layout', '{not json')
    expect(loadChartLayout(SUB)).toEqual(DEFAULT_CHART_LAYOUT)
  })

  it('脏数据: 剔除非法条目、重复项与未知副图', () => {
    const layout = sanitizeChartLayout({
      version: 1,
      main: [
        { id: 'ma', outputs: ['ma5', 'ma5', 3] },
        { id: 'ma', outputs: ['ma10'] },
        { id: 'boll', outputs: [] },
        { outputs: ['x'] },
        null,
      ],
      sub: ['vol', 'gone', 'vol', 'kdj'],
    }, SUB)
    expect(layout.main).toEqual([{ id: 'ma', outputs: ['ma5'] }])
    expect(layout.sub).toEqual(['vol', 'kdj'])
  })

  it('副图动态 key: ind:{id} 放行并持久化, 非法 id 剔除', () => {
    const layout = sanitizeChartLayout({
      version: 1,
      main: [],
      sub: ['vol', 'ind:my_macd', 'ind:', 'ind:Bad-Id', 'ind:my_macd', 'indicator', 'ind:../x'],
    }, SUB)
    expect(layout.sub).toEqual(['vol', 'ind:my_macd'])

    saveChartLayout(toggleSub(layout, indicatorSubKey('zx_sub')))
    expect(loadChartLayout(SUB).sub).toEqual(['vol', 'ind:my_macd', 'ind:zx_sub'])
  })

  it('parseIndicatorSubKey', () => {
    expect(parseIndicatorSubKey('ind:zhixing')).toBe('zhixing')
    expect(parseIndicatorSubKey('vol')).toBeNull()
    expect(parseIndicatorSubKey('ind:')).toBeNull()
    expect(parseIndicatorSubKey('ind:A')).toBeNull()
  })

  it('只勾 知行.知行短期趋势线 后重载仍只有这一条', () => {
    let layout = loadChartLayout(SUB)
    layout = toggleMainOutput(layout, 'zhixing', 'zxdq')
    saveChartLayout(layout)
    const reloaded = loadChartLayout(SUB)
    expect(visibleOutputs(reloaded, 'zhixing')).toEqual(['zxdq'])
    expect(layoutIndicatorIds(reloaded)).toEqual(['zhixing'])
  })

  it('layoutIndicatorIds: 主图自定义 + 副图 ind:, 去重排序, 不含内置', () => {
    const layout = {
      version: 1 as const,
      main: [{ id: 'ma', outputs: ['ma5'] }, { id: 'zhixing', outputs: ['a'] }, { id: 'boll', outputs: ['boll'] }],
      sub: ['vol', 'ind:b_sub', 'ind:zhixing'],
    }
    expect(layoutIndicatorIds(layout)).toEqual(['b_sub', 'zhixing'])
  })

  it('删除指标时从主图与副图清除 (含持久化布局)', () => {
    const layout = {
      version: 1 as const,
      main: [{ id: 'ma', outputs: ['ma5'] }, { id: 'zhixing', outputs: ['a'] }],
      sub: ['vol', 'ind:zhixing', 'ind:other'],
    }
    expect(removeIndicatorFromLayout(layout, 'zhixing')).toEqual({
      version: 1, main: [{ id: 'ma', outputs: ['ma5'] }], sub: ['vol', 'ind:other'],
    })
    saveChartLayout(layout)
    removeIndicatorFromStoredLayout('zhixing', SUB)
    const reloaded = loadChartLayout(SUB)
    expect(layoutIndicatorIds(reloaded)).toEqual(['other'])
  })
})
