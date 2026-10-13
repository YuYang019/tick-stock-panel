// @vitest-environment jsdom
import { describe, expect, it } from 'vitest'
import { __test__ } from './echartsZoomFix'

const { normalizeForZoom } = __test__

/** 构造一个 echarts 容器: 布局尺寸 offsetW×offsetH, 视觉矩形 rect (模拟 CSS zoom)。 */
function chartDom(offsetW: number, offsetH: number, rect: { left: number; top: number; width: number; height: number }) {
  const dom = document.createElement('div')
  dom.setAttribute('_echarts_instance_', 'ec_1')
  const canvas = document.createElement('canvas')
  dom.appendChild(canvas)
  document.body.appendChild(dom)
  Object.defineProperty(dom, 'offsetWidth', { value: offsetW })
  Object.defineProperty(dom, 'offsetHeight', { value: offsetH })
  dom.getBoundingClientRect = () => ({ ...rect, right: rect.left + rect.width, bottom: rect.top + rect.height, x: rect.left, y: rect.top, toJSON: () => ({}) }) as DOMRect
  return { dom, canvas }
}

function move(target: Element, clientX: number, clientY: number) {
  const ev = new MouseEvent('mousemove', { clientX, clientY, bubbles: true }) as MouseEvent & { zrX?: number; zrY?: number }
  Object.defineProperty(ev, 'target', { value: target })
  normalizeForZoom(ev)
  return ev
}

describe('echartsZoomFix', () => {
  it('zoom=0.9: 视觉坐标换算回画布布局坐标', () => {
    // 布局 600×400, 视觉 540×360 起点 (270,180); 鼠标视觉位于图表中心
    const { canvas } = chartDom(600, 400, { left: 270, top: 180, width: 540, height: 360 })
    const ev = move(canvas, 270 + 270, 180 + 180)
    expect(ev.zrX).toBeCloseTo(300)
    expect(ev.zrY).toBeCloseTo(200)
  })

  it('未缩放时不写 zrX, 交给 zrender 原逻辑', () => {
    const { canvas } = chartDom(600, 400, { left: 10, top: 10, width: 600, height: 400 })
    const ev = move(canvas, 100, 100)
    expect(ev.zrX).toBeUndefined()
  })

  it('图表外的事件不处理', () => {
    const el = document.createElement('div')
    document.body.appendChild(el)
    expect(move(el, 5, 5).zrX).toBeUndefined()
  })
})
