/**
 * ECharts 指针坐标 × 页面 CSS zoom 修正。
 *
 * 界面缩放 (lib/pageZoom) 用 CSS zoom 作用于 <html>。新版 Chromium 的
 * MouseEvent.offsetX/Y 与 getBoundingClientRect 都是缩放后的视觉坐标, 而
 * zrender 把 offsetX 直接当画布内布局坐标用 → zoom=0.9 时十字准心/tooltip/
 * 点击命中整体缩到 0.9 倍, 即向图表左上角偏移, 离左上越远偏得越多。
 *
 * zrender normalizeEvent 遇到已带 zrX/zrY 的事件会原样使用, 这里在捕获阶段
 * 抢先按「视觉偏移 × (布局尺寸 / 视觉尺寸)」写入画布布局坐标。比值取自图表
 * 容器本身, 同时覆盖 zoom 与 transform: scale, 浏览器若不缩放 offsetX 也不受影响
 * (比值为 1 时直接跳过, 交给 zrender 原逻辑)。
 */

const POINTER_EVENTS = [
  'mousedown', 'mouseup', 'mousemove', 'mouseout', 'mouseover',
  'click', 'dblclick', 'contextmenu', 'wheel',
  'pointerdown', 'pointerup', 'pointermove', 'pointerout', 'pointerover', 'pointercancel',
] as const

type ZrEvent = MouseEvent & { zrX?: number; zrY?: number }

function normalizeForZoom(e: Event): void {
  const ev = e as ZrEvent
  if (ev.zrX != null || !(ev.target instanceof Element)) return
  const dom = ev.target.closest<HTMLElement>('[_echarts_instance_]')
  if (!dom) return
  const rect = dom.getBoundingClientRect()
  if (!rect.width || !rect.height) return
  const sx = dom.offsetWidth / rect.width
  const sy = dom.offsetHeight / rect.height
  if (Math.abs(sx - 1) < 1e-3 && Math.abs(sy - 1) < 1e-3) return
  ev.zrX = (ev.clientX - rect.left) * sx
  ev.zrY = (ev.clientY - rect.top) * sy
}

let installed = false

/** 全局安装一次 (main.tsx 启动时调用)。 */
export function installEChartsZoomFix(): void {
  if (installed || typeof window === 'undefined') return
  installed = true
  for (const type of POINTER_EVENTS) {
    window.addEventListener(type, normalizeForZoom, { capture: true, passive: true })
  }
}

export const __test__ = { normalizeForZoom }
