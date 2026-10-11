/** 指标库页的纯工具函数 (无 React 依赖, 便于测试) */

/** 1 起的行/列 → 字符串偏移 (越界时夹到合法范围) */
export function lineColToOffset(text: string, line: number | null | undefined, col: number | null | undefined): number {
  if (!line || line < 1) return 0
  let offset = 0
  let cur = 1
  while (cur < line) {
    const nl = text.indexOf('\n', offset)
    if (nl < 0) return text.length
    offset = nl + 1
    cur += 1
  }
  const lineEnd = text.indexOf('\n', offset)
  const end = lineEnd < 0 ? text.length : lineEnd
  return Math.min(end, offset + Math.max(0, (col ?? 1) - 1))
}

/** 出错位置的选区: 从偏移起到当前词尾 (至少 1 个字符, 不跨行) */
export function errorSelection(text: string, offset: number): [number, number] {
  const start = Math.max(0, Math.min(offset, text.length))
  let end = start
  while (end < text.length && !/[\s;,()+\-*/<>=:]/.test(text[end])) end += 1
  if (end === start && start < text.length && text[start] !== '\n') end = start + 1
  return [start, end]
}

function refLabel(r: unknown): string | null {
  if (typeof r === 'string') return r
  if (r && typeof r === 'object') {
    const o = r as Record<string, unknown>
    const name = o.name ?? o.label ?? o.id
    if (typeof name !== 'string') return null
    const type = typeof o.type === 'string' ? o.type : typeof o.kind === 'string' ? o.kind : ''
    const id = typeof o.id === 'string' && o.id !== name ? ` (${o.id})` : ''
    return `${type ? `${type}: ` : ''}${name}${id}`
  }
  return null
}

/**
 * 409 删除被拒: request() 把对象 detail stringify 进 error.message。
 * 容忍 {references|referenced_by|refs: [...]} 或直接数组; 元素可为字符串或 {id,name,type}。
 * 解析失败返回 null (走通用错误提示)。
 */
export function parseReferences(message: string): string[] | null {
  let parsed: unknown
  try { parsed = JSON.parse(message) } catch { return null }
  const obj = parsed as Record<string, unknown> | unknown[]
  const list = Array.isArray(obj)
    ? obj
    : (obj && typeof obj === 'object'
        ? (obj.references ?? obj.referenced_by ?? obj.refs)
        : undefined)
  if (!Array.isArray(list)) return null
  const labels = list.map(refLabel).filter((s): s is string => !!s)
  return labels.length > 0 ? labels : null
}

/** 409 时附带的可读消息 (detail.message) */
export function parseDetailMessage(message: string): string {
  try {
    const parsed = JSON.parse(message) as { message?: unknown }
    if (parsed && typeof parsed.message === 'string') return parsed.message
  } catch { /* 非 JSON */ }
  return message
}

/**
 * 函数面板条目名 → 可点击插入的别名列表。
 * 后端 catalog 用 "O / OPEN"、"STICK / COLORSTICK / VOLSTICK"、"COLORRED / COLORGREEN / ..." 这类展示名,
 * 拆成单个 token; 占位写法转为可编译的示例 (LINETHICK0-9 → LINETHICK1, COLORBBGGRR → COLOR00C0C0)。
 */
export function paletteTokens(name: string): { label: string; insert: string }[] {
  return name
    .split('/')
    .map(s => s.trim())
    .filter(s => s && !s.includes('...') && !s.includes('…'))
    .map(label => {
      if (/^LINETHICK0-9$/i.test(label)) return { label, insert: 'LINETHICK1' }
      if (/^COLORBBGGRR$/i.test(label)) return { label, insert: 'COLOR00C0C0' }
      return { label, insert: label }
    })
}
