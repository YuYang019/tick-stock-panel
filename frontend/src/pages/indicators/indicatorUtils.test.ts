import { describe, expect, it } from 'vitest'
import { errorSelection, lineColToOffset, paletteTokens, parseDetailMessage, parseReferences } from './indicatorUtils'

const F = 'A:MA(C,5);\nB:EMA(C,10),COLORRED;\nXX'

describe('lineColToOffset', () => {
  it('行列 (1 起) → 偏移', () => {
    expect(lineColToOffset(F, 1, 1)).toBe(0)
    expect(lineColToOffset(F, 2, 3)).toBe(F.indexOf('EMA'))
    expect(lineColToOffset(F, 3, 2)).toBe(F.length - 1)
  })
  it('越界夹取', () => {
    expect(lineColToOffset(F, 9, 1)).toBe(F.length)
    expect(lineColToOffset(F, 1, 999)).toBe(F.indexOf('\n'))
    expect(lineColToOffset(F, null, null)).toBe(0)
  })
})

describe('errorSelection', () => {
  it('选到词尾', () => {
    const off = F.indexOf('COLORRED')
    expect(errorSelection(F, off)).toEqual([off, off + 'COLORRED'.length])
  })
  it('分隔符处至少选 1 个字符', () => {
    const off = F.indexOf(';')
    expect(errorSelection(F, off)).toEqual([off, off + 1])
  })
})

describe('parseReferences', () => {
  it('对象 detail 的多种字段名与元素形态', () => {
    expect(parseReferences(JSON.stringify({ message: 'x', references: ['csg_a', 'csg_b'] }))).toEqual(['csg_a', 'csg_b'])
    expect(parseReferences(JSON.stringify({ referenced_by: [{ type: '信号', id: 'csg_a', name: '金叉' }] }))).toEqual(['信号: 金叉 (csg_a)'])
    expect(parseReferences(JSON.stringify([{ id: 'csg_a' }]))).toEqual(['csg_a'])
  })
  it('非 JSON / 无引用 → null', () => {
    expect(parseReferences('指标被引用')).toBeNull()
    expect(parseReferences(JSON.stringify({ message: 'x' }))).toBeNull()
    expect(parseReferences(JSON.stringify({ references: [] }))).toBeNull()
  })
  it('detail.message', () => {
    expect(parseDetailMessage(JSON.stringify({ message: '被引用' }))).toBe('被引用')
    expect(parseDetailMessage('plain')).toBe('plain')
  })
})

describe('paletteTokens', () => {
  it('拆分别名, 丢弃省略号, 占位转示例', () => {
    expect(paletteTokens('O / OPEN').map(t => t.insert)).toEqual(['O', 'OPEN'])
    expect(paletteTokens('COLORRED / COLORGREEN / ...').map(t => t.insert)).toEqual(['COLORRED', 'COLORGREEN'])
    expect(paletteTokens('LINETHICK0-9')).toEqual([{ label: 'LINETHICK0-9', insert: 'LINETHICK1' }])
    expect(paletteTokens('COLORBBGGRR')[0].insert).toBe('COLOR00C0C0')
  })
})
