import { useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ChevronRight, CircleAlert, CircleCheck, Copy, LineChart, Save, TriangleAlert } from 'lucide-react'
import { toast } from '@/components/Toast'
import { EChartsCandlestick, type IndicatorOverlay } from '@/components/EChartsCandlestick'
import { getDefaultRange, toOHLC } from '@/components/StockDailyKChart'
import {
  api,
  previewOutputs,
  type IndicatorFunctionDoc,
  type IndicatorItem,
  type IndicatorPane,
  type IndicatorSavePayload,
  type TdxError,
} from '@/lib/api'
import { indicatorSubKey } from '@/lib/chartLayout'
import { klineDailyQueryOptions } from '@/lib/kline'
import { QK } from '@/lib/queryKeys'
import { errorSelection, lineColToOffset, paletteTokens } from './indicatorUtils'
import { PANE_LABEL } from './IndicatorLibrary'

const INPUT_CLS = 'w-full rounded-input border border-border bg-surface px-2.5 py-1.5 text-xs text-foreground focus:border-accent focus:outline-none'
const CHIP_CLS = 'shrink-0 rounded-btn border border-border bg-base/60 px-1.5 py-0.5 font-mono text-[10px] text-secondary transition-colors hover:border-accent/40 hover:text-accent cursor-pointer'
const DEFAULT_FORMULA = 'DIF:EMA(CLOSE,12)-EMA(CLOSE,26);\nDEA:EMA(DIF,9);\nMACD:(DIF-DEA)*2,COLORSTICK;'
const PREVIEW_ID = 'preview'
const SYMBOL_RE = /^[0-9A-Za-z]{4,8}\.[A-Za-z]{2,3}$/

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = window.setTimeout(() => setV(value), ms)
    return () => window.clearTimeout(t)
  }, [value, ms])
  return v
}

function normDoc(item: string | IndicatorFunctionDoc): IndicatorFunctionDoc {
  return typeof item === 'string' ? { name: item } : item
}

/** 保存接口返回体: 规范为 IndicatorItem, 也容忍 {indicator:{...}} / {id} */
function savedIdOf(res: unknown, fallback?: string): string | undefined {
  const r = res as { id?: unknown; indicator?: { id?: unknown } } | null
  const id = r?.id ?? r?.indicator?.id
  return typeof id === 'string' ? id : fallback
}

function errorPos(e: TdxError): string {
  if (!e.line) return ''
  return `第 ${e.line} 行${e.col ? ` 第 ${e.col} 列` : ''}`
}

interface Props {
  editId: string
  /** 基于 editId 另存为新指标 (预设指标只能走这条) */
  saveAs: boolean
  onSaved: (id: string) => void
}

/** 指标编辑器: 公式 → 防抖校验 → 预览 → 保存 / 另存为 */
export function IndicatorEditor({ editId, saveAs, onSaved }: Props) {
  const qc = useQueryClient()
  const list = useQuery({ queryKey: QK.indicators.list, queryFn: api.listIndicators })
  const source: IndicatorItem | undefined = editId ? list.data?.find(i => i.id === editId) : undefined
  // 预设不能覆盖: 即使从 URL 直接进入编辑也按另存为处理
  const mode: 'new' | 'edit' | 'copy' = !source ? 'new' : source.builtin || saveAs ? 'copy' : 'edit'

  const [name, setName] = useState('')
  const [newId, setNewId] = useState('')
  const [pane, setPane] = useState<IndicatorPane>('main')
  const [description, setDescription] = useState('')
  const [formula, setFormula] = useState(editId ? '' : DEFAULT_FORMULA)
  const [loaded, setLoaded] = useState(!editId)
  const [copyOpen, setCopyOpen] = useState(false)
  const [copyName, setCopyName] = useState('')
  const [copyId, setCopyId] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // 编辑/另存为: 列表到达后一次性载入表单 (组件以 editId 为 key 重挂载, 不会串到别的指标)
  useEffect(() => {
    if (loaded || !source) return
    setName(mode === 'copy' ? `${source.name} 副本` : source.name)
    setPane(source.pane)
    setDescription(source.description ?? '')
    setFormula(source.formula)
    setLoaded(true)
  }, [loaded, source, mode])

  // ===== 校验 (防抖 400ms; 按公式缓存, 回退到改过的旧公式不重复请求) =====
  const debouncedFormula = useDebounced(formula, 400)
  const validation = useQuery({
    queryKey: QK.indicators.validate(debouncedFormula),
    queryFn: () => api.validateIndicator(debouncedFormula),
    enabled: debouncedFormula.trim().length > 0,
    staleTime: Infinity,
    placeholderData: prev => prev,
    retry: false,
  })
  const v = validation.data
  const validating = formula !== debouncedFormula || validation.isFetching
  const formulaOk = !validating && v?.ok === true && formula.trim().length > 0

  const functions = useQuery({ queryKey: QK.indicators.functions, queryFn: api.getIndicatorFunctions, staleTime: Infinity })
  const [paletteQuery, setPaletteQuery] = useState('')
  const palette = useMemo(() => {
    const kw = paletteQuery.trim().toLowerCase()
    const match = (d: IndicatorFunctionDoc) => !kw || `${d.name} ${d.signature ?? ''} ${d.desc ?? ''}`.toLowerCase().includes(kw)
    const data = functions.data
    return [
      { key: 'functions', label: '函数', items: (data?.functions ?? []).map(normDoc).filter(match), call: true },
      { key: 'variables', label: '变量', items: (data?.variables ?? []).map(normDoc).filter(match), call: false },
      { key: 'modifiers', label: '修饰符', items: (data?.modifiers ?? []).map(normDoc).filter(match), call: false },
    ]
  }, [functions.data, paletteQuery])

  // ===== 预览 =====
  const [symbolInput, setSymbolInput] = useState('')
  const [symbol, setSymbol] = useState('')
  const debouncedSymbolQuery = useDebounced(symbolInput.trim(), 300)
  const suggestions = useQuery({
    queryKey: QK.instrumentSearch(debouncedSymbolQuery, 'stock'),
    queryFn: () => api.instrumentSearch(debouncedSymbolQuery, 10, 'stock'),
    enabled: debouncedSymbolQuery.length > 0 && !SYMBOL_RE.test(debouncedSymbolQuery),
    staleTime: 60_000,
  })
  const commitSymbol = (raw: string) => {
    const s = raw.trim().toUpperCase()
    const hit = suggestions.data?.results.find(r => r.symbol.toUpperCase() === s || r.code === raw.trim() || r.name === raw.trim())
    const next = hit?.symbol ?? (SYMBOL_RE.test(s) ? s : '')
    if (next) { setSymbol(next); setSymbolInput(next) }
  }
  const range = useMemo(() => getDefaultRange(), [])
  const kline = useQuery({ ...klineDailyQueryOptions(symbol, range), enabled: !!symbol })
  const rows = useMemo(() => toOHLC(kline.data?.rows ?? []), [kline.data?.rows])
  const previewFormula = useDebounced(formula, 800)
  const preview = useQuery({
    queryKey: QK.indicators.preview(previewFormula, symbol, range.start, range.end),
    queryFn: () => api.previewIndicator({ formula: previewFormula, symbol, start_date: range.start, end_date: range.end }),
    enabled: !!symbol && formulaOk && previewFormula === formula,
    staleTime: 60_000,
    retry: false,
    // 改公式时保留上一版预览线, 切股不透传
    placeholderData: (prev, prevQuery) => (prevQuery?.queryKey[3] === symbol ? prev : undefined),
  })
  const overlays = useMemo<IndicatorOverlay[]>(() => {
    const res = preview.data
    if (!res || res.symbol?.toUpperCase?.() !== symbol) return []
    const outputs = previewOutputs(res).filter(o => o.visible)
    return outputs.length > 0 ? [{ id: PREVIEW_ID, name: name.trim() || '预览', pane, dates: res.dates, outputs }] : []
  }, [preview.data, symbol, name, pane])
  const previewIndicators = useMemo(() => (pane === 'sub' ? ['vol', indicatorSubKey(PREVIEW_ID)] : ['vol']), [pane])

  // ===== 保存 =====
  const invalidateAll = () => {
    void qc.invalidateQueries({ queryKey: QK.indicators.all })
    // 输出线注册为因子 → 信号字段下拉 / 因子库同步刷新
    void qc.invalidateQueries({ queryKey: QK.customSignalsOptions })
    void qc.invalidateQueries({ queryKey: ['factors-library'] })
  }
  const save = useMutation({
    mutationFn: (target: { kind: 'update' } | { kind: 'create'; name: string; id: string }) => {
      const base = { formula, pane, description: description.trim() || undefined }
      if (target.kind === 'update' && source) return api.updateIndicator(source.id, { ...base, name: name.trim() })
      const payload: IndicatorSavePayload = { ...base, name: (target.kind === 'create' ? target.name : name).trim() }
      if (target.kind === 'create' && target.id) payload.id = target.id
      return api.createIndicator(payload)
    },
    onSuccess: (res, target) => {
      invalidateAll()
      const id = savedIdOf(res, target.kind === 'update' ? source?.id : undefined)
      toast(target.kind === 'update' ? `已保存指标 ${name.trim()}` : `已创建指标 ${id ?? ''}`, 'success')
      setCopyOpen(false)
      if (id && (target.kind === 'create' || id !== editId)) onSaved(id)
    },
  })
  const saveDisabledReason = !name.trim() ? '请填写名称' : !formulaOk ? (validating ? '校验中…' : '公式未通过校验') : ''

  const sanitizeId = (raw: string) => raw.toLowerCase().replace(/[^a-z0-9_]/g, '').slice(0, 40)

  // 错误定位: 行列 → 选中出错片段
  const locate = (e: TdxError) => {
    const el = textareaRef.current
    if (!el || !e.line) return
    const [start, end] = errorSelection(formula, lineColToOffset(formula, e.line, e.col))
    el.focus()
    el.setSelectionRange(start, end)
  }

  // 面板点击插入到光标处
  const insert = (snippet: string) => {
    const el = textareaRef.current
    const start = el?.selectionStart ?? formula.length
    const end = el?.selectionEnd ?? start
    setFormula(formula.slice(0, start) + snippet + formula.slice(end))
    requestAnimationFrame(() => {
      if (!el) return
      el.focus()
      const pos = start + snippet.length
      el.setSelectionRange(pos, pos)
    })
  }

  if (editId && list.isLoading) return <div className="p-6 text-center text-xs text-muted">加载指标…</div>
  if (editId && !list.isLoading && !source) {
    return (
      <div className="m-3 rounded-btn border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger" role="alert">
        {list.isError ? `指标列表加载失败 · ${(list.error as Error).message}` : `指标 ${editId} 不存在或已删除`}
      </div>
    )
  }

  const errors = v?.errors ?? []
  const warnings = v?.warnings ?? []
  const errorListId = 'indicator-formula-errors'

  return (
    <div className="grid h-full min-h-0 grid-cols-1 gap-3 overflow-hidden xl:grid-cols-2">
      {/* ── 左栏: 元数据 + 公式 + 函数面板 ── */}
      <section aria-label="指标公式" className="flex min-h-0 flex-col gap-3 rounded-card border border-border bg-surface/80 p-3 xl:overflow-y-auto">
        <div className="flex flex-wrap items-center gap-2">
          <LineChart className="h-4 w-4 shrink-0 text-accent" aria-hidden="true" />
          <h2 className="text-sm font-medium text-foreground">
            {mode === 'new' ? '新建指标' : mode === 'edit' ? `编辑 ${source!.name}` : `另存为 · 基于 ${source!.name}`}
          </h2>
          {mode === 'copy' && source?.builtin && (
            <span className="rounded-btn bg-accent/10 px-1.5 py-0.5 text-[10px] text-accent">预设指标不可覆盖, 将保存为新指标</span>
          )}
          {mode === 'edit' && <span className="font-mono text-[10px] text-muted">{source!.id}</span>}
        </div>

        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <label className="space-y-1">
            <span className="text-[11px] text-muted">名称 <span className="text-danger" aria-hidden="true">*</span></span>
            <input value={name} onChange={e => setName(e.target.value)} maxLength={40} required aria-required="true" className={`${INPUT_CLS} h-8`} placeholder="如 知行" />
          </label>
          {mode !== 'edit' ? (
            <label className="space-y-1">
              <span className="text-[11px] text-muted">标识 (可空, 按名称生成)</span>
              <input value={newId} onChange={e => setNewId(sanitizeId(e.target.value))} className={`${INPUT_CLS} h-8 font-mono`} placeholder="小写字母/数字/下划线" aria-describedby="indicator-id-hint" />
              <span id="indicator-id-hint" className="sr-only">保存后不可修改; 输出线因子为 ind_标识_输出键</span>
            </label>
          ) : (
            <div className="space-y-1">
              <span className="text-[11px] text-muted">标识</span>
              <div className="flex h-8 items-center font-mono text-xs text-secondary">{source!.id}</div>
            </div>
          )}
          <fieldset className="space-y-1">
            <legend className="text-[11px] text-muted">显示位置</legend>
            <div className="flex h-8 items-center gap-3 text-xs text-secondary">
              {(['main', 'sub'] as const).map(p => (
                <label key={p} className="inline-flex cursor-pointer items-center gap-1">
                  <input type="radio" name="indicator-pane" value={p} checked={pane === p} onChange={() => setPane(p)} />
                  {PANE_LABEL[p]}{p === 'main' ? ' (叠加在 K 线上)' : ' (独立副图)'}
                </label>
              ))}
            </div>
          </fieldset>
          <label className="space-y-1">
            <span className="text-[11px] text-muted">说明</span>
            <input value={description} onChange={e => setDescription(e.target.value)} maxLength={500} className={`${INPUT_CLS} h-8`} placeholder="可空" />
          </label>
        </div>

        <div>
          <label htmlFor="indicator-formula" className="mb-1 block text-xs font-medium text-secondary">公式 (通达信语法)</label>
          <textarea
            id="indicator-formula"
            ref={textareaRef}
            value={formula}
            onChange={e => setFormula(e.target.value)}
            spellCheck={false}
            rows={10}
            aria-invalid={v?.ok === false && !validating}
            aria-describedby={errors.length > 0 ? errorListId : undefined}
            className={`w-full resize-y rounded-input border bg-base/60 px-3 py-2 font-mono text-xs leading-relaxed text-foreground focus:outline-none ${v?.ok === false && !validating ? 'border-danger/60 focus:border-danger' : 'border-border focus:border-accent'}`}
            placeholder={'名称:表达式,修饰符;\n如 MA5:MA(CLOSE,5),COLORYELLOW;'}
          />
          <div className="mt-1 text-[10px] text-muted">
            <code className="font-mono">名称:表达式;</code> 输出线 · <code className="font-mono">名称:=表达式;</code> 中间变量 · <code className="font-mono">{'{...}'}</code> 注释 · DRAWTEXT 等绘图函数所在语句会跳过并警告
          </div>
        </div>

        <details open className="group rounded-btn border border-border/60 bg-base/40 px-2.5 py-1.5 text-[10px]">
          <summary className="flex cursor-pointer select-none items-center gap-1 font-medium text-foreground transition-colors hover:text-accent [&::-webkit-details-marker]:hidden">
            <ChevronRight className="h-3 w-3 shrink-0 text-accent transition-transform group-open:rotate-90" aria-hidden="true" />
            函数 / 变量 / 修饰符 <span className="font-normal text-muted">· 点击插入</span>
          </summary>
          <input
            type="search"
            value={paletteQuery}
            onChange={e => setPaletteQuery(e.target.value)}
            placeholder="搜索函数, 如 EMA / 交叉"
            aria-label="搜索函数"
            className={`${INPUT_CLS} mt-1.5 h-7`}
          />
          {functions.isLoading && <div className="mt-1.5 text-muted">加载函数清单…</div>}
          {functions.isError && <div className="mt-1.5 text-danger" role="alert">函数清单加载失败</div>}
          <div className="mt-1.5 max-h-72 space-y-2 overflow-y-auto">
            {palette.map(group => group.items.length > 0 && (
              <div key={group.key} role="group" aria-label={`插入${group.label}`}>
                <div className="mb-0.5 font-medium text-secondary">{group.label} <span className="font-normal text-muted">({group.items.length})</span></div>
                {group.call ? (
                  <div className="space-y-0.5">
                    {group.items.map(fn => (
                      <button
                        key={fn.name}
                        type="button"
                        onClick={() => insert(`${fn.name}(`)}
                        title={`插入 ${fn.signature ?? fn.name}`}
                        className="flex w-full cursor-pointer items-baseline gap-2 rounded px-1 py-px text-left leading-4 transition-colors hover:bg-accent/10"
                      >
                        <code className="w-40 shrink-0 truncate font-mono text-accent">{fn.signature ?? `${fn.name}()`}</code>
                        <span className="text-muted">{fn.desc}</span>
                      </button>
                    ))}
                  </div>
                ) : (
                  <ul className="space-y-0.5">
                    {group.items.map(item => (
                      <li key={item.name} className="flex flex-wrap items-baseline gap-1">
                        {paletteTokens(item.name).map(tok => (
                          <button
                            key={tok.label}
                            type="button"
                            onClick={() => insert(group.key === 'modifiers' ? `,${tok.insert}` : tok.insert)}
                            title={`插入 ${tok.insert}${item.desc ? ` — ${item.desc}` : ''}`}
                            className={CHIP_CLS}
                          >
                            <span className="text-accent">{tok.label}</span>
                          </button>
                        ))}
                        {item.desc && <span className="text-muted">{item.desc}</span>}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </details>
      </section>

      {/* ── 右栏: 校验 / 预览 / 保存 ── */}
      <section aria-label="校验与预览" className="flex min-h-0 flex-col gap-3 rounded-card border border-border bg-surface/80 p-3 xl:overflow-y-auto">
        <div aria-live="polite" className="space-y-2">
          {!formula.trim() ? (
            <div className="rounded-btn border border-dashed border-border px-3 py-4 text-center text-xs text-muted">输入公式后自动校验</div>
          ) : validating && !v ? (
            <div className="text-xs text-muted">校验中…</div>
          ) : validation.isError ? (
            <div className="rounded-btn border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger" role="alert">
              校验请求失败 · {(validation.error as Error).message}
            </div>
          ) : v?.ok ? (
            <div className={`flex items-center gap-2 rounded-btn border border-bull/30 bg-bull/10 px-3 py-2 text-xs text-bull ${validating ? 'opacity-60' : ''}`}>
              <CircleCheck className="h-4 w-4 shrink-0" aria-hidden="true" />
              公式有效 · {v.outputs.length} 条输出 · 预热 {v.warmup_bars} 个交易日{validating && ' (校验中…)'}
            </div>
          ) : v ? (
            <div id={errorListId} className={`space-y-1 rounded-btn border border-danger/30 bg-danger/10 px-3 py-2 ${validating ? 'opacity-60' : ''}`}>
              <div className="flex items-center gap-2 text-xs font-medium text-danger">
                <CircleAlert className="h-4 w-4 shrink-0" aria-hidden="true" />公式未通过校验 (点击定位):
              </div>
              <ul className="space-y-0.5">
                {errors.map((e, i) => (
                  <li key={i}>
                    <button
                      type="button"
                      onClick={() => locate(e)}
                      disabled={!e.line}
                      className="block w-full text-left font-mono text-[11px] leading-relaxed text-danger hover:underline disabled:no-underline"
                    >
                      {errorPos(e) && <span className="mr-1.5 font-sans font-medium">{errorPos(e)}</span>}
                      {e.message}
                      <span className="ml-1 text-danger/60">[{e.code}]</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {warnings.length > 0 && (
            <div className="space-y-0.5 rounded-btn border border-amber-400/30 bg-amber-400/5 px-3 py-2">
              <div className="flex items-center gap-1.5 text-[11px] font-medium text-amber-500">
                <TriangleAlert className="h-3.5 w-3.5" aria-hidden="true" />{warnings.length} 条警告
              </div>
              <ul className="space-y-0.5">
                {warnings.map((w, i) => (
                  <li key={i}>
                    <button type="button" onClick={() => locate(w)} disabled={!w.line} className="text-left text-[11px] leading-relaxed text-amber-500/90 hover:underline disabled:no-underline">
                      {errorPos(w) && <span className="mr-1.5 font-medium">{errorPos(w)}</span>}{w.message}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {v?.ok && v.outputs.length > 0 && (
            <div className="rounded-btn border border-border bg-base/40 px-3 py-2 text-[11px]">
              <div className="mb-1 text-muted">输出线 (信号/策略中以 <code className="font-mono">ind_{mode === 'edit' ? source!.id : (newId || '<标识>')}_键</code> 引用)</div>
              <ul className="flex flex-wrap gap-1">
                {v.outputs.map(o => (
                  <li key={o.key} className={`inline-flex items-center gap-1 rounded-btn border border-border px-1.5 py-0.5 ${o.visible ? 'text-secondary' : 'text-muted'}`}>
                    <span className="inline-block h-0.5 w-2.5 rounded" style={{ backgroundColor: o.color ?? 'currentColor' }} aria-hidden="true" />
                    {o.name}
                    <span className="font-mono text-[10px] text-muted">{o.key} · {o.kind}{o.visible ? '' : ' · NODRAW'}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* 预览 */}
        <div className="space-y-1.5">
          <form
            className="flex flex-wrap items-end gap-2"
            onSubmit={e => { e.preventDefault(); commitSymbol(symbolInput) }}
          >
            <label className="min-w-[12rem] flex-1 space-y-1">
              <span className="text-[11px] text-muted">预览股票</span>
              <input
                value={symbolInput}
                onChange={e => setSymbolInput(e.target.value)}
                onBlur={() => commitSymbol(symbolInput)}
                list="indicator-preview-symbols"
                placeholder="代码或名称, 如 600519.SH"
                className={`${INPUT_CLS} h-8 font-mono`}
              />
              <datalist id="indicator-preview-symbols">
                {(suggestions.data?.results ?? []).map(r => (
                  <option key={r.symbol} value={r.symbol}>{r.name}</option>
                ))}
              </datalist>
            </label>
            <button type="submit" className="h-8 rounded-btn border border-border px-3 text-xs text-secondary transition-colors hover:border-accent/40 hover:text-accent">
              预览
            </button>
          </form>
          {!symbol && <div className="rounded-btn border border-dashed border-border px-3 py-4 text-center text-xs text-muted">选择股票后实时预览 (近 6 个月日K)</div>}
          {symbol && (
            <div className="rounded-btn border border-border bg-base/40 p-1.5" aria-label={`${symbol} 指标预览`}>
              <div className="flex items-center gap-2 px-1 pb-1 text-[10px] text-muted" aria-live="polite">
                <span className="font-mono text-secondary">{symbol}</span>
                {kline.data?.stock_info?.name && <span>{kline.data.stock_info.name}</span>}
                {(kline.isFetching || preview.isFetching) && <span>计算中…</span>}
                {!formulaOk && !validating && <span className="text-amber-500">公式未通过校验, 暂不更新预览</span>}
              </div>
              {preview.isError && (
                <div className="px-1 pb-1 text-[10px] text-danger" role="alert">预览失败 · {(preview.error as Error).message}</div>
              )}
              {preview.data?.errors && Object.values(preview.data.errors).length > 0 && (
                <div className="px-1 pb-1 text-[10px] text-danger" role="alert">{Object.values(preview.data.errors).join('；')}</div>
              )}
              {kline.isError && <div className="px-1 text-xs text-danger" role="alert">日K加载失败</div>}
              {kline.isLoading && <div className="px-1 py-6 text-center text-xs text-muted">加载日K…</div>}
              {!kline.isLoading && rows.length === 0 && !kline.isError && <div className="px-1 py-6 text-center text-xs text-muted">无日K数据</div>}
              {rows.length > 0 && (
                <EChartsCandlestick
                  data={rows}
                  height={pane === 'sub' ? 420 : 340}
                  maLines={[]}
                  showMarkers={false}
                  symbol={symbol}
                  visibleBars={90}
                  activeIndicators={previewIndicators}
                  overlays={overlays}
                />
              )}
            </div>
          )}
        </div>

        {/* 保存 */}
        <div className="mt-auto space-y-2 rounded-btn border border-border bg-base/40 p-2.5">
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => save.mutate(mode === 'edit' ? { kind: 'update' } : { kind: 'create', name, id: newId })}
              disabled={!!saveDisabledReason || save.isPending}
              title={saveDisabledReason || undefined}
              className="inline-flex h-8 items-center gap-1.5 whitespace-nowrap rounded-btn bg-accent px-3 text-xs font-medium text-white transition-colors hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Save className="h-3.5 w-3.5" aria-hidden="true" />
              {save.isPending ? '保存中…' : mode === 'edit' ? '保存' : '保存为新指标'}
            </button>
            {mode === 'edit' && (
              <button
                type="button"
                onClick={() => { setCopyOpen(o => !o); setCopyName(`${name.trim()} 副本`); setCopyId('') }}
                aria-expanded={copyOpen}
                aria-controls="indicator-saveas"
                className="inline-flex h-8 items-center gap-1.5 whitespace-nowrap rounded-btn border border-border px-3 text-xs text-secondary transition-colors hover:border-accent/40 hover:text-accent"
              >
                <Copy className="h-3.5 w-3.5" aria-hidden="true" />另存为…
              </button>
            )}
            {saveDisabledReason && <span className="text-[10px] text-muted">{saveDisabledReason}</span>}
          </div>
          {copyOpen && mode === 'edit' && (
            <div id="indicator-saveas" className="flex flex-wrap items-end gap-2">
              <label className="min-w-[10rem] flex-1 space-y-1">
                <span className="text-[11px] text-muted">新名称</span>
                <input value={copyName} onChange={e => setCopyName(e.target.value)} maxLength={40} className={`${INPUT_CLS} h-8`} />
              </label>
              <label className="min-w-[10rem] flex-1 space-y-1">
                <span className="text-[11px] text-muted">新标识 (可空)</span>
                <input value={copyId} onChange={e => setCopyId(sanitizeId(e.target.value))} className={`${INPUT_CLS} h-8 font-mono`} />
              </label>
              <button
                type="button"
                onClick={() => save.mutate({ kind: 'create', name: copyName, id: copyId })}
                disabled={!copyName.trim() || !formulaOk || save.isPending}
                className="h-8 rounded-btn bg-accent px-3 text-xs font-medium text-white transition-colors hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                确认另存
              </button>
            </div>
          )}
          {save.isError && (
            <div className="text-[11px] text-danger" role="alert">保存失败 · {(save.error as Error).message}</div>
          )}
          <div className="text-[10px] leading-relaxed text-muted">
            保存时服务端会重新编译, 未通过校验的公式不会保存。保存后可在 K 线「主图 ▾」下拉或副图胶囊中开启, 输出线出现在自定义信号字段「自定义指标」分组。
          </div>
        </div>
      </section>
    </div>
  )
}
