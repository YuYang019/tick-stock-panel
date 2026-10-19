/**
 * 技术面心法设置页。
 *
 * 三个 Tab:
 *   心法库 — 出厂/用户心法列表, 复制、设为全局、按场景指定、删除;
 *   编辑器 — 模块开关/权重/特征参数/原则正文/检查项/情绪周期对照表;
 *   预览 — 选股票跑规则层评估并查看拼装后的提示词 (不调用 AI)。
 *
 * 心法契约在后端 app/analysis/method.py; 前端不硬编码阈值或模块清单。
 */
import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { BookOpen, Check, Copy, Eye, Loader2, RotateCcw, Save, Trash2 } from 'lucide-react'
import {
  api,
  type TaMethod,
  type TaMethodModule,
  type TaMethodSummary,
  type TaSelection,
} from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { toast } from '@/components/Toast'
import { cn } from '@/lib/cn'

const SCENARIOS: { key: string; label: string }[] = [
  { key: 'stock_analysis', label: '个股分析' },
  { key: 'watchlist_review', label: '复盘·自选股' },
  { key: 'market_review', label: '大盘复盘' },
]

const LEVEL_LABELS: Record<string, string> = {
  worldview: '总纲', method: '方法论', structure: '结构框架',
  market: '市场层', sector: '板块层', stock: '个股层',
}

type Tab = 'library' | 'editor' | 'preview'

export function SettingsTaMethodPanel() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<Tab>('library')
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const library = useQuery({ queryKey: QK.taMethods, queryFn: api.taMethods })
  const selection = library.data?.selection ?? { active: null, per_scenario: {} }
  const methods = library.data?.methods ?? []

  const methodQuery = useQuery({
    queryKey: QK.taMethod(selectedId ?? ''),
    queryFn: () => api.taMethod(selectedId as string),
    enabled: !!selectedId && tab !== 'library',
  })

  const [draft, setDraft] = useState<TaMethod | null>(null)
  const current = draft ?? methodQuery.data?.method ?? null
  const isFactory = methodQuery.data?.factory ?? selectedId === 'fusion-v1'

  const selectionMut = useMutation({
    mutationFn: (next: TaSelection) => api.updateTaSelection(next),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QK.taMethods })
      toast('心法选择已保存', 'success')
    },
  })

  const copyMut = useMutation({
    mutationFn: ({ id }: { id: string }) => api.copyTaMethod(id),
    onSuccess: res => {
      qc.invalidateQueries({ queryKey: QK.taMethods })
      setSelectedId(res.method.id)
      setDraft(null)
      setTab('editor')
      toast('已复制为新心法', 'success')
    },
    onError: (e: Error) => toast(e.message, 'error'),
  })

  const saveMut = useMutation({
    mutationFn: (method: TaMethod) => api.saveTaMethod(method.id, method),
    onSuccess: res => {
      qc.invalidateQueries({ queryKey: QK.taMethods })
      qc.invalidateQueries({ queryKey: QK.taMethod(res.method.id) })
      setDraft(null)
      toast('心法已保存', 'success')
    },
    onError: (e: Error) => toast(e.message, 'error'),
  })

  const deleteMut = useMutation({
    mutationFn: (id: string) => api.deleteTaMethod(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: QK.taMethods })
      setSelectedId(null)
      toast('已删除', 'success')
    },
    onError: (e: Error) => toast(e.message, 'error'),
  })

  const updateDraft = (fn: (m: TaMethod) => TaMethod) => {
    const base = current
    if (!base) return
    setDraft(fn(JSON.parse(JSON.stringify(base)) as TaMethod))
  }

  return (
    <div className="max-w-4xl">
      <div className="mb-4 flex items-center gap-2.5">
        <BookOpen className="h-4 w-4 text-secondary" />
        <h2 className="text-sm font-medium text-foreground">技术面心法</h2>
        <span className="text-[11px] text-muted">
          规则层算事实, AI 做解读; 模块、权重、原则正文均可编辑
        </span>
      </div>

      <div className="mb-3 flex gap-1 rounded-lg bg-elevated/40 p-1">
        {([
          ['library', '心法库'],
          ['editor', '编辑器'],
          ['preview', '预览'],
        ] as [Tab, string][]).map(([key, label]) => (
          <button key={key} onClick={() => setTab(key)}
            className={cn('rounded-md px-3 py-1 text-xs transition-colors',
              tab === key ? 'bg-surface text-foreground shadow-sm' : 'text-muted hover:text-secondary')}>
            {label}
          </button>
        ))}
      </div>

      {tab === 'library' && (
        <Library
          methods={methods}
          selection={selection}
          loading={library.isLoading}
          onSelect={id => { setSelectedId(id); setDraft(null); setTab('editor') }}
          onCopy={id => copyMut.mutate({ id })}
          onDelete={id => deleteMut.mutate(id)}
          onSetActive={id => selectionMut.mutate({ ...selection, active: id })}
          onSetScenario={(scenario, id) =>
            selectionMut.mutate({
              ...selection,
              per_scenario: { ...selection.per_scenario, [scenario]: id },
            })}
        />
      )}

      {tab === 'editor' && (
        <Editor
          method={current}
          loading={methodQuery.isLoading}
          isFactory={isFactory}
          dirty={!!draft}
          onReset={() => setDraft(null)}
          onChange={updateDraft}
          onSave={() => current && saveMut.mutate(current)}
          saving={saveMut.isPending}
        />
      )}

      {tab === 'preview' && <Preview methods={methods} defaultId={selectedId} />}
    </div>
  )
}

function Library({
  methods, selection, loading, onSelect, onCopy, onDelete, onSetActive, onSetScenario,
}: {
  methods: TaMethodSummary[]
  selection: TaSelection
  loading: boolean
  onSelect: (id: string) => void
  onCopy: (id: string) => void
  onDelete: (id: string) => void
  onSetActive: (id: string) => void
  onSetScenario: (scenario: string, id: string | null) => void
}) {
  if (loading) {
    return <div className="flex items-center gap-2 py-10 text-xs text-muted"><Loader2 className="h-4 w-4 animate-spin" /> 加载中…</div>
  }
  return (
    <div className="space-y-2">
      <p className="text-[11px] leading-relaxed text-muted">
        全局心法用于未单独指定的场景; 每个场景可覆盖。出厂心法只读, 复制后可自由编辑。
      </p>
      {methods.map(m => {
        const active = selection.active === m.id
        return (
          <div key={m.id} className={cn(
            'rounded-card border bg-surface px-4 py-3',
            active ? 'border-accent/50' : 'border-border',
          )}>
            <div className="flex items-start gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-foreground">{m.name}</span>
                  <span className="rounded bg-elevated px-1.5 py-px font-mono text-[10px] text-muted">{m.id}</span>
                  <span className="text-[10px] text-muted">v{m.version}</span>
                  {m.factory && <span className="rounded bg-accent/15 px-1.5 py-px text-[10px] text-accent">出厂</span>}
                  {active && <span className="rounded bg-bear/15 px-1.5 py-px text-[10px] text-bear">全局</span>}
                </div>
                <p className="mt-1 text-[11px] leading-relaxed text-muted">
                  {m.module_count} 个模块 · {m.maxim_count} 条原则{m.description ? ` · ${m.description}` : ''}
                </p>
              </div>
              <div className="flex shrink-0 items-center gap-1">
                <button onClick={() => onSelect(m.id)}
                  className="rounded-lg px-2 py-1 text-[11px] text-secondary hover:bg-elevated hover:text-foreground">编辑</button>
                <button onClick={() => onCopy(m.id)}
                  className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-[11px] text-secondary hover:bg-elevated hover:text-foreground">
                  <Copy className="h-3 w-3" /> 复制
                </button>
                {!active && (
                  <button onClick={() => onSetActive(m.id)}
                    className="rounded-lg px-2 py-1 text-[11px] text-accent hover:bg-accent/10">设为全局</button>
                )}
                {!m.factory && (
                  <button onClick={() => onDelete(m.id)}
                    className="rounded-lg p-1 text-muted hover:bg-danger/10 hover:text-danger" title="删除">
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-border/40 pt-2">
              <span className="text-[10px] text-muted">场景覆盖:</span>
              {SCENARIOS.map(s => (
                <label key={s.key} className="flex items-center gap-1 text-[10px] text-muted">
                  {s.label}
                  <select
                    value={selection.per_scenario[s.key] ?? ''}
                    onChange={e => onSetScenario(s.key, e.target.value || null)}
                    className="rounded border border-border bg-base px-1 py-0.5 text-[10px] text-foreground">
                    <option value="">跟随全局</option>
                    {methods.map(opt => <option key={opt.id} value={opt.id}>{opt.name}</option>)}
                  </select>
                </label>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function Editor({
  method, loading, isFactory, dirty, onReset, onChange, onSave, saving,
}: {
  method: TaMethod | null
  loading: boolean
  isFactory: boolean
  dirty: boolean
  onReset: () => void
  onChange: (fn: (m: TaMethod) => TaMethod) => void
  onSave: () => void
  saving: boolean
}) {
  if (loading) {
    return <div className="flex items-center gap-2 py-10 text-xs text-muted"><Loader2 className="h-4 w-4 animate-spin" /> 加载心法…</div>
  }
  if (!method) {
    return <p className="py-10 text-center text-xs text-muted">从「心法库」选择一个心法开始编辑</p>
  }

  const patchModule = (id: string, patch: Partial<TaMethodModule>) =>
    onChange(m => ({ ...m, modules: m.modules.map(mod => mod.id === id ? { ...mod, ...patch } : mod) }))

  const patchMaxim = (id: string, patch: Partial<TaMethod['maxims'][number]>) =>
    onChange(m => ({ ...m, maxims: m.maxims.map(x => x.id === id ? { ...x, ...patch } : x) }))

  return (
    <div className="space-y-3">
      {isFactory && (
        <div className="rounded-lg border border-warning/30 bg-warning/5 px-3 py-2 text-[11px] text-warning">
          出厂心法只读。请先在心法库中「复制」, 再编辑副本。
        </div>
      )}

      <section className="rounded-card border border-border bg-surface p-4">
        <h3 className="mb-2 text-xs font-medium text-foreground">基础信息</h3>
        <div className="grid gap-2 sm:grid-cols-2">
          <label className="block text-[11px] text-muted">
            名称
            <input value={method.name} disabled={isFactory}
              onChange={e => onChange(m => ({ ...m, name: e.target.value }))}
              className="mt-1 w-full rounded border border-border bg-base px-2 py-1 text-xs text-foreground disabled:opacity-60" />
          </label>
          <label className="block text-[11px] text-muted">
            版本
            <input value={method.version} readOnly
              className="mt-1 w-full rounded border border-border bg-elevated px-2 py-1 font-mono text-xs text-muted" />
          </label>
        </div>
      </section>

      <section className="rounded-card border border-border bg-surface p-4">
        <h3 className="mb-2 text-xs font-medium text-foreground">模块与权重</h3>
        <div className="space-y-2">
          {method.modules.map(mod => (
            <div key={mod.id} className="rounded-lg border border-border/60 bg-base/30 px-3 py-2">
              <div className="flex flex-wrap items-center gap-2">
                <input type="checkbox" checked={mod.enabled} disabled={isFactory}
                  onChange={e => patchModule(mod.id, { enabled: e.target.checked })} />
                <span className="text-xs font-medium text-foreground">{mod.name}</span>
                <span className="rounded bg-elevated px-1 py-px font-mono text-[10px] text-muted">{mod.id}</span>
                {mod.scope === 'market' && (
                  <span className="rounded bg-accent/10 px-1 py-px text-[10px] text-accent">市场层</span>
                )}
                <span className="ml-auto flex items-center gap-1 text-[10px] text-muted">
                  权重
                  <input type="number" min={0} max={1} step={0.01} value={mod.weight} disabled={isFactory}
                    onChange={e => patchModule(mod.id, { weight: Number(e.target.value) })}
                    className="w-16 rounded border border-border bg-base px-1 py-0.5 text-right font-mono text-[11px] text-foreground" />
                </span>
              </div>
              <div className="mt-1 flex flex-wrap gap-1">
                {mod.features.map(f => (
                  <span key={f.id} className="rounded bg-elevated/60 px-1.5 py-0.5 font-mono text-[10px] text-secondary">
                    {f.id}
                  </span>
                ))}
              </div>
              <textarea value={mod.principles} disabled={isFactory} rows={3}
                onChange={e => patchModule(mod.id, { principles: e.target.value })}
                placeholder="该维度的心法正文 (进入提示词)"
                className="mt-2 w-full resize-y rounded border border-border bg-base px-2 py-1.5 text-[11px] leading-relaxed text-foreground disabled:opacity-60" />
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-card border border-border bg-surface p-4">
        <h3 className="mb-2 text-xs font-medium text-foreground">投资原则 ({method.maxims.length})</h3>
        <div className="space-y-2">
          {method.maxims.map(maxim => (
            <div key={maxim.id} className="rounded-lg border border-border/60 bg-base/30 px-3 py-2">
              <div className="flex flex-wrap items-center gap-2">
                <input type="checkbox" checked={maxim.enabled} disabled={isFactory}
                  onChange={e => patchMaxim(maxim.id, { enabled: e.target.checked })} />
                <span className="text-xs font-medium text-foreground">{maxim.title}</span>
                <span className="rounded bg-elevated px-1 py-px text-[10px] text-muted">
                  {LEVEL_LABELS[maxim.level] ?? maxim.level}
                </span>
                <span className="ml-auto text-[10px] text-muted">
                  {maxim.hooks?.length ? `${maxim.hooks.length} 条规则钩子` : '纯文本原则'}
                </span>
              </div>
              <textarea value={maxim.prompt_principle} disabled={isFactory} rows={2}
                onChange={e => patchMaxim(maxim.id, { prompt_principle: e.target.value })}
                placeholder="要求模型遵循的原则 (进入提示词)"
                className="mt-2 w-full resize-y rounded border border-border bg-base px-2 py-1.5 text-[11px] leading-relaxed text-foreground disabled:opacity-60" />
            </div>
          ))}
        </div>
      </section>

      {!isFactory && (
        <div className="sticky bottom-0 flex items-center gap-2 rounded-card border border-border bg-surface/95 px-4 py-2.5 backdrop-blur">
          <button onClick={onSave} disabled={!dirty || saving}
            className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accent/90 disabled:opacity-50">
            {saving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />} 保存
          </button>
          <button onClick={onReset} disabled={!dirty}
            className="inline-flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs text-secondary hover:bg-elevated disabled:opacity-50">
            <RotateCcw className="h-3 w-3" /> 放弃修改
          </button>
          {dirty && <span className="text-[10px] text-warning">有未保存修改</span>}
        </div>
      )}
    </div>
  )
}

function Preview({ methods, defaultId }: { methods: TaMethodSummary[]; defaultId: string | null }) {
  const [symbol, setSymbol] = useState('600519.SH')
  const [methodId, setMethodId] = useState(defaultId ?? '')
  const preview = useMutation({
    mutationFn: () => api.previewTaMethod(symbol.trim(), methodId || undefined),
  })

  const evaluation = preview.data?.evaluation as Record<string, unknown> | undefined
  const modules = useMemo(
    () => (evaluation?.modules as Record<string, { score: number | null; features: { id: string; state: string; score: number | null }[] }>) ?? {},
    [evaluation],
  )

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-end gap-2 rounded-card border border-border bg-surface p-3">
        <label className="text-[11px] text-muted">
          标的
          <input value={symbol} onChange={e => setSymbol(e.target.value)}
            className="mt-1 block w-40 rounded border border-border bg-base px-2 py-1 font-mono text-xs text-foreground" />
        </label>
        <label className="text-[11px] text-muted">
          心法
          <select value={methodId} onChange={e => setMethodId(e.target.value)}
            className="mt-1 block rounded border border-border bg-base px-2 py-1 text-xs text-foreground">
            <option value="">当前场景心法</option>
            {methods.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </label>
        <button onClick={() => preview.mutate()} disabled={preview.isPending || !symbol.trim()}
          className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accent/90 disabled:opacity-50">
          {preview.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <Eye className="h-3 w-3" />} 运行预览
        </button>
        {preview.error && <span className="text-[11px] text-danger">{(preview.error as Error).message}</span>}
      </div>

      {evaluation && (
        <>
          <section className="rounded-card border border-border bg-surface p-4">
            <h3 className="mb-2 flex items-center gap-1.5 text-xs font-medium text-foreground">
              <Check className="h-3.5 w-3.5 text-accent" /> 规则层评估
            </h3>
            <div className="grid gap-2 sm:grid-cols-4">
              <Metric label="综合评分" value={String(evaluation.composite ?? '—')} />
              <Metric label="状态标签" value={String(evaluation.rule_label ?? '—')} />
              <Metric label="置信度" value={String(evaluation.confidence ?? '—')} />
              <Metric label="情绪周期" value={String(evaluation.cycle_phase ?? '—')} />
            </div>
            <div className="mt-3 overflow-x-auto">
              <table className="w-full text-left text-[11px]">
                <thead className="text-muted">
                  <tr>
                    <th className="py-1 pr-2">模块</th>
                    <th className="py-1 pr-2 text-right">权重</th>
                    <th className="py-1 pr-2 text-right">评分</th>
                    <th className="py-1">特征状态</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(modules).map(([id, mod]) => (
                    <tr key={id} className="border-t border-border/40">
                      <td className="py-1 pr-2 font-mono text-secondary">{id}</td>
                      <td className="py-1 pr-2 text-right text-muted">
                        {String((evaluation.effective_weights as Record<string, number>)?.[id] ?? '—')}
                      </td>
                      <td className="py-1 pr-2 text-right">{mod.score ?? '—'}</td>
                      <td className="py-1 text-muted">
                        {mod.features.map(f => `${f.id}:${f.state}`).join(' · ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="rounded-card border border-border bg-surface p-4">
            <h3 className="mb-2 text-xs font-medium text-foreground">拼装后的提示词</h3>
            <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-lg bg-base p-3 text-[10px] leading-relaxed text-secondary">
              {preview.data?.prompt.system}
            </pre>
            <pre className="mt-2 max-h-60 overflow-auto whitespace-pre-wrap rounded-lg bg-base p-3 text-[10px] leading-relaxed text-muted">
              {preview.data?.prompt.user}
            </pre>
          </section>
        </>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border/50 bg-elevated/30 px-2.5 py-1.5">
      <div className="text-[10px] text-muted">{label}</div>
      <div className="mt-0.5 font-mono text-xs font-semibold text-foreground">{value}</div>
    </div>
  )
}
