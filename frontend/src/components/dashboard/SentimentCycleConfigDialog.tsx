/**
 * 情绪周期口径配置对话框。
 *
 * 可编辑: 防抖天数 / 回看窗口 / 大面与高位核按钮阈值 / 各阶段进入条件的
 * 比较符、阈值、权重; 支持恢复出厂口径。
 *
 * 口径契约在后端 app/services/sentiment_cycle.py: 前端不硬编码任何阈值,
 * 全部由 GET /api/sentiment-cycle/config 返回的结构渲染。
 */
import { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { Info, Loader2, RotateCcw, X } from 'lucide-react'
import { api, type SentimentCycleConfig } from '@/lib/api'
import { useDialogBackdrop } from '@/lib/useDialogBackdrop'
import { toast } from '@/components/Toast'

type Condition = { key: string; cmp: string; value: number; weight: number }
type Draft = {
  lookback: number
  confirm_days: number
  big_loss_pct: number
  high_nuke_pct: number
  include_bj: boolean
  score_threshold: number
  style_retreat_discount: number
  phases: Record<string, { conditions: Condition[] }>
  sector_phases: Record<string, { conditions: Condition[] }>
}

type Section = 'phases' | 'sector_phases'

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v))

const CMP_OPTIONS = [
  { value: 'gte', label: '≥' },
  { value: 'lte', label: '≤' },
  { value: 'gt', label: '>' },
  { value: 'lt', label: '<' },
  { value: 'pct_gte', label: '分位≥' },
  { value: 'pct_lte', label: '分位≤' },
  { value: 'trend3_gt', label: '3日变化>' },
  { value: 'trend3_lt', label: '3日变化<' },
]

const METRIC_LABELS: Record<string, string> = {
  limit_up: '涨停家数',
  limit_down: '跌停家数',
  broken: '炸板家数',
  first_board: '首板家数',
  max_height: '最高连板',
  height_broken: '最高板断板',
  seal_rate: '封板率',
  seal_rate_drop: '封板率回落',
  promo_1_2: '1进2',
  promo_2_3: '2进3',
  promo_3p: '3板以上晋级',
  premium_lu: '涨停溢价',
  premium_multi: '连板溢价',
  premium_broken: '炸板溢价',
  big_loss: '大面数',
  high_nuke: '高位核按钮',
  up_ratio: '上涨占比',
  avg_pct: '平均涨跌',
}

const PHASE_COLORS: Record<string, string> = {
  冰点: '#12B76A', 修复: '#84CC16', 发酵: '#F59E0B',
  高潮: '#F04438', 分歧: '#FB923C', 退潮: '#6B7280',
}

export function SentimentCycleConfigDialog({ onClose }: { onClose: () => void }) {
  const backdrop = useDialogBackdrop(onClose)
  const [payload, setPayload] = useState<SentimentCycleConfig | null>(null)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [section, setSection] = useState<Section>('phases')

  useEffect(() => {
    let alive = true
    api.sentimentCycleConfig()
      .then(res => {
        if (!alive) return
        setPayload(res)
        setDraft(clone(res.config as unknown as Draft))
      })
      .catch(e => alive && setError(String(e)))
      .finally(() => alive && setLoading(false))
    return () => { alive = false }
  }, [])

  const setField = <K extends keyof Draft>(key: K, v: Draft[K]) =>
    setDraft(d => d && { ...d, [key]: v })

  const setCond = (phase: string, idx: number, field: keyof Condition, v: string | number) =>
    setDraft(d => {
      if (!d) return d
      const rules = d[section]
      const conditions = rules[phase].conditions.map((c, i) =>
        i === idx ? { ...c, [field]: v } : c,
      )
      return { ...d, [section]: { ...rules, [phase]: { conditions } } }
    })

  const save = async () => {
    if (!draft) return
    setSaving(true); setError('')
    try {
      const res = await api.saveSentimentCycleConfig(draft as unknown as Record<string, unknown>)
      setPayload(res)
      toast('情绪周期口径已保存', 'success')
      onClose()
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  const reset = async () => {
    setSaving(true); setError('')
    try {
      const res = await api.saveSentimentCycleConfig(null)
      setPayload(res)
      setDraft(clone(res.config as unknown as Draft))
      toast('已恢复出厂口径', 'success')
    } catch (e) {
      setError(String(e))
    } finally {
      setSaving(false)
    }
  }

  const phases = payload?.phases ?? []
  const transitions = payload?.transitions ?? {}

  return createPortal(
    <div className="fixed inset-0 z-[60] flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" {...backdrop} />
      <div className="relative mx-4 flex max-h-[88vh] w-full max-w-3xl flex-col overflow-hidden rounded-dialog border border-border bg-surface shadow-2xl">
        <div className="flex items-center gap-2.5 border-b border-border px-5 py-3.5">
          <h3 className="text-sm font-medium text-foreground">情绪周期 · 判定口径</h3>
          {payload?.customized && (
            <span className="rounded-full bg-accent/15 px-2 py-0.5 text-[10px] font-medium leading-none text-accent">已自定义</span>
          )}
          <button onClick={onClose} className="ml-auto rounded p-0.5 text-secondary hover:bg-elevated hover:text-foreground" aria-label="关闭">
            <X className="h-4 w-4" />
          </button>
        </div>

        {loading ? (
          <div className="flex items-center justify-center gap-2 py-20 text-xs text-muted">
            <Loader2 className="h-4 w-4 animate-spin" /> 加载口径…
          </div>
        ) : !draft ? (
          <div className="px-5 py-16 text-center text-xs text-danger">{error || '加载失败'}</div>
        ) : (
          <>
            <div className="flex-1 overflow-y-auto px-5 py-4">
              <div className="rounded-lg border border-border/60 bg-elevated/30 px-3.5 py-2.5">
                <div className="flex items-start gap-2 text-[11px] leading-relaxed text-secondary">
                  <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent/70" />
                  <div>
                    状态机按 <span className="text-foreground">{phases.join(' → ')}</span> 顺序推进，
                    只允许配置的合法转移。每个阶段的条件按权重合成得分，
                    达到阈值且连续 <span className="text-foreground">防抖天数</span> 满足才切换。
                    分位比较基于回看窗口；3 日变化为当前值与 3 日前之差。
                  </div>
                </div>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
                <NumberField label="回看窗口（交易日）" value={draft.lookback}
                  onChange={v => setField('lookback', v)} min={10} max={250} />
                <NumberField label="防抖天数" value={draft.confirm_days}
                  onChange={v => setField('confirm_days', v)} min={1} max={10} />
                <NumberField label="判定阈值（0-1）" value={draft.score_threshold}
                  onChange={v => setField('score_threshold', v)} min={0} max={1} step={0.05} />
                <NumberField label="大面阈值（%）" value={draft.big_loss_pct}
                  onChange={v => setField('big_loss_pct', v)} min={-20} max={0} />
                <NumberField label="高位核按钮阈值（%）" value={draft.high_nuke_pct}
                  onChange={v => setField('high_nuke_pct', v)} min={-20} max={0} />
                <NumberField label="风格联动·退潮阈值下调" value={draft.style_retreat_discount}
                  onChange={v => setField('style_retreat_discount', v)} min={0} max={0.5} step={0.05} />
                <label className="flex items-end gap-1.5 pb-1.5 text-[11px] text-secondary">
                  <input type="checkbox" checked={draft.include_bj}
                    onChange={e => setField('include_bj', e.target.checked)} />
                  纳入北交所
                </label>
              </div>

              <div className="mt-4 flex gap-1 rounded-lg bg-elevated/40 p-1">
                {([
                  ['phases', '全市场规则'],
                  ['sector_phases', '板块规则'],
                ] as [Section, string][]).map(([key, label]) => (
                  <button key={key} type="button" onClick={() => setSection(key)}
                    className={`rounded-md px-3 py-1 text-[11px] transition-colors ${section === key ? 'bg-surface text-foreground shadow-sm' : 'text-muted hover:text-secondary'}`}>
                    {label}
                  </button>
                ))}
              </div>
              <p className="mt-1.5 text-[10px] leading-relaxed text-muted">
                {section === 'phases'
                  ? '全市场情绪周期的阶段条件。'
                  : '板块周期只统计成分股 ≥10 只的板块, 以板块自身历史分位与平均涨跌为主, 不使用全市场的绝对家数阈值。'}
                机构主导/风格切换中时, 最新一日「退潮」的判定阈值按上方设置下调。
              </p>

              <div className="mt-3 space-y-2.5">
                {phases.map(phase => (
                  <div key={phase} className="rounded-lg border border-border/60 bg-base/30">
                    <div className="flex items-center gap-2 border-b border-border/40 px-3 py-1.5">
                      <span className="h-2 w-2 rounded-full" style={{ backgroundColor: PHASE_COLORS[phase] ?? '#6B7280' }} />
                      <span className="text-xs font-medium text-foreground">{phase}</span>
                      <span className="ml-auto text-[10px] text-muted">
                        可由 {Object.entries(transitions).filter(([, to]) => to.includes(phase)).map(([from]) => from).join('、') || '—'} 进入
                      </span>
                    </div>
                    <div className="grid grid-cols-[1fr_86px_72px_56px] items-center gap-2 px-3 pb-1 pt-1.5 text-[10px] text-muted">
                      <span>指标</span><span>比较</span><span className="text-right">阈值</span><span className="text-right">权重</span>
                    </div>
                    {draft[section]?.[phase]?.conditions.map((c, idx) => (
                      <div key={`${phase}-${c.key}`} className="grid grid-cols-[1fr_86px_72px_56px] items-center gap-2 px-3 pb-1.5">
                        <span className="truncate text-[11px] text-secondary">{METRIC_LABELS[c.key] ?? c.key}</span>
                        <select value={c.cmp} onChange={e => setCond(phase, idx, 'cmp', e.target.value)}
                          className="h-6 rounded border border-border bg-base px-1 text-[11px] text-foreground focus:outline-none">
                          {CMP_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                        </select>
                        <input type="number" step={0.01} value={c.value}
                          onChange={e => setCond(phase, idx, 'value', Number(e.target.value))}
                          className="h-6 w-full rounded border border-border bg-base px-1.5 text-right font-mono text-[11px] text-foreground focus:outline-none" />
                        <input type="number" min={0} step={0.1} value={c.weight}
                          onChange={e => setCond(phase, idx, 'weight', Number(e.target.value))}
                          className="h-6 w-full rounded border border-border bg-base px-1.5 text-right font-mono text-[11px] text-foreground focus:outline-none" />
                      </div>
                    ))}
                  </div>
                ))}
              </div>
              {error && <p className="mt-3 text-[11px] text-danger">{error}</p>}
            </div>

            <div className="flex items-center gap-2 border-t border-border px-5 py-3">
              <button onClick={reset} disabled={saving}
                className="inline-flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs text-secondary hover:bg-elevated hover:text-foreground disabled:opacity-50">
                <RotateCcw className="h-3 w-3" /> 恢复出厂
              </button>
              <span className="ml-auto" />
              <button onClick={onClose} className="rounded-lg px-3 py-1.5 text-xs text-secondary hover:bg-elevated">取消</button>
              <button onClick={save} disabled={saving}
                className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white hover:bg-accent/90 disabled:opacity-50">
                {saving && <Loader2 className="h-3 w-3 animate-spin" />} 保存
              </button>
            </div>
          </>
        )}
      </div>
    </div>,
    document.body,
  )
}

function NumberField({ label, value, onChange, min, max, step = 1 }: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[10px] text-muted">{label}</span>
      <input type="number" value={value} min={min} max={max} step={step}
        onChange={e => onChange(Number(e.target.value))}
        className="h-7 w-full rounded border border-border bg-base px-2 text-right font-mono text-[11px] text-foreground focus:outline-none focus:ring-1 focus:ring-accent/50" />
    </label>
  )
}
