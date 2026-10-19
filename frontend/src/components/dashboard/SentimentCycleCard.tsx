import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Activity, Settings2 } from 'lucide-react'
import { api, type SentimentCycleState } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { MiniMetric } from './shared'
import { SentimentCycleConfigDialog } from './SentimentCycleConfigDialog'

const PHASE_COLORS: Record<string, string> = {
  冰点: '#12B76A',
  修复: '#84CC16',
  发酵: '#F59E0B',
  高潮: '#F04438',
  分歧: '#FB923C',
  退潮: '#6B7280',
}

const STYLE_TONE: Record<string, string> = {
  题材主导: 'text-accent',
  机构主导: 'text-foreground',
  普涨: 'text-bull',
  普跌: 'text-bear',
  切换中: 'text-warning',
  均衡: 'text-muted',
}

/** 情绪周期卡片: 阶段标签 + 近 20 日色带 + 核心指标 + 极致预警。
 *  纯规则计算, 不依赖 AI Key。 */
export function SentimentCycleCard() {
  const [configOpen, setConfigOpen] = useState(false)
  const cycle = useQuery({
    queryKey: QK.sentimentCycle(),
    queryFn: () => api.sentimentCycle(),
    refetchInterval: 5 * 60_000,
  })
  const style = useQuery({
    queryKey: QK.marketStyle(),
    queryFn: () => api.marketStyle(),
    refetchInterval: 5 * 60_000,
  })
  const sectors = useQuery({
    queryKey: QK.sectorCycles('concept'),
    queryFn: () => api.sectorCycles('concept'),
    refetchInterval: 5 * 60_000,
  })

  const data = cycle.data
  const phase = data?.phase ?? '—'
  const color = PHASE_COLORS[phase] ?? '#6B7280'

  return (
    <section
      className="h-full overflow-hidden rounded-card border bg-surface/80 p-1.5 shadow-[0_1px_2px_hsl(var(--border)/0.4)] backdrop-blur-sm"
      style={{ borderColor: `${color}55` }}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <span className="h-3 w-0.5 rounded-full bg-gradient-to-b from-accent to-accent/30" />
          <Activity className="h-3.5 w-3.5 text-accent" />
          <h2 className="text-xs font-semibold text-foreground">情绪周期</h2>
        </div>
        <div className="flex items-center gap-1.5">
          {data?.extreme && (
            <span
              className={`rounded px-1 text-[9px] ${data.extreme === '赚钱' ? 'bg-bull/15 text-bull' : 'bg-bear/15 text-bear'}`}
              title={(data.extreme_items ?? []).join('、')}
            >
              极致{data.extreme}
            </span>
          )}
          <button
            type="button"
            onClick={() => setConfigOpen(true)}
            title="配置情绪周期口径"
            aria-label="配置情绪周期口径"
            className="rounded p-0.5 text-muted transition-colors hover:bg-elevated hover:text-accent"
          >
            <Settings2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {!data?.available ? (
        <p className="px-1 text-[11px] text-muted">
          {cycle.isLoading ? '计算中…' : (data?.reason || cycle.error?.message || '暂无数据')}
        </p>
      ) : (
        <div className="space-y-2">
          <div className="flex items-end justify-between gap-2 px-1">
            <div>
              <div className="text-lg font-semibold leading-none" style={{ color }}>
                {phase}
                {data.pending && (
                  <span className="ml-1.5 align-middle text-[10px] font-normal text-muted">
                    候选 {data.pending}
                  </span>
                )}
              </div>
              <div className="mt-1 text-[10px] text-muted">
                第 {data.days_in_phase} 天 · 置信度 {(data.confidence * 100).toFixed(0)}%
              </div>
            </div>
            <div className="text-right">
              <div className="text-[10px] text-muted">市场风格</div>
              <div className={`text-xs font-medium ${STYLE_TONE[style.data?.style ?? ''] ?? 'text-muted'}`}>
                {style.data?.style ?? '—'}
              </div>
            </div>
          </div>

          <PhaseStrip history={data.history} current={phase} />

          <div className="grid grid-cols-3 gap-1 px-0.5">
            <MiniMetric label="涨停" value={String(data.metrics.limit_up ?? '—')} cls="text-bull" />
            <MiniMetric label="最高板" value={String(data.metrics.max_height ?? '—')} />
            <MiniMetric
              label="涨停溢价"
              value={fmtPct(data.metrics.premium_lu)}
              cls={num(data.metrics.premium_lu) >= 0 ? 'text-bull' : 'text-bear'}
            />
            <MiniMetric label="封板率" value={fmtPct(data.metrics.seal_rate, 0, '%')} />
            <MiniMetric label="炸板" value={String(data.metrics.broken ?? '—')} />
            <MiniMetric label="大面" value={String(data.metrics.big_loss ?? '—')} cls="text-bear" />
          </div>

          {sectors.data?.available && sectors.data.sectors.length > 0 && (
            <div className="space-y-0.5 px-1" title="当日领涨概念的板块周期 (规则判定)">
              <div className="text-[10px] text-muted">主线板块周期</div>
              {sectors.data.sectors.slice(0, 4).map(sec => (
                <div key={sec.name} className="flex items-center gap-1.5 text-[10px]">
                  <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: PHASE_COLORS[sec.phase] ?? '#6B7280' }} />
                  <span className="min-w-0 flex-1 truncate text-secondary" title={sec.name}>{sec.name}</span>
                  <span className="shrink-0" style={{ color: PHASE_COLORS[sec.phase] ?? undefined }}>
                    {sec.phase}{sec.pending ? `→${sec.pending}?` : ''}
                  </span>
                </div>
              ))}
            </div>
          )}

          {data.evidence.length > 0 && (
            <p className="line-clamp-2 px-1 text-[10px] leading-relaxed text-muted" title={data.evidence.join('；')}>
              {data.evidence.slice(0, 3).join('；')}
            </p>
          )}
        </div>
      )}

      {configOpen && <SentimentCycleConfigDialog onClose={() => setConfigOpen(false)} />}
    </section>
  )
}

function PhaseStrip({ history, current }: { history: SentimentCycleState['history']; current: string }) {
  const items = (history ?? []).slice(-20)
  if (!items.length) return null
  return (
    <div className="flex items-center gap-0.5 px-1" title="近 20 个交易日阶段轨迹">
      {items.map((h, i) => (
        <span
          key={`${h.date}-${i}`}
          className="h-2.5 flex-1 rounded-[1px]"
          style={{ backgroundColor: PHASE_COLORS[h.phase] ?? '#6B7280', opacity: h.phase === current ? 1 : 0.55 }}
          title={`${h.date} ${h.phase}`}
        />
      ))}
    </div>
  )
}

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0
}

function fmtPct(v: unknown, digits = 2, suffix = '%') {
  const x = typeof v === 'number' && Number.isFinite(v) ? v : null
  return x == null ? '—' : `${x >= 0 && digits === 2 ? '+' : ''}${x.toFixed(digits)}${suffix}`
}
