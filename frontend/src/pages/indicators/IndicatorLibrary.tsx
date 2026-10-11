import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Copy, LineChart, PenLine, Plus, Search, Trash2, TriangleAlert } from 'lucide-react'
import { Modal } from '@/components/Modal'
import { toast } from '@/components/Toast'
import { SUB_CHARTS } from '@/components/EChartsCandlestick'
import { api, ApiError, type IndicatorItem } from '@/lib/api'
import { removeIndicatorFromStoredLayout } from '@/lib/chartLayout'
import { QK } from '@/lib/queryKeys'
import { parseDetailMessage, parseReferences } from './indicatorUtils'

const INPUT_CLS = 'rounded-input border border-border bg-surface px-2.5 py-1.5 text-xs focus:border-accent focus:outline-none'
const SUB_KEYS = SUB_CHARTS.map(s => s.key)

export const PANE_LABEL: Record<IndicatorItem['pane'], string> = { main: '主图', sub: '副图' }

export function IndicatorLibrary({ onEdit }: { onEdit: (id?: string, asCopy?: boolean) => void }) {
  const [query, setQuery] = useState('')
  const [pane, setPane] = useState<'all' | 'main' | 'sub'>('all')
  const [deleting, setDeleting] = useState<IndicatorItem | null>(null)

  const list = useQuery({ queryKey: QK.indicators.list, queryFn: api.listIndicators })
  const items = useMemo(() => list.data ?? [], [list.data])

  const filtered = useMemo(() => {
    const kw = query.trim().toLowerCase()
    return items.filter(item => {
      if (pane !== 'all' && item.pane !== pane) return false
      if (kw && !`${item.id} ${item.name} ${item.description ?? ''}`.toLowerCase().includes(kw)) return false
      return true
    })
  }, [items, query, pane])

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-card border border-border bg-surface/80">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2.5">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3 w-3 -translate-y-1/2 text-muted" aria-hidden="true" />
          <input
            type="search"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="搜索指标 (id / 名称)"
            aria-label="搜索指标"
            className={`${INPUT_CLS} w-56 pl-7`}
          />
        </div>
        <select value={pane} onChange={e => setPane(e.target.value as typeof pane)} className={`${INPUT_CLS} w-24`} aria-label="显示位置">
          <option value="all">全部位置</option>
          <option value="main">主图</option>
          <option value="sub">副图</option>
        </select>
        <span className="text-[10px] text-muted" aria-live="polite">
          {list.isLoading ? '加载中…' : `${filtered.length} / ${items.length} 个指标`}
        </span>
        <button
          type="button"
          onClick={() => onEdit()}
          className="ml-auto inline-flex h-8 items-center gap-1.5 rounded-btn bg-accent px-3 text-xs font-medium text-white transition-colors hover:bg-accent/90"
        >
          <Plus className="h-3.5 w-3.5" aria-hidden="true" />新建指标
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-auto">
        {list.isError && (
          <div className="m-3 rounded-btn border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger" role="alert">
            指标列表加载失败 · {String((list.error as Error).message)}
          </div>
        )}
        {list.isLoading && <div className="p-6 text-center text-xs text-muted">指标库加载中…</div>}
        {!list.isLoading && !list.isError && filtered.length === 0 && (
          <div className="p-6 text-center text-xs text-muted">
            {items.length === 0 ? '还没有指标, 点右上角「新建指标」用通达信公式创建一个。' : '无匹配指标'}
          </div>
        )}
        {filtered.length > 0 && (
          <table className="w-full min-w-[760px] text-xs">
            <thead className="sticky top-0 bg-elevated text-left text-[11px] text-secondary">
              <tr>
                <th scope="col" className="px-3 py-2.5 font-medium">指标</th>
                <th scope="col" className="px-3 py-2.5 font-medium">位置</th>
                <th scope="col" className="px-3 py-2.5 font-medium">输出线</th>
                <th scope="col" className="px-3 py-2.5 font-medium">说明</th>
                <th scope="col" className="w-48 px-3 py-2.5 text-right font-medium">操作</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(item => (
                <tr key={item.id} className="border-t border-border/70 align-top transition-colors hover:bg-elevated/40">
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-1.5">
                      <LineChart className="h-3.5 w-3.5 shrink-0 text-accent" aria-hidden="true" />
                      <span className="font-medium text-foreground">{item.name}</span>
                      {item.builtin && (
                        <span className="rounded-btn bg-accent/10 px-1.5 py-0.5 text-[10px] font-medium text-accent" title="预设指标: 可另存为后修改, 不可删除">预设</span>
                      )}
                      {(item.warnings?.length ?? 0) > 0 && (
                        <span
                          className="inline-flex items-center gap-0.5 text-[10px] text-amber-500"
                          title={item.warnings!.map(w => w.message).join('\n')}
                        >
                          <TriangleAlert className="h-3 w-3" aria-hidden="true" />{item.warnings!.length} 条警告
                        </span>
                      )}
                    </div>
                    <div className="mt-0.5 font-mono text-[10px] text-muted">{item.id}</div>
                  </td>
                  <td className="px-3 py-2.5 text-secondary">{PANE_LABEL[item.pane] ?? item.pane}</td>
                  <td className="max-w-[22rem] px-3 py-2.5">
                    <div className="flex flex-wrap gap-1">
                      {item.outputs.map(o => (
                        <span
                          key={o.key}
                          className={`inline-flex items-center gap-1 rounded-btn border border-border bg-base/50 px-1.5 py-0.5 text-[10px] ${o.visible ? 'text-secondary' : 'text-muted line-through'}`}
                          title={`${o.name} · ind_${item.id}_${o.key}${o.visible ? '' : ' · NODRAW (不上图, 仍可被引用)'}`}
                        >
                          <span className="inline-block h-0.5 w-2.5 rounded" style={{ backgroundColor: o.color ?? 'currentColor' }} aria-hidden="true" />
                          {o.name}
                        </span>
                      ))}
                      {item.outputs.length === 0 && <span className="text-muted">—</span>}
                    </div>
                  </td>
                  <td className="max-w-[16rem] px-3 py-2.5">
                    <span className="line-clamp-2 text-secondary" title={item.description}>{item.description || '—'}</span>
                  </td>
                  <td className="px-3 py-2.5 text-right">
                    <div className="inline-flex flex-wrap justify-end gap-1">
                      {!item.builtin && (
                        <button type="button" onClick={() => onEdit(item.id)} className="inline-flex items-center gap-1 rounded-btn px-2 py-1 text-[10px] text-accent transition-colors hover:bg-accent/10" aria-label={`编辑 ${item.name}`}>
                          <PenLine className="h-3 w-3" aria-hidden="true" />编辑
                        </button>
                      )}
                      <button type="button" onClick={() => onEdit(item.id, true)} className="inline-flex items-center gap-1 rounded-btn px-2 py-1 text-[10px] text-secondary transition-colors hover:bg-elevated hover:text-foreground" aria-label={`另存为 ${item.name}`}>
                        <Copy className="h-3 w-3" aria-hidden="true" />另存为
                      </button>
                      <button
                        type="button"
                        onClick={() => setDeleting(item)}
                        disabled={item.builtin}
                        title={item.builtin ? '预设指标不可删除' : undefined}
                        className="inline-flex items-center gap-1 rounded-btn px-2 py-1 text-[10px] text-danger transition-colors hover:bg-danger/10 disabled:cursor-not-allowed disabled:opacity-40"
                        aria-label={`删除 ${item.name}`}
                      >
                        <Trash2 className="h-3 w-3" aria-hidden="true" />删除
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {deleting && <DeleteIndicatorDialog item={deleting} onClose={() => setDeleting(null)} />}
    </div>
  )
}

function DeleteIndicatorDialog({ item, onClose }: { item: IndicatorItem; onClose: () => void }) {
  const qc = useQueryClient()
  const [blocked, setBlocked] = useState<{ message: string; refs: string[] } | null>(null)
  const [error, setError] = useState('')
  const remove = useMutation({
    mutationFn: () => api.deleteIndicator(item.id),
    onSuccess: () => {
      // 布局里的残留 id 一并清掉 (图表重新挂载时读取); 输出线因子随之注销 → 信号字段/因子库失效
      removeIndicatorFromStoredLayout(item.id, SUB_KEYS)
      void qc.invalidateQueries({ queryKey: QK.indicators.all })
      void qc.invalidateQueries({ queryKey: QK.customSignalsOptions })
      void qc.invalidateQueries({ queryKey: ['factors-library'] })
      toast(`已删除指标 ${item.name}`, 'success')
      onClose()
    },
    onError: (err: Error) => {
      const refs = err instanceof ApiError && err.status === 409 ? parseReferences(err.message) : null
      if (refs) {
        setBlocked({ message: parseDetailMessage(err.message), refs })
        return
      }
      setError(err instanceof ApiError && err.status === 409 ? parseDetailMessage(err.message) : err.message)
    },
  })

  return (
    <Modal
      onClose={onClose}
      labelledBy="indicator-delete-title"
      panelClassName="w-[92vw] max-w-md border border-border bg-surface shadow-xl rounded-card"
    >
      <div className="space-y-3 px-4 py-4">
        <h2 id="indicator-delete-title" className="text-sm font-semibold text-foreground">删除指标「{item.name}」?</h2>
        <p className="text-xs leading-relaxed text-secondary">
          删除后其输出线因子 <code className="font-mono">ind_{item.id}_*</code> 一并注销, K 线图上的该指标也会移除。此操作不可撤销。
        </p>
        {blocked && (
          <div className="rounded-btn border border-danger/40 bg-danger/10 px-3 py-2 text-[11px] leading-relaxed text-danger" role="alert">
            <div className="font-medium">{blocked.message && !blocked.message.startsWith('{') ? blocked.message : '该指标仍被以下对象引用, 请先移除引用后再删除:'}</div>
            <ul className="mt-1 list-disc space-y-0.5 pl-4 font-mono text-[10px]">
              {blocked.refs.map(r => <li key={r}>{r}</li>)}
            </ul>
          </div>
        )}
        {error && <div className="rounded-btn border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger" role="alert">删除失败 · {error}</div>}
      </div>
      <div className="flex justify-end gap-2 border-t border-border px-4 py-3">
        <button type="button" onClick={onClose} className="rounded-btn border border-border px-3 py-1.5 text-xs text-secondary transition-colors hover:text-foreground">
          {blocked ? '关闭' : '取消'}
        </button>
        {!blocked && (
          <button
            type="button"
            onClick={() => { setError(''); remove.mutate() }}
            disabled={remove.isPending}
            className="rounded-btn bg-danger px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-danger/90 disabled:opacity-50"
          >
            {remove.isPending ? '删除中…' : '确认删除'}
          </button>
        )}
      </div>
    </Modal>
  )
}
