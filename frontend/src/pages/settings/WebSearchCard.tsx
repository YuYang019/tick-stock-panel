import { useEffect, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Check, Globe, Loader2 } from 'lucide-react'
import { api } from '@/lib/api'

const USE_IN_LABELS: { key: string; label: string }[] = [
  { key: 'review', label: '每日复盘·消息催化' },
  { key: 'watchlist', label: '复盘·自选股消息' },
  { key: 'stock_analysis', label: '个股分析·消息面' },
  { key: 'assistant', label: '对话助手' },
]

/** 联网搜索配置卡: 独立于 AI 服务商, 直连 DeepSeek Search/Fetch。 */
export function WebSearchCard() {
  const qc = useQueryClient()
  const { data, isLoading } = useQuery({
    queryKey: ['settings', 'web-search'],
    queryFn: api.webSearchSettings,
  })

  const [enabled, setEnabled] = useState(false)
  const [apiKey, setApiKey] = useState('')
  const [budget, setBudget] = useState('200')
  const [minScore, setMinScore] = useState('1.0')
  const [sites, setSites] = useState('')
  const [useIn, setUseIn] = useState<Record<string, boolean>>({})
  const [testResult, setTestResult] = useState<{ ok: boolean; msg: string } | null>(null)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (!data) return
    setEnabled(data.config.enabled)
    setBudget(String(data.config.daily_budget))
    setMinScore(String(data.config.min_score))
    setSites(data.config.sites.join(', '))
    setUseIn(data.config.use_in)
  }, [data])

  const save = useMutation({
    mutationFn: () =>
      api.saveWebSearchSettings({
        enabled,
        api_key: apiKey.trim() ? apiKey.trim() : undefined,
        daily_budget: Number(budget) || 0,
        min_score: Number(minScore) || 0,
        sites: sites.split(/[,，\s]+/).map(s => s.trim()).filter(Boolean),
        use_in: useIn,
      }),
    onSuccess: () => {
      setSaved(true)
      setApiKey('')
      setTimeout(() => setSaved(false), 1500)
      qc.invalidateQueries({ queryKey: ['settings', 'web-search'] })
    },
  })

  const test = useMutation({
    mutationFn: () => api.testWebSearch(),
    onSuccess: r => {
      setTestResult({
        ok: r.ok,
        msg: r.ok ? `连通正常, 返回 ${r.count ?? 0} 条` : (r.error || '测试失败'),
      })
      qc.invalidateQueries({ queryKey: ['settings', 'web-search'] })
    },
    onError: (e: Error) => setTestResult({ ok: false, msg: e.message }),
  })

  const key = data?.key
  const usage = data?.usage

  return (
    <section className="overflow-hidden rounded-card border border-border bg-surface">
      <div className="flex items-start gap-3 px-5 py-4">
        <Globe className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-medium text-foreground">联网搜索</h3>
            <span className="rounded-full border border-border/60 px-1.5 py-0.5 text-[10px] text-muted">
              DeepSeek Search / Fetch
            </span>
          </div>
          <p className="mt-1 text-[11px] leading-relaxed text-muted">
            为复盘消息催化、个股消息面与对话助手提供实时网页检索。按次计费(5 元/千次),
            有日预算与当日缓存; 未启用时上述功能退回纯行情推断。
          </p>
        </div>
        <button type="button" role="switch" aria-checked={enabled}
          onClick={() => setEnabled(v => !v)}
          className={`relative mt-1 h-5 w-9 shrink-0 rounded-full transition-colors ${enabled ? 'bg-accent' : 'bg-border'}`}>
          <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${enabled ? 'left-[18px]' : 'left-0.5'}`} />
        </button>
      </div>

      <div className="space-y-3 border-t border-border/40 px-5 py-4">
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="block">
            <span className="mb-1 block text-[11px] text-muted">API Key</span>
            <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)}
              placeholder={key?.configured ? `${key.masked}（留空保持不变）` : '留空则复用直连 DeepSeek 的 AI Key'}
              className="w-full rounded-lg border border-border bg-base px-2.5 py-1.5 text-xs text-foreground placeholder:text-muted/60 focus:border-accent/60 focus:outline-none" />
            <span className="mt-1 block text-[10px] text-muted/80">
              {key?.source === 'ai_key'
                ? '当前复用 AI Key（AI 直连 api.deepseek.com）'
                : key?.source === 'dedicated'
                  ? '当前使用专用 Key'
                  : key?.can_reuse_ai_key
                    ? '未配置; 保存后可直接复用 AI Key'
                    : 'AI 走中转服务, 需单独填写 DeepSeek Key'}
            </span>
          </label>
          <label className="block">
            <span className="mb-1 block text-[11px] text-muted">每日调用上限</span>
            <input type="number" min={0} max={10000} value={budget} onChange={e => setBudget(e.target.value)}
              className="w-full rounded-lg border border-border bg-base px-2.5 py-1.5 text-xs text-foreground focus:border-accent/60 focus:outline-none" />
            <span className="mt-1 block text-[10px] text-muted/80">
              今日已用 {usage?.calls ?? 0} 次 · 缓存 {usage?.cached ?? 0} 条
            </span>
          </label>
          <label className="block">
            <span className="mb-1 block text-[11px] text-muted">最低相关度</span>
            <input type="number" step="0.1" min={-3} max={3} value={minScore} onChange={e => setMinScore(e.target.value)}
              className="w-full rounded-lg border border-border bg-base px-2.5 py-1.5 text-xs text-foreground focus:border-accent/60 focus:outline-none" />
            <span className="mt-1 block text-[10px] text-muted/80">低于该 rerank 分数的结果丢弃（0–3, 1.5 以上为优质）</span>
          </label>
          <label className="block">
            <span className="mb-1 block text-[11px] text-muted">站点白名单</span>
            <input value={sites} onChange={e => setSites(e.target.value)}
              placeholder="cninfo.com.cn, eastmoney.com"
              className="w-full rounded-lg border border-border bg-base px-2.5 py-1.5 text-xs text-foreground placeholder:text-muted/60 focus:border-accent/60 focus:outline-none" />
            <span className="mt-1 block text-[10px] text-muted/80">逗号分隔; 留空表示不限制站点</span>
          </label>
        </div>

        <div>
          <span className="mb-1.5 block text-[11px] text-muted">启用场景</span>
          <div className="flex flex-wrap gap-2">
            {USE_IN_LABELS.map(item => {
              const on = useIn[item.key] !== false
              return (
                <button key={item.key} type="button" aria-pressed={on}
                  onClick={() => setUseIn(prev => ({ ...prev, [item.key]: !on }))}
                  className={`rounded-full border px-2.5 py-1 text-[11px] transition-colors ${on ? 'border-accent/50 bg-accent/[0.08] text-accent' : 'border-border/60 text-muted hover:text-secondary'}`}>
                  {item.label}
                </button>
              )
            })}
          </div>
        </div>

        <div className="flex items-center gap-2 pt-1">
          <button type="button" onClick={() => save.mutate()} disabled={save.isPending}
            className="inline-flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-accent/90 disabled:opacity-50">
            {save.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : saved ? <Check className="h-3 w-3" /> : null}
            {saved ? '已保存' : '保存'}
          </button>
          <button type="button" onClick={() => test.mutate()} disabled={test.isPending || !key?.configured}
            className="inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs text-secondary transition-colors hover:bg-elevated hover:text-foreground disabled:opacity-50">
            {test.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
            {test.isPending ? '测试中' : '测试连通性'}
          </button>
          {testResult && (
            <span className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] ${testResult.ok ? 'border-bear/25 bg-bear/[0.06] text-bear' : 'border-danger/25 bg-danger/[0.06] text-danger'}`}>
              <span className={`h-1 w-1 rounded-full ${testResult.ok ? 'bg-bear' : 'bg-danger'}`} />
              {testResult.msg}
            </span>
          )}
          {isLoading && <Loader2 className="h-3 w-3 animate-spin text-muted" />}
        </div>
      </div>
    </section>
  )
}
