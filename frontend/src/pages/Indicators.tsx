import { useSearchParams } from 'react-router-dom'
import { LibrarySquare, PenLine } from 'lucide-react'
import { PageHeader } from '@/components/PageHeader'
import { IndicatorLibrary } from './indicators/IndicatorLibrary'
import { IndicatorEditor } from './indicators/IndicatorEditor'

type Tab = 'library' | 'editor'

const TABS: Record<Tab, { title: string; icon: typeof LibrarySquare }> = {
  library: { title: '指标库', icon: LibrarySquare },
  editor: { title: '编辑器', icon: PenLine },
}

/**
 * 通达信指标页: ?tab=library (默认) | ?tab=editor[&edit=<id>][&saveas=1]
 * 编辑器以 edit/saveas 为 key 重新挂载 → 新建态与编辑态的临时表单互不串。
 */
export function Indicators() {
  const [searchParams, setSearchParams] = useSearchParams()
  const activeTab: Tab = searchParams.get('tab') === 'editor' ? 'editor' : 'library'
  const editId = searchParams.get('edit') ?? ''
  const saveAs = searchParams.get('saveas') === '1'

  const openEditor = (id?: string, asCopy = false) => {
    const next = new URLSearchParams()
    next.set('tab', 'editor')
    if (id) next.set('edit', id)
    if (id && asCopy) next.set('saveas', '1')
    setSearchParams(next)
  }
  const changeTab = (tab: Tab) => {
    const next = new URLSearchParams()
    next.set('tab', tab)
    setSearchParams(next, { replace: true })
  }

  return (
    <div className="flex min-h-full flex-col bg-base xl:h-full">
      <PageHeader
        title="指标"
        subtitle={<span className="hidden md:inline">通达信公式指标 · 主图叠加 / 副图 · 输出线可在信号与策略中引用</span>}
        className="shrink-0 flex-wrap gap-x-4 gap-y-2 bg-base/95 px-3 lg:flex-nowrap lg:px-5"
        right={(
          <nav aria-label="指标视图">
            <div className="inline-flex items-center gap-0.5 rounded-btn border border-border bg-surface/80 p-0.5">
              {(Object.keys(TABS) as Tab[]).map(tab => {
                const mode = TABS[tab]
                const Icon = mode.icon
                const active = activeTab === tab
                return (
                  <button
                    key={tab}
                    type="button"
                    onClick={() => changeTab(tab)}
                    aria-current={active ? 'page' : undefined}
                    className={`inline-flex h-7 items-center gap-1.5 rounded-[5px] px-2.5 text-xs font-medium transition-colors ${active
                      ? 'bg-accent text-white shadow-sm'
                      : 'text-secondary hover:bg-elevated hover:text-foreground'
                    }`}
                  >
                    <Icon className="h-3.5 w-3.5" aria-hidden="true" />
                    {mode.title}
                  </button>
                )
              })}
            </div>
          </nav>
        )}
      />
      <main className="min-h-0 flex-1 px-3 pb-3 pt-3 lg:px-4 lg:pb-4 xl:overflow-hidden">
        {activeTab === 'library' && <IndicatorLibrary onEdit={openEditor} />}
        {activeTab === 'editor' && (
          <IndicatorEditor
            key={`${editId}|${saveAs ? 'copy' : ''}`}
            editId={editId}
            saveAs={saveAs}
            onSaved={id => openEditor(id)}
          />
        )}
      </main>
    </div>
  )
}
