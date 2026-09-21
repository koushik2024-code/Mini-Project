import { useState } from 'react'
import { TierBadge } from './RoutingReceipt'

export interface HistoryEntry {
  id: number
  query: string
  response: string
  predicted_class: string
  confidence: number
  selected_model: string
  fallback_used: boolean
  latency_ms: number
  created_at: string
}

interface HistorySidebarProps {
  entries: HistoryEntry[]
  total: number
  isLoading: boolean
  error: string | null
  activeId: number | null
  isOpen: boolean
  onClose: () => void
  onSelect: (entry: HistoryEntry) => void
  onDelete: (id: number) => void
  onClearAll: () => void
  onSearch: (term: string) => void
}

function HistorySidebar({
  entries,
  total,
  isLoading,
  error,
  activeId,
  isOpen,
  onClose,
  onSelect,
  onDelete,
  onClearAll,
  onSearch
}: HistorySidebarProps) {
  const [term, setTerm] = useState('')
  const [confirmClear, setConfirmClear] = useState(false)

  const handleTermChange = (value: string) => {
    setTerm(value)
    onSearch(value)
  }

  const counts = {
    easy: entries.filter(e => e.predicted_class === 'easy').length,
    medium: entries.filter(e => e.predicted_class === 'medium').length,
    hard: entries.filter(e => e.predicted_class === 'hard').length
  }
  const shown = counts.easy + counts.medium + counts.hard

  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-ink/40 z-30 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed lg:sticky top-0 left-0 z-40 h-screen flex-shrink-0 bg-sunken overflow-hidden border-r transition-[transform,width] duration-200 ${
          isOpen
            ? 'w-[296px] translate-x-0 border-line'
            : 'w-[296px] -translate-x-full lg:w-0 lg:translate-x-0 border-transparent'
        }`}
        aria-label="Search history"
        inert={!isOpen}
      >
        {/* Fixed-width inner column so the content does not squash while the
            aside animates its width shut on desktop. */}
        <div className="w-[296px] h-screen flex flex-col">
        {/* Header + filter */}
        <div className="px-4 pt-5 pb-3.5 flex-shrink-0">
          <div className="flex items-baseline justify-between gap-2">
            <h2 className="font-display text-[0.92rem] font-semibold m-0">Search history</h2>
            <div className="flex items-center gap-2">
              <span className="font-data text-[0.72rem] text-ink-3">{total}</span>
              <button onClick={onClose} className="btn btn-ghost btn-icon !h-9 !w-9 !border-0" aria-label="Hide search history">
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                  <path d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          <label htmlFor="history-filter" className="sr-only">Filter history</label>
          <input
            id="history-filter"
            type="search"
            value={term}
            onChange={(e) => handleTermChange(e.target.value)}
            placeholder="Filter history"
            className="input mt-3.5"
          />
        </div>

        {/* Entries */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-3 pb-2">
          {isLoading && entries.length === 0 && (
            <p className="px-2 py-4 text-[0.82rem] text-ink-3">Loading history&hellip;</p>
          )}

          {error && (
            <p className="px-2 py-4 text-[0.82rem] text-hard" role="alert">{error}</p>
          )}

          {!isLoading && !error && entries.length === 0 && (
            <div className="mt-3 px-4 py-6 border border-dashed border-[#d5d1c6] rounded-xl text-center">
              <svg className="w-6 h-6 mx-auto" viewBox="0 0 24 24" fill="none" stroke="#a8a2af" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="9" />
                <path d="M12 7v5l3 2" />
              </svg>
              <p className="mt-3 text-[0.8rem] leading-relaxed text-ink-3">
                {term ? 'No matching queries.' : 'Nothing here yet. Your questions will be saved as you ask them.'}
              </p>
            </div>
          )}

          <ul className="flex flex-col gap-[3px] list-none p-0 m-0">
            {entries.map((entry) => (
              <li key={entry.id} className="relative group">
                <button
                  type="button"
                  className="history-item"
                  aria-current={activeId === entry.id}
                  onClick={() => onSelect(entry)}
                >
                  <span className="block text-[0.83rem] leading-snug line-clamp-2">{entry.query}</span>
                  <span className="mt-1.5 flex items-center gap-2">
                    <TierBadge tier={entry.predicted_class} />
                    <span className="font-data text-[0.68rem] text-ink-3">{formatTime(entry.created_at)}</span>
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => onDelete(entry.id)}
                  className="absolute top-0 right-0 w-11 h-11 flex items-center justify-center text-ink-3 opacity-0 group-hover:opacity-100 focus:opacity-100 hover:text-hard transition-opacity"
                  aria-label={`Delete history entry: ${entry.query}`}
                >
                  <svg className="w-[15px] h-[15px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        </div>

        {/* Where they went + clear */}
        {entries.length > 0 && (
          <div className="flex-shrink-0 px-4 py-4 border-t border-line">
            <div className="eyebrow">Where they went</div>

            <div className="mt-2.5 flex h-[7px] gap-[2px] rounded-full overflow-hidden" aria-hidden="true">
              {counts.easy > 0 && <span className="bg-easy" style={{ width: `${(counts.easy / shown) * 100}%` }} />}
              {counts.medium > 0 && <span className="bg-medium" style={{ width: `${(counts.medium / shown) * 100}%` }} />}
              {counts.hard > 0 && <span className="bg-hard" style={{ width: `${(counts.hard / shown) * 100}%` }} />}
            </div>

            <div className="mt-2 flex justify-between font-data text-[0.68rem] text-ink-2">
              <span>{counts.easy} easy</span>
              <span>{counts.medium} medium</span>
              <span>{counts.hard} hard</span>
            </div>

            {confirmClear ? (
              <div className="mt-4 flex items-center gap-2">
                <button
                  onClick={() => { onClearAll(); setConfirmClear(false) }}
                  className="btn btn-danger flex-1 !text-[0.8rem]"
                >
                  Delete all
                </button>
                <button onClick={() => setConfirmClear(false)} className="btn btn-secondary flex-1 !text-[0.8rem]">
                  Cancel
                </button>
              </div>
            ) : (
              <button onClick={() => setConfirmClear(true)} className="btn btn-ghost w-full mt-4 !text-[0.8rem]">
                Clear all history
              </button>
            )}
          </div>
        )}
        </div>
      </aside>
    </>
  )
}

function formatTime(iso: string) {
  // Backend stores UTC ISO timestamps
  const date = new Date(/[Z+]|-\d\d:\d\d$/.test(iso) ? iso : `${iso}Z`)
  if (Number.isNaN(date.getTime())) return iso

  const now = new Date()
  const sameDay = date.toDateString() === now.toDateString()

  return sameDay
    ? date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
    : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export default HistorySidebar
