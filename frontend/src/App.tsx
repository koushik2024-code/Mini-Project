import { useState, useRef, useEffect, useCallback } from 'react'
import HistorySidebar, { type HistoryEntry } from './HistorySidebar'
import LoginScreen from './LoginScreen'
import Markdown from './Markdown'
import RoutingReceipt, { type Routing } from './RoutingReceipt'
import { API_BASE, type AuthUser, authFetch, errorDetail, UnauthorizedError } from './auth'
import { useAuth } from './useAuth'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  routing?: Routing
  /** Routing is known, the answer is still being generated. */
  pending?: boolean
}

interface ChatResponse {
  query: string
  predicted_class: string
  confidence: number
  selected_model: string
  response: string
  fallback_used: boolean
  latency_ms: number
  confidence_threshold?: number
  history_id?: number
}

interface RouteResponse {
  predicted_class: string
  confidence: number
  selected_model: string
  fallback_used: boolean
  confidence_threshold: number
}

interface HistoryListResponse {
  entries: HistoryEntry[]
  total: number
  limit: number
  offset: number
}

const EXAMPLES = [
  { text: 'What is 2+2?', tier: 'easy', label: 'Easy', model: 'qwen3:1.7b' },
  { text: 'Explain how a binary search tree works', tier: 'medium', label: 'Medium', model: 'qwen3:4b' },
  { text: 'Write a Python function to implement a distributed consensus algorithm', tier: 'hard', label: 'Hard', model: 'llama3.2:3b' }
]

function App() {
  const { user, status, error, signInWithGoogle, signOut, handleExpired } = useAuth()
  const [serverConfigured, setServerConfigured] = useState<boolean | undefined>(undefined)
  const [threshold, setThreshold] = useState<number | undefined>(undefined)

  // /health is public and cheap: it tells us whether the backend can do
  // Google sign-in at all, and what the fallback threshold is.
  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE}/health`)
      .then(r => (r.ok ? r.json() : null))
      .then(data => {
        if (cancelled || !data) return
        setServerConfigured(Boolean(data.auth_configured))
        if (typeof data.confidence_threshold === 'number') setThreshold(data.confidence_threshold)
      })
      .catch(() => { /* the login screen still works, it just can't pre-warn */ })
    return () => { cancelled = true }
  }, [])

  if (status === 'loading') {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <span className="font-data text-[0.8rem] text-ink-3 dot-pulse">Loading&hellip;</span>
      </div>
    )
  }

  if (status === 'signed-out' || !user) {
    return (
      <LoginScreen
        onCredential={signInWithGoogle}
        error={error}
        serverConfigured={serverConfigured}
      />
    )
  }

  return <RouterApp user={user} onSignOut={signOut} onExpired={handleExpired} threshold={threshold} />
}

interface RouterAppProps {
  user: AuthUser
  onSignOut: () => void
  onExpired: () => void
  threshold?: number
}

function RouterApp({ user, onSignOut, onExpired, threshold }: RouterAppProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [historyTotal, setHistoryTotal] = useState(0)
  const [historyLoading, setHistoryLoading] = useState(true)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [historyTerm, setHistoryTerm] = useState('')
  const [activeHistoryId, setActiveHistoryId] = useState<number | null>(null)
  // Open by default where there is room for it, closed on phones
  const [sidebarOpen, setSidebarOpen] = useState(
    () => typeof window === 'undefined' || window.matchMedia('(min-width: 1024px)').matches
  )

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // On phones the sidebar covers the chat, so get out of the way after a
  // choice. On desktop it sits beside the chat and should stay put.
  const closeSidebarIfOverlaying = () => {
    if (!window.matchMedia('(min-width: 1024px)').matches) setSidebarOpen(false)
  }

  const loadHistory = useCallback(async (term: string) => {
    setHistoryLoading(true)
    try {
      const params = new URLSearchParams({ limit: '100' })
      if (term.trim()) params.set('q', term.trim())

      const response = await authFetch(`/history?${params}`)
      if (!response.ok) throw new Error('Failed to load history')

      const data: HistoryListResponse = await response.json()
      setHistory(data.entries)
      setHistoryTotal(data.total)
      setHistoryError(null)
    } catch (err) {
      if (err instanceof UnauthorizedError) { onExpired(); return }
      setHistoryError('Could not load search history')
    } finally {
      setHistoryLoading(false)
    }
  }, [onExpired])

  // Debounce the filter so typing doesn't fire a request per keystroke
  useEffect(() => {
    const timer = setTimeout(() => loadHistory(historyTerm), historyTerm ? 250 : 0)
    return () => clearTimeout(timer)
  }, [historyTerm, loadHistory])

  const ask = async (query: string) => {
    if (!query.trim() || isLoading) return

    const userMessage = query.trim()
    setInput('')
    setIsLoading(true)
    setError(null)
    setActiveHistoryId(null)
    closeSidebarIfOverlaying()

    const pendingId = crypto.randomUUID()

    setMessages(prev => [
      ...prev,
      { id: crypto.randomUUID(), role: 'user', content: userMessage },
      { id: pendingId, role: 'assistant', content: '', pending: true }
    ])

    // Phase 1: routing only. Classification takes milliseconds while
    // generation takes seconds, so show the decision straight away rather
    // than leaving a blank space until the model finishes. Best-effort:
    // if it fails, the /chat response below carries the same fields.
    authFetch('/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: userMessage })
    })
      .then(r => (r.ok ? r.json() : null))
      .then((decision: RouteResponse | null) => {
        if (!decision) return
        setMessages(prev => prev.map(m =>
          m.id === pendingId && m.pending
            ? { ...m, routing: toRouting(decision, threshold) }
            : m
        ))
      })
      .catch(() => { /* handled by phase 2 */ })

    // Phase 2: the answer
    try {
      const response = await authFetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage })
      })

      if (!response.ok) throw new Error(await errorDetail(response, 'Failed to get response'))

      const data: ChatResponse = await response.json()

      setMessages(prev => prev.map(m =>
        m.id === pendingId
          ? {
              id: pendingId,
              role: 'assistant',
              content: data.response,
              pending: false,
              routing: {
                predictedClass: data.predicted_class,
                confidence: data.confidence,
                selectedModel: data.selected_model,
                fallbackUsed: data.fallback_used,
                latencyMs: data.latency_ms,
                threshold: data.confidence_threshold ?? threshold
              }
            }
          : m
      ))

      if (data.history_id !== undefined) setActiveHistoryId(data.history_id)
      loadHistory(historyTerm)
    } catch (err) {
      if (err instanceof UnauthorizedError) { onExpired(); return }

      setError(err instanceof Error ? err.message : 'An unexpected error occurred')
      setMessages(prev => prev.filter(m => m.id !== pendingId))
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    ask(input)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      ask(input)
    }
  }

  const handleSelectHistory = (entry: HistoryEntry) => {
    setActiveHistoryId(entry.id)
    closeSidebarIfOverlaying()
    setError(null)
    setMessages([
      { id: `history-q-${entry.id}`, role: 'user', content: entry.query },
      {
        id: `history-a-${entry.id}`,
        role: 'assistant',
        content: entry.response,
        routing: {
          predictedClass: entry.predicted_class,
          confidence: entry.confidence,
          selectedModel: entry.selected_model,
          fallbackUsed: entry.fallback_used,
          latencyMs: entry.latency_ms,
          threshold
        }
      }
    ])
  }

  const handleDeleteHistory = async (id: number) => {
    try {
      const response = await authFetch(`/history/${id}`, { method: 'DELETE' })
      if (!response.ok) throw new Error('Failed to delete entry')

      setHistory(prev => prev.filter(entry => entry.id !== id))
      setHistoryTotal(prev => Math.max(0, prev - 1))
      if (activeHistoryId === id) setActiveHistoryId(null)
    } catch (err) {
      if (err instanceof UnauthorizedError) { onExpired(); return }
      setHistoryError('Could not delete that entry')
    }
  }

  const handleClearHistory = async () => {
    try {
      const response = await authFetch('/history', { method: 'DELETE' })
      if (!response.ok) throw new Error('Failed to clear history')

      setHistory([])
      setHistoryTotal(0)
      setActiveHistoryId(null)
      setHistoryError(null)
    } catch (err) {
      if (err instanceof UnauthorizedError) { onExpired(); return }
      setHistoryError('Could not clear history')
    }
  }

  const handleNewChat = () => {
    setMessages([])
    setActiveHistoryId(null)
    setError(null)
    setInput('')
    inputRef.current?.focus()
  }

  return (
    <div className="min-h-screen flex">
      <HistorySidebar
        entries={history}
        total={historyTotal}
        isLoading={historyLoading}
        error={historyError}
        activeId={activeHistoryId}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelect={handleSelectHistory}
        onDelete={handleDeleteHistory}
        onClearAll={handleClearHistory}
        onSearch={setHistoryTerm}
      />

      <div className="flex-1 min-w-0 flex flex-col h-screen">

        <header className="h-16 flex-shrink-0 border-b border-line flex items-center justify-between pl-2 pr-3 sm:pl-6 sm:pr-5">
          <div className="flex items-center gap-2 sm:gap-3">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="btn btn-ghost btn-icon !border-0"
                aria-label="Show search history"
              >
                <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" aria-hidden="true">
                  <path d="M4 7h16M4 12h16M4 17h16" />
                </svg>
              </button>
            )}
            <span className="w-[26px] h-[26px] rounded-[7px] bg-ink flex items-center justify-center flex-shrink-0">
              <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="#f6f4ef" strokeWidth="2.1" strokeLinecap="round" aria-hidden="true">
                <path d="M3 12h4l3-7 4 14 3-7h4" />
              </svg>
            </span>
            <span className="wordmark hidden sm:block">LLM ROUTER</span>
          </div>

          <div className="flex items-center gap-2">
            <button onClick={handleNewChat} className="btn btn-secondary">
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
                <path d="M12 5v14M5 12h14" />
              </svg>
              <span className="hidden sm:inline">New chat</span>
            </button>
            <UserMenu user={user} onSignOut={onSignOut} />
          </div>
        </header>

        <main className="flex-1 overflow-y-auto scrollbar-thin">
          <div className="max-w-[760px] mx-auto px-4 sm:px-8 py-7">
            {messages.length === 0 ? (
              <EmptyState onPick={(text) => { setInput(text); inputRef.current?.focus() }} />
            ) : (
              <div className="flex flex-col gap-6">
                {messages.map((msg) => <Message key={msg.id} message={msg} />)}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </main>

        <div className="flex-shrink-0 border-t border-line px-4 sm:px-8 py-4">
          <div className="max-w-[760px] mx-auto">
            {error && (
              <div className="bubble-notice mb-3 flex items-start gap-2" role="alert">
                <span className="flex-1">{error}</span>
                <button onClick={() => setError(null)} className="font-semibold cursor-pointer" aria-label="Dismiss error">&times;</button>
              </div>
            )}

            <form onSubmit={handleSubmit} className="flex items-end gap-2.5">
              <label htmlFor="composer" className="sr-only">Ask a question</label>
              <textarea
                id="composer"
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask anything"
                disabled={isLoading}
                rows={1}
                className="input composer flex-1"
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="btn btn-primary btn-icon !h-[3.25rem] !w-[3.25rem] !rounded-xl flex-shrink-0"
                aria-label="Send"
              >
                <svg className="w-[19px] h-[19px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M5 12h13M12 5l7 7-7 7" />
                </svg>
              </button>
            </form>

            <p className="mt-2.5 font-data text-[0.66rem] text-ink-3">
              Routing happens before generation &mdash; you see the model within milliseconds
              {threshold !== undefined && `. Tick on each meter marks the fallback threshold, ${threshold.toFixed(2)}`}.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function EmptyState({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="py-10 sm:py-16 flex flex-col items-center">
      <h1 className="font-display text-[1.8rem] sm:text-[2.2rem] font-semibold text-center m-0">
        What do you want to ask?
      </h1>
      <p className="mt-3.5 text-[0.95rem] leading-relaxed text-ink-2 max-w-[460px] text-center">
        Your query is classified before it is answered. You will see which model handled it,
        and how sure the router was.
      </p>

      <div className="mt-9 grid gap-3.5 w-full sm:grid-cols-3">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.text}
            type="button"
            onClick={() => onPick(ex.text)}
            className="card p-4 min-h-[140px] flex flex-col justify-between text-left hover:bg-sunken transition-colors cursor-pointer"
          >
            <span className="text-[0.88rem] leading-snug font-medium">{ex.text}</span>
            <span className="flex items-center gap-2 mt-3">
              <span className={`badge badge-${ex.tier}`}>{ex.label}</span>
              <span className="font-data text-[0.68rem] text-ink-3">{ex.model}</span>
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}

function Message({ message }: { message: ChatMessage }) {
  const { role, content, routing, pending } = message

  if (role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="bubble-user max-w-[80%]">{content}</div>
      </div>
    )
  }

  return (
    <div>
      {routing && <RoutingReceipt routing={routing} />}

      {pending ? (
        <div className="bubble-assistant">
          <div className="flex items-center gap-2.5">
            <span
              className="w-[7px] h-[7px] rounded-full dot-pulse"
              style={{ backgroundColor: routing ? tierColor(routing.predictedClass) : 'var(--color-ink-3)' }}
            />
            <span className="font-data text-[0.75rem] text-ink-2">
              {routing ? `${routing.selectedModel} is answering` : 'Routing…'}
            </span>
          </div>
          <div className="mt-3.5 flex flex-col gap-2.5">
            <span className="skeleton-line w-[96%]" />
            <span className="skeleton-line w-[88%]" />
            <span className="skeleton-line w-[54%]" />
          </div>
        </div>
      ) : (
        <div className="bubble-assistant">
          <Markdown content={content} />
        </div>
      )}
    </div>
  )
}

function UserMenu({ user, onSignOut }: { user: AuthUser; onSignOut: () => void }) {
  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  // Close when clicking outside or pressing Escape
  useEffect(() => {
    if (!open) return

    const onPointerDown = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }

    document.addEventListener('mousedown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('mousedown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setOpen(v => !v)}
        className="flex items-center gap-2.5 h-11 pl-1 pr-2.5 rounded-full hover:bg-sunken transition-colors cursor-pointer"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Account menu for ${user.email}`}
      >
        <Avatar user={user} />
        <span className="hidden sm:block text-[0.83rem] text-ink-2 max-w-[9rem] truncate">
          {user.name || user.email}
        </span>
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-60 card overflow-hidden z-50 shadow-lg" role="menu">
          <div className="px-4 py-3 border-b border-line flex items-center gap-3">
            <Avatar user={user} />
            <div className="min-w-0">
              <p className="text-[0.85rem] font-medium truncate m-0">{user.name || 'Signed in'}</p>
              <p className="text-[0.72rem] text-ink-3 truncate m-0">{user.email}</p>
            </div>
          </div>
          <button
            onClick={() => { setOpen(false); onSignOut() }}
            className="w-full text-left px-4 py-3 text-[0.85rem] text-ink-2 hover:bg-sunken transition-colors cursor-pointer"
            role="menuitem"
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  )
}

function Avatar({ user }: { user: AuthUser }) {
  const [failed, setFailed] = useState(false)

  if (user.picture && !failed) {
    return (
      <img
        src={user.picture}
        alt=""
        referrerPolicy="no-referrer"
        onError={() => setFailed(true)}
        className="w-8 h-8 rounded-full flex-shrink-0 object-cover"
      />
    )
  }

  // Google's CDN can fail or be blocked - fall back to an initial
  const initial = (user.name || user.email || '?').trim().charAt(0).toUpperCase()
  return (
    <span className="w-8 h-8 rounded-full flex-shrink-0 bg-easy-tint text-easy font-display text-[0.85rem] font-semibold flex items-center justify-center">
      {initial}
    </span>
  )
}

function tierColor(tier: string) {
  if (tier === 'easy') return 'var(--color-easy)'
  if (tier === 'medium') return 'var(--color-medium)'
  if (tier === 'hard') return 'var(--color-hard)'
  return 'var(--color-ink-3)'
}

function toRouting(decision: RouteResponse, fallbackThreshold?: number): Routing {
  return {
    predictedClass: decision.predicted_class,
    confidence: decision.confidence,
    selectedModel: decision.selected_model,
    fallbackUsed: decision.fallback_used,
    threshold: decision.confidence_threshold ?? fallbackThreshold
  }
}

export default App
