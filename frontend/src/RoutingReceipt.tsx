export interface Routing {
  predictedClass: string
  confidence: number
  selectedModel: string
  fallbackUsed: boolean
  latencyMs?: number
  /** Fallback threshold from the backend; the meter draws a tick here. */
  threshold?: number
}

const TIER_COLOR: Record<string, string> = {
  easy: 'var(--color-easy)',
  medium: 'var(--color-medium)',
  hard: 'var(--color-hard)'
}

const TIER_LABEL: Record<string, string> = {
  easy: 'Easy',
  medium: 'Medium',
  hard: 'Hard'
}

export function TierBadge({ tier }: { tier?: string }) {
  if (!tier || !TIER_LABEL[tier]) return null
  return <span className={`badge badge-${tier}`}>{TIER_LABEL[tier]}</span>
}

/**
 * The routing decision, shown above the answer it produced.
 *
 * The meter is the point: the tick marks the fallback threshold, so you can
 * see at a glance whether the classifier cleared it. Past the tick the
 * predicted tier was used; short of it the policy dropped to the fallback
 * tier, and the Fallback badge appears alongside.
 */
function RoutingReceipt({ routing }: { routing: Routing }) {
  const { predictedClass, confidence, selectedModel, fallbackUsed, latencyMs, threshold } = routing

  const pct = Math.max(0, Math.min(1, confidence)) * 100
  const color = TIER_COLOR[predictedClass] ?? 'var(--color-ink-2)'
  const tickPct = threshold === undefined ? null : Math.max(0, Math.min(1, threshold)) * 100

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 pb-2.5">
      <TierBadge tier={predictedClass} />

      {fallbackUsed && (
        <span className="badge badge-fallback" title="Confidence fell below the threshold, so the router dropped to the fallback tier">
          <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5" />
          </svg>
          Fallback
        </span>
      )}

      <span className="font-data text-[0.75rem] text-ink">{selectedModel}</span>

      <span className="flex items-center gap-2">
        <span
          className="meter w-[92px]"
          role="meter"
          aria-valuenow={Math.round(pct)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label="Router confidence"
        >
          <span className="meter-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
          {tickPct !== null && (
            <span className="meter-tick" style={{ left: `${tickPct}%` }} />
          )}
        </span>
        <span className="font-data text-[0.72rem] text-ink-2">{pct.toFixed(1)}%</span>
      </span>

      {latencyMs !== undefined && (
        <span className="font-data text-[0.72rem] text-ink-3">{formatLatency(latencyMs)}</span>
      )}
    </div>
  )
}

function formatLatency(ms: number) {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  const mins = Math.floor(ms / 60000)
  const secs = Math.round((ms % 60000) / 1000)
  return `${mins}m ${secs}s`
}

export default RoutingReceipt
