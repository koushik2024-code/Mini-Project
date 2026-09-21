import { useEffect, useRef, useState } from 'react'
import { GOOGLE_CLIENT_ID } from './auth'

const GSI_SRC = 'https://accounts.google.com/gsi/client'

// Minimal shape of the Google Identity Services API we use
interface GoogleIdentity {
  accounts: {
    id: {
      initialize: (config: {
        client_id: string
        callback: (response: { credential: string }) => void
      }) => void
      renderButton: (parent: HTMLElement, options: Record<string, unknown>) => void
    }
  }
}

declare global {
  interface Window {
    google?: GoogleIdentity
  }
}

/** Load the Google Identity Services script once, shared across mounts. */
function useGoogleScript(enabled: boolean) {
  // Already-loaded script (e.g. after a remount) starts out ready
  const [state, setState] = useState<'loading' | 'ready' | 'error'>(
    () => (window.google?.accounts?.id ? 'ready' : 'loading')
  )

  useEffect(() => {
    if (!enabled || state === 'ready') return

    const existing = document.querySelector<HTMLScriptElement>(`script[src="${GSI_SRC}"]`)
    const script = existing ?? document.createElement('script')

    const onLoad = () => setState('ready')
    const onError = () => setState('error')

    script.addEventListener('load', onLoad)
    script.addEventListener('error', onError)

    if (!existing) {
      script.src = GSI_SRC
      script.async = true
      script.defer = true
      document.head.appendChild(script)
    }

    return () => {
      script.removeEventListener('load', onLoad)
      script.removeEventListener('error', onError)
    }
  }, [enabled, state])

  return state
}

interface LoginScreenProps {
  onCredential: (credential: string) => void
  error: string | null
  /** From GET /health; false means the backend has no GOOGLE_CLIENT_ID. */
  serverConfigured?: boolean
}

function LoginScreen({ onCredential, error, serverConfigured }: LoginScreenProps) {
  const buttonRef = useRef<HTMLDivElement>(null)
  const configured = Boolean(GOOGLE_CLIENT_ID)
  const scriptState = useGoogleScript(configured)

  // Keep the latest callback without re-initializing Google on every render
  const callbackRef = useRef(onCredential)
  useEffect(() => {
    callbackRef.current = onCredential
  }, [onCredential])

  useEffect(() => {
    if (scriptState !== 'ready' || !buttonRef.current || !window.google) return

    window.google.accounts.id.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: (response) => callbackRef.current(response.credential)
    })

    window.google.accounts.id.renderButton(buttonRef.current, {
      theme: 'outline',
      size: 'large',
      shape: 'pill',
      text: 'continue_with',
      width: 340
    })
  }, [scriptState])

  return (
    <div className="min-h-screen flex flex-col lg:flex-row">

      {/* Left: what the router actually does */}
      <section className="lg:w-[56%] bg-ink text-paper px-7 py-10 sm:px-14 sm:py-14 flex flex-col justify-between gap-12">
        <div className="flex items-center gap-3">
          <span className="w-[30px] h-[30px] rounded-lg bg-paper flex items-center justify-center flex-shrink-0">
            <svg className="w-[18px] h-[18px]" viewBox="0 0 24 24" fill="none" stroke="#1a1822" strokeWidth="2.1" strokeLinecap="round" aria-hidden="true">
              <path d="M3 12h4l3-7 4 14 3-7h4" />
            </svg>
          </span>
          <span className="wordmark !text-[#c9c4bc]">LLM ROUTER</span>
        </div>

        <div>
          <h1 className="font-display text-[2.1rem] sm:text-[3rem] leading-[1.08] font-semibold max-w-[560px] m-0">
            Every question gets the model it actually needs.
          </h1>
          <p className="mt-5 text-[0.97rem] leading-relaxed text-[#a5a0a8] max-w-[480px]">
            A small neural network reads your query, predicts how hard it is, and hands it
            to the smallest model that can answer it. You see the decision, not just the answer.
          </p>
        </div>

        <div className="max-w-[400px]">
          <FlowBox>
            <svg className="w-[15px] h-[15px] flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="#6e6878" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <path d="M4 6h16M4 12h10M4 18h13" />
            </svg>
            <span className="text-[0.85rem] text-[#c9c4bc]">Your query</span>
          </FlowBox>

          <Connector />

          <div className="px-4 py-3.5 border border-[#38343f] rounded-[10px] bg-[#211f2a]">
            <div className="text-[0.85rem] font-semibold">MLP classifier</div>
            <div className="mt-1 font-data text-[0.7rem] text-[#8b8594]">384-d embedding &rarr; 3 classes</div>
          </div>

          <Connector />

          <div className="flex flex-col gap-[7px]">
            <TierRow color="#35a488" label="Easy" model="qwen3:1.7b" />
            <TierRow color="#d2a03c" label="Medium" model="qwen3:4b" />
            <TierRow color="#d8705a" label="Hard" model="llama3.2:3b" />
          </div>
        </div>
      </section>

      {/* Right: sign in */}
      <section className="flex-1 flex items-center justify-center px-6 py-12 sm:p-12">
        <div className="card w-full max-w-[436px] p-8 sm:p-11">
          <h2 className="font-display text-[1.7rem] font-semibold m-0">Sign in</h2>
          <p className="mt-3 text-[0.9rem] leading-relaxed text-ink-2">
            Use your Google account. Your search history is private to you &mdash; no one
            else can read it or delete it.
          </p>

          <div className="mt-7 flex justify-center min-h-[48px]">
            {!configured && (
              <p className="text-[0.85rem] text-hard leading-relaxed" role="alert">
                Google sign-in is not configured in the browser. Set{' '}
                <code className="font-data">VITE_GOOGLE_CLIENT_ID</code> in{' '}
                <code className="font-data">frontend/.env</code> and restart the dev server.
              </p>
            )}
            {configured && serverConfigured === false && (
              <p className="text-[0.85rem] text-hard leading-relaxed" role="alert">
                The server has no <code className="font-data">GOOGLE_CLIENT_ID</code>. Set it
                in <code className="font-data">.env</code> and restart the backend.
              </p>
            )}
            {configured && serverConfigured !== false && scriptState === 'loading' && (
              <span className="font-data text-[0.78rem] text-ink-3 dot-pulse">Loading sign-in&hellip;</span>
            )}
            {configured && serverConfigured !== false && scriptState === 'error' && (
              <p className="text-[0.85rem] text-hard" role="alert">
                Could not reach Google sign-in. Check your connection and reload.
              </p>
            )}
            <div ref={buttonRef} className={scriptState === 'ready' && serverConfigured !== false ? '' : 'hidden'} />
          </div>

          {error && (
            <div className="mt-6 bubble-notice" role="alert">{error}</div>
          )}

          <div className="mt-8 pt-6 border-t border-line">
            <div className="eyebrow">What we store</div>
            <ul className="mt-3.5 flex flex-col gap-2.5 list-none p-0 m-0">
              <StoredItem>Your name, email and profile picture</StoredItem>
              <StoredItem>The questions you ask and their answers</StoredItem>
            </ul>
            <p className="mt-4 text-[0.78rem] leading-relaxed text-ink-3">
              You can delete any single search, or clear everything, at any time.
            </p>
          </div>
        </div>
      </section>
    </div>
  )
}

function FlowBox({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2.5 px-4 py-3 border border-[#38343f] rounded-[10px] bg-[#211f2a]">
      {children}
    </div>
  )
}

function Connector() {
  return <span className="block w-px h-[18px] bg-[#38343f] ml-7" aria-hidden="true" />
}

function TierRow({ color, label, model }: { color: string; label: string; model: string }) {
  return (
    <div
      className="flex items-center gap-3 px-4 py-2.5 rounded-[10px] bg-[#211f2a] border-l-[3px]"
      style={{ borderLeftColor: color }}
    >
      <span className="w-[62px] text-[0.8rem] font-semibold">{label}</span>
      <span className="font-data text-[0.75rem] text-[#a5a0a8]">{model}</span>
    </div>
  )
}

function StoredItem({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex items-start gap-2.5 text-[0.85rem] text-ink-2">
      <svg className="w-[13px] h-[13px] mt-1 flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="var(--color-easy)" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M20 6L9 17l-5-5" />
      </svg>
      <span>{children}</span>
    </li>
  )
}

export default LoginScreen
