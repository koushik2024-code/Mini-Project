import { useCallback, useEffect, useState } from 'react'
import {
  API_BASE,
  type AuthUser,
  authFetch,
  clearToken,
  errorDetail,
  setToken
} from './auth'

export type AuthStatus = 'loading' | 'signed-in' | 'signed-out'

export interface UseAuth {
  user: AuthUser | null
  status: AuthStatus
  error: string | null
  signInWithGoogle: (credential: string) => Promise<void>
  signOut: () => Promise<void>
  /** Drop the local session after the backend rejects our token. */
  handleExpired: () => void
}

export function useAuth(): UseAuth {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [status, setStatus] = useState<AuthStatus>('loading')
  const [error, setError] = useState<string | null>(null)

  // Restore an existing session on first load
  useEffect(() => {
    let cancelled = false

    const restore = async () => {
      try {
        const response = await authFetch('/auth/me')
        if (cancelled) return

        if (!response.ok) {
          setStatus('signed-out')
          return
        }

        setUser(await response.json())
        setStatus('signed-in')
      } catch {
        // No token, expired token, or backend unreachable
        if (!cancelled) setStatus('signed-out')
      }
    }

    restore()
    return () => { cancelled = true }
  }, [])

  const signInWithGoogle = useCallback(async (credential: string) => {
    setError(null)
    try {
      const response = await fetch(`${API_BASE}/auth/google`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ credential })
      })

      if (!response.ok) {
        throw new Error(await errorDetail(response, 'Sign-in failed'))
      }

      const data = await response.json()
      setToken(data.access_token)
      setUser(data.user)
      setStatus('signed-in')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign-in failed')
      setStatus('signed-out')
    }
  }, [])

  const signOut = useCallback(async () => {
    try {
      await authFetch('/auth/logout', { method: 'POST' })
    } catch {
      // Token already rejected - signing out locally is all that's left
    }
    clearToken()
    setUser(null)
    setStatus('signed-out')
  }, [])

  const handleExpired = useCallback(() => {
    clearToken()
    setUser(null)
    setStatus('signed-out')
    setError('Your session expired. Please sign in again.')
  }, [])

  return { user, status, error, signInWithGoogle, signOut, handleExpired }
}
