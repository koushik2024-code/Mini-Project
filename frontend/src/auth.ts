/** Session token storage and an authenticated fetch wrapper. */

const TOKEN_KEY = 'llm_router_session'

export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
export const GOOGLE_CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID || ''

export interface AuthUser {
  id: number
  email: string
  name: string
  picture: string
}

/** Thrown when the backend rejects our session, so callers can sign out. */
export class UnauthorizedError extends Error {
  constructor(message = 'Session expired') {
    super(message)
    this.name = 'UnauthorizedError'
  }
}

// localStorage throws in private mode and with site data blocked, so every
// access is guarded and the app still works (sign-in just won't persist).
export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function setToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token)
  } catch {
    /* session lasts for this tab only */
  }
}

export function clearToken(): void {
  try {
    localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* nothing to clear */
  }
}

/** fetch() with the session token attached; throws UnauthorizedError on 401. */
export async function authFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const token = getToken()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)

  const response = await fetch(`${API_BASE}${path}`, { ...init, headers })

  if (response.status === 401) {
    clearToken()
    throw new UnauthorizedError()
  }

  return response
}

/** Read `detail` out of a FastAPI error body, falling back to a default. */
export async function errorDetail(response: Response, fallback: string): Promise<string> {
  try {
    const body = await response.json()
    return typeof body?.detail === 'string' ? body.detail : fallback
  } catch {
    return fallback
  }
}
