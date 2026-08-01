/**
 * src/api/client.ts
 * =================
 *
 * The one axios instance every request in the app goes through, wired exactly as
 * `07 §5` specifies: attach the JWT on the way out, normalise the error envelope
 * on the way back, and sign the user out on a 401.
 *
 * WHY THE TOKEN LIVES HERE AND NOT IN REACT STATE
 * -----------------------------------------------
 * The request interceptor is a plain function called by axios outside of React's
 * render cycle — it cannot read a context. So the token is held in a module-level
 * variable that `AuthContext` writes to whenever the session changes. React
 * remains the owner of *when* the token changes; this module is just the place
 * the interceptor can synchronously read it from.
 *
 * That also keeps `expo-secure-store` out of the request path. Secure store is
 * async and would make every interceptor an `await`; instead the token is read
 * from disk once at launch and mirrored here.
 *
 * WHY ERRORS ARE NORMALISED AT THE BOUNDARY
 * -----------------------------------------
 * Screens should never touch `err.response.data.error.code`. Three things can go
 * wrong on a call — the server returned an envelope, the server returned
 * something unexpected, or there was no response at all — and without
 * normalisation every screen would have to handle all three. `normaliseError`
 * collapses them into one `ApiError` with a `code` from `03 §9` (plus
 * `NETWORK_ERROR` / `TIMEOUT` for the no-response cases), which is what `07 §7`'s
 * error-copy mapping keys off.
 *
 * WHY 401 TRIGGERS A LOGOUT
 * -------------------------
 * `07 §1`'s auth gate reads the token at launch, but a token can expire mid
 * session (the backend issues 60-minute tokens). Without this the user would sit
 * on a screen retrying a request that can never succeed. The interceptor emits a
 * logout event rather than navigating directly, because this module has no
 * business knowing about navigation — `AuthContext` subscribes and clears the
 * session, and `RootNavigator` follows automatically.
 *
 * A 423 is deliberately *not* treated the same way. An account locked mid-session
 * still needs its explanation screen (`07 §7`), and silently bouncing the user to
 * Login would lose that message.
 */

import axios, { AxiosError, type AxiosInstance } from 'axios';

import type { ApiErrorBody, ApiErrorCode } from '@/types/api';

/**
 * Expo inlines `EXPO_PUBLIC_*` at build time (SDK 49+), so this needs no extra
 * dependency and no `app.json` plumbing. The fallback is the Android emulator's
 * alias for the host machine's localhost; a real device needs the LAN IP set in
 * `.env` (see `mobile/.env.example`).
 */
export const API_BASE_URL =
  process.env.EXPO_PUBLIC_API_BASE_URL ?? 'http://10.0.2.2:8000/api/v1';

/** Generous enough for a cold USDA lookup, short enough to fail before the user gives up. */
const REQUEST_TIMEOUT_MS = 15000;

/** The error every screen and hook in the app sees. Never an AxiosError. */
export class ApiError extends Error {
  readonly code: ApiErrorCode;
  readonly status: number | null;
  readonly details: unknown;

  constructor(body: ApiErrorBody, status: number | null) {
    super(body.message);
    this.name = 'ApiError';
    this.code = body.code;
    this.status = status;
    this.details = body.details;
  }
}

// --- Token holder (see the header note on why this is not React state) -------
let authToken: string | null = null;

export function setAuthToken(token: string | null): void {
  authToken = token;
}

export function getAuthToken(): string | null {
  return authToken;
}

// --- Unauthorized subscription ----------------------------------------------
type UnauthorizedListener = () => void;
const unauthorizedListeners = new Set<UnauthorizedListener>();

/** `AuthContext` subscribes; returns an unsubscribe for cleanup. */
export function onUnauthorized(listener: UnauthorizedListener): () => void {
  unauthorizedListeners.add(listener);
  return () => unauthorizedListeners.delete(listener);
}

function emitUnauthorized(): void {
  unauthorizedListeners.forEach((listener) => listener());
}

/**
 * Collapse anything axios can reject with into one `ApiError`.
 *
 * The three cases, in the order they are checked: a well-formed `03 §9`
 * envelope; a response whose body is something else (a proxy's HTML error page,
 * for instance); and no response at all.
 */
export function normaliseError(error: unknown): ApiError {
  if (!axios.isAxiosError(error)) {
    return new ApiError(
      { code: 'INTERNAL_ERROR', message: 'Something went wrong.' },
      null,
    );
  }

  const axiosError = error as AxiosError<{ error?: ApiErrorBody }>;

  if (axiosError.response) {
    const envelope = axiosError.response.data?.error;
    if (envelope?.code && envelope?.message) {
      return new ApiError(envelope, axiosError.response.status);
    }
    // The server answered, but not in the shape the contract promises.
    return new ApiError(
      { code: 'INTERNAL_ERROR', message: 'The server returned an unexpected response.' },
      axiosError.response.status,
    );
  }

  if (axiosError.code === 'ECONNABORTED') {
    return new ApiError(
      { code: 'TIMEOUT', message: 'The request took too long. Please try again.' },
      null,
    );
  }

  return new ApiError(
    { code: 'NETWORK_ERROR', message: "You're offline. Check your connection and try again." },
    null,
  );
}

export const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: REQUEST_TIMEOUT_MS,
});

api.interceptors.request.use((config) => {
  const token = getAuthToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    const normalised = normaliseError(error);
    // 401 only: a 423 (locked) still needs its own explanation screen (07 §7).
    if (normalised.status === 401) {
      emitUnauthorized();
    }
    return Promise.reject(normalised);
  },
);
