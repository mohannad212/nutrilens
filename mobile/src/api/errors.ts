import type { AxiosError } from 'axios';

// Mirrors the shared error envelope from backend/app/core/errors.py and
// 03-api-specification.md §9: every non-2xx response is { error: { code, message, details? } }.
export interface ApiErrorEnvelope {
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}

export class ApiError extends Error {
  code: string;
  status: number | undefined;
  details: unknown;

  constructor(code: string, message: string, status?: number, details?: unknown) {
    super(message);
    this.code = code;
    this.status = status;
    this.details = details;
  }

  /** Remaining login attempts before lockout, when present (401 on /auth/login). */
  get remainingAttempts(): number | undefined {
    if (
      this.details &&
      typeof this.details === 'object' &&
      'remaining_attempts' in this.details
    ) {
      return (this.details as { remaining_attempts: number }).remaining_attempts;
    }
    return undefined;
  }
}

// Codes from 03-api-specification.md / core/errors.py - kept here so screens
// can switch on a stable string instead of magic literals scattered around.
export const ErrorCode = {
  DUPLICATE_EMAIL: 'DUPLICATE_EMAIL',
  DUPLICATE_USERNAME: 'DUPLICATE_USERNAME',
  INVALID_CREDENTIALS: 'INVALID_CREDENTIALS',
  ACCOUNT_LOCKED: 'ACCOUNT_LOCKED',
  UNAUTHORIZED: 'UNAUTHORIZED',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  NETWORK_ERROR: 'NETWORK_ERROR',
} as const;

/** Turns any axios failure (server error envelope, network failure, etc.) into an ApiError. */
export function normaliseError(err: AxiosError<ApiErrorEnvelope>): ApiError {
  const body = err.response?.data;
  if (body && 'error' in body) {
    return new ApiError(body.error.code, body.error.message, err.response?.status, body.error.details);
  }
  if (err.response) {
    return new ApiError(
      'UNKNOWN_ERROR',
      `Request failed with status ${err.response.status}.`,
      err.response.status,
    );
  }
  return new ApiError(
    ErrorCode.NETWORK_ERROR,
    "Can't reach the server. Check your connection and try again.",
  );
}
