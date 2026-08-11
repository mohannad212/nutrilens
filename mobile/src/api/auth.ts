import type { LoginRequest, RegisterRequest, RegisterResponse, TokenResponse } from '@/types/auth';
import { api } from './client';

// Typed wrappers mirroring 03-api-specification.md §2 exactly.
//
// These deliberately do not catch. `client.ts`'s response interceptor is the one
// place that turns a failure into an `ApiError` (07 §5), and wrapping it in a
// second normaliser here discarded the server's real code — a 422 with field
// details or a 409 DUPLICATE_USERNAME both arrived at the screen as
// NETWORK_ERROR, which the copy renders as "You're offline".

export async function registerRequest(payload: RegisterRequest): Promise<RegisterResponse> {
  const res = await api.post<RegisterResponse>('/auth/register', payload);
  return res.data;
}

export async function loginRequest(payload: LoginRequest): Promise<TokenResponse> {
  const res = await api.post<TokenResponse>('/auth/login', payload);
  return res.data;
}
