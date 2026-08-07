import type { AxiosError } from 'axios';

import type { LoginRequest, RegisterRequest, RegisterResponse, TokenResponse } from '@/types/auth';
import { api } from './client';
import { type ApiErrorEnvelope, normaliseError } from './errors';

// Typed wrappers mirroring 03-api-specification.md §2 exactly.

export async function registerRequest(payload: RegisterRequest): Promise<RegisterResponse> {
  try {
    const res = await api.post<RegisterResponse>('/auth/register', payload);
    return res.data;
  } catch (err) {
    throw normaliseError(err as AxiosError<ApiErrorEnvelope>);
  }
}

export async function loginRequest(payload: LoginRequest): Promise<TokenResponse> {
  try {
    const res = await api.post<TokenResponse>('/auth/login', payload);
    return res.data;
  } catch (err) {
    throw normaliseError(err as AxiosError<ApiErrorEnvelope>);
  }
}
