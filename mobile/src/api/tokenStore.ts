import * as SecureStore from 'expo-secure-store';

import type { LoginUser } from '@/types/auth';

// expo-secure-store is async, but the axios request interceptor needs a
// synchronous read to attach the Authorization header. We keep an in-memory
// mirror that's hydrated once at boot (AuthContext) and kept in sync on every
// login/logout, so the interceptor never has to await SecureStore itself.
const TOKEN_KEY = 'nutrilens.token';
const USER_KEY = 'nutrilens.user';

let cachedToken: string | null = null;

export function getTokenSync(): string | null {
  return cachedToken;
}

export async function loadSession(): Promise<{ token: string; user: LoginUser } | null> {
  const [token, userJson] = await Promise.all([
    SecureStore.getItemAsync(TOKEN_KEY),
    SecureStore.getItemAsync(USER_KEY),
  ]);
  cachedToken = token;
  if (!token || !userJson) return null;
  try {
    return { token, user: JSON.parse(userJson) as LoginUser };
  } catch {
    // Corrupt/old-shape cache - treat as logged out rather than crash.
    return null;
  }
}

export async function saveSession(token: string, user: LoginUser): Promise<void> {
  cachedToken = token;
  await Promise.all([
    SecureStore.setItemAsync(TOKEN_KEY, token),
    SecureStore.setItemAsync(USER_KEY, JSON.stringify(user)),
  ]);
}

export async function clearSession(): Promise<void> {
  cachedToken = null;
  await Promise.all([SecureStore.deleteItemAsync(TOKEN_KEY), SecureStore.deleteItemAsync(USER_KEY)]);
}
