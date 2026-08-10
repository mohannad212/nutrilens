/**
 * src/store/AuthContext.tsx
 * =========================
 *
 * Session state for the whole app (`07 §4`): the JWT, the signed-in user's
 * basic info, whether the user is signed in, and the actions that change that.
 * Backed by `expo-secure-store` so both survive a relaunch.
 */

import * as SecureStore from 'expo-secure-store';
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

import { onUnauthorized, setAuthToken } from '@/api/client';
import type { LoginUser } from '@/types/auth';

const TOKEN_KEY = 'nutrilens.access_token';
const USER_KEY = 'nutrilens.user';

interface AuthContextValue {
  token: string | null;
  user: LoginUser | null;
  isAuthenticated: boolean;
  /** True until secure store has been read once — see the header note. */
  isRestoring: boolean;
  login: (token: string, user: LoginUser) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<LoginUser | null>(null);
  const [isRestoring, setIsRestoring] = useState(true);

  // Restore the persisted session once, at launch (07 §1's auth gate).
  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const [storedToken, storedUser] = await Promise.all([
          SecureStore.getItemAsync(TOKEN_KEY),
          SecureStore.getItemAsync(USER_KEY),
        ]);
        if (!cancelled && storedToken) {
          setToken(storedToken);
          setAuthToken(storedToken);
          if (storedUser) setUser(JSON.parse(storedUser) as LoginUser);
        }
      } catch {
        // A secure-store read can fail on a device with no keychain access.
        // Treating that as "signed out" is correct: the user can log in again.
      } finally {
        if (!cancelled) setIsRestoring(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (nextToken: string, nextUser: LoginUser) => {
    setToken(nextToken);
    setUser(nextUser);
    setAuthToken(nextToken);
    await Promise.all([
      SecureStore.setItemAsync(TOKEN_KEY, nextToken),
      SecureStore.setItemAsync(USER_KEY, JSON.stringify(nextUser)),
    ]);
  }, []);

  const logout = useCallback(async () => {
    setToken(null);
    setUser(null);
    setAuthToken(null);
    await Promise.all([
      SecureStore.deleteItemAsync(TOKEN_KEY),
      SecureStore.deleteItemAsync(USER_KEY),
    ]);
  }, []);

  // An expired or rejected token anywhere in the app ends the session here.
  useEffect(() => {
    return onUnauthorized(() => {
      void logout();
    });
  }, [logout]);

  const value = useMemo<AuthContextValue>(
    () => ({
      token,
      user,
      isAuthenticated: token !== null,
      isRestoring,
      login,
      logout,
    }),
    [token, user, isRestoring, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider');
  return ctx;
}