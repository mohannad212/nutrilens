/**
 * src/store/AuthContext.tsx
 * =========================
 *
 * Session state for the whole app (`07 §4`): the JWT, whether the user is signed
 * in, and the two actions that change that. Backed by `expo-secure-store` so the
 * token survives a relaunch — the M2 Definition of Done's "token persists across
 * relaunch".
 *
 * WHY THE TOKEN IS MIRRORED INTO THE API CLIENT
 * ---------------------------------------------
 * Axios interceptors run outside React, so they cannot read this context. Every
 * change to the token is therefore written to three places at once: React state
 * (so the UI re-renders and the auth gate flips), the API client's module-level
 * holder (so the next request carries it), and secure store (so the next launch
 * finds it). Keeping all three writes inside `login`/`logout` is what stops them
 * drifting apart — the classic bug being a logout that clears React state but
 * leaves the interceptor holding a valid token.
 *
 * WHY THERE IS AN `isRestoring` FLAG
 * ----------------------------------
 * Reading secure store is asynchronous, so on the very first render the token is
 * unknown — neither present nor confirmed absent. Without this flag the auth gate
 * would treat "not loaded yet" as "signed out" and flash the Login screen for a
 * frame before bouncing an already-signed-in user to Home. `RootNavigator` waits
 * on it instead.
 *
 * WHY A 401 CLEARS THE SESSION FROM HERE
 * --------------------------------------
 * The API client emits an unauthorized event rather than navigating (it has no
 * business knowing about screens). This is where that event becomes a state
 * change: the token is dropped, `isAuthenticated` flips, and `RootNavigator`
 * shows `AuthStack`. Tokens expire after an hour, so this is a path real users
 * hit, not just an error case.
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

const TOKEN_KEY = 'nutrilens.access_token';

interface AuthContextValue {
  token: string | null;
  isAuthenticated: boolean;
  /** True until secure store has been read once — see the header note. */
  isRestoring: boolean;
  login: (token: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [isRestoring, setIsRestoring] = useState(true);

  // Restore the persisted session once, at launch (07 §1's auth gate).
  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const stored = await SecureStore.getItemAsync(TOKEN_KEY);
        if (!cancelled && stored) {
          setToken(stored);
          setAuthToken(stored);
        }
      } catch {
        // A secure-store read can fail on a device with no keychain access.
        // Treating that as "signed out" is correct: the user can log in again,
        // whereas crashing at launch leaves them with no way forward.
      } finally {
        if (!cancelled) setIsRestoring(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (next: string) => {
    setToken(next);
    setAuthToken(next);
    await SecureStore.setItemAsync(TOKEN_KEY, next);
  }, []);

  const logout = useCallback(async () => {
    setToken(null);
    setAuthToken(null);
    await SecureStore.deleteItemAsync(TOKEN_KEY);
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
      isAuthenticated: token !== null,
      isRestoring,
      login,
      logout,
    }),
    [token, isRestoring, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider');
  return ctx;
}
