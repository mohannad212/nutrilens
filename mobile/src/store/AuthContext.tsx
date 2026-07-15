import { createContext, useContext, useMemo, useState, type ReactNode } from 'react';

interface AuthState {
  token: string | null;
  isAuthenticated: boolean;
}

interface AuthContextValue extends AuthState {
  login: (token: string) => void;
  logout: () => void;
}

// M2 wires this to POST /auth/login and expo-secure-store; M0 only needs the auth gate to exist.
const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null);

  const value = useMemo<AuthContextValue>(
    () => ({
      token,
      isAuthenticated: token !== null,
      login: (t: string) => setToken(t),
      logout: () => setToken(null),
    }),
    [token],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider');
  return ctx;
}
