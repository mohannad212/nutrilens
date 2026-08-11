import { useMutation } from '@tanstack/react-query';

import { loginRequest } from '@/api/auth';
import type { ApiError } from '@/api/client';
import { useAuth } from '@/store/AuthContext';
import type { LoginRequest, TokenResponse } from '@/types/auth';

// One hook per endpoint (07-mobile-app.md §4). Saves the session on success
// so callers just need to handle navigation + error display.
export function useLogin() {
  const { login } = useAuth();

  return useMutation<TokenResponse, ApiError, LoginRequest>({
    mutationFn: loginRequest,
    onSuccess: async (data) => {
      await login(data.access_token, data.user);
    },
  });
}
