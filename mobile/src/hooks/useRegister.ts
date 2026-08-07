import { useMutation } from '@tanstack/react-query';

import { registerRequest } from '@/api/auth';
import type { ApiError } from '@/api/errors';
import type { RegisterRequest, RegisterResponse } from '@/types/auth';

// 03 §2: register returns the created profile only (no token) - the caller
// follows up with /auth/login. See Register.tsx.
export function useRegister() {
  return useMutation<RegisterResponse, ApiError, RegisterRequest>({
    mutationFn: registerRequest,
  });
}
