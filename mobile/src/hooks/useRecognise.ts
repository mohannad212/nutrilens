import { useMutation } from '@tanstack/react-query';

import { recogniseRequest } from '@/api/recognition';
import type { ApiError } from '@/api/errors';
import type { RecognitionResponse } from '@/types/api';

// One hook per endpoint (07-mobile-app.md §4).
export function useRecognise() {
  return useMutation<RecognitionResponse, ApiError, string>({
    mutationFn: recogniseRequest,
  });
}
