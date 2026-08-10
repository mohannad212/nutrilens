import { useQuery } from '@tanstack/react-query';

import { getRecommendations } from '@/api/endpoints';
import type { RecommendationsResponse } from '@/types/api';

export function useInsights() {
  return useQuery<RecommendationsResponse>({
    queryKey: ['recommendations'],
    queryFn: getRecommendations,
  });
}