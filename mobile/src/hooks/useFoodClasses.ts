import { useQuery } from '@tanstack/react-query';

import { getFoodClasses } from '@/api/endpoints';
import type { FoodClassesResponse } from '@/types/api';

/**
 * The 101 predictable classes, for the manual-correction list.
 *
 * `staleTime: Infinity` because the list changes only when the model is
 * retrained and redeployed — refetching it on every focus would be pure noise.
 */
export function useFoodClasses() {
  return useQuery<FoodClassesResponse>({
    queryKey: ['food-classes'],
    queryFn: getFoodClasses,
    staleTime: Infinity,
  });
}
