import { useQuery } from '@tanstack/react-query';

import { getMeals } from '@/api/endpoints';
import type { MealsListResponse } from '@/types/api';

function startOfToday(): string {
  const now = new Date();
  return new Date(now.getFullYear(), now.getMonth(), now.getDate()).toISOString();
}

/** Today's confirmed meals — the data behind the Home dashboard. */
export function useTodayMeals() {
  const from = startOfToday();
  return useQuery<MealsListResponse>({
    queryKey: ['meals', 'today', from],
    queryFn: () => getMeals({ from, limit: 50 }),
  });
}