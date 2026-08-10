import { useQuery } from '@tanstack/react-query';

import { getMeals } from '@/api/endpoints';
import type { MealsListResponse } from '@/types/api';

function todayRange() {
  const now = new Date();
  const start = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  return { from: start.toISOString(), to: now.toISOString() };
}

/** Today's confirmed meals — the data behind the Home dashboard. */
export function useTodayMeals() {
  const { from, to } = todayRange();
  return useQuery<MealsListResponse>({
    queryKey: ['meals', 'today', from],
    queryFn: () => getMeals({ from, to, limit: 50 }),
  });
}