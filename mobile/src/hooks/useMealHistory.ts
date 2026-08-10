import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { deleteMeal, getMeals } from '@/api/endpoints';
import type { MealsListResponse } from '@/types/api';

const PAGE_SIZE = 20;

/** Full meal history, newest first, paginated (03 §11). No date bound — all history. */
export function useMealHistory(offset: number) {
  return useQuery<MealsListResponse>({
    queryKey: ['meals', 'history', offset],
    queryFn: () => getMeals({ limit: PAGE_SIZE, offset }),
    placeholderData: (prev) => prev,
  });
}

export function useDeleteMeal() {
  const queryClient = useQueryClient();
  return useMutation<void, unknown, number>({
    mutationFn: deleteMeal,
    onSuccess: () => {
      // Removing a meal changes both the history list and the Home totals.
      void queryClient.invalidateQueries({ queryKey: ['meals'] });
    },
  });
}