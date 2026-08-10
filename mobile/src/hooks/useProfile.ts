import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { getProfile, updateGoal, updateMetrics } from '@/api/endpoints';
import type { UpdateGoalRequest, UpdateMetricsRequest, UserProfile } from '@/types/api';

export function useProfile() {
  return useQuery<UserProfile>({
    queryKey: ['profile'],
    queryFn: getProfile,
  });
}

export function useUpdateGoal() {
  const queryClient = useQueryClient();
  return useMutation<UserProfile, unknown, UpdateGoalRequest>({
    mutationFn: updateGoal,
    onSuccess: (data) => queryClient.setQueryData(['profile'], data),
  });
}

export function useUpdateMetrics() {
  const queryClient = useQueryClient();
  return useMutation<UserProfile, unknown, UpdateMetricsRequest>({
    mutationFn: updateMetrics,
    onSuccess: (data) => queryClient.setQueryData(['profile'], data),
  });
}