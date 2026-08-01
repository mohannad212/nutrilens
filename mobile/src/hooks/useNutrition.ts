/**
 * src/hooks/useNutrition.ts
 * =========================
 *
 * `07 §4` — one hook per endpoint. These three cover M5: read nutrition for a
 * confirmed item, read the cooking-method list, and log the meal.
 *
 * WHY NUTRITION IS A `useQuery` DESPITE BEING A POST
 * --------------------------------------------------
 * `POST /nutrition` is a POST for the reasons in the backend router (a
 * computation, and a first call writes a cache row), but from the app's side it
 * is a *read*: same inputs, same answer, and it should be re-fetched when the
 * user changes the portion or the cooking method. Modelling it as a query gives
 * exactly that — the query key is the request body, so adjusting the slider
 * refetches, and sliding back to a value already seen is served from cache with
 * no request at all. As a mutation it would need manual re-invocation on every
 * change and would cache nothing.
 *
 * `placeholderData: keepPreviousData` is what makes the slider feel continuous:
 * while the new figure loads, the previous one stays on screen instead of the
 * whole panel collapsing to a spinner on every drag.
 *
 * WHY THE MEAL LOG IS A `useMutation`
 * -----------------------------------
 * It genuinely creates something, it is not idempotent (`03 §6`), and it must
 * only run when the user presses the button. It invalidates `['meals']` and
 * `['recommendations']` on success, per `07 §4` — the meal that was just logged
 * changes both, and the seventh one may have unlocked insights entirely.
 */

import {
  keepPreviousData,
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
  type UseQueryResult,
} from '@tanstack/react-query';

import { getCookingMethods, postMeal, postNutrition } from '@/api/endpoints';
import type { ApiError } from '@/api/client';
import type {
  CookingMethod,
  MealCreateRequest,
  MealCreateResponse,
  NutritionRequest,
  NutritionResponse,
} from '@/types/api';

/**
 * `03 §5` — macros for one confirmed item.
 *
 * `enabled` lets a screen hold the hook while the inputs are still incomplete
 * (no food chosen yet) without firing a request that would 422.
 */
export function useNutrition(
  request: NutritionRequest | null,
): UseQueryResult<NutritionResponse, ApiError> {
  return useQuery<NutritionResponse, ApiError>({
    // The body *is* the identity of the result, so it is the whole key: change
    // the portion or the method and this is a different question.
    queryKey: ['nutrition', request],
    queryFn: () => postNutrition(request as NutritionRequest),
    enabled: request !== null,
    // Keep the previous figure visible while a slider drag refetches.
    placeholderData: keepPreviousData,
    // Nutrition for a fixed (food, method, portion) does not change minute to
    // minute; the backend cache has the same view (08 §6).
    staleTime: 5 * 60 * 1000,
  });
}

/** `03 §8` — the picker's options, fetched once per session (see CookingMethodPicker). */
export function useCookingMethods(): UseQueryResult<CookingMethod[], ApiError> {
  return useQuery<CookingMethod[], ApiError>({
    queryKey: ['meta', 'cooking-methods'],
    queryFn: async () => (await getCookingMethods()).cooking_methods,
    staleTime: Infinity, // an enum; it does not change while the app is open
  });
}

/** `03 §6` — log the confirmed meal (FR10). */
export function useCreateMeal(): UseMutationResult<
  MealCreateResponse,
  ApiError,
  MealCreateRequest
> {
  const queryClient = useQueryClient();

  return useMutation<MealCreateResponse, ApiError, MealCreateRequest>({
    mutationFn: postMeal,
    onSuccess: () => {
      // 07 §4: the new meal changes history, and it may have just unlocked FR12.
      void queryClient.invalidateQueries({ queryKey: ['meals'] });
      void queryClient.invalidateQueries({ queryKey: ['recommendations'] });
    },
  });
}
