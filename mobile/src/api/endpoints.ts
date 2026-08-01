/**
 * src/api/endpoints.ts
 * ====================
 *
 * Typed wrappers around the endpoints M5 needs (`07 §5`: "Typed endpoint wrappers
 * mirror `03` exactly"). Each function is one line of axios plus the two types
 * that bind it to the contract.
 *
 * The value is not the abstraction — it is that the URL string and the request
 * shape appear exactly once each. A screen calling `postNutrition(body)` cannot
 * misspell the path, cannot forget that `/nutrition` is a POST, and cannot pass a
 * body the backend would 422. When `03` changes, the compiler lists every call
 * site.
 */

import { api } from './client';
import type {
  CookingMethodsResponse,
  MealCreateRequest,
  MealCreateResponse,
  NutritionRequest,
  NutritionResponse,
} from '@/types/api';

/** `03 §5` — macros for one confirmed item, scaled to its confirmed portion. */
export async function postNutrition(body: NutritionRequest): Promise<NutritionResponse> {
  const { data } = await api.post<NutritionResponse>('/nutrition', body);
  return data;
}

/** `03 §6` — persist a confirmed meal. The server computes every derived value. */
export async function postMeal(body: MealCreateRequest): Promise<MealCreateResponse> {
  const { data } = await api.post<MealCreateResponse>('/meals', body);
  return data;
}

/** `03 §8` — the cooking-method enum, so the picker cannot drift from the validator. */
export async function getCookingMethods(): Promise<CookingMethodsResponse> {
  const { data } = await api.get<CookingMethodsResponse>('/meta/cooking-methods');
  return data;
}
