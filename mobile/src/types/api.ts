/**
 * src/types/api.ts
 * ================
 *
 * The TypeScript half of the API contract. `03-api-specification.md` says it
 * outright: "the Pydantic schemas in `app/schemas/` and the TypeScript types in
 * `mobile/src/types/` must match exactly."
 *
 * WHY THESE ARE HAND-WRITTEN AND NOT GENERATED
 * --------------------------------------------
 * FastAPI publishes an OpenAPI schema, so `openapi-typescript` could generate
 * this file. It is written by hand anyway, for two reasons. Generation adds a
 * build step that must run against a *live backend* before the app will
 * typecheck, which breaks CI whenever the two are worked on separately. And the
 * generated output mirrors whatever the server currently does, so a backend
 * field that accidentally drifts from `03` propagates silently into the app
 * instead of failing the compile — which is precisely the failure these types
 * exist to catch.
 *
 * Each type below is annotated with the section of `03` it binds to. If a field
 * here disagrees with that section, one of the two is a bug.
 */

/** `03 §10` — enumerations, the single source of truth. */
export type HealthGoal = 'weight_loss' | 'maintenance' | 'weight_gain';
export type MealType = 'breakfast' | 'lunch' | 'dinner' | 'snack';
export type CookingMethod =
  | 'grilled'
  | 'fried'
  | 'boiled'
  | 'baked'
  | 'roasted'
  | 'steamed'
  | 'raw'
  | 'other';
export type PortionMethod = 'reference_object' | 'container_prior' | 'shape_only';
export type NutritionSource = 'usda' | 'cache';

/**
 * `03 §9` — the error envelope, identical for every non-2xx response.
 *
 * The app's single axios interceptor normalises every failure into this shape,
 * so screens branch on `code` rather than on HTTP status. That is what lets
 * `USDA_UNAVAILABLE` get its own friendly copy (`07 §7`) without every screen
 * knowing that it happens to be a 502.
 */
export type ApiErrorCode =
  | 'VALIDATION_ERROR'
  | 'INVALID_CREDENTIALS'
  | 'ACCOUNT_LOCKED'
  | 'DUPLICATE_EMAIL'
  | 'DUPLICATE_USERNAME'
  | 'UNAUTHORIZED'
  | 'FORBIDDEN'
  | 'NOT_FOUND'
  | 'IMAGE_TOO_LARGE'
  | 'IMAGE_BAD_FORMAT'
  | 'IMAGE_TOO_SMALL'
  | 'IMAGE_POOR_LIGHTING'
  | 'IMAGE_TOO_BLURRY'
  | 'INFERENCE_FAILED'
  | 'USDA_RATE_LIMITED'
  | 'USDA_UNAVAILABLE'
  | 'NO_NUTRITION_MATCH'
  | 'INTERNAL_ERROR'
  // Not from the server: the interceptor's own codes for a request that never
  // got an answer. Screens need to distinguish "the server said no" from "the
  // server said nothing" to offer a retry rather than an explanation.
  | 'NETWORK_ERROR'
  | 'TIMEOUT';

export interface ApiErrorBody {
  code: ApiErrorCode;
  message: string;
  details?: unknown;
}

/** `03 §4` — recognition, produced upstream and carried into the M5 flow. */
export interface Portion {
  min_g: number;
  max_g: number;
  point_g: number;
  method: PortionMethod;
  reference: string | null;
  scale_source: string;
}

export interface Prediction {
  rank: number;
  food101_class: string;
  display_name: string;
  confidence: number;
  portion: Portion | null;
}

export interface Alternative {
  food101_class: string;
  display_name: string;
  confidence: number;
}

export interface RecognitionResponse {
  request_id: string;
  items: Prediction[];
  low_confidence: boolean;
  alternatives: Alternative[];
}

/** `03 §5` — nutrition. */
export interface MacroSet {
  /** Whole kcal — the server rounds per `03 §1`; the app never re-rounds. */
  calories: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
}

export interface NutritionRequest {
  food101_class: string;
  cooking_method: CookingMethod;
  portion_g: number;
}

export interface NutritionResponse {
  food_id: number;
  food_name: string;
  /** Null when the cache row could not record provenance — see the backend's nutrition_cache. */
  usda_fdc_id: string | null;
  cooking_method: CookingMethod;
  per_100g: MacroSet;
  portion_g: number;
  scaled: MacroSet;
  source: NutritionSource;
  disclaimer: string;
}

/** `03 §6` — meals. Note the absence of any nutrition field on the request. */
export interface MealItemIn {
  food101_class: string;
  cooking_method: CookingMethod;
  portion_confirmed_g: number;
  portion_min_g?: number | null;
  portion_max_g?: number | null;
  confidence_score?: number | null;
}

export interface MealCreateRequest {
  meal_type: MealType;
  items: MealItemIn[];
}

export interface MealItemOut {
  item_id: number;
  food_name: string;
  calories_est: number;
  protein_est_g: number;
  carbs_est_g: number;
  fat_est_g: number;
  portion_confirmed_g: number;
}

export interface MealCreateResponse {
  log_id: number;
  logged_at: string;
  meal_type: MealType;
  total_calories: number;
  items: MealItemOut[];
  meals_logged_total: number;
  recommendations_unlocked: boolean;
  disclaimer: string;
}

/** `03 §8` — meta, used to keep the cooking-method picker in sync with the enum. */
export interface CookingMethodsResponse {
  cooking_methods: CookingMethod[];
}
/** `03 §11` — GET /meals list response, one row per logged meal today. */
export interface MealSummary {
  log_id: number;
  logged_at: string;
  meal_type: MealType;
  total_calories: number;
  item_count: number;
  items: MealItemOut[];
}

export interface MealsListResponse {
  total: number;
  limit: number;
  offset: number;
  meals: MealSummary[];
}
/** `03 §3` — the authenticated user's profile. */
export type Sex = 'male' | 'female';
export type ActivityLevel =
  | 'sedentary'
  | 'light'
  | 'moderate'
  | 'active'
  | 'very_active';

export interface UserProfile {
  user_id: number;
  username: string;
  email: string;
  health_goal: HealthGoal;
  created_at: string;
  height_cm: number | null;
  weight_kg: number | null;
  birth_year: number | null;
  sex: Sex | null;
  activity_level: ActivityLevel | null;
}

export interface UpdateGoalRequest {
  health_goal: HealthGoal;
}

/** `03 §3` — any subset; an omitted key is unchanged, an explicit null clears it. */
export interface UpdateMetricsRequest {
  height_cm?: number | null;
  weight_kg?: number | null;
  birth_year?: number | null;
  sex?: Sex | null;
  activity_level?: ActivityLevel | null;
}
/** `03 §7` — FR12: unlocks after `MIN_MEALS_FOR_RECOMMENDATION` logged meals. */
export interface Insight {
  rec_id: number;
  insight_text: string;
  nutrient_focus: string;
  was_read: boolean;
}

export interface RecommendationsResponse {
  unlocked: boolean;
  insights: Insight[];
  generated_at: string;
  meals_logged: number;
  meals_required: number;
  message: string;
  disclaimer: string;
}