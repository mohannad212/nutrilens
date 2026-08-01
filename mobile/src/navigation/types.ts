/**
 * src/navigation/types.ts
 * =======================
 *
 * Param lists for every navigator, in one place so a screen can type its own
 * route without importing the navigator that hosts it (which would be circular).
 *
 * These are what make `navigation.navigate('NutritionSummary', params)` a
 * compile-time contract. The capture flow hands a fair amount of state forward —
 * the food class, the confirmed portion, the classifier's confidence — and every
 * one of those is required by `POST /nutrition` or `POST /meals`. Typing the
 * route means M3's camera screen cannot navigate here having forgotten one.
 */

import type { NutritionSummaryParams } from '@/screens/NutritionSummary';
import type { RecognitionResponse } from '@/types/api';

/** `07 §1` — the capture flow: Camera -> RecognitionResults -> NutritionSummary. */
export type CaptureStackParamList = {
  Camera: undefined;
  /** M3/M4 populate this from the `/recognise` response. */
  RecognitionResults: { recognition: RecognitionResponse } | undefined;
  NutritionSummary: NutritionSummaryParams;
};

export type AppTabsParamList = {
  Home: undefined;
  History: undefined;
  Capture: undefined;
  Insights: undefined;
  Profile: undefined;
};

export type AuthStackParamList = {
  Login: undefined;
  Register: undefined;
};
