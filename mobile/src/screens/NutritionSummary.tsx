/**
 * src/screens/NutritionSummary.tsx
 * ================================
 *
 * Screen 3 of `07 §2` and the last step of the core loop: the user sees what
 * their meal contains and presses "Log This Meal". M5's Definition of Done is
 * measured here — "photo->log in <= 3 interactions", of which this screen owns
 * the last one.
 *
 * WHAT IT DOES
 * ------------
 * Takes the item the user confirmed upstream (food class, portion, the
 * classifier's confidence), asks `POST /nutrition` for its macros, renders them,
 * and on confirmation sends `POST /meals`. The cooking-method picker lives here
 * rather than on the recognition screen because changing it changes the numbers
 * on *this* screen — FR07 appends it to the USDA query — so the cause and its
 * effect are visible together.
 *
 * WHY THE MEAL IS NOT LOGGED FROM THE NUTRITION RESPONSE
 * -----------------------------------------------------
 * The obvious shortcut is to send the macros this screen just received along with
 * the meal, saving the server a lookup. `03 §6` forbids it and the request type
 * has no field for it: derived values are the server's (`02 §6`). So the log
 * request carries only what the user asserted — the food, the method, the portion
 * — and the backend recomputes. The numbers shown here and the numbers stored
 * agree because both come from the same `nutrition_service`, not because the app
 * passed them along.
 *
 * WHY THE MACRO SPLIT IS BY ENERGY
 * --------------------------------
 * The bars show each macro's share of the meal's *calories*, computed with the
 * Atwater factors (4/4/9 kcal per gram). By mass, fat always looks like a minor
 * component; by energy it frequently dominates, and energy is what the meal is
 * being judged on. The percentages are derived from the macro grams rather than
 * from the reported calorie figure, so the three bars always sum to 100% — using
 * the reported total would leave a visible gap wherever USDA's calorie value and
 * its macro values disagree slightly, which for lab-measured foods they often do.
 *
 * WHY THE DISCLAIMER IS RENDERED FROM THE RESPONSE
 * ------------------------------------------------
 * `07 §3` marks it required on this screen. It is passed straight through from
 * `NutritionResponse.disclaimer` rather than hard-coded, so the text on screen is
 * the text the server sent (see `DisclaimerBanner`).
 *
 * WHY A USDA FAILURE BLOCKS LOGGING
 * ---------------------------------
 * If nutrition cannot be fetched, the screen shows an error with a retry and the
 * log button does not appear. Logging anyway would write a meal whose macros the
 * server could not determine either — the failure is upstream of both calls. It
 * is better to keep the user on a screen that can succeed on retry than to record
 * a meal with nothing in it.
 */

import { useMemo, useState } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import CalorieRing from '@/components/CalorieRing';
import CookingMethodPicker from '@/components/CookingMethodPicker';
import DisclaimerBanner from '@/components/DisclaimerBanner';
import MacroBar from '@/components/MacroBar';
import PrimaryButton from '@/components/PrimaryButton';
import { ErrorState, LoadingState, Screen, messageForError } from '@/components/states';
import { useCookingMethods, useCreateMeal, useNutrition } from '@/hooks/useNutrition';
import { theme } from '@/theme';
import type { CookingMethod, MealType, NutritionRequest } from '@/types/api';

/** Atwater factors — kcal per gram. Used only for the on-screen energy split. */
const KCAL_PER_G = { protein: 4, carbs: 4, fat: 9 } as const;

/** The eight values from `03 §10`, used until `/meta/cooking-methods` answers. */
const FALLBACK_METHODS: CookingMethod[] = [
  'grilled',
  'fried',
  'boiled',
  'baked',
  'roasted',
  'steamed',
  'raw',
  'other',
];

export interface NutritionSummaryParams {
  food101_class: string;
  display_name: string;
  portion_confirmed_g: number;
  portion_min_g?: number | null;
  portion_max_g?: number | null;
  confidence_score?: number | null;
  meal_type: MealType;
  /** Preselected method, if the previous screen already asked. */
  cooking_method?: CookingMethod;
}

interface NutritionSummaryProps {
  params: NutritionSummaryParams;
  /** Called once the meal is logged — the flow returns to Home (`07 §1`). */
  onLogged: (logId: number) => void;
  onCancel?: () => void;
}

export default function NutritionSummary({
  params,
  onLogged,
  onCancel,
}: NutritionSummaryProps) {
  const [method, setMethod] = useState<CookingMethod>(params.cooking_method ?? 'grilled');

  const methodsQuery = useCookingMethods();
  const methods = methodsQuery.data ?? FALLBACK_METHODS;

  const request = useMemo<NutritionRequest>(
    () => ({
      food101_class: params.food101_class,
      cooking_method: method,
      portion_g: params.portion_confirmed_g,
    }),
    [params.food101_class, params.portion_confirmed_g, method],
  );

  const nutrition = useNutrition(request);
  const createMeal = useCreateMeal();

  // Energy split, from the macro grams rather than the reported calories — see
  // the header note on why those are not the same denominator.
  const split = useMemo(() => {
    const macros = nutrition.data?.scaled;
    if (!macros) return { protein: 0, carbs: 0, fat: 0 };

    const proteinKcal = macros.protein_g * KCAL_PER_G.protein;
    const carbsKcal = macros.carbs_g * KCAL_PER_G.carbs;
    const fatKcal = macros.fat_g * KCAL_PER_G.fat;
    const total = proteinKcal + carbsKcal + fatKcal;
    if (total <= 0) return { protein: 0, carbs: 0, fat: 0 };

    return {
      protein: proteinKcal / total,
      carbs: carbsKcal / total,
      fat: fatKcal / total,
    };
  }, [nutrition.data]);

  function handleLog() {
    createMeal.mutate(
      {
        meal_type: params.meal_type,
        items: [
          {
            food101_class: params.food101_class,
            cooking_method: method,
            portion_confirmed_g: params.portion_confirmed_g,
            portion_min_g: params.portion_min_g ?? null,
            portion_max_g: params.portion_max_g ?? null,
            confidence_score: params.confidence_score ?? null,
            // No macros here, deliberately: 02 §6 makes those the server's.
          },
        ],
      },
      { onSuccess: (meal) => onLogged(meal.log_id) },
    );
  }

  // First load only — a refetch triggered by changing the method keeps the
  // previous figures on screen (`keepPreviousData` in useNutrition).
  if (nutrition.isPending) {
    return (
      <Screen>
        <LoadingState label="Looking up nutrition…" />
      </Screen>
    );
  }

  if (nutrition.isError) {
    return (
      <Screen>
        <ErrorState error={nutrition.error} onRetry={() => void nutrition.refetch()} />
      </Screen>
    );
  }

  const data = nutrition.data;

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.heading}>{params.display_name}</Text>
        <Text style={styles.subheading}>
          {Math.round(params.portion_confirmed_g)} g · {data.food_name}
        </Text>

        <View style={styles.ring}>
          <CalorieRing consumed={data.scaled.calories} caption="this meal" />
        </View>

        <View style={styles.card}>
          <MacroBar
            label="Protein"
            grams={data.scaled.protein_g}
            pct={split.protein}
            color={theme.color.protein}
          />
          <MacroBar
            label="Carbs"
            grams={data.scaled.carbs_g}
            pct={split.carbs}
            color={theme.color.carbs}
          />
          <MacroBar
            label="Fat"
            grams={data.scaled.fat_g}
            pct={split.fat}
            color={theme.color.fat}
          />
        </View>

        <CookingMethodPicker value={method} options={methods} onChange={setMethod} />

        {/* Quiet provenance. `03 §5`'s `source` is what makes the cache fallback
            visible; showing it costs a line and is the honest thing to surface
            when a value may be older than today. */}
        <Text style={styles.provenance}>
          {data.source === 'cache' ? 'From cached USDA data' : 'From USDA FoodData Central'}
          {nutrition.isFetching ? ' · updating…' : ''}
        </Text>

        <DisclaimerBanner text={data.disclaimer} />

        {createMeal.isError ? (
          <Text style={styles.error}>{messageForError(createMeal.error)}</Text>
        ) : null}

        <View style={styles.actions}>
          <PrimaryButton
            label="Log This Meal"
            onPress={handleLog}
            loading={createMeal.isPending}
          />
          {onCancel ? (
            <PrimaryButton label="Cancel" variant="secondary" onPress={onCancel} />
          ) : null}
        </View>
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: {
    padding: theme.space(5),
    paddingBottom: theme.space(10),
  },
  heading: {
    color: theme.color.textPrimary,
    fontSize: 24,
    fontWeight: '500',
  },
  subheading: {
    color: theme.color.textSecondary,
    fontSize: 14,
    marginTop: theme.space(1),
  },
  ring: {
    alignItems: 'center',
    marginVertical: theme.space(6),
  },
  card: {
    backgroundColor: theme.color.surface,
    borderRadius: theme.radius.lg,
    borderWidth: 1,
    borderColor: theme.color.border,
    padding: theme.space(4),
    paddingBottom: theme.space(2),
    marginBottom: theme.space(5),
  },
  provenance: {
    color: theme.color.textSecondary,
    fontSize: 12,
  },
  error: {
    color: theme.color.danger,
    fontSize: 13,
    marginTop: theme.space(4),
  },
  actions: {
    marginTop: theme.space(6),
    gap: theme.space(3),
  },
});
