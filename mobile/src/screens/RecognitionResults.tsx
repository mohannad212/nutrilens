import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useMemo, useState } from 'react';
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import FoodClassPicker from '@/components/FoodClassPicker';
import PredictionCard from '@/components/PredictionCard';
import PrimaryButton from '@/components/PrimaryButton';
import { Screen } from '@/components/states';
import { useFoodClasses } from '@/hooks/useFoodClasses';
import type { CaptureStackParamList } from '@/navigation/types';
import { theme } from '@/theme';
import type { FoodClass, MealType } from '@/types/api';

type ResultsRoute = RouteProp<CaptureStackParamList, 'RecognitionResults'>;
type Navigation = NativeStackNavigationProp<CaptureStackParamList, 'RecognitionResults'>;

const PORTION_STEP_G = 10;

const MEAL_TYPES: { value: MealType; label: string }[] = [
  { value: 'breakfast', label: 'Breakfast' },
  { value: 'lunch', label: 'Lunch' },
  { value: 'dinner', label: 'Dinner' },
  { value: 'snack', label: 'Snack' },
];

/** Simple time-of-day default — the user can still be wrong, this just saves a tap. */
function inferMealType(): MealType {
  const hour = new Date().getHours();
  if (hour < 11) return 'breakfast';
  if (hour < 16) return 'lunch';
  if (hour < 21) return 'dinner';
  return 'snack';
}

export default function RecognitionResults() {
  const navigation = useNavigation<Navigation>();
  const { params } = useRoute<ResultsRoute>();
  const { imageUri, result } = params;

  const top = result.items[0];
  const [selectedClass, setSelectedClass] = useState(top.food101_class);
  const [pickerOpen, setPickerOpen] = useState(false);
  /**
   * Set when the user overrules the model from the full 101 list. Held separately
   * because such a choice has no confidence to report — see `correctedConfidence`.
   */
  const [override, setOverride] = useState<FoodClass | null>(null);

  const foodClasses = useFoodClasses();

  const selected = useMemo(() => {
    if (override) {
      return { ...top, food101_class: override.food101_class, display_name: override.display_name };
    }
    if (!result.low_confidence) return top;
    const alt = result.alternatives.find((a) => a.food101_class === selectedClass);
    return alt
      ? { ...top, food101_class: alt.food101_class, display_name: alt.display_name, confidence: alt.confidence }
      : top;
  }, [override, result, selectedClass, top]);

  /**
   * A human correction is not a model score. Sending the classifier's confidence
   * for a food the user chose would record a number the model never asserted, and
   * `meal_item.confidence_score` is meant to capture how sure the *model* was.
   */
  const correctedConfidence = override ? null : selected.confidence;

  const handleOverride = (choice: FoodClass) => {
    setOverride(choice);
    setSelectedClass(choice.food101_class);
    setPickerOpen(false);
  };

  /**
   * `inferMealType()` is a default, not a verdict: it reads the clock, so a late
   * dinner is filed as a "snack" with no way to say otherwise. The meal type is
   * the axis M7's insights group by, so a wrong one quietly skews the analysis.
   */
  const [mealType, setMealType] = useState<MealType>(inferMealType());

  const [portionG, setPortionG] = useState(top.portion?.point_g ?? 100);
  const minG = top.portion?.min_g ?? 50;
  const maxG = top.portion?.max_g ?? 500;

  const adjustPortion = (delta: number) => {
    setPortionG((current) => Math.min(maxG, Math.max(minG, current + delta)));
  };

  const handleConfirm = () => {
    navigation.navigate('NutritionSummary', {
      food101_class: selected.food101_class,
      display_name: selected.display_name,
      portion_confirmed_g: portionG,
      portion_min_g: top.portion?.min_g ?? null,
      portion_max_g: top.portion?.max_g ?? null,
      confidence_score: correctedConfidence,
      meal_type: mealType,
    });
  };

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.content}>
        <Image source={{ uri: imageUri }} style={styles.photo} resizeMode="cover" />

        {override ? (
          <View style={styles.section}>
            <Text style={styles.title}>You chose</Text>
            <PredictionCard displayName={override.display_name} selected />
          </View>
        ) : result.low_confidence ? (
          <View style={styles.section}>
            <Text style={styles.title}>Not quite sure — pick the closest match</Text>
            {result.alternatives.map((alt) => (
              <PredictionCard
                key={alt.food101_class}
                displayName={alt.display_name}
                confidence={alt.confidence}
                selected={alt.food101_class === selectedClass}
                onPress={() => setSelectedClass(alt.food101_class)}
              />
            ))}
          </View>
        ) : (
          <View style={styles.section}>
            <Text style={styles.title}>We think this is</Text>
            <PredictionCard displayName={top.display_name} confidence={top.confidence} selected />
          </View>
        )}

        {/*
          Always available, including on a high-confidence result — that is the
          case `alternatives` cannot cover, and the one where a silent error does
          the most damage.
        */}
        <Pressable
          onPress={() => setPickerOpen(true)}
          style={styles.correctLink}
          accessibilityRole="button"
        >
          <Text style={styles.correctLinkText}>
            {override ? 'Choose a different food' : 'Not the right food? Choose another'}
          </Text>
        </Pressable>

        <View style={styles.section}>
          <Text style={styles.title}>Estimated portion</Text>
          <View style={styles.portionRow}>
            <Pressable
              onPress={() => adjustPortion(-PORTION_STEP_G)}
              style={styles.portionButton}
              accessibilityRole="button"
              accessibilityLabel="Decrease portion"
            >
              <Text style={styles.portionButtonText}>−</Text>
            </Pressable>
            <Text style={styles.portionValue}>{Math.round(portionG)} g</Text>
            <Pressable
              onPress={() => adjustPortion(PORTION_STEP_G)}
              style={styles.portionButton}
              accessibilityRole="button"
              accessibilityLabel="Increase portion"
            >
              <Text style={styles.portionButtonText}>+</Text>
            </Pressable>
          </View>
        </View>

        <View style={styles.section}>
          <Text style={styles.title}>Meal</Text>
          <View style={styles.mealTypeRow}>
            {MEAL_TYPES.map((option) => {
              const isSelected = option.value === mealType;
              return (
                <Pressable
                  key={option.value}
                  onPress={() => setMealType(option.value)}
                  style={[styles.mealChip, isSelected ? styles.mealChipSelected : null]}
                  accessibilityRole="radio"
                  accessibilityState={{ selected: isSelected }}
                >
                  <Text
                    style={[styles.mealChipText, isSelected ? styles.mealChipTextSelected : null]}
                  >
                    {option.label}
                  </Text>
                </Pressable>
              );
            })}
          </View>
        </View>

        <PrimaryButton label="Confirm" onPress={handleConfirm} />
        <PrimaryButton label="Retake photo" variant="secondary" onPress={() => navigation.replace('Camera')} />
      </ScrollView>

      <FoodClassPicker
        visible={pickerOpen}
        classes={foodClasses.data?.food_classes ?? []}
        selectedClass={selected.food101_class}
        onSelect={handleOverride}
        onClose={() => setPickerOpen(false)}
        isLoading={foodClasses.isPending}
        error={foodClasses.error}
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: {
    paddingHorizontal: theme.space(4),
    paddingVertical: theme.space(4),
    gap: theme.space(3),
  },
  photo: {
    width: '100%',
    aspectRatio: 4 / 3,
    borderRadius: theme.radius.lg,
    backgroundColor: theme.color.surface,
    marginBottom: theme.space(2),
  },
  section: { marginBottom: theme.space(4) },
  // 44pt minimum touch target (07 §9).
  correctLink: { minHeight: 44, justifyContent: 'center', marginBottom: theme.space(4) },
  correctLinkText: { color: theme.color.accent, fontSize: 15, textAlign: 'center' },
  mealTypeRow: { flexDirection: 'row', gap: theme.space(2) },
  mealChip: {
    flex: 1,
    minHeight: 44, // 07 §9 touch target
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: theme.color.border,
    borderRadius: theme.radius.md,
    paddingVertical: theme.space(2),
    alignItems: 'center',
  },
  mealChipSelected: { borderColor: theme.color.accent, backgroundColor: theme.color.surface },
  mealChipText: { color: theme.color.textSecondary, fontSize: 13 },
  mealChipTextSelected: { color: theme.color.textPrimary, fontWeight: '500' },
  title: {
    color: theme.color.textPrimary,
    fontSize: 18,
    fontWeight: '600',
    marginBottom: theme.space(3),
  },
  portionRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: theme.space(6),
  },
  portionButton: {
    width: 44,
    height: 44,
    borderRadius: theme.radius.pill,
    borderWidth: 1,
    borderColor: theme.color.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  portionButtonText: { color: theme.color.textPrimary, fontSize: 22 },
  portionValue: { color: theme.color.textPrimary, fontSize: 20, fontWeight: '500', minWidth: 80, textAlign: 'center' },
});