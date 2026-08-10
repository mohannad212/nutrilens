import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useMemo, useState } from 'react';
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import PredictionCard from '@/components/PredictionCard';
import PrimaryButton from '@/components/PrimaryButton';
import { Screen } from '@/components/states';
import type { CaptureStackParamList } from '@/navigation/types';
import { theme } from '@/theme';
import type { MealType } from '@/types/api';

type ResultsRoute = RouteProp<CaptureStackParamList, 'RecognitionResults'>;
type Navigation = NativeStackNavigationProp<CaptureStackParamList, 'RecognitionResults'>;

const PORTION_STEP_G = 10;

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

  const selected = useMemo(() => {
    if (!result.low_confidence) return top;
    const alt = result.alternatives.find((a) => a.food101_class === selectedClass);
    return alt
      ? { ...top, food101_class: alt.food101_class, display_name: alt.display_name, confidence: alt.confidence }
      : top;
  }, [result, selectedClass, top]);

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
      confidence_score: selected.confidence,
      meal_type: inferMealType(),
    });
  };

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.content}>
        <Image source={{ uri: imageUri }} style={styles.photo} resizeMode="cover" />

        {result.low_confidence ? (
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

        <PrimaryButton label="Confirm" onPress={handleConfirm} />
        <PrimaryButton label="Retake photo" variant="secondary" onPress={() => navigation.replace('Camera')} />
      </ScrollView>
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