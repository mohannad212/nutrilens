import { useMemo } from 'react';
import { FlatList, StyleSheet, Text, View } from 'react-native';
import type { BottomTabScreenProps } from '@react-navigation/bottom-tabs';

import CalorieRing from '@/components/CalorieRing';
import MacroBar from '@/components/MacroBar';
import PrimaryButton from '@/components/PrimaryButton';
import { ErrorState, LoadingState, Screen } from '@/components/states';
import { useTodayMeals } from '@/hooks/useMeals';
import { useAuth } from '@/store/AuthContext';
import { theme } from '@/theme';
import type { AppTabsParamList } from '@/navigation/AppTabs';

type Props = BottomTabScreenProps<AppTabsParamList, 'Home'>;

const KCAL_PER_G = { protein: 4, carbs: 4, fat: 9 } as const;

export default function Home({ navigation }: Props) {
  const { user } = useAuth();
  const mealsQuery = useTodayMeals();

  const { totalCalories, macros } = useMemo(() => {
    const meals = mealsQuery.data?.meals ?? [];
    let calories = 0;
    let protein = 0;
    let carbs = 0;
    let fat = 0;
    for (const meal of meals) {
      calories += meal.total_calories;
      for (const item of meal.items) {
        protein += item.protein_est_g;
        carbs += item.carbs_est_g;
        fat += item.fat_est_g;
      }
    }
    return { totalCalories: calories, macros: { protein, carbs, fat } };
  }, [mealsQuery.data]);

  const split = useMemo(() => {
    const proteinKcal = macros.protein * KCAL_PER_G.protein;
    const carbsKcal = macros.carbs * KCAL_PER_G.carbs;
    const fatKcal = macros.fat * KCAL_PER_G.fat;
    const total = proteinKcal + carbsKcal + fatKcal;
    if (total <= 0) return { protein: 0, carbs: 0, fat: 0 };
    return { protein: proteinKcal / total, carbs: carbsKcal / total, fat: fatKcal / total };
  }, [macros]);

  if (mealsQuery.isPending) {
    return (
      <Screen>
        <LoadingState label="Loading today's meals…" />
      </Screen>
    );
  }

  if (mealsQuery.isError) {
    return (
      <Screen>
        <ErrorState error={mealsQuery.error} onRetry={() => void mealsQuery.refetch()} />
      </Screen>
    );
  }

  const meals = mealsQuery.data.meals;

  return (
    <Screen>
      <FlatList
        contentContainerStyle={styles.content}
        data={meals}
        keyExtractor={(item) => String(item.log_id)}
        ListHeaderComponent={
          <>
            <Text style={styles.heading}>{user ? `Hi, ${user.username}` : 'Today'}</Text>

            <View style={styles.ring}>
              <CalorieRing consumed={totalCalories} caption="today" />
            </View>

            <View style={styles.card}>
              <MacroBar
                label="Protein"
                grams={macros.protein}
                pct={split.protein}
                color={theme.color.protein}
              />
              <MacroBar
                label="Carbs"
                grams={macros.carbs}
                pct={split.carbs}
                color={theme.color.carbs}
              />
              <MacroBar label="Fat" grams={macros.fat} pct={split.fat} color={theme.color.fat} />
            </View>

            <Text style={styles.sectionTitle}>Recent meals</Text>
          </>
        }
        renderItem={({ item }) => (
          <View style={styles.mealRow}>
            <Text style={styles.mealType}>{item.meal_type}</Text>
            <Text style={styles.mealCalories}>{Math.round(item.total_calories)} kcal</Text>
          </View>
        )}
        ListEmptyComponent={<Text style={styles.empty}>No meals logged yet today.</Text>}
      />

      <View style={styles.fab}>
        <PrimaryButton label="Log a meal" onPress={() => navigation.navigate('Capture')} />
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: { padding: theme.space(5), paddingBottom: theme.space(20) },
  heading: { color: theme.color.textPrimary, fontSize: 22, fontWeight: '500' },
  ring: { alignItems: 'center', marginVertical: theme.space(6) },
  card: {
    backgroundColor: theme.color.surface,
    borderRadius: theme.radius.lg,
    borderWidth: 1,
    borderColor: theme.color.border,
    padding: theme.space(4),
    paddingBottom: theme.space(2),
    marginBottom: theme.space(5),
  },
  sectionTitle: {
    color: theme.color.textPrimary,
    fontSize: 16,
    fontWeight: '500',
    marginBottom: theme.space(2),
  },
  mealRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: theme.space(3),
    borderBottomWidth: 1,
    borderBottomColor: theme.color.border,
  },
  mealType: { color: theme.color.textPrimary, fontSize: 14, textTransform: 'capitalize' },
  mealCalories: { color: theme.color.textSecondary, fontSize: 14 },
  empty: { color: theme.color.textSecondary, fontSize: 14, marginTop: theme.space(3) },
  fab: { position: 'absolute', left: theme.space(5), right: theme.space(5), bottom: theme.space(5) },
});