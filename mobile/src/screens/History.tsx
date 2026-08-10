import { useState } from 'react';
import { Alert, FlatList, Pressable, StyleSheet, Text, View } from 'react-native';

import { ErrorState, LoadingState, Screen } from '@/components/states';
import { useDeleteMeal, useMealHistory } from '@/hooks/useMealHistory';
import { theme } from '@/theme';
import type { MealSummary } from '@/types/api';

const PAGE_SIZE = 20;

function formatDate(iso: string): string {
  const date = new Date(iso);
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export default function History() {
  const [offset, setOffset] = useState(0);
  const mealsQuery = useMealHistory(offset);
  const deleteMeal = useDeleteMeal();

  const handleDelete = (meal: MealSummary) => {
    Alert.alert('Delete meal?', `Remove this ${meal.meal_type} log (${Math.round(meal.total_calories)} kcal)?`, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => deleteMeal.mutate(meal.log_id),
      },
    ]);
  };

  if (mealsQuery.isPending) {
    return (
      <Screen>
        <LoadingState label="Loading your meal history…" />
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

  const { meals, total } = mealsQuery.data;
  const canLoadMore = offset + meals.length < total;
  const canGoBack = offset > 0;

  return (
    <Screen>
      <FlatList
        contentContainerStyle={styles.content}
        data={meals}
        keyExtractor={(item) => String(item.log_id)}
        ListHeaderComponent={<Text style={styles.heading}>Meal history</Text>}
        renderItem={({ item }) => (
          <View style={styles.row}>
            <View style={styles.rowMain}>
              {/*
                What was eaten leads; the meal type is context. Showing only
                `meal_type` made every row read "Snack" or "Dinner", which is the
                one thing the user already knows and never what they came to look
                up. `items` is already in the list response — no extra request.
              */}
              <Text style={styles.mealFoods} numberOfLines={1}>
                {item.items.length > 0
                  ? item.items.map((i) => i.display_name).join(', ')
                  : item.meal_type}
              </Text>
              <Text style={styles.mealDate}>
                {item.meal_type} · {formatDate(item.logged_at)}
              </Text>
            </View>
            <View style={styles.rowEnd}>
              <Text style={styles.mealCalories}>{Math.round(item.total_calories)} kcal</Text>
              <Pressable
                onPress={() => handleDelete(item)}
                accessibilityRole="button"
                accessibilityLabel={`Delete ${item.meal_type} log`}
                style={styles.deleteButton}
              >
                <Text style={styles.deleteText}>Delete</Text>
              </Pressable>
            </View>
          </View>
        )}
        ListEmptyComponent={<Text style={styles.empty}>No meals logged yet.</Text>}
        ListFooterComponent={
          // Only when paging is actually possible. Two permanently greyed-out
          // buttons under a single meal read as broken controls, not as an
          // accurate statement that there is nothing to page to.
          canGoBack || canLoadMore ? (
            <View style={styles.pager}>
              <Pressable
                disabled={!canGoBack}
                onPress={() => setOffset((o) => Math.max(0, o - PAGE_SIZE))}
                style={[styles.pagerButton, !canGoBack && styles.pagerButtonDisabled]}
              >
                <Text style={styles.pagerText}>Newer</Text>
              </Pressable>
              <Pressable
                disabled={!canLoadMore}
                onPress={() => setOffset((o) => o + PAGE_SIZE)}
                style={[styles.pagerButton, !canLoadMore && styles.pagerButtonDisabled]}
              >
                <Text style={styles.pagerText}>Older</Text>
              </Pressable>
            </View>
          ) : null
        }
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: { padding: theme.space(5), paddingBottom: theme.space(10) },
  heading: {
    color: theme.color.textPrimary,
    fontSize: 22,
    fontWeight: '500',
    marginBottom: theme.space(4),
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: theme.space(3),
    borderBottomWidth: 1,
    borderBottomColor: theme.color.border,
  },
  // USDA descriptions run long ("Potatoes, frozen, french fried, par fried,
  // extruded, unprepared"). Without `flex: 1` the text sizes to its content and
  // pushes the calories and Delete button off the right edge.
  rowMain: { flex: 1, marginRight: theme.space(3) },
  mealFoods: { color: theme.color.textPrimary, fontSize: 15 },
  // Capitalised because the meal type now leads this line ("Snack · 10 Aug").
  mealDate: {
    color: theme.color.textSecondary,
    fontSize: 12,
    marginTop: 2,
    textTransform: 'capitalize',
  },
  rowEnd: { alignItems: 'flex-end', gap: theme.space(1), flexShrink: 0 },
  mealCalories: { color: theme.color.textSecondary, fontSize: 14 },
  deleteButton: { paddingVertical: 2 },
  deleteText: { color: theme.color.danger, fontSize: 12 },
  empty: { color: theme.color.textSecondary, fontSize: 14, marginTop: theme.space(6), textAlign: 'center' },
  pager: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: theme.space(5),
  },
  pagerButton: {
    borderWidth: 1,
    borderColor: theme.color.border,
    borderRadius: theme.radius.md,
    paddingHorizontal: theme.space(4),
    paddingVertical: theme.space(2),
  },
  pagerButtonDisabled: { opacity: 0.4 },
  pagerText: { color: theme.color.textPrimary, fontSize: 13 },
});