import { FlatList, StyleSheet, Text, View } from 'react-native';

import { ErrorState, LoadingState, Screen } from '@/components/states';
import DisclaimerBanner from '@/components/DisclaimerBanner';
import { useInsights } from '@/hooks/useInsights';
import { theme } from '@/theme';
import type { Insight } from '@/types/api';

export default function Insights() {
  const insightsQuery = useInsights();

  if (insightsQuery.isPending) {
    return (
      <Screen>
        <LoadingState label="Loading your insights…" />
      </Screen>
    );
  }

  if (insightsQuery.isError) {
    return (
      <Screen>
        <ErrorState error={insightsQuery.error} onRetry={() => void insightsQuery.refetch()} />
      </Screen>
    );
  }

  const data = insightsQuery.data;

  // FR12: locked until MIN_MEALS_FOR_RECOMMENDATION meals are logged.
  if (!data.unlocked) {
    return (
      <Screen>
        <View style={styles.locked}>
          <Text style={styles.lockedTitle}>Insights aren&apos;t ready yet</Text>
          <Text style={styles.lockedBody}>{data.message}</Text>
          <View style={styles.progressTrack}>
            <View
              style={[
                styles.progressFill,
                {
                  width: `${Math.min(
                    100,
                    (data.meals_logged / Math.max(1, data.meals_required)) * 100,
                  )}%`,
                },
              ]}
            />
          </View>
          <Text style={styles.progressLabel}>
            {data.meals_logged} / {data.meals_required} meals logged
          </Text>
        </View>
      </Screen>
    );
  }

  return (
    <Screen>
      <FlatList
        contentContainerStyle={styles.content}
        data={data.insights}
        keyExtractor={(item) => String(item.rec_id)}
        ListHeaderComponent={
          <>
            <Text style={styles.heading}>Your Daily Insights</Text>
            <Text style={styles.subheading}>
              Actionable recommendations based on your recent nutritional logging.
            </Text>
          </>
        }
        renderItem={({ item }: { item: Insight }) => (
          <View style={styles.card}>
            <Text style={styles.focus}>{item.nutrient_focus}</Text>
            <Text style={styles.insightText}>{item.insight_text}</Text>
          </View>
        )}
        ListFooterComponent={<DisclaimerBanner text={data.disclaimer} />}
        ListEmptyComponent={<Text style={styles.empty}>No insights yet.</Text>}
      />
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: { padding: theme.space(5), paddingBottom: theme.space(10) },
  heading: { color: theme.color.textPrimary, fontSize: 22, fontWeight: '500' },
  subheading: {
    color: theme.color.textSecondary,
    fontSize: 14,
    marginTop: theme.space(1),
    marginBottom: theme.space(5),
  },
  card: {
    backgroundColor: theme.color.surface,
    borderRadius: theme.radius.lg,
    borderWidth: 1,
    borderColor: theme.color.border,
    padding: theme.space(4),
    marginBottom: theme.space(3),
  },
  focus: {
    color: theme.color.accent,
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    marginBottom: theme.space(2),
  },
  insightText: { color: theme.color.textPrimary, fontSize: 14, lineHeight: 20 },
  empty: { color: theme.color.textSecondary, fontSize: 14, textAlign: 'center', marginTop: theme.space(6) },
  locked: {
    flex: 1,
    padding: theme.space(6),
    justifyContent: 'center',
    gap: theme.space(3),
  },
  lockedTitle: { color: theme.color.textPrimary, fontSize: 18, fontWeight: '600', textAlign: 'center' },
  lockedBody: { color: theme.color.textSecondary, fontSize: 14, textAlign: 'center', lineHeight: 20 },
  progressTrack: {
    height: 8,
    borderRadius: theme.radius.pill,
    backgroundColor: theme.color.surface,
    overflow: 'hidden',
    marginTop: theme.space(4),
  },
  progressFill: { height: '100%', backgroundColor: theme.color.accent },
  progressLabel: {
    color: theme.color.textSecondary,
    fontSize: 12,
    textAlign: 'center',
    marginTop: theme.space(2),
  },
});