import { Pressable, StyleSheet, Text, View } from 'react-native';

import { theme } from '@/theme';
import ConfidenceChip from './ConfidenceChip';

interface PredictionCardProps {
  displayName: string;
  confidence: number;
  selected?: boolean;
  onPress?: () => void;
}

// 07-mobile-app.md §3: one recognised item - name, confidence chip.
// Portion range joins this card in M4 once portion estimation ships.
export default function PredictionCard({ displayName, confidence, selected, onPress }: PredictionCardProps) {
  const content = (
    <View style={[styles.card, selected && styles.cardSelected]}>
      <Text style={styles.name}>{displayName}</Text>
      <ConfidenceChip confidence={confidence} />
    </View>
  );

  if (!onPress) return content;

  return (
    <Pressable
      onPress={onPress}
      accessibilityRole="button"
      accessibilityState={{ selected: !!selected }}
    >
      {content}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: theme.color.surface,
    borderRadius: theme.radius.md,
    borderWidth: 1,
    borderColor: theme.color.border,
    paddingHorizontal: theme.space(4),
    paddingVertical: theme.space(3),
    marginBottom: theme.space(2),
  },
  cardSelected: {
    borderColor: theme.color.accent,
  },
  name: {
    color: theme.color.textPrimary,
    fontSize: 16,
    fontWeight: '500',
  },
});
