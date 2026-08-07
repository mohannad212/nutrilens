import { StyleSheet, Text, View } from 'react-native';

import { theme } from '@/theme';

interface ConfidenceChipProps {
  confidence: number; // 0..1
}

// 07-mobile-app.md §3: colour-coded (green >=0.75, amber 0.6-0.75, red <0.6).
// Colour is never the only signal (§9 accessibility) - the percentage text carries the same info.
export default function ConfidenceChip({ confidence }: ConfidenceChipProps) {
  const pct = Math.round(confidence * 100);
  const level = confidence >= 0.75 ? 'high' : confidence >= 0.6 ? 'medium' : 'low';
  const color =
    level === 'high' ? theme.color.success : level === 'medium' ? theme.color.warning : theme.color.danger;

  return (
    <View style={[styles.chip, { backgroundColor: `${color}26`, borderColor: color }]}>
      <Text style={[styles.text, { color }]}>{pct}%</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  chip: {
    borderWidth: 1,
    borderRadius: theme.radius.pill,
    paddingHorizontal: theme.space(2),
    paddingVertical: theme.space(1),
    alignSelf: 'flex-start',
  },
  text: {
    fontSize: 12,
    fontWeight: '600',
  },
});
