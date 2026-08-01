/**
 * src/components/MacroBar.tsx
 * ===========================
 *
 * `07 §3` — "labelled progress bar per macro, props `{label, grams, pct, color}`".
 * Three of these under the calorie ring make up the macro breakdown on the
 * Nutrition Summary screen.
 *
 * WHAT `pct` MEANS, AND WHY IT IS PASSED IN
 * -----------------------------------------
 * It is the macro's share of the meal's energy, not its share by mass. Those are
 * very different numbers — fat carries 9 kcal/g against 4 for protein and carbs,
 * so a meal that is 20% fat by weight can be 40% fat by calories — and the
 * energy split is the one that means something nutritionally, and the one `06`'s
 * insights are computed on.
 *
 * The bar does not compute it. The caller does, because only the caller can see
 * all three macros at once, and a component that derived its own percentage from
 * a single value would have to guess at the total.
 *
 * WHY COLOUR IS NEVER THE ONLY SIGNAL
 * -----------------------------------
 * `07 §9` requires it, and this is the component where it would be easiest to get
 * wrong: three bars distinguished only by the palette in `07 §8` are three
 * identical bars to a user with colour-vision deficiency. Each one is labelled in
 * text and carries an `accessibilityLabel` that states the macro, the grams and
 * the share, so the colour is reinforcement rather than information.
 */

import { StyleSheet, Text, View } from 'react-native';

import { theme } from '@/theme';

interface MacroBarProps {
  label: string;
  grams: number;
  /** Share of the meal's *energy*, 0-1. See the header note. */
  pct: number;
  color: string;
}

export default function MacroBar({ label, grams, pct, color }: MacroBarProps) {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(pct) ? pct : 0));
  const percentLabel = `${Math.round(clamped * 100)}%`;

  return (
    <View
      style={styles.container}
      accessibilityRole="progressbar"
      accessibilityLabel={`${label}: ${grams} grams, ${percentLabel} of calories`}
    >
      <View style={styles.row}>
        <Text style={styles.label}>{label}</Text>
        <Text style={styles.value}>
          {grams} g · {percentLabel}
        </Text>
      </View>
      <View style={styles.track}>
        <View style={[styles.fill, { width: `${clamped * 100}%`, backgroundColor: color }]} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: theme.space(3),
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
    marginBottom: theme.space(1.5),
  },
  label: {
    color: theme.color.textPrimary,
    fontSize: 14,
  },
  value: {
    color: theme.color.textSecondary,
    fontSize: 13,
  },
  track: {
    height: 8,
    borderRadius: theme.radius.pill,
    backgroundColor: theme.color.surfaceAlt,
    overflow: 'hidden',
  },
  fill: {
    height: '100%',
    borderRadius: theme.radius.pill,
  },
});
