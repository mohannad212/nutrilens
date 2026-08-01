/**
 * src/components/CookingMethodPicker.tsx
 * ======================================
 *
 * `07 §3` — "dropdown bound to `GET /meta/cooking-methods`". FR07's cooking
 * method is appended to the USDA query, so this control is what makes the
 * difference between grilled and fried chicken's nutrition.
 *
 * WHY THE OPTIONS COME FROM THE API
 * ---------------------------------
 * They could be read from the `CookingMethod` union in `src/types/api.ts`, which
 * is already correct and costs no request. Fetching them means the picker and the
 * backend validator are the *same list at runtime*, not two lists that happen to
 * agree at compile time — and the failure this prevents is specific: add a ninth
 * method to the backend enum, and a hard-coded picker keeps offering eight
 * forever with nothing to flag it. `useCookingMethods` caches the response for the
 * session, so the cost is one request per launch.
 *
 * The typed union still earns its place: it is what makes the *selected* value
 * type-safe everywhere it flows onward.
 *
 * WHY A ROW OF CHIPS AND NOT A NATIVE DROPDOWN
 * --------------------------------------------
 * `07 §2` calls it a dropdown, and a `Picker` would be the literal reading. Chips
 * are used instead because eight short options fit on screen, and this control
 * sits inside the three-interaction budget the core loop is measured against
 * (`00`, and M5's DoD). A dropdown costs an open, a scroll and a select — three
 * interactions for one field. A chip row costs one tap, and the current selection
 * is readable without opening anything.
 */

import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import { theme } from '@/theme';
import type { CookingMethod } from '@/types/api';

const LABELS: Record<CookingMethod, string> = {
  grilled: 'Grilled',
  fried: 'Fried',
  boiled: 'Boiled',
  baked: 'Baked',
  roasted: 'Roasted',
  steamed: 'Steamed',
  raw: 'Raw',
  other: 'Other',
};

interface CookingMethodPickerProps {
  value: CookingMethod;
  options: CookingMethod[];
  onChange: (method: CookingMethod) => void;
  label?: string;
}

export default function CookingMethodPicker({
  value,
  options,
  onChange,
  label = 'How was it cooked?',
}: CookingMethodPickerProps) {
  return (
    <View style={styles.container}>
      <Text style={styles.label}>{label}</Text>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.row}
      >
        {options.map((method) => {
          const selected = method === value;
          return (
            <Pressable
              key={method}
              onPress={() => onChange(method)}
              style={[styles.chip, selected && styles.chipSelected]}
              accessibilityRole="radio"
              accessibilityState={{ selected }}
              accessibilityLabel={LABELS[method] ?? method}
            >
              <Text style={[styles.chipText, selected && styles.chipTextSelected]}>
                {LABELS[method] ?? method}
              </Text>
            </Pressable>
          );
        })}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginBottom: theme.space(4),
  },
  label: {
    color: theme.color.textPrimary,
    fontSize: 14,
    marginBottom: theme.space(2),
  },
  row: {
    gap: theme.space(2),
    paddingRight: theme.space(4),
  },
  chip: {
    // 44 px minimum touch target (07 §9).
    minHeight: 44,
    justifyContent: 'center',
    paddingHorizontal: theme.space(4),
    borderRadius: theme.radius.pill,
    borderWidth: 1,
    borderColor: theme.color.border,
    backgroundColor: theme.color.surface,
  },
  chipSelected: {
    backgroundColor: theme.color.accent,
    borderColor: theme.color.accent,
  },
  chipText: {
    color: theme.color.textSecondary,
    fontSize: 14,
  },
  chipTextSelected: {
    color: theme.color.bg,
    fontWeight: '500',
  },
});
