/**
 * src/components/DisclaimerBanner.tsx
 * ===================================
 *
 * `07 §3` marks this component **required on Nutrition (3) and Insights (4)**,
 * and `03 §5`/`§7` put the text on every response that carries estimates or
 * suggestions.
 *
 * WHY THE TEXT IS A PROP AND NOT A CONSTANT IN THIS FILE
 * -----------------------------------------------------
 * The obvious implementation hard-codes the sentence here. It is passed in
 * instead, straight from the API response, so the disclaimer the user reads is
 * the one the server actually sent. A copy in the app is a second source of
 * truth that can quietly fall out of step with `core/constants.py` after a
 * wording change — and a legal statement that differs between the API contract
 * and the screen is worse than either version alone.
 *
 * It also means the component cannot render a disclaimer for a response that did
 * not include one: no text, nothing rendered, and the omission is visible in the
 * UI rather than papered over with a hard-coded default.
 */

import { StyleSheet, Text, View } from 'react-native';

import { theme } from '@/theme';

export default function DisclaimerBanner({ text }: { text: string | undefined }) {
  if (!text) return null;

  return (
    <View style={styles.container} accessibilityRole="alert">
      <Text style={styles.text}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: theme.color.surfaceAlt,
    borderRadius: theme.radius.md,
    borderWidth: 1,
    borderColor: theme.color.border,
    padding: theme.space(3),
    marginTop: theme.space(4),
  },
  text: {
    color: theme.color.textSecondary,
    fontSize: 12,
    lineHeight: 18,
  },
});
