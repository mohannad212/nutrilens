/**
 * src/components/states.tsx
 * =========================
 *
 * `07 §7`: "Every screen renders one of: `LoadingState`, `ErrorState(retry)`,
 * `EmptyState`, or content." These are those three, plus `Screen` — the
 * safe-area wrapper from `07 §3` that every screen sits inside.
 *
 * WHY THE ERROR COPY IS MAPPED FROM `error.code`
 * ----------------------------------------------
 * `07 §7` asks for it, and the reason is that the server's message is written for
 * an API consumer, not for someone standing in a restaurant. "The nutrition
 * database is unavailable and this food is not cached" is accurate and unhelpful;
 * "We couldn't fetch nutrition right now — try again in a moment" tells the user
 * what to do. The mapping is keyed on the machine-readable `code` from `03 §9`
 * precisely so the wording on both sides can change independently.
 *
 * Codes with no entry fall back to the server's own message, which is better than
 * a generic "something went wrong" — the server usually knows something specific,
 * and a validation error in particular names the field.
 *
 * WHY `ErrorState` ALWAYS TAKES A RETRY
 * -------------------------------------
 * Most of these failures are transient — an outage, a rate limit, a dropped
 * connection. A dead-end error screen makes the user re-navigate the whole
 * capture flow to try again, which after photographing a meal is enough friction
 * to lose them.
 */

import { ActivityIndicator, StyleSheet, Text, View, type ViewStyle } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import PrimaryButton from './PrimaryButton';
import { theme } from '@/theme';
import type { ApiErrorCode } from '@/types/api';

/** `07 §7` — friendly copy per `03 §9` code. Unmapped codes use the server's message. */
const ERROR_COPY: Partial<Record<ApiErrorCode, string>> = {
  USDA_UNAVAILABLE:
    "We couldn't fetch nutrition right now — showing cached values where possible. Try again in a moment.",
  USDA_RATE_LIMITED:
    "We've looked up a lot of foods recently. Give it a minute and try again.",
  NO_NUTRITION_MATCH:
    "We don't have nutrition data for this food. Try a different cooking method, or pick another match.",
  NETWORK_ERROR: "You're offline. Check your connection and try again.",
  TIMEOUT: 'That took too long. Try again.',
  INTERNAL_ERROR: 'Something went wrong on our end. Please try again.',
  IMAGE_TOO_LARGE: 'That photo is too large. Take another and we’ll shrink it for you.',
  IMAGE_POOR_LIGHTING: 'That photo is too dark or too bright to read. Retake it with even lighting.',
  IMAGE_TOO_BLURRY: 'That photo is a little blurry. Hold steady and retake it.',
};

export function messageForError(error: unknown): string {
  const code = (error as { code?: ApiErrorCode })?.code;
  if (code && ERROR_COPY[code]) return ERROR_COPY[code] as string;

  const message = (error as { message?: string })?.message;
  return message && message.length > 0 ? message : 'Something went wrong. Please try again.';
}

export function Screen({ children, style }: { children: React.ReactNode; style?: ViewStyle }) {
  return <SafeAreaView style={[styles.screen, style]}>{children}</SafeAreaView>;
}

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return (
    <View style={styles.centred} accessibilityRole="progressbar" accessibilityLabel={label}>
      <ActivityIndicator size="large" color={theme.color.accent} />
      <Text style={styles.body}>{label}</Text>
    </View>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry: () => void }) {
  return (
    <View style={styles.centred}>
      <Text style={styles.title}>We hit a snag</Text>
      <Text style={styles.body}>{messageForError(error)}</Text>
      <View style={styles.action}>
        <PrimaryButton label="Try again" onPress={onRetry} />
      </View>
    </View>
  );
}

export function EmptyState({ title, body }: { title: string; body?: string }) {
  return (
    <View style={styles.centred}>
      <Text style={styles.title}>{title}</Text>
      {body ? <Text style={styles.body}>{body}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: theme.color.bg,
  },
  centred: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: theme.space(6),
    gap: theme.space(3),
  },
  title: {
    color: theme.color.textPrimary,
    fontSize: 18,
    fontWeight: '500',
    textAlign: 'center',
  },
  body: {
    color: theme.color.textSecondary,
    fontSize: 14,
    lineHeight: 20,
    textAlign: 'center',
  },
  action: {
    marginTop: theme.space(2),
    alignSelf: 'stretch',
  },
});
