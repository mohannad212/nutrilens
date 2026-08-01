/**
 * src/components/PrimaryButton.tsx
 * ================================
 *
 * `07 §3`'s `PrimaryButton`. The one call-to-action style in the app, so "Log
 * This Meal" and "Sign in" cannot drift apart visually.
 *
 * The `loading` state disables the button as well as changing its label, because
 * the alternative — a button that still responds while a mutation is in flight —
 * is how a user logs the same meal twice. `03 §6` has no idempotency key (two
 * identical meals are a legitimate thing to eat), so preventing the double tap is
 * the client's job.
 */

import { ActivityIndicator, Pressable, StyleSheet, Text, View } from 'react-native';

import { theme } from '@/theme';

interface PrimaryButtonProps {
  label: string;
  onPress: () => void;
  loading?: boolean;
  disabled?: boolean;
  variant?: 'primary' | 'secondary';
}

export default function PrimaryButton({
  label,
  onPress,
  loading = false,
  disabled = false,
  variant = 'primary',
}: PrimaryButtonProps) {
  const isInactive = disabled || loading;
  const isSecondary = variant === 'secondary';

  return (
    <Pressable
      onPress={onPress}
      disabled={isInactive}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ disabled: isInactive, busy: loading }}
      style={({ pressed }) => [
        styles.button,
        isSecondary && styles.secondary,
        isInactive && styles.inactive,
        pressed && !isInactive && styles.pressed,
      ]}
    >
      <View style={styles.content}>
        {loading ? <ActivityIndicator size="small" color={theme.color.bg} /> : null}
        <Text style={[styles.label, isSecondary && styles.secondaryLabel]}>{label}</Text>
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    minHeight: 52, // comfortably past the 44 px minimum (07 §9)
    borderRadius: theme.radius.md,
    backgroundColor: theme.color.accent,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: theme.space(5),
  },
  secondary: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: theme.color.border,
  },
  inactive: {
    opacity: 0.5,
  },
  pressed: {
    opacity: 0.85,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.space(2),
  },
  label: {
    color: theme.color.bg,
    fontSize: 16,
    fontWeight: '500',
  },
  secondaryLabel: {
    color: theme.color.textPrimary,
  },
});
