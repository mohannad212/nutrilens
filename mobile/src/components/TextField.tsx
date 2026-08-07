import { forwardRef } from 'react';
import { StyleSheet, Text, TextInput, View, type TextInputProps } from 'react-native';

import { theme } from '@/theme';

interface TextFieldProps extends TextInputProps {
  label: string;
  error?: string;
}

// Shared form input (07-mobile-app.md §3). Forwards ref so react-hook-form's
// register-by-ref pattern (via Controller) works without extra glue.
const TextField = forwardRef<TextInput, TextFieldProps>(({ label, error, style, ...rest }, ref) => {
  return (
    <View style={styles.wrapper}>
      <Text style={styles.label}>{label}</Text>
      <TextInput
        ref={ref}
        style={[styles.input, error ? styles.inputError : null, style]}
        placeholderTextColor={theme.color.textSecondary}
        accessibilityLabel={label}
        {...rest}
      />
      {error ? <Text style={styles.error}>{error}</Text> : null}
    </View>
  );
});
TextField.displayName = 'TextField';

export default TextField;

const styles = StyleSheet.create({
  wrapper: {
    marginBottom: theme.space(4),
  },
  label: {
    color: theme.color.textSecondary,
    fontSize: 13,
    marginBottom: theme.space(1),
  },
  input: {
    backgroundColor: theme.color.surface,
    borderColor: theme.color.border,
    borderWidth: 1,
    borderRadius: theme.radius.md,
    paddingHorizontal: theme.space(3),
    paddingVertical: theme.space(3),
    color: theme.color.textPrimary,
    fontSize: 16,
  },
  inputError: {
    borderColor: theme.color.danger,
  },
  error: {
    color: theme.color.danger,
    fontSize: 12,
    marginTop: theme.space(1),
  },
});
