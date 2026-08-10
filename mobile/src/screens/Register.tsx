import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import PrimaryButton from '@/components/PrimaryButton';
import TextField from '@/components/TextField';
import { Screen, messageForError } from '@/components/states';
import { useLogin } from '@/hooks/useLogin';
import { useRegister } from '@/hooks/useRegister';
import { theme } from '@/theme';
import type { HealthGoal, RegisterRequest } from '@/types/auth';
import type { AuthStackParamList } from '@/navigation/AuthStack';

type Props = NativeStackScreenProps<AuthStackParamList, 'Register'>;

const GOALS: { value: HealthGoal; label: string }[] = [
  { value: 'weight_loss', label: 'Weight loss' },
  { value: 'maintenance', label: 'Maintenance' },
  { value: 'weight_gain', label: 'Weight gain' },
];

export default function Register({ navigation }: Props) {
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [healthGoal, setHealthGoal] = useState<HealthGoal>('maintenance');
  const [fieldErrors, setFieldErrors] = useState<{
    email?: string;
    username?: string;
    password?: string;
  }>({});

  const register = useRegister();
  const login = useLogin();

  function validate(): boolean {
    const errors: typeof fieldErrors = {};
    if (!email.trim()) errors.email = 'Email is required.';
    if (!username.trim()) errors.username = 'Username is required.';
    // Mirrors the backend's own rule (schemas/auth.py password_has_letter_and_digit)
    // so an invalid password is caught here rather than coming back as a 422.
    if (password.length < 8) errors.password = 'Password must be at least 8 characters.';
    else if (!/[a-zA-Z]/.test(password) || !/[0-9]/.test(password))
      errors.password = 'Password must contain at least one letter and one digit.';
    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  }

  function handleSubmit() {
    if (!validate()) return;
    const body: RegisterRequest = {
      email: email.trim(),
      username: username.trim(),
      password,
      health_goal: healthGoal,
    };
    register.mutate(body, {
      // 03 §2: register doesn't return a token, so log in right after with the
      // same credentials. On success, AuthContext flips isAuthenticated and
      // RootNavigator takes over — no manual navigation needed here either.
      onSuccess: () => {
        login.mutate({ email: body.email, password: body.password });
      },
    });
  }

  const isSubmitting = register.isPending || login.isPending;
  const submitError = register.error ?? login.error;

  return (
    <Screen>
      <View style={styles.content}>
        <Text style={styles.heading}>Create your account</Text>
        <Text style={styles.subheading}>Track meals and see where your nutrition stands.</Text>

        <TextField
          label="Email"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
          error={fieldErrors.email}
        />
        <TextField
          label="Username"
          value={username}
          onChangeText={setUsername}
          autoCapitalize="none"
          error={fieldErrors.username}
        />
        <TextField
          label="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          error={fieldErrors.password}
        />

        <Text style={styles.label}>Goal</Text>
        <View style={styles.goalRow}>
          {GOALS.map((goal) => {
            const selected = goal.value === healthGoal;
            return (
              <Pressable
                key={goal.value}
                onPress={() => setHealthGoal(goal.value)}
                style={[styles.goalChip, selected ? styles.goalChipSelected : null]}
                accessibilityRole="radio"
                accessibilityState={{ selected }}
              >
                <Text style={[styles.goalText, selected ? styles.goalTextSelected : null]}>
                  {goal.label}
                </Text>
              </Pressable>
            );
          })}
        </View>

        {submitError ? <Text style={styles.error}>{messageForError(submitError)}</Text> : null}

        <View style={styles.actions}>
          <PrimaryButton label="Create account" onPress={handleSubmit} loading={isSubmitting} />
          <PrimaryButton
            label="I already have an account"
            variant="secondary"
            onPress={() => navigation.navigate('Login')}
          />
        </View>
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: { flexGrow: 1, justifyContent: 'center', padding: theme.space(5) },
  heading: { color: theme.color.textPrimary, fontSize: 24, fontWeight: '500' },
  subheading: {
    color: theme.color.textSecondary,
    fontSize: 14,
    marginTop: theme.space(1),
    marginBottom: theme.space(6),
  },
  label: { color: theme.color.textSecondary, fontSize: 13, marginBottom: theme.space(2) },
  goalRow: { flexDirection: 'row', gap: theme.space(2), marginBottom: theme.space(5) },
  goalChip: {
    flex: 1,
    borderWidth: 1,
    borderColor: theme.color.border,
    borderRadius: theme.radius.md,
    paddingVertical: theme.space(3),
    alignItems: 'center',
  },
  goalChipSelected: { borderColor: theme.color.accent, backgroundColor: theme.color.surface },
  goalText: { color: theme.color.textSecondary, fontSize: 13 },
  goalTextSelected: { color: theme.color.textPrimary, fontWeight: '500' },
  error: { color: theme.color.danger, fontSize: 13, marginTop: theme.space(2) },
  actions: { marginTop: theme.space(6), gap: theme.space(3) },
});