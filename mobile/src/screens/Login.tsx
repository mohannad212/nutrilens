import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';

import PrimaryButton from '@/components/PrimaryButton';
import TextField from '@/components/TextField';
import { Screen, messageForError } from '@/components/states';
import { useLogin } from '@/hooks/useLogin';
import { theme } from '@/theme';
import type { LoginRequest } from '@/types/auth';
import type { AuthStackParamList } from '@/navigation/AuthStack';

type Props = NativeStackScreenProps<AuthStackParamList, 'Login'>;

export default function Login({ navigation }: Props) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fieldErrors, setFieldErrors] = useState<{ email?: string; password?: string }>({});

  const login = useLogin();

  function validate(): boolean {
    const errors: typeof fieldErrors = {};
    // `03 §2` authenticates on email, not username, and typing the username is
    // the natural mistake — say so here rather than round-tripping to the
    // server's "an email address must have an @-sign", which describes the
    // parser's complaint instead of the user's actual error.
    if (!email.trim()) errors.email = 'Email is required.';
    else if (!email.includes('@')) errors.email = 'Use your email address, not your username.';
    if (!password) errors.password = 'Password is required.';
    setFieldErrors(errors);
    return Object.keys(errors).length === 0;
  }

  function handleSubmit() {
    if (!validate()) return;
    const body: LoginRequest = { email: email.trim(), password };
    // No manual navigation on success: AuthContext.login() flips
    // isAuthenticated, and RootNavigator swaps to the main app on its own.
    login.mutate(body);
  }

  return (
    <Screen>
      <View style={styles.content}>
        <Text style={styles.heading}>Welcome back</Text>
        <Text style={styles.subheading}>Log in to keep tracking your meals.</Text>

        <TextField
          label="Email"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
          error={fieldErrors.email}
        />
        <TextField
          label="Password"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          error={fieldErrors.password}
        />

        {login.isError ? <Text style={styles.error}>{messageForError(login.error)}</Text> : null}

        <View style={styles.actions}>
          <PrimaryButton label="Log In" onPress={handleSubmit} loading={login.isPending} />
          <PrimaryButton
            label="Create an account"
            variant="secondary"
            onPress={() => navigation.navigate('Register')}
          />
        </View>
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: { flex: 1, justifyContent: 'center', padding: theme.space(5) },
  heading: { color: theme.color.textPrimary, fontSize: 24, fontWeight: '500' },
  subheading: {
    color: theme.color.textSecondary,
    fontSize: 14,
    marginTop: theme.space(1),
    marginBottom: theme.space(6),
  },
  error: { color: theme.color.danger, fontSize: 13, marginTop: theme.space(2) },
  actions: { marginTop: theme.space(6), gap: theme.space(3) },
});