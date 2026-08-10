import { useEffect, useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';

import PrimaryButton from '@/components/PrimaryButton';
import TextField from '@/components/TextField';
import { ErrorState, LoadingState, Screen, messageForError } from '@/components/states';
import { useProfile, useUpdateGoal, useUpdateMetrics } from '@/hooks/useProfile';
import { useAuth } from '@/store/AuthContext';
import { theme } from '@/theme';
import type { HealthGoal } from '@/types/api';

const GOALS: { value: HealthGoal; label: string }[] = [
  { value: 'weight_loss', label: 'Weight loss' },
  { value: 'maintenance', label: 'Maintenance' },
  { value: 'weight_gain', label: 'Weight gain' },
];

export default function Profile() {
  const { logout } = useAuth();
  const profileQuery = useProfile();
  const updateGoal = useUpdateGoal();
  const updateMetrics = useUpdateMetrics();

  const [height, setHeight] = useState('');
  const [weight, setWeight] = useState('');
  const [birthYear, setBirthYear] = useState('');

  // Populate the form once the profile loads (not on every refetch, so the
  // user's in-progress edits aren't overwritten by a background refresh).
  useEffect(() => {
    if (profileQuery.data) {
      setHeight(profileQuery.data.height_cm != null ? String(profileQuery.data.height_cm) : '');
      setWeight(profileQuery.data.weight_kg != null ? String(profileQuery.data.weight_kg) : '');
      setBirthYear(
        profileQuery.data.birth_year != null ? String(profileQuery.data.birth_year) : '',
      );
    }
  }, [profileQuery.data]);

  if (profileQuery.isPending) {
    return (
      <Screen>
        <LoadingState label="Loading your profile…" />
      </Screen>
    );
  }

  if (profileQuery.isError) {
    return (
      <Screen>
        <ErrorState error={profileQuery.error} onRetry={() => void profileQuery.refetch()} />
      </Screen>
    );
  }

  const profile = profileQuery.data;

  const handleSaveMetrics = () => {
    updateMetrics.mutate({
      height_cm: height.trim() ? Number(height) : null,
      weight_kg: weight.trim() ? Number(weight) : null,
      birth_year: birthYear.trim() ? Number(birthYear) : null,
    });
  };

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.content}>
        <Text style={styles.heading}>{profile.username}</Text>
        <Text style={styles.subheading}>{profile.email}</Text>

        <Text style={styles.sectionTitle}>Goal</Text>
        <View style={styles.goalRow}>
          {GOALS.map((goal) => {
            const selected = goal.value === profile.health_goal;
            return (
              <Pressable
                key={goal.value}
                onPress={() => updateGoal.mutate({ health_goal: goal.value })}
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
        {updateGoal.isError ? (
          <Text style={styles.error}>{messageForError(updateGoal.error)}</Text>
        ) : null}

        <Text style={styles.sectionTitle}>Body metrics</Text>
        <TextField
          label="Height (cm)"
          value={height}
          onChangeText={setHeight}
          keyboardType="numeric"
        />
        <TextField
          label="Weight (kg)"
          value={weight}
          onChangeText={setWeight}
          keyboardType="numeric"
        />
        <TextField
          label="Birth year"
          value={birthYear}
          onChangeText={setBirthYear}
          keyboardType="numeric"
        />
        {updateMetrics.isError ? (
          <Text style={styles.error}>{messageForError(updateMetrics.error)}</Text>
        ) : null}
        <PrimaryButton
          label="Save metrics"
          onPress={handleSaveMetrics}
          loading={updateMetrics.isPending}
        />

        <View style={styles.logoutRow}>
          <PrimaryButton label="Log out" variant="secondary" onPress={() => void logout()} />
        </View>
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: { padding: theme.space(5), paddingBottom: theme.space(10) },
  heading: { color: theme.color.textPrimary, fontSize: 22, fontWeight: '500' },
  subheading: { color: theme.color.textSecondary, fontSize: 14, marginBottom: theme.space(6) },
  sectionTitle: {
    color: theme.color.textPrimary,
    fontSize: 16,
    fontWeight: '500',
    marginBottom: theme.space(3),
    marginTop: theme.space(2),
  },
  goalRow: { flexDirection: 'row', gap: theme.space(2), marginBottom: theme.space(2) },
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
  error: { color: theme.color.danger, fontSize: 13, marginBottom: theme.space(3) },
  logoutRow: { marginTop: theme.space(8) },
});