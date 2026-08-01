/**
 * src/screens/Capture.tsx
 * =======================
 *
 * Screen 5 of `07 §2` — the camera. Still a placeholder: `expo-camera`, the
 * framing guide, client-side compression and the `POST /recognise` call all
 * belong to M3/M4 (`07 §10` step 3).
 *
 * WHY THERE IS A DEVELOPMENT SHORTCUT BELOW
 * -----------------------------------------
 * M5 built the last screen of a flow whose first two screens do not exist yet, so
 * without this there is no way to reach the Nutrition Summary on a device and no
 * way to see M5 working end to end against a running backend. The shortcut pushes
 * straight to that screen with a stand-in for what recognition *would* have
 * produced.
 *
 * Two things about it are deliberate. It is wrapped in `__DEV__`, so it is
 * stripped from any production build and cannot ship. And the values it supplies
 * are only the ones the camera and the portion slider would have supplied — a
 * food class, a portion, a confidence. **No nutrition is invented**: the macros
 * on the next screen come from the real `POST /nutrition` against the real
 * backend, which is the whole point of being able to run it.
 *
 * M3 deletes this block when the real camera lands.
 */

import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { StyleSheet, Text, View } from 'react-native';

import PrimaryButton from '@/components/PrimaryButton';
import { Screen } from '@/components/states';
import { theme } from '@/theme';
import type { CaptureStackParamList } from '@/navigation/types';

type Navigation = NativeStackNavigationProp<CaptureStackParamList, 'Camera'>;

/** Stands in for a `/recognise` result until M3's camera exists. */
const DEV_SAMPLE_ITEM = {
  food101_class: 'grilled_salmon',
  display_name: 'Grilled Salmon',
  portion_confirmed_g: 210,
  portion_min_g: 180,
  portion_max_g: 240,
  confidence_score: 0.91,
  meal_type: 'lunch',
  cooking_method: 'grilled',
} as const;

export default function Capture() {
  const navigation = useNavigation<Navigation>();

  return (
    <Screen>
      <View style={styles.container}>
        <Text style={styles.title}>Camera</Text>
        <Text style={styles.body}>
          Live capture arrives with M3 — see 07-mobile-app.md §2, Screen 5.
        </Text>

        {__DEV__ ? (
          <View style={styles.devBlock}>
            <Text style={styles.devLabel}>Development shortcut</Text>
            <Text style={styles.body}>
              Skips straight to the M5 nutrition screen with a stand-in recognition result.
              The nutrition itself still comes from the live API.
            </Text>
            <PrimaryButton
              label="Open Nutrition Summary"
              onPress={() => navigation.navigate('NutritionSummary', DEV_SAMPLE_ITEM)}
            />
          </View>
        ) : null}
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    padding: theme.space(6),
    gap: theme.space(3),
  },
  title: {
    color: theme.color.textPrimary,
    fontSize: 22,
    fontWeight: '500',
  },
  body: {
    color: theme.color.textSecondary,
    fontSize: 14,
    lineHeight: 20,
  },
  devBlock: {
    marginTop: theme.space(8),
    padding: theme.space(4),
    borderRadius: theme.radius.md,
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: theme.color.border,
    gap: theme.space(3),
  },
  devLabel: {
    color: theme.color.warning,
    fontSize: 12,
    fontWeight: '500',
  },
});
