/**
 * src/navigation/CaptureStack.tsx
 * ===============================
 *
 * `07 §1` — "Capture flow is a modal stack pushed over the tabs:
 * `Camera → RecognitionResults → NutritionSummary → (back to Home)`".
 *
 * WHY THE STACK EXISTS NOW, WITH TWO PLACEHOLDER SCREENS IN IT
 * ------------------------------------------------------------
 * M5 owns only the last screen. Camera and RecognitionResults belong to M3/M4 and
 * are still stubs. The stack is defined in full anyway because the *route
 * contract* is the thing M3 and M4 need to build against: `CaptureStackParamList`
 * states exactly what NutritionSummary requires to be handed, so when the camera
 * screen lands, "navigate to the summary" is a typed call that the compiler
 * checks rather than a shape two milestones have to agree on by memory.
 *
 * WHY `NutritionSummary` IS WRAPPED RATHER THAN USED DIRECTLY
 * -----------------------------------------------------------
 * The screen takes plain props (`params`, `onLogged`, `onCancel`) instead of
 * React Navigation's `route`/`navigation` objects. That keeps it renderable in
 * isolation — a test or a future preview can pass it an object and a callback
 * with no navigation container in scope. The adapter below is the four lines that
 * connect it to the router, and it is the only part that knows a router exists.
 *
 * WHERE "BACK TO HOME" HAPPENS
 * ----------------------------
 * On a successful log the whole capture stack is popped to its root and the tab
 * navigator is sent to Home (`07 §1`). Popping matters: leaving the flow on the
 * stack would let the back gesture return the user to a summary for a meal they
 * have already logged, where pressing "Log This Meal" again would log a duplicate
 * (`03 §6` has no idempotency key).
 */

import { createNativeStackNavigator } from '@react-navigation/native-stack';
import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import { useNavigation } from '@react-navigation/native';

import Capture from '@/screens/Capture';
import NutritionSummary from '@/screens/NutritionSummary';
import PlaceholderScreen from '@/screens/PlaceholderScreen';
import { theme } from '@/theme';
import type { CaptureStackParamList } from './types';

const Stack = createNativeStackNavigator<CaptureStackParamList>();

// M3/M4 replace this with the real Screen 2 (07 §2).
function RecognitionResults() {
  return <PlaceholderScreen label="Recognition Results" />;
}

type SummaryProps = NativeStackScreenProps<CaptureStackParamList, 'NutritionSummary'>;

/** Adapter: route params in, navigation side effects out. See the header note. */
function NutritionSummaryRoute({ route, navigation }: SummaryProps) {
  const rootNavigation = useNavigation();

  return (
    <NutritionSummary
      params={route.params}
      onLogged={() => {
        // Clear the flow before leaving, so back cannot return to a logged meal.
        navigation.popToTop();
        rootNavigation.navigate('Home' as never);
      }}
      onCancel={() => navigation.goBack()}
    />
  );
}

export default function CaptureStack() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: { backgroundColor: theme.color.bg },
        headerTintColor: theme.color.textPrimary,
        headerShadowVisible: false,
        contentStyle: { backgroundColor: theme.color.bg },
      }}
    >
      <Stack.Screen name="Camera" component={Capture} options={{ headerShown: false }} />
      <Stack.Screen
        name="RecognitionResults"
        component={RecognitionResults}
        options={{ title: 'Confirm your meal' }}
      />
      <Stack.Screen
        name="NutritionSummary"
        component={NutritionSummaryRoute}
        options={{ title: 'Nutrition' }}
      />
    </Stack.Navigator>
  );
}
