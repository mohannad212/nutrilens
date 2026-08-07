import { createNativeStackNavigator } from '@react-navigation/native-stack';

import Camera from '@/screens/Camera';
import RecognitionResults from '@/screens/RecognitionResults';
import type { RecognitionResponse } from '@/types/recognition';
import AppTabs from './AppTabs';

// Root stack for authenticated users. AppTabs is one screen in here (not the
// root itself) so the capture flow can push Camera -> RecognitionResults as
// modals that cover the tab bar (07-mobile-app.md §1: "a modal stack pushed
// over the tabs"). NutritionSummary joins this stack in M5.
export type AppStackParamList = {
  Tabs: undefined;
  Camera: undefined;
  RecognitionResults: { imageUri: string; result: RecognitionResponse };
};

const Stack = createNativeStackNavigator<AppStackParamList>();

export default function AppStack() {
  return (
    <Stack.Navigator screenOptions={{ headerShown: false }}>
      <Stack.Screen name="Tabs" component={AppTabs} />
      <Stack.Screen name="Camera" component={Camera} options={{ presentation: 'fullScreenModal' }} />
      <Stack.Screen
        name="RecognitionResults"
        component={RecognitionResults}
        options={{ presentation: 'modal' }}
      />
    </Stack.Navigator>
  );
}
