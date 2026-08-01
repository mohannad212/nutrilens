import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import { theme } from '@/theme';
import CaptureStack from './CaptureStack';
import History from '@/screens/History';
import Home from '@/screens/Home';
import Insights from '@/screens/Insights';
import Profile from '@/screens/Profile';
import type { AppTabsParamList } from './types';

export type { AppTabsParamList };

const Tab = createBottomTabNavigator<AppTabsParamList>();

export default function AppTabs() {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: theme.color.accent,
        tabBarInactiveTintColor: theme.color.textSecondary,
        tabBarStyle: { backgroundColor: theme.color.surface, borderTopColor: theme.color.border },
      }}
    >
      <Tab.Screen name="Home" component={Home} />
      <Tab.Screen name="History" component={History} />
      {/* The capture tab hosts the whole flow (07 §1), not a single screen. */}
      <Tab.Screen name="Capture" component={CaptureStack} />
      <Tab.Screen name="Insights" component={Insights} />
      <Tab.Screen name="Profile" component={Profile} />
    </Tab.Navigator>
  );
}
