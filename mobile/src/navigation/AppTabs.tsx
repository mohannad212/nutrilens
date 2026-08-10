import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { Image } from 'react-native';

import { theme } from '@/theme';
import CaptureStack from './CaptureStack';
import History from '@/screens/History';
import Home from '@/screens/Home';
import Insights from '@/screens/Insights';
import Profile from '@/screens/Profile';
import type { AppTabsParamList } from './types';

export type { AppTabsParamList };

const Tab = createBottomTabNavigator<AppTabsParamList>();

const ICONS: Record<keyof AppTabsParamList, ReturnType<typeof require>> = {
  Home: require('../../assets/icons/home.png'),
  History: require('../../assets/icons/history.png'),
  Capture: require('../../assets/icons/capture.png'),
  Insights: require('../../assets/icons/insights.png'),
  Profile: require('../../assets/icons/profile.png'),
};

export default function AppTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        headerShown: false,
        tabBarActiveTintColor: theme.color.accent,
        tabBarInactiveTintColor: theme.color.textSecondary,
        tabBarStyle: { backgroundColor: theme.color.surface, borderTopColor: theme.color.border },
        tabBarIcon: ({ color, size }) => (
          <Image
            source={ICONS[route.name as keyof AppTabsParamList]}
            style={{ width: size, height: size, tintColor: color }}
            resizeMode="contain"
          />
        ),
      })}
    >
      <Tab.Screen name="Home" component={Home} />
      <Tab.Screen name="History" component={History} />
      <Tab.Screen name="Capture" component={CaptureStack} />
      <Tab.Screen name="Insights" component={Insights} />
      <Tab.Screen name="Profile" component={Profile} />
    </Tab.Navigator>
  );
}