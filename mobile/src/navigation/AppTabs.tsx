import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import { theme } from '@/theme';
import Capture from '@/screens/Capture';
import History from '@/screens/History';
import Home from '@/screens/Home';
import Insights from '@/screens/Insights';
import Profile from '@/screens/Profile';

export type AppTabsParamList = {
  Home: undefined;
  History: undefined;
  Capture: undefined;
  Insights: undefined;
  Profile: undefined;
};

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
      <Tab.Screen name="Capture" component={Capture} />
      <Tab.Screen name="Insights" component={Insights} />
      <Tab.Screen name="Profile" component={Profile} />
    </Tab.Navigator>
  );
}
