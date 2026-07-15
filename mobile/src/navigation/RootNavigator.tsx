import { NavigationContainer } from '@react-navigation/native';

import { useAuth } from '@/store/AuthContext';
import AppTabs from './AppTabs';
import AuthStack from './AuthStack';

// Auth gate: valid JWT in secure storage -> AppTabs/Home, else AuthStack/Login (07 §1).
export default function RootNavigator() {
  const { isAuthenticated } = useAuth();

  return <NavigationContainer>{isAuthenticated ? <AppTabs /> : <AuthStack />}</NavigationContainer>;
}
