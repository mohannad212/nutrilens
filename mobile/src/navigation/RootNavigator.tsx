import { NavigationContainer } from '@react-navigation/native';

import { LoadingState, Screen } from '@/components/states';
import { useAuth } from '@/store/AuthContext';
import AppTabs from './AppTabs';
import AuthStack from './AuthStack';

// Auth gate: valid JWT in secure storage -> AppTabs/Home, else AuthStack/Login (07 §1).
export default function RootNavigator() {
  const { isAuthenticated, isRestoring } = useAuth();

  // Secure store is read asynchronously, so on the first frame the session is
  // unknown rather than absent. Rendering AuthStack here would flash the Login
  // screen at a user who is already signed in.
  if (isRestoring) {
    return (
      <Screen>
        <LoadingState label="" />
      </Screen>
    );
  }

  return <NavigationContainer>{isAuthenticated ? <AppTabs /> : <AuthStack />}</NavigationContainer>;
}
