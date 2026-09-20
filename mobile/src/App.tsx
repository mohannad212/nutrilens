import { StatusBar } from 'expo-status-bar';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import RootNavigator from '@/navigation/RootNavigator';
import { AuthProvider } from '@/store/AuthContext';

const queryClient = new QueryClient();

export default function App() {
  return (
    // Every screen wraps in <Screen> (src/components/Screen.tsx), which reads
    // insets via react-native-safe-area-context's SafeAreaView. That hook
    // throws without a SafeAreaProvider ancestor — this is it.
    <SafeAreaProvider>
      <QueryClientProvider client={queryClient}>
        <AuthProvider>
          <StatusBar style="light" />
          <RootNavigator />
        </AuthProvider>
      </QueryClientProvider>
    </SafeAreaProvider>
  );
}
