import { StyleSheet, Text, View } from 'react-native';

import { theme } from '@/theme';

// Generic stand-in until each screen's real implementation lands (see 09-build-phases.md).
export default function PlaceholderScreen({ label }: { label: string }) {
  return (
    <View style={styles.container}>
      <Text style={styles.text}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.color.bg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  text: {
    color: theme.color.textPrimary,
    fontSize: 18,
  },
});
