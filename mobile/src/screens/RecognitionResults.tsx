import { useNavigation, useRoute, type RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { useState } from 'react';
import { Alert, Image, ScrollView, StyleSheet, Text, View } from 'react-native';

import PredictionCard from '@/components/PredictionCard';
import PrimaryButton from '@/components/PrimaryButton';
import Screen from '@/components/Screen';
import type { AppStackParamList } from '@/navigation/AppStack';
import { theme } from '@/theme';

type ResultsRoute = RouteProp<AppStackParamList, 'RecognitionResults'>;

// Screen 2 (07-mobile-app.md §2): show predictions + let the user pick when
// unsure (FR04). Portion estimation (M4) and nutrition/logging (M5) aren't
// built yet, so "Confirm" ends the flow here for now instead of a dead click.
export default function RecognitionResults() {
  const navigation = useNavigation<NativeStackNavigationProp<AppStackParamList>>();
  const { params } = useRoute<ResultsRoute>();
  const { imageUri, result } = params;

  const top = result.items[0];
  const [selectedClass, setSelectedClass] = useState(top.food101_class);

  const selectedLabel = result.low_confidence
    ? (result.alternatives.find((a) => a.food101_class === selectedClass)?.display_name ??
      top.display_name)
    : top.display_name;

  const handleConfirm = () => {
    Alert.alert(
      `${selectedLabel} confirmed`,
      "Portion estimation and nutrition logging aren't built yet (M4/M5) - this is as far as the flow goes for now.",
      [{ text: 'OK', onPress: () => navigation.navigate('Tabs') }],
    );
  };

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.content}>
        <Image source={{ uri: imageUri }} style={styles.photo} resizeMode="cover" />

        {result.low_confidence ? (
          <View style={styles.section}>
            <Text style={styles.title}>Not quite sure - pick the closest match</Text>
            {result.alternatives.map((alt) => (
              <PredictionCard
                key={alt.food101_class}
                displayName={alt.display_name}
                confidence={alt.confidence}
                selected={alt.food101_class === selectedClass}
                onPress={() => setSelectedClass(alt.food101_class)}
              />
            ))}
          </View>
        ) : (
          <View style={styles.section}>
            <Text style={styles.title}>We think this is</Text>
            <PredictionCard displayName={top.display_name} confidence={top.confidence} selected />
          </View>
        )}

        <PrimaryButton label="Confirm" onPress={handleConfirm} />
        <PrimaryButton label="Retake photo" onPress={() => navigation.replace('Camera')} />
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  content: {
    paddingHorizontal: theme.space(4),
    paddingVertical: theme.space(4),
    gap: theme.space(3),
  },
  photo: {
    width: '100%',
    aspectRatio: 4 / 3,
    borderRadius: theme.radius.lg,
    backgroundColor: theme.color.surface,
    marginBottom: theme.space(2),
  },
  section: {
    marginBottom: theme.space(4),
  },
  title: {
    color: theme.color.textPrimary,
    fontSize: 18,
    fontWeight: '600',
    marginBottom: theme.space(3),
  },
});
