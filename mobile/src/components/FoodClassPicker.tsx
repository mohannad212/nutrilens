/**
 * src/components/FoodClassPicker.tsx
 * ==================================
 *
 * A searchable list of all 101 classes, for when the model is *confidently*
 * wrong.
 *
 * `/recognise` only returns `alternatives` below the confidence threshold
 * (`03 §4`), so the case this exists for is precisely the one FR04 does not
 * cover: a high-confidence prediction that happens to be incorrect. Observed in
 * testing — a plate of penne came back as "French Fries" at 0.71, above the 0.60
 * threshold, so no pick-list was offered and the user had no way to say
 * otherwise. Every downstream value (USDA match, macros, the logged meal) then
 * inherits that error silently.
 *
 * Search is a plain case-insensitive substring over the display name. 101 items
 * is small enough that anything cleverer — fuzzy matching, an index — would be
 * complexity with no user-visible payoff.
 */

import { useMemo, useState } from 'react';
import { FlatList, Modal, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';

import { theme } from '@/theme';
import type { FoodClass } from '@/types/api';

interface Props {
  visible: boolean;
  classes: FoodClass[];
  /** Highlighted in the list so the current choice is obvious on open. */
  selectedClass?: string;
  onSelect: (choice: FoodClass) => void;
  onClose: () => void;
  isLoading?: boolean;
  error?: unknown;
}

export default function FoodClassPicker({
  visible,
  classes,
  selectedClass,
  onSelect,
  onClose,
  isLoading,
  error,
}: Props) {
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return classes;
    return classes.filter((c) => c.display_name.toLowerCase().includes(q));
  }, [classes, query]);

  return (
    <Modal visible={visible} animationType="slide" onRequestClose={onClose} transparent={false}>
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Choose the right food</Text>
          <Pressable onPress={onClose} style={styles.close} accessibilityRole="button">
            <Text style={styles.closeText}>Cancel</Text>
          </Pressable>
        </View>

        <TextInput
          value={query}
          onChangeText={setQuery}
          placeholder="Search foods…"
          placeholderTextColor={theme.color.textSecondary}
          autoCapitalize="none"
          autoCorrect={false}
          style={styles.search}
          accessibilityLabel="Search foods"
        />

        {isLoading ? (
          <Text style={styles.message}>Loading foods…</Text>
        ) : error ? (
          <Text style={styles.message}>Couldn&apos;t load the food list. Close and try again.</Text>
        ) : filtered.length === 0 ? (
          <Text style={styles.message}>No food matches “{query.trim()}”.</Text>
        ) : (
          <FlatList
            data={filtered}
            keyExtractor={(item) => item.food101_class}
            keyboardShouldPersistTaps="handled"
            renderItem={({ item }) => {
              const isSelected = item.food101_class === selectedClass;
              return (
                <Pressable
                  onPress={() => onSelect(item)}
                  style={[styles.row, isSelected ? styles.rowSelected : null]}
                  accessibilityRole="button"
                  accessibilityState={{ selected: isSelected }}
                >
                  <Text style={[styles.rowText, isSelected ? styles.rowTextSelected : null]}>
                    {item.display_name}
                  </Text>
                </Pressable>
              );
            }}
          />
        )}
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.color.bg, paddingTop: theme.space(12) },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: theme.space(4),
    marginBottom: theme.space(3),
  },
  title: { color: theme.color.textPrimary, fontSize: 18, fontWeight: '600' },
  // 44pt minimum touch target (07 §9).
  close: { minHeight: 44, justifyContent: 'center', paddingHorizontal: theme.space(2) },
  closeText: { color: theme.color.accent, fontSize: 16 },
  search: {
    marginHorizontal: theme.space(4),
    marginBottom: theme.space(3),
    paddingHorizontal: theme.space(4),
    height: 48,
    borderRadius: theme.radius.md,
    borderWidth: 1,
    borderColor: theme.color.border,
    backgroundColor: theme.color.surface,
    color: theme.color.textPrimary,
    fontSize: 16,
  },
  message: {
    color: theme.color.textSecondary,
    fontSize: 14,
    textAlign: 'center',
    paddingHorizontal: theme.space(6),
    paddingVertical: theme.space(6),
  },
  row: {
    minHeight: 48,
    justifyContent: 'center',
    paddingHorizontal: theme.space(5),
    borderBottomWidth: 1,
    borderBottomColor: theme.color.border,
  },
  rowSelected: { backgroundColor: theme.color.surface },
  rowText: { color: theme.color.textPrimary, fontSize: 16 },
  rowTextSelected: { color: theme.color.accent, fontWeight: '600' },
});
