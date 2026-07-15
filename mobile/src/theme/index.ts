// Design tokens — see 07-mobile-app.md §8. No hard-coded hex outside this file.
export const theme = {
  color: {
    bg: '#0E0F12',
    surface: '#17191F',
    surfaceAlt: '#1F222A',
    textPrimary: '#F5F7FA',
    textSecondary: '#A2A8B3',
    accent: '#3DDC97',
    protein: '#4C9AFF',
    carbs: '#FFB74D',
    fat: '#FF7597',
    danger: '#E5484D',
    warning: '#F5A623',
    success: '#3DDC97',
    border: 'rgba(255,255,255,0.08)',
  },
  radius: { sm: 8, md: 12, lg: 16, pill: 999 },
  space: (n: number): number => n * 4,
  font: { regular: 'System', medium: 'System' },
} as const;

export type Theme = typeof theme;
