/**
 * src/components/CalorieRing.tsx
 * ==============================
 *
 * `07 §3` — "SVG ring (react-native-svg), props `{consumed, target}`". The
 * headline number on both the Nutrition Summary screen and Home.
 *
 * HOW THE ARC IS DRAWN
 * --------------------
 * Two concentric circles: a full-circumference track, and a progress circle
 * whose `strokeDasharray` is the circumference and whose `strokeDashoffset` is
 * the unfilled remainder. Rotating -90° puts the start of the arc at twelve
 * o'clock, which is where a progress ring is expected to begin. This is the
 * standard trick and it avoids computing arc paths by hand — a full circle with a
 * dash gap is exactly an arc.
 *
 * WHY PROGRESS IS CLAMPED BUT THE NUMBER IS NOT
 * ---------------------------------------------
 * Eating past the target is normal and the ring simply fills; letting the arc
 * continue past 100% would wrap it back over itself and read as *less* progress
 * than a full ring. So the arc clamps at full while the figure below it keeps
 * counting. The user's actual intake is never misreported — only the geometry is
 * bounded.
 *
 * WHY THE TARGET IS OPTIONAL
 * --------------------------
 * On the Nutrition Summary screen this ring shows a single meal, which has no
 * target to be a proportion of. With no target the ring renders full in the
 * accent colour and the caption drops the "of N kcal" — it is a display of a
 * figure, not a progress bar. Inventing a daily target here would put a number on
 * screen the backend never computed, and the one place NutriLens does estimate
 * energy needs is `06 §2`, which requires body metrics the user may not have
 * given.
 */

import { StyleSheet, Text, View } from 'react-native';
import Svg, { Circle, G } from 'react-native-svg';

import { theme } from '@/theme';

interface CalorieRingProps {
  consumed: number;
  /** Omit for a single meal — see the header note on why no target is invented. */
  target?: number | null;
  size?: number;
  strokeWidth?: number;
  caption?: string;
}

export default function CalorieRing({
  consumed,
  target = null,
  size = 180,
  strokeWidth = 14,
  caption,
}: CalorieRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  const hasTarget = typeof target === 'number' && target > 0;
  const rawProgress = hasTarget ? consumed / target : 1;
  // Clamped for geometry only; the printed number below is never clamped.
  const progress = Math.max(0, Math.min(1, rawProgress));
  const dashOffset = circumference * (1 - progress);

  const accessibilityLabel = hasTarget
    ? `${Math.round(consumed)} of ${Math.round(target)} kilocalories`
    : `${Math.round(consumed)} kilocalories`;

  return (
    <View
      style={[styles.container, { width: size, height: size }]}
      accessibilityRole="progressbar"
      accessibilityLabel={accessibilityLabel}
    >
      <Svg width={size} height={size}>
        {/* -90° so the arc starts at the top rather than at three o'clock. */}
        <G rotation={-90} originX={size / 2} originY={size / 2}>
          <Circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={theme.color.surfaceAlt}
            strokeWidth={strokeWidth}
            fill="none"
          />
          <Circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={theme.color.accent}
            strokeWidth={strokeWidth}
            strokeDasharray={`${circumference} ${circumference}`}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            fill="none"
          />
        </G>
      </Svg>

      <View style={styles.centre} pointerEvents="none">
        <Text style={styles.value}>{Math.round(consumed)}</Text>
        <Text style={styles.unit}>kcal</Text>
        {caption ? <Text style={styles.caption}>{caption}</Text> : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  centre: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  value: {
    color: theme.color.textPrimary,
    fontSize: 40,
    fontWeight: '500',
  },
  unit: {
    color: theme.color.textSecondary,
    fontSize: 13,
  },
  caption: {
    color: theme.color.textSecondary,
    fontSize: 12,
    marginTop: theme.space(1),
  },
});
