import { useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import { useRef, useState } from 'react';
import { ActivityIndicator, Alert, Pressable, StyleSheet, Text, View } from 'react-native';

import PrimaryButton from '@/components/PrimaryButton';
import { Screen, messageForError } from '@/components/states';
import { useRecognise } from '@/hooks/useRecognise';
import type { CaptureStackParamList } from '@/navigation/types';
import { theme } from '@/theme';

const MIN_WIDTH = 640;
const MIN_HEIGHT = 480;

/**
 * The upload's long edge. Was 1600, which cost encode time and bytes for detail
 * nothing downstream reads: the classifier resizes to EfficientNet-B4's 380 px
 * input and the segmenter to its own smaller one. 1024 is still ~2.7x the model
 * input and stays clear of the server's 640x480 minimum (03 §4) even for a 4:3
 * frame, which lands at 1024x768.
 */
const MAX_LONG_EDGE = 1024;
const JPEG_QUALITY = 0.8;

/**
 * Capture quality, not upload quality. The frame is re-encoded a moment later at
 * `JPEG_QUALITY` anyway, so asking the camera for a near-lossless 12 MP JPEG
 * first only buys a slower shutter and a larger bitmap to decode and downscale.
 */
const CAPTURE_QUALITY = 0.6;

type Navigation = NativeStackNavigationProp<CaptureStackParamList, 'Camera'>;

/**
 * Target width for the upload, or null to send it untouched.
 *
 * Capping the *long* edge alone is not safe, because the server's floor is
 * per-axis (`width >= 640 AND height >= 480`, `03 §4`) and the short edge is what
 * a cap pushes under it. A 16:9 portrait frame scaled to a 1024 long edge lands
 * at 576x1024 — rejected on width, even though it is the taller image. Aspect
 * ratio varies by device and camera mode, so this has to be computed rather than
 * assumed.
 *
 * The shrink target and the server's minimums pull in opposite directions, so
 * take whichever scale is larger: shrink as far as MAX_LONG_EDGE asks, but never
 * past the point where either axis breaches its minimum. `ceil` because a
 * rounding-down of half a pixel is the difference between 640 and a 400.
 */
function downscaledWidth(width: number, height: number): number | null {
  const shrink = Math.min(1, MAX_LONG_EDGE / Math.max(width, height));
  const floor = Math.max(MIN_WIDTH / width, MIN_HEIGHT / height);
  const scale = Math.max(shrink, floor);
  return scale >= 1 ? null : Math.ceil(width * scale);
}

export default function Camera() {
  const navigation = useNavigation<Navigation>();
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const recognise = useRecognise();

  const processAndSubmit = async (uri: string, width: number, height: number) => {
    if (width < MIN_WIDTH || height < MIN_HEIGHT) {
      Alert.alert(
        'Photo too small',
        `Please use a photo at least ${MIN_WIDTH}x${MIN_HEIGHT}. This one is ${width}x${height}.`,
      );
      return;
    }

    setIsCapturing(true);
    try {
      const targetWidth = downscaledWidth(width, height);
      const resizeAction = targetWidth ? [{ resize: { width: targetWidth } }] : [];

      const manipulated = await ImageManipulator.manipulateAsync(uri, resizeAction, {
        compress: JPEG_QUALITY,
        format: ImageManipulator.SaveFormat.JPEG,
      });

      const result = await recognise.mutateAsync(manipulated.uri);
      navigation.replace('RecognitionResults', { imageUri: manipulated.uri, result });
    } catch (err) {
      // Map on `code` (07 §7) rather than showing the server's raw wording —
      // NO_FOOD_DETECTED in particular is a normal outcome, not a failure, so it
      // gets its own title instead of "Recognition failed".
      const code = (err as { code?: string })?.code;
      Alert.alert(
        code === 'NO_FOOD_DETECTED' ? 'No food detected' : 'Recognition failed',
        messageForError(err),
      );
    } finally {
      setIsCapturing(false);
    }
  };

  const handleShutter = async () => {
    if (!cameraRef.current || isCapturing) return;
    const photo = await cameraRef.current.takePictureAsync({ quality: CAPTURE_QUALITY });
    if (!photo) return;
    await processAndSubmit(photo.uri, photo.width, photo.height);
  };

  const handleGallery = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('Permission needed', 'Allow photo library access to upload a meal photo.');
      return;
    }
    const picked = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
    });
    if (picked.canceled || picked.assets.length === 0) return;
    const asset = picked.assets[0];
    await processAndSubmit(asset.uri, asset.width, asset.height);
  };

  if (!permission) {
    return (
      <Screen>
        <View style={styles.center} />
      </Screen>
    );
  }

  if (!permission.granted) {
    return (
      <Screen>
        <View style={styles.center}>
          <Text style={styles.permissionText}>
            NutriLens needs camera access to photograph your meals.
          </Text>
          <PrimaryButton label="Grant camera access" onPress={requestPermission} />
          <Pressable onPress={handleGallery} style={styles.linkRow} accessibilityRole="button">
            <Text style={styles.linkText}>Or upload from gallery instead</Text>
          </Pressable>
          <Pressable
            onPress={() => navigation.goBack()}
            style={styles.linkRow}
            accessibilityRole="button"
          >
            <Text style={styles.linkText}>Cancel</Text>
          </Pressable>
        </View>
      </Screen>
    );
  }

  return (
    <View style={styles.flex}>
      <CameraView ref={cameraRef} style={styles.flex} facing="back" />
      <View style={styles.overlay}>
        <Pressable
          onPress={() => navigation.goBack()}
          style={styles.closeButton}
          accessibilityRole="button"
          accessibilityLabel="Close camera"
        >
          <Text style={styles.closeText}>✕</Text>
        </Pressable>

        <View style={styles.controls}>
          <Pressable
            onPress={handleGallery}
            style={styles.secondaryButton}
            accessibilityRole="button"
            accessibilityLabel="Upload from gallery"
          >
            <Text style={styles.secondaryText}>Gallery</Text>
          </Pressable>

          <Pressable
            onPress={handleShutter}
            disabled={isCapturing}
            style={[styles.shutter, isCapturing && styles.shutterDisabled]}
            accessibilityRole="button"
            accessibilityLabel="Take photo"
          />

          <View style={styles.secondaryButton} />
        </View>

        {/*
          The shutter is disabled for the whole round trip (capture, downscale,
          upload, inference), so without a moving indicator the screen reads as
          frozen rather than busy — the reason this felt like "waiting a lot".
        */}
        {isCapturing ? (
          <View style={styles.statusRow}>
            <ActivityIndicator color={theme.color.accent} />
            <Text style={styles.statusText}>Recognising…</Text>
          </View>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: '#000' },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: theme.space(6),
    gap: theme.space(4),
  },
  permissionText: {
    color: theme.color.textPrimary,
    fontSize: 16,
    textAlign: 'center',
    marginBottom: theme.space(4),
  },
  linkRow: { marginTop: theme.space(4) },
  linkText: { color: theme.color.textSecondary, fontSize: 14 },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'space-between',
  },
  closeButton: {
    alignSelf: 'flex-end',
    margin: theme.space(4),
    width: 44,
    height: 44,
    borderRadius: theme.radius.pill,
    backgroundColor: 'rgba(0,0,0,0.5)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeText: { color: '#fff', fontSize: 18 },
  controls: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: theme.space(8),
    paddingBottom: theme.space(10),
  },
  shutter: {
    width: 72,
    height: 72,
    borderRadius: theme.radius.pill,
    backgroundColor: '#fff',
    borderWidth: 4,
    borderColor: 'rgba(255,255,255,0.4)',
  },
  shutterDisabled: { opacity: 0.5 },
  secondaryButton: { width: 60, alignItems: 'center' },
  secondaryText: { color: '#fff', fontSize: 13 },
  statusRow: {
    position: 'absolute',
    top: '45%',
    alignSelf: 'center',
    flexDirection: 'row',
    alignItems: 'center',
    gap: theme.space(2),
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: theme.space(4),
    paddingVertical: theme.space(2),
    borderRadius: theme.radius.md,
  },
  statusText: { color: '#fff', fontSize: 16 },
});