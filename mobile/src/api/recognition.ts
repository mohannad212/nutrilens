import type { AxiosError } from 'axios';

import type { RecognitionResponse } from '@/types/recognition';
import { api } from './client';
import { type ApiErrorEnvelope, normaliseError } from './errors';

/**
 * POST /recognise (03-api-specification.md §4) - the one multipart endpoint.
 * `imageUri` is a local file URI from expo-camera/expo-image-picker, already
 * resized/compressed client-side (07-mobile-app.md §5/§6).
 *
 * Deliberately don't set a Content-Type header by hand: React Native's
 * FormData/XHR layer generates the multipart boundary itself, and a
 * hand-set 'multipart/form-data' header without that boundary is a classic
 * "backend can't parse the upload" bug.
 */
export async function recogniseRequest(imageUri: string): Promise<RecognitionResponse> {
  const formData = new FormData();
  const filename = imageUri.split('/').pop() ?? 'meal.jpg';
  const extMatch = /\.(\w+)$/.exec(filename);
  const ext = extMatch ? extMatch[1].toLowerCase() : 'jpg';
  const type = ext === 'png' ? 'image/png' : 'image/jpeg';

  // React Native's fetch/XHR polyfill accepts this {uri, name, type} shape
  // for a file field - not a real Blob, but this is the documented pattern.
  formData.append('image', { uri: imageUri, name: filename, type } as unknown as Blob);

  try {
    const res = await api.post<RecognitionResponse>('/recognise', formData, {
      timeout: 30000, // inference can be slower than the default 15s
    });
    return res.data;
  } catch (err) {
    throw normaliseError(err as AxiosError<ApiErrorEnvelope>);
  }
}
