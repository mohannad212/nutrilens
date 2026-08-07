// App-wide runtime config. Expo SDK 49+ inlines any process.env var prefixed
// EXPO_PUBLIC_ at build time from .env files - no extra dotenv plugin needed.
// See mobile/.env.example. For a physical device, point this at your machine's
// LAN IP (e.g. http://192.168.1.23:8000) - localhost on-device means the phone
// itself, not your dev machine.
const API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL ?? 'http://localhost:8000';

// Every backend route lives under this prefix (03-api-specification.md §Base URL).
const API_V1_PREFIX = '/api/v1';

export const Config = {
  API_BASE_URL,
  API_V1_PREFIX,
} as const;
