# 07 — React Native Mobile App

React Native 0.73+ (Expo recommended), TypeScript strict. Nine screens matching the Phase-1
wireframes (report Figures 5.7–5.8). The app is **thin**: it renders, captures images, calls the API,
and holds only UI/session state. No nutrition math, no secrets.

---

## 1. Navigation structure

```
RootNavigator (native-stack)
├── AuthStack           (unauthenticated)
│   ├── Login
│   └── Register
└── AppTabs             (authenticated — bottom tab bar, persistent)
    ├── Home            (Screen 1: Dashboard / recent meals)
    ├── History         (Screen 7: meal history)
    ├── Capture (FAB)   (Screen 5 → 2 → 3 flow)
    ├── Insights        (Screen 4: daily insights)
    └── Profile         (Screen 8: profile & goals)
```
Auth gate: on launch, read JWT from secure storage → if valid, land on `AppTabs/Home`, else `AuthStack/Login`.
Capture flow is a modal stack pushed over the tabs: `Camera → RecognitionResults → NutritionSummary → (back to Home)`.

## 2. Screens (component + behaviour spec)

| # | Screen | Purpose | Key elements | Calls |
|---|--------|---------|--------------|-------|
| 1 | **Home / Dashboard** | Daily summary + entry point | Calorie ring (today), macro totals, "Recent Meals" list, capture FAB | `GET /meals?from=today` |
| 2 | **Recognition Results** | Show predictions + portion | Photo preview, ranked items w/ confidence, portion range, **cooking method dropdown**, confirm | uses `/recognise` result |
| 3 | **Nutrition Summary** | Show macros for the meal | Calorie ring, protein/carb/fat bars, **disclaimer**, "Log This Meal" | `POST /nutrition`, `POST /meals` |
| 4 | **Daily Insights** | Recommendations | Insight cards w/ `nutrient_focus` icon, disclaimer, locked-state progress | `GET /recommendations` |
| 5 | **Camera Capture** | Take/upload photo | Live camera, shutter, "upload from gallery", framing guide | `POST /recognise` |
| 6 | **Portion Estimation** | Confirm/adjust portion (FR06) | Estimated range, **slider**, live macro preview | (local; then `/nutrition`) |
| 7 | **Meal History** | Chronological log (FR11) | Date-grouped list, per-meal calories, tap→detail, delete | `GET /meals`, `DELETE /meals/{id}` |
| 8 | **Profile & Goals** | Account + goal (FR13) | Avatar/initials, health-goal selector, macro summary, logout, **delete account** (FR14) | `GET /users/me`, `PATCH /users/me/goal`, `DELETE /users/me` |
| 9 | **Home (alt/empty state)** | First-run dashboard | Empty-state prompt to log first meal | — |

Screen order in the capture flow (maps FR03→FR09): **5 → 2 → 6 → 3 → log**. (Screen 6 can be embedded
in Screen 2 as a section; keep as a step for clarity.)

## 3. Component library (`src/components/`)

- `CalorieRing` — SVG ring (react-native-svg), props `{consumed, target}`.
- `MacroBar` — labelled progress bar per macro, props `{label, grams, pct, color}`.
- `PredictionCard` — one recognised item: name, confidence chip, portion range.
- `ConfidenceChip` — colour-coded (green ≥0.75, amber 0.6–0.75, red <0.6 → prompts top-3).
- `PortionSlider` — min–max range slider (FR06); emits confirmed grams; shows live macro preview.
- `CookingMethodPicker` — dropdown bound to `GET /meta/cooking-methods`.
- `InsightCard` — recommendation text + focus icon + read state.
- `DisclaimerBanner` — the statutory disclaimer; **required on Nutrition (3) and Insights (4)**.
- `MealListItem` — history row: thumbnail-less (no image stored) icon, name, kcal, time.
- `PrimaryButton`, `TextField`, `Screen` (safe-area wrapper), `EmptyState`, `LoadingState`, `ErrorState`.

## 4. State management

- **Server state:** TanStack Query (React Query). One hook per endpoint in `src/hooks/`
  (`useLogin`, `useRegister`, `useRecognise`, `useNutrition`, `useMeals`, `useCreateMeal`,
  `useRecommendations`, `useProfile`, `useUpdateGoal`, `useDeleteAccount`). Query keys namespaced by
  user. Invalidate `['meals']` and `['recommendations']` after `POST /meals`.
- **Auth/session state:** a lightweight `AuthContext` holding `{token, user, isAuthenticated}`,
  backed by `expo-secure-store`. Provides `login()`, `logout()`, `register()`.
- **Ephemeral capture state:** local component state / a `useReducer` for the capture flow
  (image URI → recognition result → confirmed portions → cooking methods).
- **No global Redux needed** at this scope.

## 5. API client (`src/api/`)

```ts
// client.ts
const api = axios.create({ baseURL: Config.API_BASE_URL, timeout: 15000 });
api.interceptors.request.use(cfg => {                 // attach JWT
  const t = getToken(); if (t) cfg.headers.Authorization = `Bearer ${t}`; return cfg;
});
api.interceptors.response.use(r => r, err => {         // 401 → logout; normalise error envelope
  if (err.response?.status === 401) authEvents.emit('logout');
  return Promise.reject(normaliseError(err));          // → { code, message, details }
});
```
- Typed endpoint wrappers mirror `03` exactly; response types live in `src/types/` and must match the
  Pydantic schemas field-for-field.
- `/recognise` uses `FormData` with the image; set `Content-Type: multipart/form-data`.
- Image capture: compress client-side to keep ≤ 10 MB and ≥ 640×480 before upload (expo-image-manipulator).

## 6. Camera & image handling

- `expo-camera` for capture; `expo-image-picker` for gallery. Request permissions with graceful denial UI.
- Enforce constraints client-side (mirror server FR03): resize longest edge to ≤ 1600 px, JPEG quality
  0.8, assert ≥ 640×480; show a friendly error otherwise (don't rely only on the server 400).
- **The app never caches the food image** beyond the in-flight request (privacy parity with backend).

## 7. Error, loading, offline states

- Every screen renders one of: `LoadingState`, `ErrorState(retry)`, `EmptyState`, or content.
- Map API `error.code` → friendly copy (e.g. `USDA_UNAVAILABLE` → "We couldn't fetch nutrition right
  now — showing cached values where possible.").
- Offline: React Query `onError` + a top banner "You're offline"; queue nothing destructive.
- `ACCOUNT_LOCKED` (423) on login → explain lockout + how to proceed.

## 8. Design tokens (`src/theme/`)

Match the dark wireframes (report Fig 5.7–5.8): dark UI, high-contrast accents.

```ts
export const theme = {
  color: {
    bg: '#0E0F12', surface: '#17191F', surfaceAlt: '#1F222A',
    textPrimary: '#F5F7FA', textSecondary: '#A2A8B3',
    accent: '#3DDC97',            // calorie ring / primary actions
    protein: '#4C9AFF', carbs: '#FFB74D', fat: '#FF7597',
    danger: '#E5484D', warning: '#F5A623', success: '#3DDC97',
    border: 'rgba(255,255,255,0.08)',
  },
  radius: { sm: 8, md: 12, lg: 16, pill: 999 },
  space: (n:number)=> n*4,
  font: { regular:'System', medium:'System' },   // weights 400/500 only
};
```
Typography: two weights (400/500), sentence case, no all-caps. Charts use the macro colours above
consistently (protein blue, carbs amber, fat pink) across every screen.

## 9. Accessibility (per report §5.4)

- All interactive elements have `accessibilityLabel` / `accessibilityRole`.
- Minimum touch target 44×44. Colour is never the only signal (confidence chip also shows text).
- Support dynamic font scaling; test with large text.

## 10. App build order (aligns with `09`)

1. Scaffolding, theme, navigation shells, `AuthContext` (M1/M2).
2. Login/Register wired to `/auth` (M2).
3. Camera → `/recognise` → Recognition Results + portion slider (M3–M4).
4. Cooking method + `/nutrition` → Nutrition Summary + `POST /meals` (M5).
5. History + Profile/goal (M6).
6. Insights screen + locked state (M7).
7. Delete account + polish/empty states (M8).

---

Prev: [`06-recommendation-engine.md`](./06-recommendation-engine.md) · Next: [`08-backend-services.md`](./08-backend-services.md)
