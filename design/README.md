# design/ — UI mockups

The eight screen designs for the NutriLens app, exported from Google Stitch.

## How to use these

They are the **visual target**, not code to paste. `HANDOFF.md §3` has the full
reasoning, but in short: any generated implementation has to be reconciled with
four things that already exist in this repo —

1. the theme tokens in `mobile/src/theme/index.ts` (`07 §8`) — no hex colours in screens
2. the component library in `mobile/src/components/`
3. the typed API layer in `mobile/src/api/endpoints.ts`
4. `mobile/src/types/api.ts`, which is bound field-for-field to `03-api-specification.md`

`mobile/src/screens/NutritionSummary.tsx` is a finished screen that does all four
correctly — the fastest way to see the intended shape of a screen in this codebase.

## Screens

Drop the exported images in `screens/`, named after the screen they show, so they
line up with `07-mobile-app.md §2`:

| File | Screen (`07 §2`) |
|---|---|
| `01-home.png`                 | 1 — Home / Dashboard |
| `02-recognition-results.png`  | 2 — Recognition Results |
| `03-nutrition-summary.png`    | 3 — Nutrition Summary (**already built**) |
| `04-insights.png`             | 4 — Daily Insights |
| `05-camera.png`               | 5 — Camera Capture |
| `06-portion.png`              | 6 — Portion Estimation |
| `07-history.png`              | 7 — Meal History |
| `08-profile.png`              | 8 — Profile & Goals |

If Stitch also exported code, put it in `stitch-export/` beside this file. Keep it
as reference material — it is not wired into the build, and nothing in `mobile/`
imports from here.

## Why these live in the repo

So the design is versioned alongside the code that implements it, and so anyone
who clones has them without needing access to a Stitch account.
