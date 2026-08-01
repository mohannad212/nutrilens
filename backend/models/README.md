# backend/models/ — model weights

The trained weight files (`*.pth`) are **not in git** — they're large binaries, so
git ignores them (see root `.gitignore`). Only the tiny label maps
(`labels_food101.json`, `labels_container.json`) are committed. After cloning, you
must **download the weights into this folder** for the backend to run the full
pipeline.

## Files needed here

| File | Purpose | ~Size | In git? |
|---|---|---|---|
| `food101_efficientnet_b4.pth` | food recognition (M3) | 72 MB | ❌ download |
| `labels_food101.json` | class labels | 2 KB | ✅ tracked |
| `container_classifier.pth` | portion Stage 1 (bowl/plate) | 6.5 MB | ❌ download |
| `labels_container.json` | container labels | <1 KB | ✅ tracked |
| `food_segmenter.pth` | portion segmentation (food/bg) | 44 MB | ❌ download |

(The Stage-2 reference detector is an off-the-shelf COCO model the backend
downloads itself when `REFERENCE_DETECTOR_ENABLED=true` — nothing to place here.)

## Where to get the weights

**Option A — GitHub Release (canonical).** Download the `.pth` files from the repo's
Releases page (tag e.g. `models-v1`) and drop them into this folder.

**Option B — shared team drive.** If a teammate has them, copy the three `.pth`
files here directly.

The backend loads whatever is present at startup and **degrades gracefully** if a
file is missing (e.g. no segmenter → classical fallback), so you can start with a
subset.

## Verify

```bash
ls backend/models/*.pth   # expect the three files above
```
On backend startup the logs show, e.g.:
`Portion estimator ready (container=True, reference=…, segmenter=True)`
