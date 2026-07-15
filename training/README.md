# Training — EfficientNet-B4 on Food-101 (spec: `04-ml-model.md`)

One script does everything: `food101_train.py`. Recipe: warm-up (3 epochs, frozen backbone)
→ fine-tune (≤30 epochs) with **RandAugment, Random Erasing, MixUp/CutMix, EMA weights,
LR warmup → cosine decay**, and horizontal-flip **TTA** at the final test evaluation.
It seeds everything (42), carves a fixed 100-img/class validation set, checkpoints
**every epoch**, and **auto-resumes** — critical on free GPU tiers that disconnect.

## Time budget (free tiers)

| | GPU | Session limit | Weekly cap | Realistic total |
|---|---|---|---|---|
| **Kaggle (recommended)** | T4×2 / P100 | 12 h | 30 GPU-h/week | **2–3 sessions, ~3 days** |
| **Free Colab** | T4 | ~3–4 h realistic | daily caps | **5–7 days of daily sessions** |

Total compute ≈ 14–20 GPU-hours (early stop usually kicks in around epoch 20–25).
Expected result: **88–91% top-1** on the test split (target ≥ 80%; TTA adds ~0.3–0.5%).

## Option A — Kaggle (recommended)

1. kaggle.com → sign up → verify phone (required for GPU access).
2. Create Notebook → Settings (right panel) → **Accelerator: GPU T4 x2**,
   **Persistence: Files** (keeps `/kaggle/working` between sessions).
3. **Add Input → search "food41" (by kmader) → attach.** The script auto-detects it at
   `/kaggle/input/food41` and skips the 5 GB download entirely. (If you skip this step it
   downloads via torchvision instead — also fine, ~10 min per session.)
4. File → Upload `food101_train.py` into the notebook's working dir, then run:
   ```
   !python food101_train.py --data-root /kaggle/tmp/data --ckpt-dir /kaggle/working/ckpt --workers 2
   ```
   (Batch size auto-scales to 16 per GPU; on T4×2 that's 32 with grad-accum ×2 = effective 64.)
5. "Save Version → Save & Run All (Commit)" lets it run up to 12 h unattended (even with
   your browser closed). Checkpoints persist in `/kaggle/working/ckpt`.
6. If it stops before finishing, run the same command again — it resumes automatically.
7. When training ends it auto-exports; you can also re-run with `--export-only` any time to
   evaluate on the test split and produce `food101_efficientnet_b4.pth` +
   `labels_food101.json` + `artifacts/metrics.json` — download all of them.

## Option B — Free Colab

1. Upload `food101_train.py` to Google Drive at `MyDrive/nutrilens/food101_train.py`.
2. Open `food101_colab.ipynb` in Colab → Runtime → Change runtime type → **T4 GPU**.
3. Run all cells. Keep the tab open and interact occasionally (free tier disconnects idle sessions).
4. After every disconnect: reconnect → Run all again. It re-downloads the dataset (~10 min)
   and **resumes from your Drive checkpoint** — no progress lost.
5. Repeat daily until early stop / 25 epochs, then run the export cell.

## After training (either option)

1. Check `artifacts/metrics.json` → `test_top1` must be ≥ 0.80 (if not, see spec `04 §9` fallbacks).
2. Create a GitHub Release `model-v1.0.0` on the project repo; attach
   `food101_efficientnet_b4.pth` and `labels_food101.json`; put `{test_top1, test_top5, date}`
   in the release notes.
3. The backend downloads these into `backend/models/` at deploy time (spec `01 §3`, `08 §10`).

## Watch out for

- **OOM (CUDA out of memory):** batch auto-scales to 16/GPU with grad-accum ×2. If it
  still OOMs: `--batch-size 8 --grad-accum 4`.
- **First val score looks low:** normal — warm-up only trains the head (~55–65%). Accuracy
  jumps once fine-tune unfreezes the backbone.
- **Training loss looks "stuck" around 2–3 with MixUp/CutMix:** normal — mixed soft targets
  keep the loss high by design. Watch **val top-1**, not train loss.
- **`ema` beats `raw` in the log:** expected — the EMA weights are what get exported.
- **Never delete `checkpoints/split_val.json`** — it pins the val split so resumed runs and
  the final report stay consistent.
