# 04 — ML Model: EfficientNet-B4 on Food-101

**Decision (locked):** we train the classifier ourselves via transfer learning. This is the core
academic contribution. This doc covers the full pipeline: data → training → evaluation → export →
serving. Training runs **offline** (Colab Pro / Kaggle GPU); serving runs **on the backend**.

Target: **top-1 ≥ 80%** on the Food-101 test split (NFR Accuracy). Documented fallback path if we
miss it (see §9).

---

## 1. Dataset — Food-101

- 101 classes × 1,000 images = **101,000** images.
- Official split: **750 train / 250 test per class** (75,750 train / 25,250 test). Use the official
  `train.txt` / `test.txt` split files — do **not** random-split (breaks comparability with
  published results, report §3.2.2).
- Carve a **validation** set from train: hold out 100 images/class (→ 650 train / 100 val / 250 test
  per class). Keep it fixed (seed = 42) and store the split lists in `training/configs/split_val.txt`.
- Known dataset quirk: Food-101 train labels contain some noise; that is expected and part of the
  benchmark. Do not attempt to clean it for GP2.

Download: `torchvision.datasets.Food101(root, split="train"/"test", download=True)` — but override
with the official split files for the val carve-out.

## 2. Preprocessing & augmentation

Input size: **380 × 380** (EfficientNet-B4's native resolution; matches `preprocess.py` in serving).

Training transforms (torchvision):
```python
train_tf = T.Compose([
    T.RandomResizedCrop(380, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
])
eval_tf = T.Compose([
    T.Resize(400), T.CenterCrop(380), T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
Augmentation rationale ties directly to risk **R1** in the report (under-represented dishes):
rotation/brightness/crop expand effective coverage.

## 3. Model

```python
import torchvision.models as M
model = M.efficientnet_b4(weights=M.EfficientNet_B4_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features            # 1792
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4, inplace=True),
    nn.Linear(in_features, 101),                         # 101 Food-101 classes
)
```

### Two-phase transfer learning
1. **Warm-up (freeze backbone):** train only the new head for **3 epochs**, LR = 1e-3. Lets the head
   adapt without wrecking pretrained features.
2. **Fine-tune (unfreeze):** unfreeze all layers, train **~25 epochs** with a low LR and cosine decay.

## 4. Hyperparameters (`training/configs/food101_b4.yaml`)

| Param | Value |
|-------|-------|
| image_size | 380 |
| batch_size | 32 (reduce to 16 if VRAM-limited; use grad-accum ×2) |
| optimizer | AdamW |
| head warm-up LR | 1e-3, 3 epochs, backbone frozen |
| fine-tune LR | 3e-4 with cosine annealing to 1e-6 |
| weight_decay | 1e-4 |
| epochs (fine-tune) | 25 (early-stop on val top-1, patience 5) |
| label_smoothing | 0.1 |
| loss | CrossEntropyLoss(label_smoothing=0.1) |
| mixed precision | AMP (`torch.cuda.amp`) — ~2× speed, less VRAM |
| grad clipping | max_norm 1.0 |
| seed | 42 (torch, numpy, random all seeded) |

## 5. Training loop (essentials for `training/food101_train.py`)

```python
scaler = torch.cuda.amp.GradScaler()
for epoch in range(epochs):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
    scheduler.step()
    val_top1, val_top5 = evaluate(model, val_loader)     # see §6
    save_checkpoint_if_best(model, val_top1, epoch)      # keeps best + last
    log_to_tensorboard(epoch, loss, val_top1, val_top5)
```
Checkpointing: save `best.pth` (highest val top-1) **and** `last.pth` every epoch to Google Drive so a
Colab disconnect never loses progress. Each checkpoint stores `{epoch, model_state, optimizer_state,
val_top1, class_to_idx}`.

## 6. Evaluation & metrics

Report on the held-out **test** split (never touched during training/tuning):
- **Top-1 accuracy** (primary, target ≥ 80%).
- **Top-5 accuracy** (sanity; EfficientNet-B4 on Food-101 typically ≥ 95%).
- **Per-class accuracy** → identify the weakest classes (feeds report discussion + R1 mitigation).
- **Confusion matrix** (101×101 heatmap saved to `training/artifacts/confusion.png`).
- **Macro F1** for balanced view.

Deliverables saved to `training/artifacts/`: `metrics.json`, `confusion.png`, `per_class_acc.csv`,
`training_curves.png`. These go straight into the GP2 report's results chapter.

## 7. Export for serving

Two acceptable formats — pick TorchScript for portability:
```python
model.eval()
example = torch.randn(1, 3, 380, 380)
traced = torch.jit.trace(model, example)
traced.save("food101_efficientnet_b4.pth")     # goes to backend/models/
# also dump labels
json.dump(idx_to_class, open("labels_food101.json", "w"))
```
Version the artifact via **GitHub Releases** (e.g. `model-v1.0.0`), not git-tracked (`01 §3`).
Record `{git_sha, val_top1, test_top1, date}` in the release notes for rollback/audit (Maintainability NFR).

## 8. Serving — `backend/app/ml/classifier.py`

Loads once at startup (FastAPI lifespan), runs on CPU by default (`DEVICE=cpu`) or CUDA if available.

```python
class FoodClassifier:
    def __init__(self, weights_path: str, labels_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = torch.jit.load(weights_path, map_location=self.device).eval()
        self.idx_to_class = json.load(open(labels_path))
        self.tf = build_eval_transform()          # same 380 center-crop + ImageNet norm as §2

    @torch.inference_mode()
    def predict(self, pil_image, top_k: int = 3) -> list[Prediction]:
        x = self.tf(pil_image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        conf, idx = probs.topk(top_k)
        return [Prediction(food101_class=self.idx_to_class[str(i.item())],
                           confidence=round(c.item(), 4)) for c, i in zip(conf, idx)]
```
Contract with `/recognise` (`03 §4`): top-1 if `confidence ≥ CONFIDENCE_THRESHOLD`, else return top-3
as `alternatives` with `low_confidence: true`.

Performance: on CPU, B4 inference is ~0.3–1.0 s per image — within the 5 s NFR budget when combined
with the portion pipeline. If latency is a problem in eval, allow `DEVICE=cuda` on the deploy VM or
switch the served model to EfficientNet-B0 (documented trade-off, retrain with same pipeline).

## 9. Fallback path (if top-1 < 80%)

If, after tuning, the model misses 80% on test:
1. Train **longer** (more fine-tune epochs) and/or add **TenCrop TTA** at eval.
2. Try **EfficientNet-B5** (higher accuracy, more compute) or **EfficientNetV2-S**.
3. As a last resort, load a **published pre-trained Food-101 checkpoint** and fine-tune from it.
Document whichever path was used in the results chapter. The serving interface (`§8`) is unchanged.

## 10. Multi-label (future — GP2 stretch, not core)

The report notes multi-component dish decomposition as a planned extension. For GP2 core, ship
**single-label top-1/top-3**. If time allows, add a sigmoid multi-label head trained on a
multi-label corpus; keep it behind a feature flag so the core loop is never blocked (report §5.3.5).

---

Prev: [`03-api-specification.md`](./03-api-specification.md) · Next: [`05-portion-estimation.md`](./05-portion-estimation.md)
