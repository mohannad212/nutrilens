# Training — Portion aux models · spec `05 §3, §6, §7`

Two aux models for the portion pipeline, each trained the same way (transfer
learning → TorchScript → drop into `backend/models/`):

| Model | Stage | Files | Output |
|---|---|---|---|
| **Container classifier** (bowl/plate) | 1 (`§3`) | `container_train.py` · `container_colab.ipynb` | `container_classifier.pth` + `labels_container.json` |
| **Food segmenter** (food/bg) | shared (`§6`) | `segmenter_train.py` · `segmenter_colab.ipynb` | `food_segmenter.pth` |

(The Stage-2 reference detector — fork/spoon/knife — needs **no training**: it's an
off-the-shelf COCO model the backend downloads itself.)

---

## Container classifier (bowl vs plate)

Stage 1: a tiny **MobileNetV3-Small** that classifies the container the food sits
in. Scoped to **2 classes** — `bowl` and `plate` — to fit the deadline. Same
transfer-learning recipe as the food model (`04`), minutes to train.

### Files
- `container_train.py` — one script: collect data · train · export TorchScript.
- `container_colab.ipynb` — put `container_train.py` in Drive, *Run all*.

## Data — collected automatically, no manual labelling
`--collect` pulls images that already carry `Plate` / `Bowl` **bounding boxes**
from **Open Images V7** (via FiftyOne), crops each box, and writes an ImageFolder:
```
data/containers/{train,val}/{bowl,plate}/*.jpg
```

## Run (Colab / Kaggle, a T4 is plenty)
```bash
pip install -q fiftyone
python container_train.py --collect --per-class 1500   # ~few min, ~1500 crops/class
python container_train.py --train                       # ~5–10 min on a T4
python container_train.py --export                      # TorchScript + labels
```
Outputs land in `./artifacts`:
- `container_classifier.pth` — TorchScript (load with `torch.jit.load`, like `04`)
- `labels_container.json` — `{"0": "bowl", "1": "plate"}`
- `metrics.json` — `val_acc`, per-class accuracy

## How it plugs into the backend (`app/ml/portion/`)
The `ContainerModel` protocol needs `classify(image) -> (class, confidence)`; this
model provides it. Map its output to a size prior in `container_priors.py`:

| model output | container prior to use |
|---|---|
| `plate` | `dinner_plate` |
| `bowl`  | `bowl_large` |

The **rim pixel area** the scale maths needs (`cm_per_px = √(A_real/A_px)`) is a
separate classical-CV step — already built in `app/ml/portion/container_model.py`
(`RimAreaDetector`). So dropping `container_classifier.pth` + `labels_container.json`
into `backend/models/` is all it takes to upgrade `shape_only` → `container_prior`.

### Notes / watch-outs
- If a class comes back with very few crops, raise `--per-class` or check the
  network — Open Images downloads can be flaky.
- 2 classes on a small backbone should reach high val accuracy fast; if it plateaus
  low, the crops are probably noisy (increase `MIN_BOX_PX` / `--per-class`).
- Attach the exported files to a GitHub Release like the food model (`04` README),
  so the backend can pull them at deploy time.

---

## Food segmenter (food vs background)

Shared prerequisite (`§6`): a **DeepLabV3-MobileNetV3-Large** producing a binary
food/not-food mask, so the estimator measures the *food's* area — not the plate's.
Recognition already named the dish, so we don't need per-food masks; every food id
collapses to one "food" label.

### Files
- `segmenter_train.py` — train · export TorchScript.
- `segmenter_colab.ipynb` — put `segmenter_train.py` in Drive, *Run all*.

### Data
FoodSeg103 (~7k images, pixel masks) via HuggingFace `datasets`
(`DATASET_ID = "EduardoPacheco/FoodSeg103"`). Masks are binarised (any id > 0 →
food). If that id is unavailable, swap it for another food-seg dataset with an
image + integer-mask column.

### Run (Colab, T4)
```bash
pip install -q datasets
python segmenter_train.py --train     # ~20-40 min
python segmenter_train.py --export    # -> food_segmenter.pth
```
Output: `food_segmenter.pth` (TorchScript) + `metrics.json` (`food_iou`).

### How it plugs into the backend
Drop `food_segmenter.pth` into `backend/models/`. The lifespan loads it via
`app/ml/portion/segmenter.py` and it replaces the classical GrabCut fallback —
no code change. Serving resizes to 512×512, argmaxes 2 channels, and resizes the
mask back to the original resolution.
