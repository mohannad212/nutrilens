# 05 — Portion Estimation: 3-Stage CV Pipeline

**Decision (locked):** build the **full 3-stage research pipeline**, not a stub. This is the hardest
technical piece in NutriLens and its most novel contribution. It estimates food **mass in grams**
from a single uncalibrated smartphone photo — no depth sensor, no reference card required.

**Output contract:** always a calibrated **range** `{min_g, max_g, point_g, method, reference}`
(never a fake-precise single number). Consumed by `/recognise` (`03 §4`). The user always confirms
or adjusts before nutrition is computed (FR06).

**Honesty principle:** each stage widens or narrows the uncertainty band. When a stage cannot run
(e.g. no reference object found), the pipeline **degrades gracefully** to a wider range and records
which `method` produced the estimate.

---

## 1. Why this is hard (state it plainly in the report)

A 2D photo has no absolute scale: the same pixel footprint could be a small bowl up close or a large
plate far away. Monocular depth from a single uncalibrated image is inherently ambiguous (report §3.1.5,
Im2Calories). We resolve scale using **known-size cues** in the scene and **food-class shape priors**,
then convert estimated **volume → mass** via food density. We never claim point precision; we return a
range and let the user confirm.

---

## 2. Pipeline overview

```
        input: preprocessed RGB image + predicted food class(es) + segmentation mask
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                           ▼
 STAGE 1                     STAGE 2                      STAGE 3
 Container-type              Reference-object             Food-class geometric
 classification             spatial calibration          shape priors
 "plate? bowl? cup?"        "fork/hand/coin → px per cm" "disc? hemisphere? slab?"
        │                          │                           │
        └──────────► scale estimate (cm/px) ◄──────────────────┘
                                   │
                     volume (cm³) = shape_model(area, height_prior, scale)
                                   │
                     mass (g) = volume × density(food_class)
                                   │
                     range = mass × [1 − ε_low , 1 + ε_high]   (ε from stage confidence)
                                   ▼
            output: { min_g, max_g, point_g, method, reference }
```

Segmentation (which pixels are *this* food) is a shared prerequisite — see §6.

---

## 3. Stage 1 — Container-type classification

**Goal:** classify the container/surface holding the food to obtain a **prior on physical scale**.

- Classes: `dinner_plate`, `side_plate`, `bowl_small`, `bowl_large`, `cup_mug`, `wrap_flat`,
  `takeaway_box`, `unknown`.
- Model: a small CNN (MobileNetV3-Small or EfficientNet-B0) fine-tuned on a container dataset
  (see §7 data). Output: class + confidence.
- Each container maps to a **typical diameter/width range** (cm) stored in
  `app/ml/portion/container_priors.py`:

| Container | Typical diameter/width (cm) | Range used |
|-----------|-----------------------------|------------|
| dinner_plate | 27 | 24–30 |
| side_plate | 20 | 18–22 |
| bowl_small | 13 | 11–15 |
| bowl_large | 18 | 16–21 |
| cup_mug | 8 (rim) | 7–9 |
| wrap_flat | 22 (length) | 18–26 |
| takeaway_box | 15 | 12–18 |
| unknown | — | fall through to Stage 2 only, wider ε |

**How scale is derived here:** detect the container ellipse/bounding region (OpenCV contour or a
lightweight segmentation), take its major-axis pixel length, and divide the typical diameter by it →
a *first* cm-per-pixel estimate. This is coarse (hence a range), refined by Stage 2 when possible.

## 4. Stage 2 — Reference-object spatial calibration

**Goal:** find a **known-size everyday object** in the frame to get a precise cm-per-pixel scale.
This is the strongest signal; when present, it dominates the estimate.

- Reference detector: an object detector (YOLOv8-nano or a fine-tuned SSD-MobileNet) trained to find:
  `fork`, `spoon`, `knife`, `hand`, `coin`, `credit_card`, `smartphone`, `chopsticks`.
- Known real-world dimensions in `app/ml/portion/reference_sizes.py`:

| Reference | Dimension used | Real size (cm) | Notes |
|-----------|----------------|----------------|-------|
| fork | length | 19.0 | Western table fork |
| spoon | length | 18.0 | Tablespoon |
| knife | length | 21.0 | Table knife |
| credit_card | long edge | 8.56 | ISO/IEC 7810 ID-1 — most reliable |
| coin | diameter | 2.4 | Configurable per region (default US quarter) |
| smartphone | long edge | 14.7 | Configurable; user's device known → high precision |
| hand | width across knuckles | 8.4 | Weakest; user can calibrate once in settings |
| chopsticks | length | 24.0 | |

**Scale computation:** `cm_per_px = real_size_cm / detected_pixel_length`. Prefer references in this
priority order (most reliable first): `credit_card > smartphone(known) > coin > fork/spoon/knife >
chopsticks > hand`. If several are present, use the highest-priority one and average with the next if
they agree within 15%.

**Precision/uncertainty:** a good reference gives a **tight** band (ε ≈ ±12%). If none is found, fall
back to Stage 1's container scale (ε ≈ ±25%); if neither, Stage 3 shape-only with a device-normalised
heuristic (ε ≈ ±35%) and `method = "shape_only"`.

## 5. Stage 3 — Food-class geometric shape priors

**Goal:** turn the food's pixel footprint + a scale into a **volume**, then into **mass**.

Each Food-101 class maps to a **geometric shape model** and a **density**. Store in
`app/ml/portion/shape_priors.py` and `densities.py`.

**Shape models** (pick per class; parameters = measured area A in cm² from the mask × scale, plus a
class height prior h):

| Shape model | Volume formula | Example classes |
|-------------|----------------|-----------------|
| flat_disc | `V = A × h`, h≈ small (0.8–1.5 cm) | pizza, pancakes, fried_egg, steak (slab) |
| dome / hemisphere | `V = (2/3)·π·r³`, r from A | scoop of rice, mashed_potatoes, ice_cream |
| ellipsoid | `V = (4/3)·π·a·b·c` | apple, dumplings, falafel, meatballs |
| cylinder | `V = π·r²·h` | sushi rolls, burrito, hot_dog |
| irregular_pile | `V = A × h_eff`, h_eff = k·√A (heaped) | salad, pasta, fries, noodles |
| liquid_in_bowl | `V = container_volume × fill_ratio` | soup, curry, cereal_with_milk |

Height priors and the `k` heap factor per class live in a table with a sensible default
(`h = 1.2 cm`, `k = 0.35`) for unlisted classes.

**Density table** (g/cm³) — `densities.py`, values from food-science references; examples:

| Food group | Density (g/cm³) |
|------------|-----------------|
| cooked rice / grains | 0.85 |
| lean cooked meat | 1.05 |
| leafy salad (loose) | 0.25 |
| bread / baked | 0.30 |
| soup / liquid | 1.00 |
| cheese | 1.10 |
| fried potato (fries) | 0.55 |
| default (unlisted) | 0.90 |

**Mass:** `mass_g = volume_cm3 × density`. Then apply the uncertainty band from the governing stage:
`min_g = mass·(1−ε_low)`, `max_g = mass·(1+ε_high)`, `point_g = mass`.

Clamp to sane bounds per class (e.g. a single fried egg ∈ [30, 90] g) to reject wild segmentation
errors — clamp table in `shape_priors.py`.

## 6. Shared prerequisite — food segmentation

To measure area A you need the food's pixel mask. Options, in order of preference for GP2:
1. **Lightweight semantic segmentation** (e.g. a fine-tuned `DeepLabV3-MobileNet` or `U-Net`) trained
   on food-vs-background + coarse food regions. Preferred.
2. **GrabCut / Otsu + contour** (classical OpenCV) as a no-training fallback — works acceptably on
   plated food with clean backgrounds; document its limits.
Store the mask in memory only; never persist. If segmentation fails, use the container interior as
the food region and widen ε.

## 7. Data & training for the auxiliary models (offline, `training/portion/`)

- **Container classifier:** assemble ~3–5k images across the 8 container classes (public tableware
  datasets + scraped/collected images, hand-labelled). Same transfer-learning recipe as `04` (smaller
  backbone). Save `container_classifier.pth`.
- **Reference detector:** fine-tune YOLOv8-nano on a small annotated set (~1–2k images) covering the
  reference objects; or use an off-the-shelf COCO detector for `fork/spoon/knife/cell phone` classes
  (already in COCO) and add `credit_card`/`coin` via a small custom head. Save `reference_detector.pt`.
- **Segmentation:** fine-tune on a food segmentation dataset (e.g. UEC-FoodPix, FoodSeg103) or use
  classical fallback if data/time is short.
- **Calibration validation set:** collect ~150 photos of foods **weighed on a kitchen scale** with a
  known reference object in frame. This is the ground-truth set to report **Mean Absolute Percentage
  Error (MAPE)** of the pipeline — the headline portion-estimation result for the GP2 report.

## 8. Serving interface — `app/ml/portion/__init__.py`

```python
@dataclass
class PortionEstimate:
    min_g: float; max_g: float; point_g: float
    method: str            # "reference_object" | "container_prior" | "shape_only"
    reference: str | None   # e.g. "dinner_plate", "fork", None

class PortionEstimator:
    def __init__(self, container_model, reference_detector, segmenter, tables): ...

    def estimate(self, image: "PIL.Image", food_class: str, mask=None) -> PortionEstimate:
        scale, method, ref = self._resolve_scale(image)      # Stage 1 + Stage 2
        region = mask if mask is not None else self._segment(image)   # §6
        area_cm2 = self._area(region, scale)
        volume = self._volume(food_class, area_cm2, scale)   # Stage 3 shape model
        mass = volume * density_of(food_class)
        eps_lo, eps_hi = uncertainty_for(method)
        mass = clamp_mass(food_class, mass)
        return PortionEstimate(round(mass*(1-eps_lo),1), round(mass*(1+eps_hi),1),
                               round(mass,1), method, ref)
```
Deterministic, side-effect-free, no disk writes. Unit-tested with synthetic masks + known scales
(see `11`).

## 9. Accuracy expectations & honesty in the report

- With a good reference object: MAPE typically **20–35%** — acceptable for a range estimate the user
  confirms. Report the real measured MAPE from §7's validation set.
- The report must state clearly: this is **assistive estimation**, the user is always in the loop
  (FR06 slider), and empirical accuracy is bounded by monocular ambiguity (report §3.1.5, R4).
- This framing is exactly what the committee expects: ambitious pipeline, honest evaluation.

## 10. Milestone note

The full pipeline is **M4**. To keep the core loop unblocked, build in this internal order and let the
API return whatever the best available stage yields:
1. Stage 3 shape-only + container prior (works with just segmentation) → `method="container_prior"`.
2. Add Stage 2 reference detection → upgrades to `method="reference_object"` when a reference is found.
3. Add trained segmentation model (replaces classical fallback).
Each increment strictly improves the range without changing the `/recognise` contract.

---

Prev: [`04-ml-model.md`](./04-ml-model.md) · Next: [`06-recommendation-engine.md`](./06-recommendation-engine.md)
