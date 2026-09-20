# 13 — Portion accuracy backlog (post-audit build order)

**Status:** audit done and fixes landed; this document is the build order for what remains.
**Written:** 2026-08-28, against commit `8934d86` plus uncommitted audit work.
**Updated:** 2026-09-03 — §5a, §1, §2 built; §4 attempted and rejected on
measurement; §3 built but disabled. Accuracy unchanged at MAPE 18.2%. See
*WHERE THIS STANDS* at the end before starting anything.
**Prerequisite:** the 2026-08-28 audit changes in `backend/app/ml/portion/`
(rim-detector rewrite, tilt correction, pile cap, class retunes) — everything
below assumes those are in the tree.

> **Read `backend/data/weighed/README.md` first.** It holds the ground-truth set,
> the shoot protocol, and the per-photo findings this document is derived from.
> Every claim here traces to a weighed photo; nothing is from intuition.

---

## WHY THIS FILE EXISTS

The 2026-08 audit took portion MAPE from 26.9% to 18.8% and range coverage from
66.7% to 100% (then 75% once a fourth, harder photo joined the set). Along the
way it produced more *findings* than *fixes* — several confirmed defects were
deliberately left alone because fixing them without evidence would have been
guessing. This file records them so the reasoning is not lost, and orders them by
measured impact rather than by how interesting they are.

The headline correction from the audit: **the system is currently accurate
because it degrades honestly, not because all three stages work.** Stage 3 is
solid, Stage 2 works when cutlery is present, and Stage 1 is unreliable.

---

## THE MEASURED BASELINE (do not re-derive)

Weighed photos in `data/weighed/manifest.csv`, run against the shipped pipeline.
**Current: MAPE 18.2%, coverage 4/6 (66.7%)** — see the fries entry below.

| photo | true | estimate | scale used | error | limited by |
|---|---|---|---|---|---|
| donuts + fork | 203 g | 199 g | `reference_object` (fork) | **1.9%** | — |
| donuts on scale | 203 g | 219 g | `container_prior` | 7.7% | — |
| fries + fork | 169 g | 145 g | `reference_object` (fork) | 14.1% | band misses by 6 g |
| chicken wings | 275 g | 326 g | `shape_only` | 18.4% | rim not found |
| fries on scale | 169 g | 203 g | `container_prior` | 20.1% | camera tilt |
| spaghetti | 316 g | 382 g | `shape_only` | 20.9% | rim not found |
| lentil soup (bowl) | 242 g | 310 g | `reference_object` (fork) | 28.1% | soup depth prior |

### 2026-08-30 — `french_fries` was the worst class in the table, by 10x

The two fries photos above are new, and on the pipeline as shipped they read
**484 g and 677 g against a weighed 169 g** (the 677 clipped to the 500 g clamp,
so the app showed 3.0x where the truth was 4.0x). Both stages upstream were
right: the class was correct at 0.76–0.79, and the two shots independently
resolved almost the same scale (0.0150 vs 0.01525 cm/px) from different rungs.

The defect was the shape model. `french_fries` was `IRREGULAR_PILE` at `k=0.4`,
and `h_eff = k·√A` hits `PILE_MAX_HEIGHT_CM` for any area over **156 cm²** — so
every normal plated serving was modelled as a uniform 5 cm deep heap. Fries
spread; they do not mound to 5 cm. Moved to `FLAT_DISC h=1.5` (the same remedy
`fried_rice`/`pad_thai` got in the 2026-08 audit, reached from the other end).
**Set MAPE 76.2% → 18.2%, coverage 50% → 66.7% on the identical six photos.**

`h` is the effective uniform depth; density 0.55 already accounts for the air
between fries. The two shots bracket it — 1.75 cm from the fork rung, 1.25 cm
from the tilted plate rung — so 1.5 splits them (−14% / +20%) rather than
fitting either exactly. One weighed sample does not earn more precision.

**Two things this leaves open:**
- **The fries + fork band misses by 6 g** (169 vs a `[127.7, 162.5]` band). The
  ±12% reference-object band is the tightest in the pipeline and the point is
  14% low. Do not widen the band to cover it — get more weighed fries rows and
  re-fit `h` instead.
- **`caesar_salad` and `greek_salad` are the same latent defect, unmeasured.**
  Both are `IRREGULAR_PILE` at `k=0.5`, which pins at the 5 cm ceiling above
  only **100 cm²** — a smaller serving than fries needed. They are prime
  suspects for the same 3–4x over-read. Weigh one before touching either.

Run it with:

```bash
cd backend && .venv/bin/python scripts/eval_portion_mape.py \
    --manifest data/weighed/manifest.csv --images-dir data/weighed/images \
    --out artifacts/mape.csv
```

**This is the regression gate. Run it before and after every change below.**

---

## DECISIONS ALREADY MADE (do not re-litigate)

- **Segmentation is fine — do not touch it.** Verified on all four real food
  photos; the spaghetti overlay was strand-accurate and correctly excluded the
  plate rim. It also correctly returned *zero* food on a photo of a laptop.
- **The reference detector works.** Every photo with a fork in frame produced a
  sane scale. It is under-built (§2), not broken.
- **Rejecting a doubtful rim is correct behaviour.** Falling back to the wide
  `shape_only` band is more honest than a confidently wrong ruler. Do not "fix"
  Stage 1 by loosening the rejection rules.
- **Morphological opening does not fix rim detection.** It was tried; it appeared
  to rescue the spaghetti photo, and visual inspection showed it had outlined the
  plate *plus the kitchen scale*. Do not retry it.
- **The three dead prior entries** (`burrito`, `fried_egg`, `mashed_potatoes`) are
  not in the 101-class label set and can never fire. Harmless; delete them if
  touching that file, but do not spend a weighed photo on them.

---

## THE WORK, IN PRIORITY ORDER

Priority is by **measured impact on the weighed set**, not by severity in the
abstract.

### 1. Credit-card reference detector — biggest win per hour, no training

**Why first.** It sidesteps Stage 1 entirely, and it is the only *exact*
reference available. `reference_sizes.py` already ranks `credit_card` priority 1
(above `fork` at 4) — the slot exists and is empty.

- Card: **85.60 mm exactly** (ISO/IEC 7810 ID-1), zero variance, every bank card
  / ID / licence on earth.
- Fork: 19 cm *assumed*, real forks run 17–21 cm → ~10% length variance →
  **~20% mass error**, since mass scales with area.

**Approach — classical geometry, no model.** COCO has no card class, and a card
does not need one: a fork needs ML because it is a *shape*; a card is a rigid
rectangle with a fixed aspect ratio of **1.5857**.

1. Canny edges (several thresholds — a card may be lighter or darker than the
   surface) → contours.
2. `approxPolyDP` → keep 4-sided convex contours.
3. `minAreaRect` → accept only if long/short is within **6%** of 1.5857.
4. Reject anything whose long edge is under **80 px** — too small to be a useful
   ruler anyway.
5. Report the long edge in px; `reference_sizes.py` already maps it to 8.56 cm.

**Measured on the existing photos:** step 4 is what matters. At 12% tolerance the
kitchen scale's display panel (ratio 1.42) was detected as a card in the donut
photo — a confidently wrong ruler. At 6% + 80 px: **zero false positives across
all five photos**, and a synthetic card was located to within **1–2%** of its true
size. Prototype: `backend/scripts/card_reference_prototype.py` (standalone, run it
against any photo; it is a prototype, not wired into the pipeline).

**Blocked on:** one real photo of a card on the counter (numbers covered), ideally
one top-down and one slightly tilted. The tuning above was validated against a
*pasted flat rectangle* — a real card has rounded corners, glare, shadows and
perspective skew that shifts the apparent ratio off 1.5857. Loosening tolerance
to catch tilted cards brings the false positives back; that tension is the whole
engineering problem and must be tuned against a real image.

**Do not:** tune the tolerance against synthetic images; ship without a
false-positive check over the whole weighed set; use COCO `cell phone` as a
substitute (phone sizes vary — the existing code skips it deliberately, and that
comment is correct).

#### BUILT 2026-09-03 — shipped enabled, but its hit rate is still unproven

`CardReferenceDetector` + `CompositeReferenceDetector` in `reference_detector.py`,
wired in `main.py` and in the MAPE harness (both, so the harness cannot drift from
production again). Config: `card_reference_enabled` / `card_ratio_tol` (0.06) /
`card_min_long_px` (80). It runs independently of the COCO detector and survives
that one failing to load, since it needs no weights.

**The false-positive check the section demanded is now a script and a test.**
`scripts/card_fp_audit.py` runs the detector over all six weighed photos — every
one a negative — and exits non-zero on any detection;
`test_card_detector_finds_no_card_in_the_weighed_photos` locks the same result in
CI. It reproduces the audit's finding exactly:

| setting | result over the 6 weighed photos |
|---|---|
| shipped (6% + 80 px) | **0 detections** |
| rejected (12%, no px floor) | 2 detections on `donuts_203`, ratio **1.418** — the kitchen scale's display panel |

**Accuracy when it does fire** (flat pasted rectangles, the same synthetic method
the prototype used — used only as a sanity check, not to tune anything): long edge
recovered to within **1.7%** at 90/120/160/260 px.

**Honest gaps, both unchanged by this work:**
- **Recall is patchy and unmeasured on real cards.** In the synthetic sweep it
  missed 200 px and 320 px cards outright while finding 90/120/160/260 — the
  contour simply fails to close against some backgrounds. It fails in the *safe*
  direction (miss, not hallucinate), but "it will find your card" is not yet a
  claim the evidence supports.
- **Still blocked on a real card photo**, exactly as this section said. Rounded
  corners, glare and perspective skew are not in a pasted rectangle. Nothing here
  changes that; it means the detector is built and safe, not that it is validated.
- A card over **8% of the frame area** is rejected by rule 1 (at that size it is
  the table, not a card beside a meal). Fine for meal photos, worth knowing.

**Effect on the weighed set: none — MAPE 18.2%, coverage 66.7%, unchanged.** No
photo in the set contains a card. The value of the measurement is that it did not
*regress*: no false card appeared to hijack a scale.

---

### 2. Fix the reference table's dead entries

`reference_sizes.py` advertises 8 references; only **3** are reachable
(`fork`/`spoon`/`knife` via COCO). `credit_card`, `coin`, `smartphone`,
`chopsticks` and `hand` can never fire.

**Approach:** implement `credit_card` (§1), then **delete the rest** or mark them
explicitly as unimplemented. Code that advertises capability it lacks is worse
than code that admits the gap — and a reader of the report will check.

**Do not:** implement `hand` (width varies hugely between people) or `coin`
(needs a country-specific model, and a coin is often too small in frame to
measure precisely).

#### DONE 2026-09-03

`REFERENCE_SIZES` now contains only what a detector can actually emit —
`credit_card` (§1) plus `fork`/`spoon`/`knife` (COCO). The four rejected
references moved to `UNIMPLEMENTED_REFERENCES`, each carrying its measured size
*and* the reason it is not built, so the research stays visible without
masquerading as a feature. `size_for()` and `best_reference()` never consult that
dict, so no detector can accidentally resolve a scale from one.

The priority integers are deliberately left at their `05 §4` values rather than
renumbered 1..n: the gaps (2, 3, 5, 6) are themselves evidence that the classes
were considered and rejected. Two tests lock it —
`test_reference_table_lists_only_detectable_classes` derives the expected set from
the detectors themselves, so adding a reference without a detector fails CI.

---

### 3. `LIQUID_IN_BOWL` fill depth — caused the worst measured error

**The finding.** The soup photo: 242 g over a measured 155 cm² surface implies a
real fill depth of **1.56 cm**. The priors assume **2.0–5.0 cm**. Had the dish
been classified as an actual soup class rather than `panna_cotta`, it would have
computed **620 g** — 2.6× over.

**Why depth is the hard case.** It is genuinely unobservable from a top-down
photo, and fill level varies far more than any other shape parameter. This is the
weakest model in the table and always will be.

**Approach — derive depth, do not re-guess it.** In a tapered bowl, a *smaller*
liquid surface means the liquid sits lower and is therefore shallower. So:

```
fill_ratio = food_surface_area / container_rim_area
depth      = h_max * f(fill_ratio)      # monotonic, f(1.0) -> full depth
```

This turns an unobservable into something measured from two areas we already
compute.

**Blocked on:** Stage-1 rim detection (§4), which is exactly what failed on the
soup photo. Until then the class clamps are doing the real work — leave them.

**Do not:** lower the `LIQUID_IN_BOWL` height priors to make the soup photo come
out right. n=1, and a normal full bowl genuinely *is* deeper than that shallow
serving. Tuning to it would break the common case. Collect two or three more
weighed bowls first — one shallow, one full.

#### BUILT 2026-09-03 — mechanism implemented, SHIPPED DISABLED, still blocked

`shape_priors.derived_fill_depth_cm`, wired through `PortionEstimator` behind
`portion_derive_bowl_depth` (**default False**).

**The geometry has no free parameters, which is the point.** Model the bowl as a
hemispherical cap of rim radius `R`. Liquid at depth `d` has surface radius
`r² = d(2R − d)`, so the measurable `fill_ratio = r²/R² = u(2 − u)` with `u = d/R`,
which inverts exactly to

```
d = R · (1 − sqrt(1 − fill_ratio))        R = sqrt(A_rim / pi)
```

`fill_ratio` is a ratio of two areas, so it is scale-free and can be computed in
pixels. Nothing here is fitted — the model is fixed once the bowl shape is
assumed, so there is no constant anyone can quietly tune to a photo.

**Why a hemisphere, not the cone the section implied.** A cone gives
`d = h·sqrt(fill_ratio)`, and the two disagree sharply at exactly the shallow
fills that caused the problem. On the weighed soup bowl the cone predicts
**2.63 cm** and the hemisphere **~2.0 cm**, against a measured **1.56 cm**. The
hemisphere was chosen because real crockery is rounded rather than conical — the
data point is consistent with that, but it is one point and did not decide it.

**Why it ships off, and this is not hedging.** The formula needs a container rim.
§4 was the attempt to supply one and it *failed its own measurement* (below). The
single weighed bowl produces no rim from the Otsu detector at all — so the 2.0 cm
figure above had to be computed from the §4 growth rim, which is itself
unvalidated. There is currently **no way to check this against ground truth**, and
switching it on would be tuning an unobservable to n=1. It stays disabled until
the two or three weighed bowls this section asks for exist. Its geometry is
unit-tested (closed form, monotonicity, and the cases where it must decline);
`test_bowl_depth_derivation_is_off_by_default` locks the status so it cannot drift
on by accident.

---

### 4. Stage-1 rim detection — biggest measured loss, hardest fix

**The finding.** 3 of 5 photos get no container scale. The detector is classical
Otsu thresholding: it works only when the plate strongly contrasts with the
background. A black plate on granite works; a **white plate on speckled granite
fails**, because the bright grains bridge the plate to the frame edge into one
blob that the border rule (correctly) rejects.

**Approach — use the trained segmenter, not Otsu.** The food segmenter already
understands crockery and reaches IoU 0.925 on food. Either (a) add a *container*
class and retrain it, or (b) exploit the fact that the food mask's surroundings
within the same connected surface are the container. This is the real fix, and
the only one likely to survive varied backgrounds.

**Do not:** attempt further classical-CV rescues. Otsu + contours cannot separate
a white plate from a light speckled counter — this was tested to exhaustion
during the audit. Do not loosen the border-touch or containment rules to raise
the hit rate: they are what stopped the countertop being used as a ruler on 3/3
photos before the audit.

#### ATTEMPTED 2026-09-03 — approach (b) built, measured, and REJECTED

`SurfaceGrowthRimDetector` + `ChainedRimDetector` in `container_model.py`, behind
`rim_surface_growth_enabled` (**default False**). Keeping the code, because the
negative result is worth more than the disk space and the *approach* is still the
right idea — but it does not ship.

**What was built.** Approach (b) as this section describes it. Rather than asking
"which pixels are plate?" globally, seed from the trained segmenter's food mask
(IoU 0.925), sample a thin ring *just outside* it — which is the container by
construction, since food rests on its container — and grow the connected region
matching that ring's colour in CIELAB. No image-wide statistic is involved, which
is what made Otsu brittle. It inherits every one of the old detector's safety
rules (border touches, area fraction, must-be-bigger-than-the-food) and runs
strictly *behind* Otsu, so it can only add rims, never change one.

**The colour tolerance is not a constant, deliberately.** A single fixed tolerance
proved to be a knife-edge: the grown area on the spaghetti photo moves
27% → 40% → 42% → nothing across the plausible range, and every value that rescued
one photo lost another. Picking the one that made six photos resolve would have
been fitting a constant to the test set. Instead the ladder is swept and the
**largest region that still passes every safety rule** wins — grow until it
spills, take the last size before it spilled — which needs no constant at all.

**It works, in the sense that it finds rims: coverage 3/6 → 6/6.**

**And it is worse, which is the finding:**

| | MAPE | range coverage |
|---|---|---|
| §4 off (shipped) | **18.2%** | **66.7%** |
| §4 on | 37.5% | 33.3% |

Only the three previously-`shape_only` rows moved, confirming the additive design
held. `wings` 326 g → **600 g** (clipped at the clamp; true 275 g).
`spaghetti` 382 g → 433 g (true 316 g). `soup` was unchanged because the fork
outranks any container.

**Why — established by looking at the overlays, not inferred**
(`backend/artifacts/rim_overlays/`, and this is the section's own first rule):
- **wings** — the grown hull **hugs the food** and never reaches the plate rim. A
  rim read too small makes the scale too coarse and the mass triples.
- **spaghetti** — the hull spills past the plate and **swallows the kitchen scale**
  underneath it. This is the *identical* failure the audit recorded when
  morphological opening appeared to "rescue" this same photo. Two independent
  methods, one failure mode: on this photo the plate does not separate from what
  is under it by colour, full stop.

The two errors run in **opposite directions**, so no single correction fixes both.

**And it is not a container-size problem.** The obvious rescue — "the rim is right,
the 27 cm `dinner_plate` prior is wrong" — was tested by re-running both photos
against every prior in the table:

| photo | 27 cm | 20 cm | 18 cm | 13 cm |
|---|---|---|---|---|
| wings (275 g) | 276% | 86.9% | **36.2%** | 48.7% |
| spaghetti (316 g) | 36.9% | **24.9%** | 39.1% | 68.3% |

No size makes either acceptable, and the best sizes disagree. The rim itself is
wrong.

**What this means for the section.** The colour-homogeneity assumption fails on a
scalloped white paper plate on a granite counter — the ring around the food and
the plate's outer rim are not one colour region. That is a limit of *classical*
region growth, not of the idea. **§4(a) — give the segmenter a container class and
retrain — is now the only remaining candidate**, and this result is the argument
for it: two different classical methods have now failed on the same photo for the
same reason.

**Do not** switch `rim_surface_growth_enabled` on without a weighed set that shows
it helping. `test_surface_growth_stays_off_by_default` locks that.

---

### 5. Bowl/plate classifier — confirmed wrong, currently latent

**The finding.** The soup bowl was classified `dinner_plate` at **0.809**. A
bowl/plate mixup is a **2.25×** mass error (27 cm vs 18 cm prior). The model said
"plate" on all five photos at 0.81–0.99 confidence.

**But note the priority:** `container_prior` drove the final number on exactly
**one** of five photos — and on that one it was *correct*. On the soup, the fork
outranked it. **The classifier has never yet changed a measured result.** It is a
landmine for the first fork-less bowl photo, not a present error. Rank it
accordingly.

**Root cause, established:**
- *Not* preprocessing — the 224×224 the model sees shows the bowl clearly.
- There **is** a train/serve mismatch: `container_train.py` trains on cropped
  Open Images *bounding boxes*, but is served whole scenes. Cropping to the food
  bbox at inference moves bowl 0.19 → 0.36 — real, but never enough to flip.
- The model is simply biased toward "plate". `metrics.json` (which the script
  *does* write, including per-class accuracy) was never committed, so the
  bowl-recall behind the quoted 0.874 val_acc is unknown.

**Approach, in order:**

**5a. Propagate the doubt into the band — DONE 2026-09-02.** Keep
argmax but widen the band by the classifier's own uncertainty:

```
widen = 1 + (1 - confidence) * (2.25 - 1)
```

At p=0.99 → ×1.01 (free when confident). At p=0.81 → ×1.24. A full mixture band
spanning bowl-min to plate-max is **3.5× wide** — too wide to show a user, which
is why committing-and-widening is the right trade. This is cheap, principled, and
defensible in the report.

Shipped in `container_priors.confidence_widen` / `widen_for_confidence`, applied
in `PortionEstimator._container_scale` on top of the range-derived band. Covered
by `test_container_band_widens_when_the_classifier_is_unsure` (the point estimate
must not move — this is a claim about uncertainty, not a re-estimate).

**Measured effect on the weighed set: MAPE 18.2% → 18.2%, coverage 66.7% → 66.7%.**
Nothing moved, and that is the expected result, not a disappointment: the widening
only touches `container_prior` rows, and the classifier is confident on both of
them (donuts 0.988 → ×1.014, fries-on-scale 0.951 → ×1.061). The two uncovered
rows are both `reference_object`, which this does not touch. The value is
insurance against the first fork-less bowl photo, and it is deliberately priced so
that the confident case pays ~1% for it.

**New finding — the "always plate" story is now wrong.** Printing the classifier's
own output over all six weighed photos:

| photo | container | conf | widen |
|---|---|---|---|
| donuts_203 | dinner_plate | 0.988 | ×1.014 |
| **fries_fork_169** | **bowl_large** | **0.588** | **×1.516** |
| fries_scale_169 | dinner_plate | 0.951 | ×1.061 |
| soup_bowl_242 | dinner_plate | 0.809 | ×1.239 |
| spaghetti_316 | dinner_plate | 0.973 | ×1.034 |
| wings_275 | dinner_plate | 0.975 | ×1.032 |

`fries_fork_169` is a flat plate read as `bowl_large` at 0.588 — the mixup in the
*opposite* direction from the soup, on the newest photo. The audit's "the model
says plate on all five photos at 0.81–0.99" held for the five photos it had. So
the bias is not a clean one-way prior toward "plate"; the model is simply weak,
and its confidence is at least honest enough here (0.588 is the lowest in the
table, on the one it got wrong). It still never changed a measured number — that
photo resolves scale from the fork — so the §5 priority is unchanged.
Two of six container calls now look wrong, which strengthens the case for §0+§5c.

**5b. Retrain — locally, not on Colab.** Benchmarked on this machine:

| device | throughput | full 10-epoch run |
|---|---|---|
| CPU | 13 img/s | ~33 min |
| **MPS (Apple GPU)** | **132 img/s** | **~3 min** |

MobileNetV3-Small is 2.5 M params. ~~**One-line blocker:** `container_train.py`
line ~206 reads `cuda if available else cpu` and never checks MPS, so it would
silently crawl on CPU. Fix that first.~~ **DONE 2026-09-02** — replaced by
`_pick_device()` (cuda → mps → cpu). Nothing else in the script is CUDA-only
(no AMP/GradScaler), so this was the whole blocker.

**5c. Change the taxonomy while retraining — this is the real fix.** bowl/plate
is the wrong abstraction; the maths needs *size*. `container_priors.py` already
has four: `dinner_plate` 27, `side_plate` 20, `bowl_large` 18, `bowl_small` 13.
Today a side plate is classified "plate" and scaled as 27 cm — a **1.8× error**
of exactly the same kind, sitting undetected because nothing tests it. A 4-class
size model maps straight onto priors that already exist.

**Do not:** retrain to reproduce the same 2 classes — that leaves the side-plate
error in place. Do not quote overall val_acc again; report **per-class recall and
a confusion matrix**, and commit `metrics.json` this time. Do not retrain at all
until §0 below exists, or there will be no way to tell whether it helped.

---

### 6. Recalibrate the `shape_only` frame-width prior

Assumes the photo frame spans **26 cm**. Both photos that used it came out high
by a consistent ~19% (wings +18.4%, spaghetti +20.9%), and working backwards each
independently implies **~24 cm**.

**Do not change it yet.** Two points is not evidence, and both are the same
photographer, phone and counter. Revisit once the weighed set has ~10 shape_only
rows. Recorded here so the observation is not lost.

---

## §0 — THE PREREQUISITE FOR §5: A LABELLED CONTAINER SET

You cannot fix what you cannot measure, and the classifier's real per-class recall
is currently unknown.

**Collect:** ~15 bowls and ~15 plates from crockery you already own — varied
colours and sizes, on your normal counter, shot like a real meal photo. Label
each `bowl` or `plate` (and ideally by size class for §5c). This is an afternoon
and it converts "the model seems biased" into a number.

**Also grow `data/weighed/`** per its README — the P1 list there targets exactly
the classes the audit changed *without* measurement (`ice_cream`,
`french_fries`, `fried_rice`, `pad_thai`, `spaghetti_carbonara`, `caesar_salad`,
`onion_rings`, `edamame`). Those are where a regression would hide.

**Do not** cook to obtain these — nearly all are buyable ready-made. The scale
matters, not the stove.

---

## SUGGESTED ORDER FOR THE NEXT FEW DAYS

| day | task | needs |
|---|---|---|
| ~~1~~ | ~~§5a band widening + `container_train.py` MPS one-liner~~ **done 2026-09-02** | nothing |
| ~~2~~ | ~~§1 card detector~~ **built 2026-09-03** — safe, but recall unproven | still wants the card photo |
| ~~2~~ | ~~§2 delete/mark dead reference entries~~ **done 2026-09-03** | — |
| ~~later~~ | ~~§4 rim detection~~ **attempted 2026-09-03, approach (b) REJECTED on measurement** | — |
| ~~later~~ | ~~§3 fill depth~~ **built 2026-09-03, ships disabled** — still blocked, §4 did not unblock it | weighed bowls |
| **next** | Photograph a card (§1) and ~15 bowls + ~15 plates (§0) | 20 min + an afternoon |
| then | §0 finish container set → measure current per-class recall | crockery |
| then | §5b/§5c retrain as a 4-class size model, locally | §0 set |
| then | §4(a) — add a container class to the segmenter and retrain | §0 set |

§6 still waits for data — do not touch it.

### WHERE THIS STANDS AFTER 2026-09-03

**The measured accuracy has not moved: MAPE 18.2%, coverage 66.7%, exactly the
2026-08-30 baseline.** Four sections were worked; none of them could move it, and
the reasons are individually sound:

- **§5a, §1, §2** are correctness and honesty work whose payoff is not in this
  metric. §5a only widens bands (the classifier is confident on both container
  rows); §1 finds no card because no photo has one; §2 changes no arithmetic.
- **§4 could have moved it and moved it the wrong way**, so it is off.
- **§3 cannot be evaluated at all** until there are weighed bowls.

**Everything that remains is blocked on photographs, and that is now the whole
critical path.** §5c needs the §0 crockery set; §4(a) needs it too; §3 needs
weighed bowls; §1's recall needs a real card; §6 needs ~10 shape_only rows. The
code-side backlog is finished — the next real gain comes from the camera, not the
editor.

Test count went 492 → 513. Regression gate and `scripts/card_fp_audit.py` both
pass.

---

## COMMON MISTAKES (all of these were nearly made during the audit)

- **Reporting a fix before looking at what the tool actually returned.** A
  "rescued" rim outline turned out to include the kitchen scale; a whole
  investigation ran against the wrong image file because the newest file in
  `~/Downloads` was assumed to be the right one. *Look at the artefact.*
- **Tuning a prior to a single photo.** Every class retune in the audit that was
  not backed by a weighed photo is flagged as unverified in
  `data/weighed/README.md` for exactly this reason.
- **Evaluating a different configuration than production.** The MAPE harness
  defaulted the reference detector *off* after it was enabled in config, scoring
  21.3% where production behaviour gives 18.8%. It now follows app config; keep
  it that way.
- **Trusting an overall accuracy number.** 0.874 val_acc hid a classifier that
  answers "plate" almost always.
- **Widening a range to hide an error.** The band must reflect a real, named
  source of uncertainty (§5a widens by the classifier's own confidence — that is
  legitimate). Widening merely to raise coverage is dishonest and will not
  survive a viva question.
