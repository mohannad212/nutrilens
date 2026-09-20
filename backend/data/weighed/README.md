# Weighed photo set — how to grow it, and what to shoot next

This folder is the ground truth for portion accuracy. `manifest.csv` +
`scripts/eval_portion_mape.py` turn it into the MAPE number the report quotes,
and it is the regression gate for any change to `app/ml/portion/`.

    .venv/bin/python scripts/eval_portion_mape.py \
        --manifest data/weighed/manifest.csv \
        --images-dir data/weighed/images \
        --out artifacts/mape.csv

Current set: 4 photos (wings 275 g, spaghetti 316 g, donuts 203 g, soup 242 g).
MAPE 18.8%, coverage 75%. The script's docstring suggests ~150 rows for a
stable number.

`soup_bowl_242.jpg` is filed as `lobster_bisque` because the dish (lentil soup)
has **no Food-101 class**. The label is a stand-in for "thick soup in a bowl" —
the right shape model and density family — so the row measures portion error, not
classification error. Keep that caveat in mind when quoting the number.

### What the bowl photo exposed (2026-08-28)

First bowl in the set, and it broke three things worth recording:

1. **The container classifier called an obvious square bowl a `dinner_plate`
   (conf 0.809).** First real bowl test, and it failed. It did no damage *here*
   only because a fork was in frame and Stage 2 outranks Stage 1 — without the
   fork it would have measured an ~18 cm bowl with a 27 cm plate ruler.
2. **`LIQUID_IN_BOWL` fill-depth priors are far too deep for a shallow serving.**
   242 g over a measured 155 cm² surface implies a real depth of **1.56 cm**; the
   priors assume 2.0-5.0 cm. At h=4.0 the same photo computes 620 g (2.6x over).
   Depth is genuinely unobservable from a top-down photo, which makes this the
   weakest shape model in the table — do not tune it on this one photo; the
   principled fix is to derive fill depth from the ratio of food area to bowl rim
   area (a tapered bowl with a small surface is a shallow fill), which is blocked
   on Stage-1 rim detection working.
3. **Rim detection failed again** (bright pink bowl on granite) — 3 of 5 photos
   now get no container scale.

Good news from the same photo: the food gate passed it, the fork was detected
and gave a geometrically sane scale (155 cm² ≈ a 12.5 cm square of soup, which
matches the image), and `low_confidence` correctly fired on the wrong class.

## Why not just download a public dataset

Checked (2026-08). The overlap with our 101-class label set is almost nil, so
public weighed data cannot validate our classes:

| dataset | weighed? | reference object | overlap with our 101 classes |
|---|---|---|---|
| ECUSTFD (2978 imgs, 19 classes) | yes, scale + drainage volume | 1-yuan coin, 25.0 mm | **1** — `donuts` |
| SimpleFood45 (513 imgs, 12 classes) | yes (g, mL, kcal) | 5x4 checkerboard, 1.2 cm | **1** — `pizza` |
| Nutrition5k (~5k dishes) | yes, per-ingredient mass | none in frame (RGB-D rig) | mixed cafeteria plates, not single Food-101 dishes |

Two further problems even where they overlap: none of them use **fork/spoon/knife**,
which is the only reference our Stage-2 detector can actually find, so they
exercise a scale path we do not ship; and they are shot on rigs, not phones at
arm's length, so they do not test the framing our users actually produce.

Conclusion: **our own weighed photos are worth more than any public set.** You do
not have to *cook* — nearly everything below is buyable ready-made.

## Shoot protocol

1. **Tare the plate**, then weigh — the number must be food only.
2. **Put a fork flat in frame**, fully visible, not overlapping the food. Fork is
   the only working reference (19 cm prior) and it is the difference between a
   ±12% band and a ±35% guess. Best result so far: donuts 199 g vs 203 g actual.
3. **Leave a margin around the plate.** The rim detector rejects any plate whose
   outline touches more than one frame edge, so a plate cropped by the frame gets
   no container scale.
4. **Maximise plate/background contrast** — a dark plate on a light counter (or
   the reverse). A white plate on light speckled granite is the known Stage-1
   failure: the speckle bridges the plate to the frame edge and it falls back to
   shape-only.
5. **Two shots per dish**: one top-down, one at ~45°. The oblique one tests the
   tilt correction, which currently only fires for tall (ELLIPSOID/DOME) classes.
6. Name the file `<food_class>_<grams>.jpg` and add a row to `manifest.csv`.

## What to shoot next, in priority order

### P1 — classes changed in the 2026-08 audit with no weighed photo behind them

These were retuned from reasoning and family-consistency, not measurement. They
are the likeliest place for a regression to hide.

| class | why it matters | how to get it |
|---|---|---|
| `ice_cream` | DOME→ELLIPSOID; was unbounded height | supermarket tub, scoop it |
| `french_fries` | biggest beneficiary of the new 5 cm pile cap | any fast food |
| `fried_rice` | IRREGULAR_PILE→FLAT_DISC h=2.5 | Chinese takeaway |
| `pad_thai` | IRREGULAR_PILE→FLAT_DISC h=2.0 | Thai takeaway |
| `spaghetti_carbonara` | moved with bolognese, never measured | ready meal / takeaway |
| `caesar_salad` or `greek_salad` | highest heap factor (k=0.5), pile cap | supermarket |
| `onion_rings`, `nachos` | pile cap + void-heavy density | fast food / snack |
| `edamame` | pile cap; density assumes pods not beans | frozen aisle |

### P2 — different shape models, so the test set covers the pipeline

| class | shape model exercised | how to get it |
|---|---|---|
| `pizza` | FLAT_DISC (also the one SimpleFood45 overlap) | any |
| `hamburger` | ELLIPSOID + bread_composite density — the family behind the original 747 g bug | any |
| `hot_dog`, `sushi`, `spring_rolls` | CYLINDER | any |
| `ramen` or `pho` | LIQUID_IN_BOWL, and a genuine **bowl** (we have zero bowl photos) | takeaway |
| `cheesecake` / `chocolate_cake` | cake slab | any |
| `guacamole` / `hummus` | dip in a bowl | supermarket |
| `gyoza` / `samosa` | small ELLIPSOID | frozen / takeaway |
| `waffles` / `pancakes` | thin FLAT_DISC (lowest h in the table) | frozen / cafe |

### P3 — a bowl, specifically

Every photo we have is a plate, and the bowl/plate classifier has never been
tested against ground truth. A misread there is a **2.25x** mass error
(bowl prior 18 cm vs plate 27 cm), which makes it the single biggest accuracy
lever in the system. Any soup/ramen/curry in an obvious bowl closes that gap.

## Known stale entries

`shape_priors.SHAPE_PRIORS` has 3 keys that are not in the 101-class label set
and can therefore never fire: `burrito`, `fried_egg`, `mashed_potatoes`.
Harmless, but do not spend a weighed photo on them.
