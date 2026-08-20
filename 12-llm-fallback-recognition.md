# 12 — LLM fallback recognition (Claude vision → USDA)

**Status:** designed, not built. This document is the build order.
**Written:** 2026-08-12, against commit `bc881de`.
**Prerequisite:** `bc881de` (*fix(recognise): stop refusing real food the label set
has no class for*) — the fallback hooks into the code path that commit created.

> **Do this first, independently:** § *Survey (FNDDS) is excluded from USDA
> search* is a live inconsistency in shipped code. It is worth fixing whether or
> not this feature is ever built, and it raises the ceiling this feature is
> working against — so it should land before any LLM work starts.

---

## WHY THIS FILE EXISTS

Food-101 is a closed set of 101 classes. Two failures follow from that, both
measured on a real photo (`IMG_7121.heic`, scrambled eggs on a decorated plate,
4032×2268):

1. **False refusal.** The open-set `FoodGate` scored the photo **0.950** — it was
   certain the photo was food — while Food-101's top-1 fell below
   `no_food_confidence_floor` (0.25). The service returned `400 NO_FOOD_DETECTED`
   for a plate with food on it. Fixed in `bc881de` by making rejection the gate's
   call alone.
2. **Confidently wrong.** With the refusal removed, the same photo returns
   **`carrot_cake` at 0.777** — above `confidence_threshold` (0.60), so
   `low_confidence` is false, `alternatives` is empty, and the app commits
   silently. Top-5 was `carrot_cake 0.777 / deviled_eggs 0.029 /
   breakfast_burrito 0.016 / omelette 0.015 / crab_cakes 0.013`. There is no
   scrambled-eggs class, so the model **cannot** be right — it can only be
   confidently wrong, and the user logs carrot cake's macros for a plate of eggs.

No threshold fixes (2): 0.777 clears any floor that doesn't also start refusing
legitimate hard food photos. The label set is the constraint, and the only way
past it is a recogniser that isn't bounded by 101 classes.

## WHAT PROBLEM THIS SOLVES

It gives `/recognise` a path for foods Food-101 has no class for — scrambled
eggs, and most cuisines the dataset never sampled. Claude identifies the dish
from the photo and returns **the USDA search term directly**, so the entire
existing nutrition chain (`resolve_food_item` → cache → `scale_to_portion` →
meal logging) runs unchanged.

The key insight: `usda_query_map.USDA_QUERY_OVERRIDES` is already a hand-written
Food-101 → USDA-English translation table (`beef_carpaccio` → `beef tenderloin`,
`escargots` → `snail`, `guacamole` → `avocado`). That table works only because
there are 101 entries to write by hand. An LLM does that translation for
arbitrary foods, which is exactly the piece that cannot scale.

## WHY A FALLBACK, NOT A REPLACEMENT

EfficientNet-B4 stays the primary recogniser. Claude runs only when the local
pipeline has no usable answer. Three reasons, in priority order:

1. **The trained model is the project's contribution.** Replacing a fine-tuned
   classifier (91.4% top-1) with an API call is not defensible in an assessment.
   Keeping it primary and adding a documented open-set fallback *is* the honest
   improvement `food_gate.py` already gestures at in its own docstring, which
   lists "an explicit background class" and "an OOD detector (ODIN /
   energy-based)" as the rigorous answers. This is a third, and a stronger one.
2. **Cost and latency are bounded** — it fires on a minority of photos, not on
   every shutter press.
3. **It degrades to today's behaviour.** API down, rate-limited, or disabled by
   config ⇒ `/recognise` behaves exactly as it does at `bc881de`.

---

## DECISIONS ALREADY MADE (do not re-litigate)

| # | Decision | Reason |
|---|---|---|
| D1 | Fallback only; EfficientNet stays primary | above |
| D2 | Claude returns the **USDA query term**, not just a display name | reuses the whole nutrition chain; no new endpoint, no migration |
| D3 | Result lands in the **pick-list for user confirmation**, never auto-committed | repeats the Carrot Cake mistake otherwise |
| D4 | `confidence` is **null** for an LLM-identified food | an LLM's self-reported confidence is not the classifier's softmax. Precedent already in the codebase: `RecognitionResults.tsx` sets `correctedConfidence = null` for a manual override, and `confidence_score` is nullable end-to-end (`schemas/meal.py:101`, `db/models.py:265`) |
| D5 | Off by default (`llm_fallback_enabled: bool = False`) | same posture as `reference_detector_enabled`; an opt-in external dependency |
| D6 | Anthropic key is server-side only, in `backend/.env` | `07` — "No nutrition math, no secrets"; matches `USDA_API_KEY` |
| D7 | Model `claude-opus-5`, structured outputs via Pydantic | see *The Claude call* |
| D8 | No DB migration | `food_item.food101_class` is nullable `String(50)` with **no FK and no CHECK** (`db/models.py:237`); `MealItem` joins on `food_id`. Verified. |

## OPEN DECISIONS — ASK BEFORE BUILDING

| # | Question | Why it matters |
|---|---|---|
| Q1 | **Privacy disclosure.** Sending meal photos to a third-party API is not covered by any PRD document. M3's DoD is "no image is written to disk (verified in test)" — this doesn't violate that literally (nothing touches our disk) but it does change what happens to a user's food photos. Needs a user-facing disclosure and an explicit sign-off. | Do not ship silently. This is the one item that must be settled by a human before the first request leaves the building. |
| Q2 | **Latency budget.** `/recognise` measures 1.0–2.3s against a 5s NFR. Adding a vision call may breach it. Measure before committing to `effort`, and decide whether a breach on the fallback path only is acceptable. | Could force `effort: "low"` or a smaller model. |
| Q3 | **Trigger condition** — which of the two triggers in *Where it plugs in* ship in v1. | Scopes the work. |

---

## WHERE IT PLUGS IN

`app/services/recognition_service.py`, in `recognise()`, **after** the food gate
clears and **after** `classifier.predict()`. Current shape at `bc881de`:

```
validate_and_load_image(raw, content_type)
  ↓
food_gate.inspect(image)        → below food_gate_min_prob ⇒ 400 NO_FOOD_DETECTED
  ↓  gate_cleared = True
classifier.predict(image, top_k=3)
  ↓
if not gate_cleared and top_conf < no_food_confidence_floor ⇒ 400   ← fallback-only backstop
  ↓
items[0] = Prediction(rank=1, ...) + portion
low_confidence = top_conf < confidence_threshold
```

Insert the fallback between `predict()` and building `items`.

**Trigger — the gate says food, but Food-101 has no usable answer:**

```python
llm_eligible = (
    gate_cleared                                   # it IS food; only "which" is unknown
    and settings.llm_fallback_enabled
    and llm_recogniser is not None                 # fail open, like the gate
    and top_conf < settings.llm_fallback_threshold  # default 0.60 — see below
)
```

`llm_fallback_threshold` should default to `confidence_threshold` (0.60): the
band where FR04 already admits the model is unsure is exactly the band where a
second opinion is worth paying for. **This does not catch the Carrot Cake case
(0.777).** That one is caught by the second trigger:

**Second trigger (recommended for v1, confirm via Q3) — user-initiated.** When
the user taps "Not the right food?" in `RecognitionResults.tsx:138`, call the
fallback explicitly rather than only opening the 101-item picker. This is the
only trigger that reaches a confidently-wrong prediction, because by definition
nothing in the server-side signal says 0.777 is wrong — the user does.

Needs a small endpoint, e.g. `POST /recognise/assist` taking the same multipart
image, returning the same `RecognitionResponse` shape with `source: "llm"`. The
app already holds `imageUri` on that screen, so it can re-upload.

---

## THE CLAUDE CALL

New module: **`backend/app/integrations/llm_recogniser.py`** — sits beside
`usda_client.py`, which is the right neighbourhood (both are typed wrappers over
a third-party HTTP API with their own failure taxonomy).

```bash
pip install anthropic      # add to pyproject [project.dependencies]
```

**Model:** `claude-opus-5` — $5 / $25 per MTok. Thinking is **on by default** on
this model (omitting `thinking` runs adaptive); `max_tokens` caps thinking +
response text together, so give it headroom. Control depth with
`output_config={"effort": ...}`, not `budget_tokens` (removed — returns 400).
Do **not** pass `temperature`/`top_p`/`top_k` — also removed, also 400.

**Structured output:** use `client.messages.parse()` with a Pydantic model. The
project is Pydantic throughout (`app/schemas/`), and `.parse()` validates the
response against the schema at the SDK layer so the model retries on mismatch —
no hand-parsing, no regex, no JSON-repair loop.

```python
from pydantic import BaseModel, Field

class LlmFoodGuess(BaseModel):
    is_food: bool = Field(description="False if the photo shows no edible food")
    display_name: str = Field(description="Human-readable dish name, e.g. 'Scrambled Eggs'")
    usda_query: str = Field(
        description=(
            "Search term for USDA FoodData Central, in food-science English. "
            "Prefer the ingredient over the dish name when USDA is unlikely to "
            "index the dish: 'guacamole' -> 'avocado'. Omit brand names."
        )
    )
    cooking_method: str | None = Field(
        description="One of the app's cooking methods if visually determinable, else null"
    )
    self_reported_confidence: float = Field(ge=0.0, le=1.0)
```

`cooking_method` must be constrained to the app's existing vocabulary — read it
from `GET /meta/cooking-methods` (`app/api/meta.py:71`) and put the literal list
in the schema description or as an enum. Do not invent a parallel vocabulary.

**Request shape:** one user message with a base64 `image` content block followed
by a `text` block. Image is the **already-validated in-memory PIL image**
re-encoded to JPEG — never re-read from disk (M3 DoD), and never the raw upload
bytes, so the size/format/quality gates in `preprocess.py` still apply first.

**Prompt:** state that the classifier already failed and its top guesses, and ask
for the dish. Giving Claude the failed candidates is cheap and measurably useful
context. Keep the system prompt short — Claude Opus 5's minimum cacheable prefix
is **512 tokens**, so a short prompt will silently not cache; do **not** add
`cache_control` unless you measure the prefix clearing 512.

**Handle `stop_reason` before reading content.** Claude Opus 5 ships safety
classifiers and can return HTTP 200 with `stop_reason: "refusal"` and empty or
partial `content`. Food photos are unlikely to trip them, but code that indexes
`response.content[0]` unconditionally breaks when they do. Treat a refusal as
"fallback unavailable" and degrade (see the matrix below).

**Cost/latency:** measure, don't estimate. Use `client.messages.count_tokens()`
against a real 1024×576 upload (the app's actual size — see
`mobile/src/screens/Camera.tsx`, `MAX_LONG_EDGE = 1024`). Never use `tiktoken`;
it is OpenAI's tokenizer and is wrong for Claude.

---

## USDA PATH — the one change needed downstream

`resolve_food_item(db, food101_class, cooking_method, usda)` calls
`usda.search(food101_class, cooking_method)`, which runs
`build_query_chain()` — the progressively-relaxed chain that applies
`USDA_QUERY_OVERRIDES` and strips Food-101's method adjectives. For an
LLM-supplied term that machinery is wrong: the term is already in USDA English.

**Add a sibling method, do not refactor `search()`:**

```python
def search_query(self, query: str, cooking_method: str | None = None) -> UsdaFood:
    """Look up a caller-supplied USDA term, bypassing the Food-101 query chain."""
```

It must reuse the existing `HourlyRateLimiter`, `_get_search`, `select_best`, and
the same three typed errors (`UsdaRateLimitedError` / `UsdaUnavailableError` /
`NoNutritionMatchError`). The LLM query is one rung, not a chain.

`resolve_food_item` then needs an optional `usda_query: str | None = None` that
routes to `search_query` when present. Cache key stays
`(food101_class, cooking_method)` — store the LLM's slug (e.g.
`scrambled_eggs`) in `food101_class`. It is a plain nullable `String(50)` with no
constraint, so this is legal today (D8), and it means the second user to
photograph scrambled eggs gets a cache hit and costs nothing.

---

---

## THE USDA COVERAGE CEILING

Identifying the dish is only half the chain. Both links must hold:

```
Claude identifies it   AND   USDA FoodData Central indexes it
```

USDA is a US food database, and its coverage of composite non-Western dishes is
thin. **In practice USDA coverage will bound this feature before Claude's food
knowledge does.** `usda_query_map.py` is already evidence — the override table
maps dishes *down to ingredients* precisely because the dish is absent:

```python
"escargots": "snail",
"guacamole": "avocado",   # USDA has no "guacamole" at all
```

So the interesting failure is not "Claude didn't know it". It is **Claude
identified it correctly and USDA has no row for it** — molokhia, mansaf, kabsa.

### The ladder

Five tiers, most trustworthy first. Every tier below the first two trades
accuracy for coverage, so **which tier answered must be visible to the user** —
"lab-measured" and "estimated from ingredients" must not render identically.

| Tier | Method | Provenance | Status |
|---|---|---|---|
| 1 | USDA Foundation / SR Legacy | Lab-measured, real `fdc_id` | **shipped** |
| 2 | USDA Survey (FNDDS) | USDA-derived composite, real `fdc_id` | **one-line fix — see next section** |
| 3 | Ingredient decomposition | USDA ingredients, LLM-estimated ratios | designed below, **not in v1** |
| 4 | Nearest analogue (`molokhia` → `spinach cooked`) | USDA row for a *different* food | shipped (the override table) |
| 5 | Ask the user / manual macros | User-stated | not built |

Today the code sits at tiers 1 and 4, with nothing in between — which is why a
correctly-identified molokhia collapses straight to "spinach" or to a 404.

### Tier 3 — ingredient decomposition (the real answer, deferred)

When USDA lacks the *dish* it almost always has the *ingredients*. Molokhia is
jute leaves, chicken, oil, garlic and stock; every one of those is in FDC even
though the stew is not. So instead of one `usda_query`, ask Claude for a
component breakdown, look each component up through `search_query()`, scale by
proportion, and sum to per-100 g macros:

```
molokhia → [ (jute potherb, cooked)  60%
             (chicken, cooked)       25%
             (oil, olive)             5%
             (garlic / stock)        10% ]
```

This is not improvised — it is how FNDDS itself is built, so it reproduces
USDA's own method for dishes USDA has not gotten to.

Why it fits the existing architecture:

- **Paid once.** The computed result caches under the slug via the existing
  `(food101_class, cooking_method)` unique index, so the second user to
  photograph molokhia triggers neither a Claude call nor a USDA call.
- **Rate limits survive it.** Four lookups instead of one, but only on a cache
  miss, and `HourlyRateLimiter` already gates each one.
- **Degrades component-wise.** A missing component leaves the rest usable.

What it costs, stated plainly: **the ratios are LLM-estimated, not measured.**
That must be visible in the data, not just the docs — such a `food_item` carries
a null `usda_fdc_id`, and `resolve_food_item`'s `source` (today `"cache"` /
`"usda"`) needs a third value such as `"derived"` that flows through to the app.

**Ruled out: having Claude return macros directly.** It breaks the principle the
whole nutrition layer rests on — every number traceable to a measured source —
and recalled macros carry no provenance at all. Tier 3 keeps USDA as the source
of every *number* and puts the LLM only on the *proportions*, which is a far
smaller claim to defend.

---

## SURVEY (FNDDS) IS EXCLUDED FROM USDA SEARCH

**Independent of this feature. Fix it on its own branch.**

`app/integrations/usda_client.py` ranks three data types but only ever requests
two:

```python
_DATA_TYPE_RANK: dict[str, int] = {
    "Foundation": 0,
    "SR Legacy": 1,
    "Survey (FNDDS)": 2,      # ranked...
}
SEARCH_DATA_TYPES = "Foundation,SR Legacy"   # ...but never requested
```

`select_best` is written to rank FNDDS third, but `_get_search` never asks USDA
for it, so that branch is unreachable today. FNDDS is the one USDA dataset built
around *mixed dishes as eaten* rather than single ingredients and commodities —
exactly where a composite dish would live if USDA has it at all. Enabling it is
tier 2 of the ladder above, and it is the cheapest coverage the project can buy.

**The change:** `SEARCH_DATA_TYPES = "Foundation,SR Legacy,Survey (FNDDS)"`.

**Why it is low-risk:** FNDDS already sorts last on the data-type key, so a
Foundation or SR Legacy hit still wins whenever one exists. The existing ranking
rationale holds unchanged — FNDDS values are derived rather than lab-measured,
which is why third is the right place for them.

**The one real behavioural change, and it is not obvious.** `select_best` sorts
on **preparation match first, data type second** — deliberately, per its own
docstring ("Being right about the food has to come first"). So an FNDDS row whose
description matches the user's confirmed cooking method will sort *ahead* of a
Foundation row that does not. That is consistent with the existing rationale, but
it means enabling FNDDS can change which row is returned for foods that already
resolve today. Re-run the M5 nutrition tests and spot-check a few Food-101
classes that currently return good Foundation matches.

**Tests to update:** `tests/test_m5_usda_client.py:442` asserts the exact string
`params["dataType"] == "Foundation,SR Legacy"`. That assertion is pinning the old
contract, so updating it is the deliberate change — not a fix-the-test. Consider
adding a case that proves the ranking still prefers Foundation over an FNDDS row
when neither matches the cooking method.

**Also review:** `SEARCH_PAGE_SIZE = 10` was chosen so `select_best` has enough
candidates to re-rank across preparations. A third data type widens the pool, so
confirm 10 still surfaces the good preparation matches rather than filling the
page with FNDDS variants.

---

## CONFIG ADDITIONS (`app/config.py`)

Follow the house comment style — say *why* the default is what it is, not just
what it is.

```python
# ---- LLM fallback recognition (see 12-llm-fallback-recognition.md) ----
llm_fallback_enabled: bool = False      # opt-in: external dependency + cost
anthropic_api_key: str = ""
llm_model: str = "claude-opus-5"
llm_fallback_threshold: float = 0.60    # == confidence_threshold; the FR04 unsure band
llm_timeout_seconds: float = 8.0        # under the 5 s NFR is impossible; see Q2
llm_effort: str = "low"                 # measure before raising
```

Also add `ANTHROPIC_API_KEY` to **`.env.example`** with the same
get-your-own-key note as `USDA_API_KEY`, and confirm `.env` stays gitignored.

## WIRING

- **`app/main.py` lifespan** — construct the recogniser once at startup, in the
  same try/except degrade-don't-crash pattern as `load_classifier` and
  `load_food_gate` (`main.py:95-130`). Missing/blank key ⇒
  `app.state.llm_recogniser = None` + a warning, never a crash.
- **`app/deps.py`** — `get_llm_recogniser(request) -> LlmRecogniser | None`,
  mirroring `get_food_gate` (`deps.py:186`) including its fail-open docstring.
- **`app/api/recognise.py`** — add the dependency, pass it through. The route
  stays a one-line map of what the endpoint does.
- **`recognition_service.recognise()`** — new optional param
  `llm_recogniser: LlmRecogniser | None = None`, defaulted so every existing
  test constructing the service by hand keeps compiling.

## CONTRACT CHANGE (`app/schemas/recognition.py`)

Additive, so no client breaks:

- `Prediction.confidence: float` → `float | None` (D4).
- `Prediction.source: Literal["food101", "llm"] = "food101"` — the app must be
  able to tell the user where a label came from, and must not render "78%" for
  something that has no softmax behind it.
- Mirror both in `mobile/src/types/api.ts`.
- Update **`03-api-specification.md`** §4 in the same commit. The response shape
  is spec'd there; a code change that leaves the spec stale is how the
  `NO_FOOD_DETECTED` drift happened (it appears in **no** markdown doc — verified
  by grep — because it was added to code only).

## DEGRADATION MATRIX

The fallback is an enrichment. **Nothing in this table may turn a request that
succeeds today into a failure.**

| Condition | Behaviour |
|---|---|
| `llm_fallback_enabled = False` | Skip entirely; identical to `bc881de` |
| Key missing / recogniser `None` | Skip; warn once at startup |
| Anthropic timeout / `APIConnectionError` | Log, skip, return the Food-101 result |
| `RateLimitError` (429) | Log, skip. Do **not** surface a 429 to the app |
| `stop_reason == "refusal"` | Log with the category, skip |
| `is_food = False` | Trust the `FoodGate` over the LLM — it is the calibrated, measured signal (0.017 on bare granite, 0.950 on the real plate). Log the disagreement; do not convert this into a 400 |
| USDA has no match for `usda_query` | Existing `NoNutritionMatchError` (404) path — already handled, already tested |

Catch typed SDK exceptions in a most-specific-first chain
(`RateLimitError` → `APIStatusError` → `APIConnectionError`), never a bare
`except Exception`, and never string-match error messages.

## TESTS

`tests/test_m3_recognise.py` already has the stub pattern to copy — see the
`StubGate` / `_make_gated_client` block added in `bc881de`. Add a `StubLlm` the
same way. **No test may make a real network call.**

Required:

1. Disabled by default ⇒ byte-identical response to `bc881de`.
2. Gate passes + flat Food-101 + fallback on ⇒ LLM name in `items[0]`,
   `source="llm"`, `confidence=None`, `low_confidence=True`, alternatives
   still populated from Food-101.
3. Confident Food-101 (≥ threshold) ⇒ fallback **not** called (assert the stub
   was never invoked — this is the cost guard).
4. LLM raises `RateLimitError` ⇒ 200 with the Food-101 result, not 429.
5. LLM raises timeout ⇒ 200 with the Food-101 result.
6. `stop_reason: "refusal"` ⇒ 200 with the Food-101 result.
7. LLM says `is_food=False` but the gate passed ⇒ gate wins, 200.
8. `search_query()` bypasses `build_query_chain` — assert the exact query string
   hitting `_get_search`.
9. The M3 DoD disk guard still passes (`test_no_image_written_to_disk`) with the
   fallback enabled — the JPEG re-encode for the API must be in-memory.

Baseline is **475 passed** at `bc881de`. Run `./.venv/bin/python -m pytest -q`.

## MEASURE BEFORE AND AFTER

The project's habit is measured evidence for design claims (the 0.46 laptop in
`food_gate.py`, the 0.950/0.017 gate numbers). Keep it. On a fixed set of ~20
photos including out-of-vocabulary dishes:

- p50 / p95 added latency, against the 5s NFR (Q2).
- Measured tokens and cost per call via `count_tokens`.
- Fallback trigger rate — how often it actually fires.
- Correct-identification rate on the out-of-vocabulary subset. `IMG_7121.heic`
  (scrambled eggs, currently "Carrot Cake 0.777") is regression case #1.

---

## DISCUSSION / DEFENSE QUESTIONS

**Q: "Didn't you just outsource the ML to an API?"**
A: No — the trained EfficientNet-B4 handles every photo and remains the primary
recogniser; the fallback fires only when Food-101 provably cannot answer, because
the dish is not one of its 101 classes. The measured failure is specific: gate
0.950 (certainly food), top-1 `carrot_cake` 0.777 on scrambled eggs, a class the
dataset does not contain. The fallback addresses the *label set*, not the model.

**Q: "Why not just add scrambled eggs as a 102nd class?"**
A: That is retraining plus labelled data for every dish a user might eat, which
is the closed-set problem restated rather than solved. The generalisable fix is a
recogniser that isn't bounded by a fixed vocabulary.

**Q: "Why does Claude return a USDA query instead of just a name?"**
A: Because `usda_query_map.USDA_QUERY_OVERRIDES` shows the translation is the
hard part — USDA indexes `avocado`, not `guacamole`. Hand-writing that mapping is
tractable for 101 classes and impossible for arbitrary foods.

**Q: "How do you stop it being confidently wrong in a new way?"**
A: The result is a proposal in the FR04 pick-list, never an auto-commit (D3), and
it carries `confidence: null` and `source: "llm"` so the UI cannot present it as a
calibrated model score. That is the same discipline already applied to manual
corrections.

**Q: "What if the Anthropic API is down?"**
A: Every failure mode degrades to the current behaviour — see the degradation
matrix. The fallback can never turn a working request into a failing one.

## COMMON MISTAKES

- Calling the fallback on every photo instead of gating on `llm_eligible` —
  turns a bounded cost into a per-shutter-press cost. Test #3 guards this.
- Writing the image to a temp file to send it — violates the M3 DoD. Re-encode
  the in-memory PIL image. Test #9 guards this.
- Letting an Anthropic 429 or timeout surface as a 4xx/5xx from `/recognise`.
- Putting `ANTHROPIC_API_KEY` anywhere the app can read it. It is server-side
  only; `EXPO_PUBLIC_*` values ship inside the bundle and are readable by anyone.
- Refactoring `UsdaClient.search()` instead of adding `search_query()` beside it.
  The Food-101 query chain is tested and load-bearing.
- Passing `temperature` / `top_p` / `budget_tokens` to `claude-opus-5` — all
  removed on this model, all return 400.
