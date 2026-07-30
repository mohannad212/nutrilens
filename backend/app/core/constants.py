"""
app/core/constants.py
========================

WHY THIS FILE EXISTS
---------------------
Two strings in this project are contractual: the disclaimer attached to every
nutrition response (`03 §5`) and the longer disclaimer attached to every
recommendation response (`03 §7`). They are a stated NFR and a wellbeing
requirement (`06 §5`), they are asserted verbatim by tests (`11`), and they must
read identically everywhere they appear. That makes them exactly the kind of
value that must exist once, in a module with no dependencies, rather than being
retyped in each schema that returns one.

WHAT PROBLEM IT SOLVES
------------------------
Inlined literals drift. The moment the disclaimer is typed into
`schemas/nutrition.py` and again into `schemas/recommendation.py`, a later
wording tweak lands in one and not the other, and the app shows two different
legal statements depending on which screen the user is on. Worse, a test that
asserts "the disclaimer is present" by comparing against its own copy of the
string passes while the API returns something subtly different. One constant per
disclaimer means the test and the response cannot disagree.

HOW IT WORKS INTERNALLY
--------------------------
Plain module-level strings. Nothing imports anything, so any layer (schemas,
services, ml) can import from here without a circular-import risk — which is the
reason this lives in `core/` rather than next to whichever feature happened to
need it first.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Module constants (chosen): zero machinery, importable from anywhere,
  greppable, and diffable in review — a change to a legal string shows up as a
  one-line diff in a file whose entire purpose is legal strings.
- A Pydantic field default on each response schema: still a literal per schema,
  so it does not solve duplication.
- Reading them from `config.py` / the environment: these are not environment-
  varying configuration. Making them overridable per deployment would allow a
  deployment to run with the disclaimer blanked out, which is the one thing the
  NFR forbids.
- Storing them in the database: adds a query to every response and a migration
  to every wording fix, for a value that changes roughly never.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why are there two different disclaimers rather than one?"
A: They cover different claims. A nutrition response reports *estimated numbers*
   ("these values are approximate"). A recommendation response additionally
   offers *dietary suggestions*, which is closer to advice, so `03 §7` requires
   the stronger wording plus an explicit pointer to a qualified professional.
   Using the weaker text on an insight would under-state what the app is doing.

Q: "How do you guarantee the disclaimer is never missing from a response?"
A: It is not an optional field the route has to remember to set — it is a
   non-optional field on the response schema with this constant as its default
   (see `schemas/recommendation.py`), so a response object cannot be constructed
   without it, and `11`'s tests assert its presence on every insight payload.

COMMON STUDENT MISTAKES
--------------------------
- Paraphrasing the text when copying it into code. These strings are quoted from
  `03`; a paraphrase silently breaks the contract the mobile app and the report
  both reference.
- Putting user-facing copy that *does* vary (insight templates) in here. Those
  belong with the engine that renders them (`ml/reco_targets.py`) because they
  take parameters and change as the algorithm changes.
"""

# 03 §5 — attached to nutrition/meal responses (estimated values).
NUTRITION_DISCLAIMER = (
    "Nutritional values are estimates for informational purposes only and do not "
    "constitute medical advice."
)

# 03 §7 / 06 §5 — attached to every recommendation response (suggestions, so the
# stronger wording plus the referral to a qualified professional).
RECOMMENDATION_DISCLAIMER = (
    "Nutritional information and dietary suggestions are for informational purposes "
    "only and do not constitute medical or clinical nutritional advice. Consult a "
    "qualified healthcare professional for specific dietary needs."
)
