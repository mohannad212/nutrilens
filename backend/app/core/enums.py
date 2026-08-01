"""
app/core/enums.py
====================

WHY THIS FILE EXISTS
---------------------
`03 §10` calls itself "Enumerations (single source of truth)". Most of those
enums already have exactly one home: `health_goal`, `meal_type`, `sex` and
`activity_level` are Postgres `ENUM` types, so `db/models.py` defines them once
and every schema imports from there. `cooking_method` is the exception. It is
stored as `VARCHAR(30)` (`02 §2.3`) because it is part of a cache key rather than
a constrained domain column, so there is no ORM enum to import — and without this
file its eight values would be retyped in the request schema, the meal schema,
the query map, and the `/meta` endpoint.

WHAT PROBLEM IT SOLVES
------------------------
Four copies of a list drift. The concrete failure is not hypothetical: the app's
cooking-method picker is populated from `GET /meta/cooking-methods`, so if that
endpoint and the request validator disagree by even one value, the picker offers
a choice the API then rejects with a 422 — a bug that only appears for whichever
option was added last, and only on a real device.

WHY `cooking_method` IS NOT A DATABASE ENUM
----------------------------------------------
Worth stating because it is the obvious question. The other four are `ENUM` types
because they are closed domains the database should defend. `cooking_method` is
half of the nutrition cache's lookup key, and cache keys benefit from being cheap
to extend: adding "poached" should be a code change and a new cached row, not a
migration with `ALTER TYPE`. The validation that matters happens at the edge, in
the Pydantic schema, which is where a bad value would arrive from.

WHY THE VALUES ARE ORDERED THE WAY THEY ARE
----------------------------------------------
Source order is preserved by `Enum` and is what `/meta/cooking-methods` returns,
which is what the app's picker renders. `03 §10`'s order is kept verbatim rather
than alphabetised, so the contract document and the on-screen list match.

COMMON STUDENT MISTAKES
--------------------------
- Defining the enum inside the schema module that needed it first, so the next
  module to need it imports from an unrelated feature's namespace (or, worse,
  copies it).
- Using a plain `str` field with a comment listing the allowed values — the
  values then exist only in the comment, and nothing enforces them.
"""

import enum


class CookingMethod(str, enum.Enum):
    """`03 §10` cooking_method — appended to the USDA query (FR07).

    `str` mixin so a member is directly usable wherever a plain string is
    expected (query building, the `VARCHAR(30)` column) without `.value` at every
    call site, and so it serialises to its value in JSON.
    """

    grilled = "grilled"
    fried = "fried"
    boiled = "boiled"
    baked = "baked"
    roasted = "roasted"
    steamed = "steamed"
    raw = "raw"
    other = "other"


COOKING_METHODS: list[str] = [m.value for m in CookingMethod]
