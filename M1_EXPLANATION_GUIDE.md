# M1_EXPLANATION_GUIDE.md — Database & Core Plumbing

This guide explains **M1** (see `09-build-phases.md`): the milestone where NutriLens's schema
becomes a real, running Postgres database with ORM models wired to it, seed data loaded, and the
security/dependency primitives (`security.py`, `deps.py`) that every later milestone builds on.

---

## 1. What we built

| File | Role |
|---|---|
| `app/config.py` | Typed, environment-based settings (one source of truth) |
| `app/db/base.py` | SQLAlchemy `Base`, `engine`, `SessionLocal` |
| `app/db/models.py` | 5 ORM models: `User`, `MealLog`, `FoodItem`, `MealItem`, `Recommendation` |
| `app/security.py` | bcrypt password hashing, JWT create/verify |
| `app/deps.py` | FastAPI dependency injection: `get_db`, `get_current_user` |
| `alembic/env.py`, `alembic/versions/0001_initial_schema.py` | Migration wiring + the first migration |
| `scripts/seed_food101_classes.py` | Idempotent seed of the 101 Food-101 classes |
| `scripts/seed_demo_user.py` | Dev-only demo user with 7 confirmed meals |
| `tests/test_m1_models.py` | Round-trips a hashed-password `User`; cascade/restrict checks |

**Definition of Done, verified:** `alembic upgrade head` builds the schema exactly as designed;
seeding inserts 101 food classes; `pytest tests/test_m1_models.py` passes (3/3), proving a `User`
with a hashed password round-trips correctly through the ORM.

---

## 2. Why we built it this way

The whole system depends on a correct, defensible data model *before* any business logic exists.
Getting the schema, cascade behaviour, and security primitives right in M1 means every later
milestone (auth in M2, recognition in M3, ...) builds on a foundation that's already tested, rather
than discovering a schema mistake three milestones later when it's expensive to fix.

---

## 3. Database relationships (recap of `02-database-schema.md`)

```
user (1) ──< meal_log (1) ──< meal_item >── (1) food_item
  │
  └──< recommendation
```

- `user → meal_log → meal_item`: **owned chain**, `ON DELETE CASCADE`. Deleting a user removes
  everything they own — this is what makes FR14 (account deletion) a single, safe `DELETE`.
- `meal_item → food_item`: **shared reference**, `ON DELETE RESTRICT`. `food_item` is the USDA
  nutrition cache — many users' meal items point at the same row, so it must never be deleted out
  from under someone else's history.
- `user → recommendation`: owned, `ON DELETE CASCADE`.

The schema is in **Third Normal Form (3NF)**: every non-key column depends on the whole primary key
and nothing but the key — e.g. macros live once in `food_item`, not duplicated into every
`meal_item` row, so a correction to "grilled chicken" nutrition data doesn't require updating every
meal that ever referenced it.

---

## 4. ORM concepts you should be able to explain

- **What an ORM is**: a layer that maps Python classes ↔ database tables, and object attributes ↔
  columns, so you manipulate `Session`/objects in Python instead of writing raw SQL strings.
- **`Session` = unit of work**: it tracks every object you've loaded or added and only sends SQL to
  Postgres when you `flush()`/`commit()`/query — this is why `db.flush()` appears in the seed
  scripts (assigns a `user_id` without fully committing yet, so a follow-up insert can reference it).
- **`relationship()` vs a column**: `relationship()` does not create a column — it's a Python-side
  convenience that tells SQLAlchemy how to load related rows (`user.meals` runs a `SELECT` under the
  hood). The actual foreign key is the real column (`meal_log.user_id`).
- **`back_populates`**: keeps both sides of a relationship (`User.meals` / `MealLog.user`)
  consistent in memory without an extra query when you append to one side.
- **Cascade — two layers, not one**: `cascade="all, delete-orphan"` (Python/ORM-level, only fires
  through a loaded `Session`) vs `ondelete="CASCADE"` (database-level, fires even for raw SQL). M1's
  models set **both**, because FR14 needs the guarantee to hold even outside the ORM.

---

## 5. Alembic concepts you should be able to explain

- **Why migrations exist at all**: ORM models describe the *desired* schema; nothing changes in the
  real database until a migration runs. Migrations are versioned, ordered, and replayable —
  everyone's environment (dev, CI, production) ends up with an identical schema history.
- **`--autogenerate`**: diffs the live database against `Base.metadata` (built from `db/models.py`)
  and writes the DDL needed to reconcile them — but it must always be manually reviewed.
- **Why review matters here specifically**: autogenerate can miss `ondelete=` values on foreign
  keys, and doesn't know that Postgres ENUM types must be created *before* any table that uses them
  as a column type. Migration `0001_initial_schema.py` handles both by hand: enums first, then
  tables in dependency order (`user`, `food_item` → `meal_log` → `meal_item`, `recommendation`).
- **`upgrade()` / `downgrade()`**: apply / reverse a single migration. `downgrade()` in `0001` drops
  tables in the *reverse* of creation order, then drops the enum types — respecting FK dependencies.

---

## 6. SQLAlchemy 2.0 concepts you should be able to explain

- **Declarative style**: `class User(Base): ...` with `Mapped[type]` type hints and
  `mapped_column(...)` — the current (2.0) idiom, fully compatible with static type checkers.
- **`Numeric` vs `Float`**: macros/calories use `Numeric(7,2)` (exact fixed-point decimal, no
  floating-point drift when summing many small values); `confidence_score` uses `Float` because a
  classifier's confidence is inherently an approximation, not a value meant to be summed exactly.
- **Postgres `ENUM`**: `health_goal` and `meal_type` are real Postgres enum types, not just strings
  — an invalid value is rejected by the *database* itself, a second line of defense beyond Pydantic
  validation in the API layer.

---

## 7. Likely graduation-discussion questions (with model answers)

**Q: Walk me through what happens when a user deletes their account.**
A: A single `DELETE FROM "user" WHERE user_id = :id`. Because `meal_log` and `recommendation` both
have `ON DELETE CASCADE` foreign keys to `user`, Postgres automatically deletes every meal log (and,
transitively, every meal item via `meal_item`'s own `CASCADE` to `meal_log`) and every recommendation
belonging to that user — in one statement, at the database level, so it can't be forgotten or done
partially by application code. The shared `food_item` cache rows are untouched, because that FK is
`RESTRICT`, not `CASCADE`.

**Q: Why is `food_item` not just a column inside `meal_item`?**
A: 3NF — the same food (e.g. "grilled chicken breast") appears in many users' many meals; storing its
macros once in `food_item` and referencing it by `food_id` avoids duplicating (and risking
inconsistent) nutrition data across every meal_item row that uses it.

**Q: What's the difference between an ORM-level cascade and a database-level cascade?**
A: The ORM-level `cascade="all, delete-orphan"` only fires when you delete an object through a
SQLAlchemy `Session` that has the relationship loaded. The database-level `ondelete="CASCADE"` on
the actual foreign key constraint fires *regardless* of how the delete happens — even a raw SQL
`DELETE` from a different tool. NutriLens sets both so FR14's guarantee is structural, not just an
application-code convention.

**Q: Why hash passwords with bcrypt instead of storing them encrypted (reversible)?**
A: Passwords should never be *recoverable*, even by NutriLens itself — bcrypt is one-way. Encryption
implies a key that can decrypt back to the original, which is exactly the risk model we want to avoid
(a compromised key or insider would expose every password at once).

**Q: What would you improve if you had more time?**
A: An account-unlock flow (currently accounts lock permanently until manually reset — no
self-service or timed auto-unlock exists yet), and moving from hand-verifying the first Alembic
migration to trusting a well-reviewed `--autogenerate` diff for schema changes after M1.

---

## 8. Common mistakes to watch for (and avoid saying you made)

- Forgetting `ondelete=` on a `ForeignKey` in Python even though the ORM relationship has a cascade
  — the *database* constraint then has no cascade behaviour, and a delete throws a foreign-key
  violation instead of cascading.
- Editing a live database's schema by hand "just this once" instead of through a migration — from
  then on, Alembic's model of "current state" no longer matches reality.
- Storing passwords with a fast, general-purpose hash (SHA-256/MD5) instead of a deliberately slow,
  salted one (bcrypt) — fast hashes make brute-forcing a stolen hash database far cheaper.

Prev: (none — this is the first milestone) · Next: `M2_EXPLANATION_GUIDE.md`
