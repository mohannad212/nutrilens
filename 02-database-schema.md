# 02 — Database Schema & Migrations

Database: **PostgreSQL 15**. Design: **Third Normal Form (3NF)**. This mirrors Table 5.1 of the
Phase-1 report and is the **single source of truth** for persistent data. ORM = SQLAlchemy 2.0;
migrations = Alembic.

> **Privacy invariant (enforced, tested):** submitted images are **never** stored. There is no
> image column, blob table, or file path anywhere in this schema. Images exist only in backend
> memory during the pipeline and are discarded when the meal is confirmed (report §4.5, §5.2).

---

## 1. Entity-relationship overview

```
user (1) ──< meal_log (1) ──< meal_item >── (1) food_item
  │
  └──< recommendation

user 1─N meal_log         (a user has many meals)
meal_log 1─N meal_item    (a meal has many items)
food_item 1─N meal_item   (a food can appear in many meals)  ← shared reference / USDA cache
user 1─N recommendation   (a user has many generated insights)
```

Cascade policy:
- Owned chains delete together: `user → meal_log → meal_item` use **ON DELETE CASCADE**.
- Shared reference data is protected: `meal_item.food_id → food_item` uses **ON DELETE RESTRICT**.
- `recommendation.user_id → user` uses **ON DELETE CASCADE**.

This makes **FR14 (account deletion)** a single `DELETE FROM user WHERE user_id = :id` that
irreversibly removes everything the user owns, while preserving the shared USDA cache.

---

## 2. Tables (authoritative DDL)

### 2.1 `user`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| user_id | SERIAL | PK | Auto-increment id |
| username | VARCHAR(50) | UNIQUE, NOT NULL | Display name |
| email | VARCHAR(100) | UNIQUE, NOT NULL | Login identifier |
| password_hash | VARCHAR(255) | NOT NULL | **bcrypt** salted hash (never plaintext) |
| health_goal | ENUM(`weight_loss`,`maintenance`,`weight_gain`) | NOT NULL | Recommendation target (FR13) |
| created_at | TIMESTAMP | DEFAULT now() | Account creation |
| account_locked | BOOLEAN | DEFAULT FALSE | TRUE after 5 consecutive failed logins (FR02) |
| failed_login_count | SMALLINT | DEFAULT 0 | Reset to 0 on success; increments on failure |
| height_cm | NUMERIC(4,1) | NULL | Body metric for the TDEE estimate (`06 §2`) |
| weight_kg | NUMERIC(5,1) | NULL | " |
| birth_year | SMALLINT | NULL | Year only — age in whole years is all Mifflin-St Jeor needs |
| sex | ENUM(`female`,`male`) | NULL | Sex-specific constant in the BMR equation |
| activity_level | ENUM(`sedentary`,`light`,`moderate`,`active`,`very_active`) | NULL | TDEE activity multiplier |

```sql
CREATE TYPE health_goal_enum   AS ENUM ('weight_loss', 'maintenance', 'weight_gain');
CREATE TYPE sex_enum           AS ENUM ('female', 'male');
CREATE TYPE activity_level_enum AS ENUM ('sedentary','light','moderate','active','very_active');

CREATE TABLE "user" (
    user_id             SERIAL PRIMARY KEY,
    username            VARCHAR(50)  NOT NULL UNIQUE,
    email               VARCHAR(100) NOT NULL UNIQUE,
    password_hash       VARCHAR(255) NOT NULL,
    health_goal         health_goal_enum NOT NULL,
    created_at          TIMESTAMP    NOT NULL DEFAULT now(),
    account_locked      BOOLEAN      NOT NULL DEFAULT FALSE,
    failed_login_count  SMALLINT     NOT NULL DEFAULT 0,
    -- Optional body metrics (M7). See the note below on why every one is NULL-able.
    height_cm           NUMERIC(4,1),
    weight_kg           NUMERIC(5,1),
    birth_year          SMALLINT,
    sex                 sex_enum,
    activity_level      activity_level_enum
);
CREATE INDEX idx_user_email ON "user"(email);
```

**Why the five body metrics are all nullable.** They exist only so `06`'s calorie
insight can compare a user's intake against their own estimated needs (Mifflin-St Jeor
TDEE) instead of against a population average, which in a nutrition app would be a
fabricated number presented as personal advice. Every *other* insight is a macro
**ratio** and needs no body data at all, so a user who supplies none of these still
gets the full macro analysis — only the calorie line softens to a descriptive
observation naming the fields still needed. Making them `NOT NULL` would have been a
breaking change to FR01's registration form and would have locked out existing
accounts. `sex` has exactly two values because the BMR equation takes a two-valued
constant (+5 / −161); a user who prefers not to state it leaves the column NULL,
which is a supported path rather than a blocked one.

Added by migration `0002_user_anthropometrics` (nullable columns with no default, so
the change is metadata-only on Postgres — no table rewrite).

### 2.2 `meal_log`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| log_id | SERIAL | PK | |
| user_id | INTEGER | FK → user(user_id) ON DELETE CASCADE, NOT NULL | Owner |
| logged_at | TIMESTAMP | NOT NULL DEFAULT now() | When meal was logged |
| meal_type | ENUM(`breakfast`,`lunch`,`dinner`,`snack`) | NOT NULL | |
| total_calories | NUMERIC(7,2) | NULL until computed | Aggregated from items |
| user_confirmed | BOOLEAN | DEFAULT FALSE | TRUE once user confirms recognition (FR10) |

```sql
CREATE TYPE meal_type_enum AS ENUM ('breakfast', 'lunch', 'dinner', 'snack');

CREATE TABLE meal_log (
    log_id          SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES "user"(user_id) ON DELETE CASCADE,
    logged_at       TIMESTAMP NOT NULL DEFAULT now(),
    meal_type       meal_type_enum NOT NULL,
    total_calories  NUMERIC(7,2),
    user_confirmed  BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX idx_meal_log_user      ON meal_log(user_id);
CREATE INDEX idx_meal_log_user_time ON meal_log(user_id, logged_at DESC);  -- history + 7-day window
```

### 2.3 `food_item` (shared USDA cache)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| food_id | SERIAL | PK | |
| usda_fdc_id | VARCHAR(20) | UNIQUE | USDA FoodData Central identifier |
| food_name | VARCHAR(200) | NOT NULL | Human-readable name from USDA |
| food101_class | VARCHAR(50) | NULL | Mapped Food-101 class label |
| calories_per_100g | NUMERIC(7,2) | | USDA lab-verified |
| protein_per_100g | NUMERIC(7,2) | | USDA lab-verified |
| carbs_per_100g | NUMERIC(7,2) | | USDA lab-verified |
| fat_per_100g | NUMERIC(7,2) | | USDA lab-verified |
| cooking_method | VARCHAR(30) | NULL | Preparation variant (grilled/fried/…) — part of the cache key |
| cached_at | TIMESTAMP | DEFAULT now() | For cache-freshness policy |

```sql
CREATE TABLE food_item (
    food_id            SERIAL PRIMARY KEY,
    usda_fdc_id        VARCHAR(20) UNIQUE,
    food_name          VARCHAR(200) NOT NULL,
    food101_class      VARCHAR(50),
    calories_per_100g  NUMERIC(7,2),
    protein_per_100g   NUMERIC(7,2),
    carbs_per_100g     NUMERIC(7,2),
    fat_per_100g       NUMERIC(7,2),
    cooking_method     VARCHAR(30),
    cached_at          TIMESTAMP NOT NULL DEFAULT now()
);
CREATE INDEX idx_food_item_class  ON food_item(food101_class);
CREATE UNIQUE INDEX idx_food_item_class_method ON food_item(food101_class, cooking_method);
```
> The `(food101_class, cooking_method)` unique index is the **cache lookup key**: "grilled chicken"
> and "fried chicken" are distinct cached rows (report §3.2.3 rationale for USDA).

### 2.4 `meal_item`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| item_id | SERIAL | PK | |
| log_id | INTEGER | FK → meal_log(log_id) ON DELETE CASCADE, NOT NULL | Parent meal |
| food_id | INTEGER | FK → food_item(food_id) ON DELETE RESTRICT, NOT NULL | Identified food |
| portion_min_g | NUMERIC(6,1) | | Lower bound of estimate (FR05) |
| portion_max_g | NUMERIC(6,1) | | Upper bound of estimate (FR05) |
| portion_confirmed_g | NUMERIC(6,1) | | User-confirmed mass (FR06) |
| confidence_score | FLOAT | | Classifier confidence 0–1 (FR04) |
| calories_est | NUMERIC(7,2) | | Derived: per-100g × confirmed/100 |
| protein_est_g | NUMERIC(7,2) | | Derived |
| carbs_est_g | NUMERIC(7,2) | | Derived |
| fat_est_g | NUMERIC(7,2) | | Derived |

```sql
CREATE TABLE meal_item (
    item_id              SERIAL PRIMARY KEY,
    log_id               INTEGER NOT NULL REFERENCES meal_log(log_id) ON DELETE CASCADE,
    food_id              INTEGER NOT NULL REFERENCES food_item(food_id) ON DELETE RESTRICT,
    portion_min_g        NUMERIC(6,1),
    portion_max_g        NUMERIC(6,1),
    portion_confirmed_g  NUMERIC(6,1),
    confidence_score     FLOAT,
    calories_est         NUMERIC(7,2),
    protein_est_g        NUMERIC(7,2),
    carbs_est_g          NUMERIC(7,2),
    fat_est_g            NUMERIC(7,2)
);
CREATE INDEX idx_meal_item_log ON meal_item(log_id);
```

### 2.5 `recommendation`

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| rec_id | SERIAL | PK | |
| user_id | INTEGER | FK → user(user_id) ON DELETE CASCADE, NOT NULL | Target user |
| generated_at | TIMESTAMP | NOT NULL DEFAULT now() | |
| insight_text | TEXT | NOT NULL | Plain-language insight (FR12) |
| nutrient_focus | VARCHAR(30) | NULL | e.g. `fibre`, `protein` |
| was_read | BOOLEAN | DEFAULT FALSE | UI read state |

```sql
CREATE TABLE recommendation (
    rec_id          SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES "user"(user_id) ON DELETE CASCADE,
    generated_at    TIMESTAMP NOT NULL DEFAULT now(),
    insight_text    TEXT NOT NULL,
    nutrient_focus  VARCHAR(30),
    was_read        BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX idx_recommendation_user ON recommendation(user_id, generated_at DESC);
```

---

## 3. SQLAlchemy 2.0 models (skeleton for `app/db/models.py`)

```python
from datetime import datetime
from sqlalchemy import String, ForeignKey, Numeric, Boolean, SmallInteger, Float, Text, Enum, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import enum

class Base(DeclarativeBase): ...

class HealthGoal(str, enum.Enum):
    weight_loss = "weight_loss"; maintenance = "maintenance"; weight_gain = "weight_gain"

class MealType(str, enum.Enum):
    breakfast = "breakfast"; lunch = "lunch"; dinner = "dinner"; snack = "snack"

class Sex(str, enum.Enum):
    female = "female"; male = "male"

class ActivityLevel(str, enum.Enum):
    sedentary = "sedentary"; light = "light"; moderate = "moderate"
    active = "active"; very_active = "very_active"

class User(Base):
    __tablename__ = "user"
    user_id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    health_goal: Mapped[HealthGoal] = mapped_column(Enum(HealthGoal, name="health_goal_enum"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    account_locked: Mapped[bool] = mapped_column(Boolean, default=False)
    # Optional body metrics (M7) — all nullable; see §2.1
    height_cm: Mapped[float | None] = mapped_column(Numeric(4, 1))
    weight_kg: Mapped[float | None] = mapped_column(Numeric(5, 1))
    birth_year: Mapped[int | None] = mapped_column(SmallInteger)
    sex: Mapped[Sex | None] = mapped_column(Enum(Sex, name="sex_enum"))
    activity_level: Mapped[ActivityLevel | None] = mapped_column(
        Enum(ActivityLevel, name="activity_level_enum"))
    failed_login_count: Mapped[int] = mapped_column(SmallInteger, default=0)
    meals: Mapped[list["MealLog"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    recommendations: Mapped[list["Recommendation"]] = relationship(cascade="all, delete-orphan")

class MealLog(Base):
    __tablename__ = "meal_log"
    log_id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.user_id", ondelete="CASCADE"), index=True)
    logged_at: Mapped[datetime] = mapped_column(server_default=func.now())
    meal_type: Mapped[MealType] = mapped_column(Enum(MealType, name="meal_type_enum"))
    total_calories: Mapped[float | None] = mapped_column(Numeric(7, 2))
    user_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    user: Mapped["User"] = relationship(back_populates="meals")
    items: Mapped[list["MealItem"]] = relationship(back_populates="log", cascade="all, delete-orphan")

class FoodItem(Base):
    __tablename__ = "food_item"
    food_id: Mapped[int] = mapped_column(primary_key=True)
    usda_fdc_id: Mapped[str | None] = mapped_column(String(20), unique=True)
    food_name: Mapped[str] = mapped_column(String(200))
    food101_class: Mapped[str | None] = mapped_column(String(50), index=True)
    calories_per_100g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    protein_per_100g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    carbs_per_100g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    fat_per_100g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    cooking_method: Mapped[str | None] = mapped_column(String(30))
    cached_at: Mapped[datetime] = mapped_column(server_default=func.now())

class MealItem(Base):
    __tablename__ = "meal_item"
    item_id: Mapped[int] = mapped_column(primary_key=True)
    log_id: Mapped[int] = mapped_column(ForeignKey("meal_log.log_id", ondelete="CASCADE"), index=True)
    food_id: Mapped[int] = mapped_column(ForeignKey("food_item.food_id", ondelete="RESTRICT"))
    portion_min_g: Mapped[float | None] = mapped_column(Numeric(6, 1))
    portion_max_g: Mapped[float | None] = mapped_column(Numeric(6, 1))
    portion_confirmed_g: Mapped[float | None] = mapped_column(Numeric(6, 1))
    confidence_score: Mapped[float | None] = mapped_column(Float)
    calories_est: Mapped[float | None] = mapped_column(Numeric(7, 2))
    protein_est_g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    carbs_est_g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    fat_est_g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    log: Mapped["MealLog"] = relationship(back_populates="items")

class Recommendation(Base):
    __tablename__ = "recommendation"
    rec_id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.user_id", ondelete="CASCADE"), index=True)
    generated_at: Mapped[datetime] = mapped_column(server_default=func.now())
    insight_text: Mapped[str] = mapped_column(Text)
    nutrient_focus: Mapped[str | None] = mapped_column(String(30))
    was_read: Mapped[bool] = mapped_column(Boolean, default=False)
```

---

## 4. Migrations (Alembic)

- Initialise once: `alembic init alembic`, point `sqlalchemy.url` at `DATABASE_URL`, set
  `target_metadata = Base.metadata`.
- Generate the first migration: `alembic revision --autogenerate -m "initial schema"`.
- **Manually verify** the autogenerated migration creates the two `ENUM` types *before* the tables
  that use them, and that FK `ondelete` values are present (Alembic sometimes drops them).
- Apply: `alembic upgrade head`. Every schema change after this ships as a new revision — never edit
  the DB by hand.

---

## 5. Seed data

Two seed scripts (idempotent) in `backend/scripts/`:

1. **`seed_food101_classes.py`** — inserts the 101 Food-101 class labels into `food_item.food101_class`
   with `usda_fdc_id = NULL` and null macros (they fill in lazily on first USDA lookup). Source the
   class list from `models/labels_food101.json` (see `04`).
2. **`seed_demo_user.py`** (dev only) — one demo user (`demo@nutrilens.app` / `Demo1234!`,
   goal = `maintenance`) plus 7 confirmed meals so the recommendation engine can be demoed
   immediately. **Guard behind `APP_ENV=development`.**

---

## 6. Data-integrity rules (enforced in services, tested in `11`)

- `portion_min_g ≤ portion_confirmed_g ≤ portion_max_g` is **not** a hard DB constraint (user may
  override), but the service logs a warning if confirmed is outside the range.
- `meal_log.total_calories` is always recomputed = Σ `meal_item.calories_est` on write (never trusted
  from the client).
- Derived macro fields (`*_est`) are computed server-side only; the client never sends them.
- A `meal_log` with `user_confirmed = FALSE` is a draft and is **excluded** from history and from the
  recommendation window.

---

Prev: [`01-architecture.md`](./01-architecture.md) · Next: [`03-api-specification.md`](./03-api-specification.md)
