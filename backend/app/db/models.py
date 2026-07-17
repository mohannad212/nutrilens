"""
app/db/models.py
===================

WHY THIS FILE EXISTS
---------------------
This is the ORM layer: Python classes that map 1:1 onto the five Postgres
tables designed in Chapter 5 / PRD `02-database-schema.md`
(user, meal_log, food_item, meal_item, recommendation). It is the single
source of truth for "what does our data look like" from Python's side —
Alembic reads it to generate migrations, and every service in the app
reads/writes the database exclusively through these classes.

WHAT PROBLEM IT SOLVES
------------------------
Without an ORM, every query would be hand-written SQL strings passed
around as loose dicts — no autocomplete, no type checking, no protection
against typos in column names, and SQL injection risk if you're not
careful with string formatting. The ORM gives you Python objects
(`user.email`, `meal.items`) with real relationships you can traverse
in code, while SQLAlchemy generates the parameterised SQL underneath.

HOW IT WORKS INTERNALLY
--------------------------
Each class inherits from `Base` (see `db/base.py`). `mapped_column(...)`
declares a real database column with its type and constraints. `Mapped[T]`
is a typing.  annotation SQLAlchemy 2.0 uses for static type checking AND
to infer the SQL type when possible. `relationship(...)` does NOT create a
column — it tells the ORM "when I access `user.meals`, run a SELECT on
meal_log WHERE user_id = this user's id" (lazy-loaded by default). The two
`enum.Enum` subclasses map directly onto Postgres `ENUM` types (created
once, in the first Alembic migration), matching FR01's `health_goal` and
the domain values for `meal_type`.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- SQLAlchemy 2.0 declarative style with `Mapped`/`mapped_column` (chosen):
  the current SQLAlchemy idiom — full static typing, works with mypy.
- The older SQLAlchemy 1.x `Column(...)` style: still works but has no
  type-checker support and is being phased out upstream.
- An ORM-less "raw SQL + psycopg" approach: more control, but you rebuild
  everything the ORM gives you for free (change tracking, relationships,
  Alembic diffing) — not worth it for a 5-table schema with clear
  relationships.
- `ON DELETE CASCADE` on the owned chain (user → meal_log → meal_item) vs
  `ON DELETE RESTRICT` on the shared `food_item` reference table: this
  directly implements FR14 (account deletion) as a single
  `DELETE FROM "user" WHERE user_id = :id` that cleans up everything the
  user owns, while a shared USDA cache row can never be silently deleted
  out from under other users' meal history.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why is `food_item` NOT cascade-deleted with a meal?"
A: `food_item` is a *shared reference table* (the USDA nutrition cache) —
   many users' meal_items point at the same "grilled chicken breast" row.
   Deleting a meal must never delete nutrition data other users depend on,
   so the FK uses `ondelete="RESTRICT"` instead of `CASCADE`.

Q: "Why store `health_goal` as a Postgres ENUM instead of a plain string?"
A: A DB-level ENUM rejects invalid values (`'bulking'`) at the database
   layer, not just in application code — a second line of defense if a
   bug or a different client ever writes directly to the DB.

Q: "Why `Numeric` instead of `Float` for calories/macros?"
A: `Numeric` is a fixed-precision decimal type — no floating-point
   rounding drift when summing many small values (e.g. macro totals
   across meal items). `Float` (used only for `confidence_score`) is fine
   there because a classifier confidence is inherently an approximation,
   not a value users expect to add up exactly.

Q: "What does `back_populates` do, and why does it appear in pairs?"
A: It links two `relationship()` declarations (e.g. `User.meals` and
   `MealLog.user`) so that appending to one side keeps the in-memory
   object graph consistent on the other side, without an extra query.

COMMON STUDENT MISTAKES
--------------------------
- Forgetting `ondelete=` on the FK definition in Python — SQLAlchemy will
  happily let you define a relationship without it, but then the actual
  Postgres constraint has no cascade behavior, and deleting a user throws
  a foreign-key violation instead of cascading.
- Using `cascade="all, delete-orphan"` on the SQLAlchemy relationship
  without ALSO setting `ondelete="CASCADE"` on the DB column — the first
  is an ORM-level behavior (only fires if you delete via a loaded Session
  object), the second is a DB-level guarantee that fires even for raw
  SQL deletes. NutriLens needs both for FR14 to be truly irreversible.
- Mixing up `Mapped[str | None]` (nullable column) vs `Mapped[str]`
  (NOT NULL) — this file is deliberately explicit about which columns are
  optional, matching the `NULL`/`NOT NULL` markers in `02`.
"""

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Enum,
    Float,
    ForeignKey,
    Index,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


# ---------------------------------------------------------------------------
# ENUM types — mirror Postgres CREATE TYPE ... AS ENUM (see 02 §2.1, §2.2)
# ---------------------------------------------------------------------------
class HealthGoal(str, enum.Enum):
    weight_loss = "weight_loss"
    maintenance = "maintenance"
    weight_gain = "weight_gain"


class MealType(str, enum.Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"
    snack = "snack"


# ---------------------------------------------------------------------------
# user  — FR01 (registration), FR02 (auth + lockout), FR13 (health goal)
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "user"
    # Index names mirror the Alembic migration exactly so ORM metadata and the
    # migration never drift (a future --autogenerate would otherwise churn them).
    __table_args__ = (Index("idx_user_email", "email"),)

    user_id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    health_goal: Mapped[HealthGoal] = mapped_column(
        Enum(HealthGoal, name="health_goal_enum"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    account_locked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    failed_login_count: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=0)

    meals: Mapped[list["MealLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# meal_log — one row per logged meal (FR10)
# ---------------------------------------------------------------------------
class MealLog(Base):
    __tablename__ = "meal_log"
    __table_args__ = (
        Index("idx_meal_log_user", "user_id"),
        Index("idx_meal_log_user_time", "user_id", text("logged_at DESC")),
    )

    log_id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.user_id", ondelete="CASCADE"), nullable=False
    )
    logged_at: Mapped[datetime] = mapped_column(server_default=func.now())
    meal_type: Mapped[MealType] = mapped_column(
        Enum(MealType, name="meal_type_enum"), nullable=False
    )
    total_calories: Mapped[float | None] = mapped_column(Numeric(7, 2))
    user_confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    user: Mapped["User"] = relationship(back_populates="meals")
    items: Mapped[list["MealItem"]] = relationship(
        back_populates="log", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# food_item — shared USDA nutrition cache (see 02 §2.3, "cache lookup key")
# ---------------------------------------------------------------------------
class FoodItem(Base):
    __tablename__ = "food_item"
    __table_args__ = (
        Index("idx_food_item_class", "food101_class"),
        Index("idx_food_item_class_method", "food101_class", "cooking_method", unique=True),
    )

    food_id: Mapped[int] = mapped_column(primary_key=True)
    usda_fdc_id: Mapped[str | None] = mapped_column(String(20), unique=True)
    food_name: Mapped[str] = mapped_column(String(200), nullable=False)
    food101_class: Mapped[str | None] = mapped_column(String(50))
    calories_per_100g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    protein_per_100g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    carbs_per_100g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    fat_per_100g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    cooking_method: Mapped[str | None] = mapped_column(String(30))
    cached_at: Mapped[datetime] = mapped_column(server_default=func.now())

    meal_items: Mapped[list["MealItem"]] = relationship(back_populates="food")


# ---------------------------------------------------------------------------
# meal_item — one row per identified food within a meal (FR04-FR08)
# ---------------------------------------------------------------------------
class MealItem(Base):
    __tablename__ = "meal_item"
    __table_args__ = (Index("idx_meal_item_log", "log_id"),)

    item_id: Mapped[int] = mapped_column(primary_key=True)
    log_id: Mapped[int] = mapped_column(
        ForeignKey("meal_log.log_id", ondelete="CASCADE"), nullable=False
    )
    food_id: Mapped[int] = mapped_column(
        ForeignKey("food_item.food_id", ondelete="RESTRICT"), nullable=False
    )
    portion_min_g: Mapped[float | None] = mapped_column(Numeric(6, 1))
    portion_max_g: Mapped[float | None] = mapped_column(Numeric(6, 1))
    portion_confirmed_g: Mapped[float | None] = mapped_column(Numeric(6, 1))
    confidence_score: Mapped[float | None] = mapped_column(Float)
    calories_est: Mapped[float | None] = mapped_column(Numeric(7, 2))
    protein_est_g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    carbs_est_g: Mapped[float | None] = mapped_column(Numeric(7, 2))
    fat_est_g: Mapped[float | None] = mapped_column(Numeric(7, 2))

    log: Mapped["MealLog"] = relationship(back_populates="items")
    food: Mapped["FoodItem"] = relationship(back_populates="meal_items")


# ---------------------------------------------------------------------------
# recommendation — generated dietary insights (FR12)
# ---------------------------------------------------------------------------
class Recommendation(Base):
    __tablename__ = "recommendation"
    __table_args__ = (Index("idx_recommendation_user", "user_id", text("generated_at DESC")),)

    rec_id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.user_id", ondelete="CASCADE"), nullable=False
    )
    generated_at: Mapped[datetime] = mapped_column(server_default=func.now())
    insight_text: Mapped[str] = mapped_column(Text, nullable=False)
    nutrient_focus: Mapped[str | None] = mapped_column(String(30))
    was_read: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    user: Mapped["User"] = relationship(back_populates="recommendations")
