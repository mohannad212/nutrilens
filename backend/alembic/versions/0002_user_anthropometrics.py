"""user anthropometrics for TDEE-based calorie insights

Revision ID: 0002_user_anthropometrics
Revises: 0001_initial_schema
Create Date: 2026-07-30

WHY THIS FILE EXISTS
---------------------
`06-recommendation-engine.md §2` specifies a `calorie_gap` insight that tells
the user whether their average daily intake is above or below what their goal
suggests. The M1 schema has no way to answer that: it stores no height, weight,
age, sex or activity level, so there is no maintenance-energy estimate to
compare the average against. This migration adds those five columns to `user`
so the engine can compute a Mifflin-St Jeor TDEE per user instead of judging
their intake against a population average (which would be a fabricated,
un-defendable number in a nutrition app).

WHAT PROBLEM IT SOLVES
------------------------
Without a per-user energy estimate the engine has exactly two bad options:
invent a generic target ("2000 kcal") and present it as personal advice, or
drop the calorie insight and lose a headline part of FR12. Both are worse than
asking for five optional data points. Macro *distribution* insights (protein /
carbs / fat as % of energy) are scale-free and need none of this — which is why
every new column is NULLABLE and the engine still works fully without them.

HOW IT WORKS INTERNALLY
--------------------------
Same two-step ordering rule as `0001`: the two new Postgres ENUM types
(`sex_enum`, `activity_level_enum`) are created *before* the columns that use
them, because `ALTER TABLE ... ADD COLUMN sex sex_enum` fails if the type is
not already in Postgres's type catalogue. `op.add_column` then appends each
column; because every one is nullable with no server default, this is a
metadata-only change on Postgres (no table rewrite, no lock held while rows are
backfilled) and existing rows simply get NULL.

`downgrade()` drops the columns first and the types second — a type cannot be
dropped while any column still references it.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Five nullable columns on `user` (chosen): the data is 1:1 with the user, read
  on every recommendation refresh, and tiny. A plain column read beats a join.
- A separate `user_profile` table: normalises "optional profile data" away from
  the auth-critical row, but buys nothing at this scale — it is a guaranteed
  1:1 join on every refresh for five scalars.
- `birth_year SMALLINT` rather than a full `date_of_birth`: the engine only
  needs age in whole years for the BMR equation, and storing a year instead of
  an exact birth date keeps one less piece of directly identifying data (NFR
  privacy). Age is derived as `current_year - birth_year`.
- `NUMERIC` for height/weight rather than `FLOAT`: consistent with every other
  measurement column in `02` — no binary-float surprises when a user's 70.5 kg
  round-trips through the API.
- `sex_enum` with exactly two values, and the column nullable: Mifflin-St Jeor
  needs a sex-specific constant (+5 / -161), and there is no meaningful third
  constant to offer. Nullability is what protects the user who would rather not
  say — leaving the column empty costs only the calorie insight, since every
  macro insight is a ratio and needs no body data at all.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why are these nullable when the calorie insight needs them?"
A: Because FR12's other insights don't. Macro percentages are ratios and stay
   valid at any body size, so a user who skips these fields still gets protein
   / carb / fat insights; only the calorie line softens to a descriptive
   observation. Making the columns NOT NULL would have forced a breaking change
   on FR01's registration form and locked out existing accounts.

Q: "Is adding columns to a live table safe?"
A: For nullable columns with no default, yes — Postgres records the change in
   the catalogue without rewriting existing rows. Adding a NOT NULL column with
   a default is the variant that historically rewrote the whole table.

Q: "Why not just ask the user for their calorie target directly?"
A: A self-reported target invites exactly the behaviour `06 §5` forbids — a
   user in a restrictive mindset could enter an unsafely low number and the app
   would validate it. A computed TDEE with a deliberately mild goal posture
   cannot be talked into an extreme deficit.

COMMON STUDENT MISTAKES
--------------------------
- Forgetting `create_type=False` on the `postgresql.ENUM` passed to
  `add_column` — Alembic then tries to `CREATE TYPE` a second time and the
  migration dies with "type already exists".
- Dropping the ENUM type before the column that uses it in `downgrade()` —
  Postgres refuses with a dependency error.
- Adding the columns to `db/models.py` and never writing this migration: the
  ORM and the real database silently drift, and the code works locally (where
  tests build tables from `Base.metadata`) but 500s in production.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002_user_anthropometrics"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SEX_VALUES = ("female", "male")
ACTIVITY_VALUES = ("sedentary", "light", "moderate", "active", "very_active")


def upgrade() -> None:
    # --- 1. ENUM types first (must exist before a column can be typed with them) ---
    postgresql.ENUM(*SEX_VALUES, name="sex_enum").create(op.get_bind(), checkfirst=True)
    postgresql.ENUM(*ACTIVITY_VALUES, name="activity_level_enum").create(
        op.get_bind(), checkfirst=True
    )

    # --- 2. columns (all nullable -> metadata-only change, no table rewrite) ---
    op.add_column("user", sa.Column("height_cm", sa.Numeric(4, 1), nullable=True))
    op.add_column("user", sa.Column("weight_kg", sa.Numeric(5, 1), nullable=True))
    op.add_column("user", sa.Column("birth_year", sa.SmallInteger(), nullable=True))
    op.add_column(
        "user",
        sa.Column(
            "sex",
            postgresql.ENUM(*SEX_VALUES, name="sex_enum", create_type=False),
            nullable=True,
        ),
    )
    op.add_column(
        "user",
        sa.Column(
            "activity_level",
            postgresql.ENUM(*ACTIVITY_VALUES, name="activity_level_enum", create_type=False),
            nullable=True,
        ),
    )


def downgrade() -> None:
    # Columns before types: a type cannot be dropped while a column still uses it.
    op.drop_column("user", "activity_level")
    op.drop_column("user", "sex")
    op.drop_column("user", "birth_year")
    op.drop_column("user", "weight_kg")
    op.drop_column("user", "height_cm")

    postgresql.ENUM(name="activity_level_enum").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="sex_enum").drop(op.get_bind(), checkfirst=True)
