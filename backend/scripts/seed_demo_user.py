"""
scripts/seed_demo_user.py
============================

WHY THIS FILE EXISTS
---------------------
The recommendation engine (FR12, built in M7) only activates once a user
has ≥7 confirmed meal logs. Manually creating a user and photographing 7
meals through the app every time you want to demo or test that feature is
slow. This script creates one demo account with 7 confirmed meals in a
single command, purely for local development and demos.

WHAT PROBLEM IT SOLVES
------------------------
Lets a developer (or the supervisor during a live defense) see the "core
loop complete" and "recommendations unlocked" states immediately, without
manually logging seven real meals through the mobile app first.

HOW IT WORKS INTERNALLY
--------------------------
Same idempotent "check-then-insert" pattern as
`seed_food101_classes.py`: checks whether `demo@nutrilens.app` already
exists before creating anything. If not, it hashes a known password with
`hash_password` (never storing plaintext, even for a demo account), then
inserts one `User`, seven `MealLog` rows (each `user_confirmed=True`, each
linked to a `MealItem` pointing at an existing seeded `food_item` row) so
the demo account satisfies FR10's minimum-7-meals precondition for FR12
out of the box.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- A **guarded** dev-only script (chosen), refusing to run unless
  `APP_ENV == "development"`: prevents an accidental `python
  seed_demo_user.py` in a production/staging environment from creating a
  publicly-known demo login (`demo@nutrilens.app` / `Demo1234!`) with
  guessable credentials on a real deployment.
- Running this automatically on every app startup: rejected — seeding test
  data on every restart of a real server is exactly the kind of surprising
  side effect that should require an explicit, intentional command.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why guard this behind APP_ENV instead of just telling developers not
    to run it in production?"
A: Defense in depth — humans forget or copy-paste commands between
   terminals. A hard-coded, publicly-documented demo password must be
   structurally incapable of being seeded into a real environment.

Q: "Why does the demo password still go through hash_password() instead
    of being stored in cleartext 'since it's just a demo'?"
A: There is no legitimate reason for ANY password, demo or otherwise, to
   ever exist unhashed in the users table — consistency here is itself
   part of the security posture graders will look for.

COMMON STUDENT MISTAKES
--------------------------
- Forgetting the `APP_ENV` guard entirely, so `seed_demo_user.py` quietly
  works (and is discoverable) against a production database.
- Storing the demo password in plaintext "because it's fake data anyway."
- Not marking the seeded meals `user_confirmed=True` — since the
  recommendation query filters on confirmed meals only, unconfirmed seed
  data wouldn't actually unlock FR12 for the demo.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Allow `python scripts/seed_demo_user.py` from backend/ (adds backend/ to the
# import path even when the package isn't editable-installed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db.base import SessionLocal  # noqa: E402
from app.db.models import (  # noqa: E402
    FoodItem,
    HealthGoal,
    MealItem,
    MealLog,
    MealType,
    User,
)
from app.security import hash_password  # noqa: E402

DEMO_EMAIL = "demo@nutrilens.app"
DEMO_USERNAME = "demo_user"
DEMO_PASSWORD = "Demo1234!"  # dev-only, never used against a real deployment


def seed_demo_user() -> None:
    if settings.app_env != "development":
        print(
            "Refusing to run: APP_ENV is "
            f"'{settings.app_env}', not 'development'. This script only "
            "runs in local/dev environments."
        )
        sys.exit(1)

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == DEMO_EMAIL).first()
        if existing is not None:
            print("seed_demo_user: demo user already exists, nothing to do.")
            return

        user = User(
            username=DEMO_USERNAME,
            email=DEMO_EMAIL,
            password_hash=hash_password(DEMO_PASSWORD),
            health_goal=HealthGoal.maintenance,
        )
        db.add(user)
        db.flush()  # assigns user.user_id without committing yet

        # Reuse a seeded food_item so this script can run right after
        # seed_food101_classes.py without creating a duplicate class row.
        food = db.query(FoodItem).filter(FoodItem.food101_class == "omelette").first()
        if food is None:
            raise RuntimeError("No seeded food_item found — run seed_food101_classes.py first.")

        now = datetime.now(UTC)
        for day_offset in range(7):
            meal = MealLog(
                user_id=user.user_id,
                logged_at=now - timedelta(days=day_offset),
                meal_type=MealType.breakfast,
                total_calories=350.00,
                user_confirmed=True,
            )
            db.add(meal)
            db.flush()

            db.add(
                MealItem(
                    log_id=meal.log_id,
                    food_id=food.food_id,
                    portion_min_g=180.0,
                    portion_max_g=220.0,
                    portion_confirmed_g=200.0,
                    confidence_score=0.91,
                    calories_est=350.00,
                    protein_est_g=22.0,
                    carbs_est_g=4.0,
                    fat_est_g=26.0,
                )
            )

        db.commit()
        print(
            f"seed_demo_user: created {DEMO_EMAIL} / {DEMO_PASSWORD} "
            "with 7 confirmed meals (recommendation threshold met)."
        )
    finally:
        db.close()


if __name__ == "__main__":
    seed_demo_user()
