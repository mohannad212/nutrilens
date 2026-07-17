"""
tests/test_m1_models.py
==========================

WHY THIS FILE EXISTS
---------------------
M1's Definition of Done (see `09-build-phases.md`) explicitly requires:
"a unit test round-trips a User with a hashed password." This file is
that test, plus a couple of closely related ones that exercise the ORM
relationships and cascade/restrict behavior the schema design depends on.

WHAT EACH TEST VALIDATES AND WHY IT MATTERS
-----------------------------------------------
- test_user_roundtrip_with_hashed_password: proves the full path
  works end-to-end — hash a password, save a User, reload it from the
  database, and confirm (a) the stored value is never the plaintext
  password and (b) `verify_password` still correctly authenticates the
  original plaintext against the stored hash. This is the core guarantee
  behind FR01/FR02 — if this test passed with plaintext or a broken hash
  comparison, real user passwords could be exposed by a database leak.

- test_meal_log_cascades_with_user: proves the `ON DELETE CASCADE`
  relationship between `user` and `meal_log` actually fires — deleting a
  user removes their meal logs too, which is exactly what FR14 (account
  deletion) depends on structurally, not just as an application-level
  promise.

- test_food_item_restrict_prevents_delete_while_referenced: proves the
  *opposite* policy on the shared `food_item` table — a food row that a
  `meal_item` still references cannot be deleted, protecting the shared
  USDA cache from being silently destroyed by an unrelated user's data
  cleanup.
"""

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.models import FoodItem, HealthGoal, MealItem, MealLog, MealType, User
from app.security import hash_password, verify_password


def test_user_roundtrip_with_hashed_password(db_session):
    plaintext_password = "CorrectHorseBattery9!"

    user = User(
        username="shatha",
        email="shatha@example.com",
        password_hash=hash_password(plaintext_password),
        health_goal=HealthGoal.maintenance,
    )
    db_session.add(user)
    db_session.commit()

    reloaded = db_session.query(User).filter(User.email == "shatha@example.com").one()

    # The stored value must never equal the plaintext password.
    assert reloaded.password_hash != plaintext_password
    # But the original password must still verify correctly against it.
    assert verify_password(plaintext_password, reloaded.password_hash) is True
    # And an incorrect password must NOT verify.
    assert verify_password("wrong-password", reloaded.password_hash) is False


def test_meal_log_cascades_with_user(db_session):
    user = User(
        username="ahmad",
        email="ahmad@example.com",
        password_hash=hash_password("Password123!"),
        health_goal=HealthGoal.weight_loss,
    )
    db_session.add(user)
    db_session.flush()

    meal = MealLog(user_id=user.user_id, meal_type=MealType.lunch)
    db_session.add(meal)
    db_session.commit()

    db_session.delete(user)
    db_session.commit()

    remaining = db_session.query(MealLog).filter(MealLog.user_id == user.user_id).all()
    assert remaining == []


def test_food_item_restrict_prevents_delete_while_referenced(db_session):
    user = User(
        username="referenced_user",
        email="ref@example.com",
        password_hash=hash_password("Password123!"),
        health_goal=HealthGoal.maintenance,
    )
    food = FoodItem(food_name="Omelette", food101_class="omelette")
    db_session.add_all([user, food])
    db_session.flush()

    meal = MealLog(user_id=user.user_id, meal_type=MealType.breakfast)
    db_session.add(meal)
    db_session.flush()

    db_session.add(MealItem(log_id=meal.log_id, food_id=food.food_id, portion_confirmed_g=200.0))
    db_session.commit()

    db_session.delete(food)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()
