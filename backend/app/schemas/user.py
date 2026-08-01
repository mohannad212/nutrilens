"""
app/schemas/user.py
======================

WHY THIS FILE EXISTS
---------------------
The request and response shapes for `/users` (`03 §3`): read the current profile,
change the health goal (FR13), and fill in the optional body metrics that M7's
calorie insight needs. Three small endpoints, but two of them have semantics
subtle enough to be worth stating precisely.

WHY THE PROFILE RESPONSE CARRIES THE BODY METRICS
----------------------------------------------------
`03 §3`'s example for `GET /users/me` shows five fields and none of the metrics.
This schema adds them, deliberately and additively, because the contract does not
otherwise work end to end:

- `PATCH /users/me/metrics` is specified as returning "the updated profile". If
  the profile shape excludes the very fields that endpoint edits, its response
  cannot show what it just changed.
- The Profile screen (`07 §2`, Screen 8) has to render the *current* values in
  its form. Without them it would render empty inputs over stored data, and a
  user editing one field would appear to have lost the rest.
- M7's calorie insight names exactly which fields are still missing and points
  the user here (`03 §3`). Arriving at a screen that cannot show which are set
  makes that prompt unfollowable.

Nothing is removed, so any client written against `03 §3`'s example keeps working.

THE PARTIAL-UPDATE SEMANTICS OF `MetricsUpdateRequest`
---------------------------------------------------------
`03 §3` states the rule: "Omitted keys are left unchanged; an explicit `null`
clears a field." Those are genuinely different operations on a nullable column,
and a naive implementation cannot tell them apart — both arrive as `None` on the
parsed model.

Pydantic's `model_fields_set` is what distinguishes them: it records which keys
were actually present in the incoming JSON. So `{"weight_kg": 71.0}` touches only
the weight, while `{"sex": null}` explicitly clears the sex. Getting this wrong
in the obvious direction is destructive — a screen that PATCHes one field would
wipe the other four to null, silently deleting data the user entered at
registration.

WHY VALIDATION IS SHARED WITH REGISTRATION
---------------------------------------------
`03 §3`: "Validation is identical to `/auth/register`." That is enforced
structurally by inheriting `AnthropometricsMixin` rather than by restating the
bounds, so the two endpoints cannot drift. It matters because the bounds are not
cosmetic: these values are multiplied into a calorie figure presented as personal
advice, so a unit mix-up (`height_cm: 1.78`) has to be refused at the edge rather
than surfacing as confident nonsense in an insight (`03 §2`).

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why is the goal its own endpoint rather than part of a general profile PATCH?"
A: `03 §3` separates them, and the separation is meaningful: the goal has a
   documented side effect (the next recommendation refresh targets a different
   macro distribution, `06`), while the metrics only sharpen an existing insight.
   Distinct endpoints keep that side effect attached to the one request that
   causes it.

Q: "Can a user change their email or username here?"
A: No, and there is no field for either. Both are unique login identifiers, so
   changing them is an account-identity operation with its own collision
   handling, and `03 §3` does not include it. It is out of scope (`00 §Non-goals`).

COMMON STUDENT MISTAKES
--------------------------
- Implementing the metrics PATCH with `model_dump()` instead of
  `model_dump(exclude_unset=True)`, so every unmentioned field is set to null.
- Duplicating the range checks here instead of inheriting them, so a later change
  to the registration bounds silently does not apply to edits.
- Returning the ORM `User` directly, publishing `password_hash` to the client.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr

from app.db.models import ActivityLevel, HealthGoal, Sex
from app.schemas.auth import AnthropometricsMixin


class UserProfile(BaseModel):
    """`GET /users/me` and the body of both PATCH responses (`03 §3`).

    `from_attributes` lets this be built straight from the ORM row, but only the
    fields named here are published — `password_hash`, `account_locked` and
    `failed_login_count` are absent by construction, not by remembering to strip
    them.
    """

    model_config = ConfigDict(from_attributes=True)

    user_id: int
    username: str
    email: EmailStr
    health_goal: HealthGoal
    created_at: datetime

    # Additive beyond `03 §3`'s example — see the module docstring.
    height_cm: float | None = None
    weight_kg: float | None = None
    birth_year: int | None = None
    sex: Sex | None = None
    activity_level: ActivityLevel | None = None


class GoalUpdateRequest(BaseModel):
    """`PATCH /users/me/goal` (FR13). One field, because it has one side effect."""

    health_goal: HealthGoal


class MetricsUpdateRequest(AnthropometricsMixin):
    """`PATCH /users/me/metrics`. Every field optional; see the partial-update note.

    Inherits its validation from the same mixin `RegisterRequest` uses, so the two
    entry points for these values cannot disagree about what is acceptable.
    """

    def provided_fields(self) -> dict[str, object]:
        """Only the keys actually present in the request body.

        This is the whole partial-update mechanism: `exclude_unset=True` drops
        fields the client never mentioned, while keeping a field the client sent
        as an explicit `null` — which is what turns "leave alone" and "clear"
        into two distinguishable operations (`03 §3`).
        """
        return self.model_dump(exclude_unset=True)
