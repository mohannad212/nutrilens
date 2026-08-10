"""
app/schemas/meal.py
======================

WHY THIS FILE EXISTS
---------------------
The request and response shapes for `POST /meals` (`03 §6`) — the call that turns
a confirmed recognition into a permanent row. `03` is a contract, and this is the
backend's binding half of it for the endpoint that closes the core loop.

WHAT PROBLEM IT SOLVES
------------------------
The request model is where "the server never trusts the client with derived
values" (`02 §6`, `03 §6`) stops being a policy and becomes a fact about the type
system. `MealItemIn` has fields for what the user confirmed — the food, how it was
cooked, how much of it there is — and **no field for calories, macros, or a food
name**. There is nowhere to put a forged nutrition figure, so no service further
down has to remember to ignore one. That is a stronger guarantee than validation:
the value cannot be expressed, let alone accepted.

WHY `logged_at` IS ABSENT FROM THE REQUEST
---------------------------------------------
`09` M7 asks M5 to "reject or ignore a client-supplied future `logged_at`", and
omitting the field entirely is the strongest available form of that. A meal dated
next month would sit at the top of a newest-first history forever and re-enter the
recommendation window when its date arrived. The server clock is the only source,
and an unknown key in the body is ignored by Pydantic's default behaviour rather
than silently honoured.

WHY `items` IS BOUNDED
-------------------------
At least one item, because a meal with no food in it has nothing to log and would
write a `meal_log` with a total of zero into the recommendation engine's averages.
At most ten, because each item is potentially a USDA round trip inside one request
(`08 §5`) — an unbounded list is a way for a single call to spend the hourly quota
and blow through the 5 s Performance NFR. Ten is far past any real plate.

WHY THE RESPONSE ECHOES `meals_logged_total` AND `recommendations_unlocked`
------------------------------------------------------------------------------
`03 §6` specifies both. They matter at precisely this moment and nowhere else: the
user has just logged a meal, and if it was their seventh, FR12 has this instant
become available. Without these fields the app would have to poll
`/recommendations` after every single log to find out.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Why does the response repeat each item's macros when the client just saw them
    on the nutrition screen?"
A: Because these are the values that were *persisted*, computed server-side at
   write time, and they are what history will show. Echoing them lets the client
   display exactly what was stored rather than what it last calculated, and it
   makes any divergence between the two immediately visible instead of silent.

Q: "`confidence_score` comes from the client. Isn't that a trusted value?"
A: It is the classifier's own output being handed back for storage, and it is
   descriptive metadata (`02 §2.4`) — it feeds no calculation. A wrong value
   misreports how sure the model was, which is worth having and not worth a second
   round trip to re-derive. Nothing downstream computes on it.

COMMON STUDENT MISTAKES
--------------------------
- Adding `calories` to the request body "so the client does not have to wait",
  which hands a health record's contents to the least trustworthy participant.
- Allowing an empty `items` list, which writes a meaningless zero-calorie meal
  into the recommendation window.
- Letting the response model serialise the ORM row directly, publishing `user_id`
  and every column added later without anyone deciding to.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.constants import NUTRITION_DISCLAIMER
from app.core.enums import CookingMethod
from app.db.models import MealLog, MealType
from app.schemas.nutrition import MAX_PORTION_G

MAX_ITEMS_PER_MEAL = 10


class MealItemIn(BaseModel):
    """One item of `03 §6`'s request body — user assertions only."""

    food101_class: str = Field(
        min_length=1,
        max_length=50,
        pattern=r"^[a-z0-9_]+$",
        description="Raw Food-101 class as returned by /recognise",
    )
    cooking_method: CookingMethod
    portion_confirmed_g: float = Field(
        gt=0, le=MAX_PORTION_G, description="User-confirmed mass (FR06)"
    )
    portion_min_g: float | None = Field(
        default=None, ge=0, le=MAX_PORTION_G, description="Lower bound from /recognise (FR05)"
    )
    portion_max_g: float | None = Field(
        default=None, ge=0, le=MAX_PORTION_G, description="Upper bound from /recognise (FR05)"
    )
    confidence_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Classifier confidence (FR04); metadata only"
    )


class MealCreateRequest(BaseModel):
    """`POST /meals` request body (`03 §6`). Note the absence of `logged_at`."""

    meal_type: MealType
    items: list[MealItemIn] = Field(min_length=1, max_length=MAX_ITEMS_PER_MEAL)


def display_name_for(food101_class: str | None, food_name: str) -> str:
    """The short human label for a logged item.

    `food_name` is the USDA description, which is written for a nutrient database
    rather than a person: "Potatoes, frozen, french fried, par fried, extruded,
    unprepared". Correct, and the right thing to keep as provenance — but it
    cannot be read at a glance in a history row, and truncating it is worse than
    useless ("OLIVE GARDEN, spaghetti with meat sauce" cut at the first comma
    becomes "OLIVE GARDEN").

    The Food-101 class is the label the user actually chose on the results
    screen, so it is what they expect to see again. Falls back to the USDA name
    for rows whose class was never recorded (`food101_class` is nullable — see
    the seeding note in `nutrition_cache`).
    """
    if not food101_class:
        return food_name
    return food101_class.replace("_", " ").title()


class MealItemOut(BaseModel):
    """One persisted `meal_item`, as `03 §6` returns it."""

    model_config = ConfigDict(from_attributes=True)

    item_id: int
    food_name: str
    #: Short label derived from the Food-101 class; `food_name` stays the USDA
    #: description so provenance is never lost.
    display_name: str
    calories_est: float
    protein_est_g: float
    carbs_est_g: float
    fat_est_g: float
    portion_confirmed_g: float


class MealSummary(BaseModel):
    """One meal as `GET /meals` and `GET /meals/{id}` return it (`03 §6`).

    `item_count` is published alongside `items` even though the client could take
    the list's length. It is what the History row renders ("2 items") without
    reading the nested array, and it keeps that row correct if a future page ever
    returns summaries with the items omitted for size.
    """

    log_id: int
    logged_at: datetime
    meal_type: MealType
    total_calories: float
    item_count: int
    items: list[MealItemOut]

    @classmethod
    def from_meal(cls, meal: MealLog) -> "MealSummary":
        items = [
            MealItemOut(
                item_id=item.item_id,
                food_name=item.food.food_name,
                display_name=display_name_for(item.food.food101_class, item.food.food_name),
                calories_est=float(item.calories_est or 0),
                protein_est_g=float(item.protein_est_g or 0),
                carbs_est_g=float(item.carbs_est_g or 0),
                fat_est_g=float(item.fat_est_g or 0),
                portion_confirmed_g=float(item.portion_confirmed_g or 0),
            )
            for item in meal.items
        ]
        return cls(
            log_id=meal.log_id,
            logged_at=meal.logged_at,
            meal_type=meal.meal_type,
            total_calories=float(meal.total_calories or 0),
            item_count=len(items),
            items=items,
        )


class MealListResponse(BaseModel):
    """`GET /meals` (`03 §6`) — a page of history plus what it is a page *of*.

    `total` counts every meal matching the filters, not the rows in `meals`. The
    History screen needs that to know whether to keep paging; without it the only
    signal is "a short page", which is indistinguishable from the last page.
    """

    total: int
    limit: int
    offset: int
    meals: list[MealSummary]


class MealDeleteResponse(BaseModel):
    """`DELETE /meals/{log_id}` (`03 §6`)."""

    deleted: bool = True


class MealCreateResponse(BaseModel):
    """`POST /meals` response body (`03 §6`)."""

    log_id: int
    logged_at: datetime
    meal_type: MealType
    total_calories: float
    items: list[MealItemOut]
    meals_logged_total: int
    recommendations_unlocked: bool
    # The meal's totals are estimates built from estimated portions, so the same
    # `03 §5` disclaimer that governs a nutrition response governs this one.
    disclaimer: str = NUTRITION_DISCLAIMER

    @classmethod
    def from_meal(
        cls, meal: MealLog, meals_logged_total: int, recommendations_unlocked: bool
    ) -> "MealCreateResponse":
        """Map the persisted `MealLog` (and its items) onto the contract shape.

        Reads from the committed ORM rows rather than from anything computed in
        the router, so the response is a report of what is actually in the
        database.
        """
        return cls(
            log_id=meal.log_id,
            logged_at=meal.logged_at,
            meal_type=meal.meal_type,
            total_calories=float(meal.total_calories or 0),
            items=[
                MealItemOut(
                    item_id=item.item_id,
                    food_name=item.food.food_name,
                    display_name=display_name_for(item.food.food101_class, item.food.food_name),
                    calories_est=float(item.calories_est or 0),
                    protein_est_g=float(item.protein_est_g or 0),
                    carbs_est_g=float(item.carbs_est_g or 0),
                    fat_est_g=float(item.fat_est_g or 0),
                    portion_confirmed_g=float(item.portion_confirmed_g or 0),
                )
                for item in meal.items
            ],
            meals_logged_total=meals_logged_total,
            recommendations_unlocked=recommendations_unlocked,
        )
