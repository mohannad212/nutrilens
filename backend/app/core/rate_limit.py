"""
app/core/rate_limit.py
=========================

WHY THIS FILE EXISTS
---------------------
`08 §7` requires the backend to enforce the USDA hourly request cap itself rather
than discovering it by being rejected. USDA FoodData Central issues a per-key
budget (`USDA_RATE_LIMIT_PER_HOUR`, default 3600); exceed it and *every*
subsequent lookup fails for the rest of the hour, for every user of the
deployment. This file is the counter that stops us walking into that.

WHAT PROBLEM IT SOLVES
------------------------
Without a local budget, the failure mode is cliff-shaped and shared. One runaway
client, a retry loop, or a demo with an empty cache burns the key's whole hour,
and the next user to photograph a meal gets a 429 they had no part in causing —
including, potentially, during the GP2 demo. A local counter converts that into a
gradual, self-inflicted limit we can observe and reason about: we stop *before*
USDA does, and the response says exactly why.

It is worth being clear about what this does **not** protect against. Because the
counter lives in process memory, N Gunicorn workers each hold their own budget of
N x limit in aggregate (`08 §10` runs multiple workers). Sizing the configured
limit below the real USDA cap absorbs that, and the true fix — a shared counter in
Redis — is deliberately out of scope for a single-VM evaluation deployment where
the cache absorbs almost all traffic anyway.

HOW IT WORKS INTERNALLY
--------------------------
A **fixed window**: a counter plus the timestamp the current hour began. Every
`acquire()` first checks whether an hour has elapsed and, if so, resets both. If
the counter is under the limit it increments and returns `True`; otherwise it
returns `False` without incrementing, so a rejected call cannot deepen the hole.

`time.monotonic()` is the clock, not `time.time()`: monotonic time cannot jump
backwards when NTP corrects the system clock, which would otherwise either extend
a window indefinitely or reset it early. The clock is injectable so tests can
advance an hour without sleeping.

A `threading.Lock` guards both fields together. FastAPI runs `def` routes in a
threadpool, so two requests genuinely can read-then-write the counter
concurrently; without the lock the classic lost-update race lets the limit drift
upward under exactly the load where it matters.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- **Fixed window (chosen).** Two integers and a lock. It matches the shape of the
  thing being modelled — USDA's own quota is a fixed hourly budget, so a fixed
  window under-counts in exactly the way that keeps us inside theirs.
- **Sliding window / token bucket.** Smoother, and standard for smoothing bursty
  *client* traffic. Here the resource being protected has fixed-window semantics
  itself, so the extra state buys precision against the wrong model.
- **No local limiting, just handle USDA's 429.** Simpler, and the client does
  handle that 429 regardless (a shared key can be exhausted by something outside
  this process). But it makes the cap a surprise rather than a policy, and
  `08 §7` asks for the counter explicitly.
- **Redis / database counter.** Correct across workers and the right answer for a
  multi-instance deployment. It adds an infrastructure dependency and a network
  round trip to every nutrition lookup, for a project deployed to one VM.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "What happens at the boundary of a fixed window?"
A: The classic burst: a client can spend the tail of one window and the head of
   the next back to back, briefly doubling the nominal rate. Harmless here — the
   budget is 3600/hour against a cache hit rate that should approach 100% after
   warm-up, so the practical load is nowhere near the limit.

Q: "Why return `False` instead of raising from the limiter?"
A: The limiter is a mechanism and should not decide HTTP semantics. Its caller
   (`usda_client`) is the layer that knows a refusal means `USDA_RATE_LIMITED`, and
   `nutrition_service` is the layer that knows a cached value beats the error
   entirely (`08 §6`). A raise here would force a `try/except` at both.

Q: "Does hitting this limit mean users see errors?"
A: Only for foods that are not cached. `08 §6`'s read path checks `food_item`
   first, so a saturated budget degrades the app to "no *new* foods this hour"
   rather than "no nutrition at all".

COMMON STUDENT MISTAKES
--------------------------
- Using `time.time()` and getting a window that misbehaves whenever the system
  clock is adjusted.
- Incrementing the counter before checking the limit, so rejected calls push the
  count ever higher and the window never recovers.
- Forgetting the lock because "Python has a GIL" — the GIL does not make
  read-modify-write sequences atomic across threads.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

SECONDS_PER_HOUR = 3600.0


class HourlyRateLimiter:
    """A thread-safe fixed-window counter: at most `limit` acquisitions per hour.

    `clock` returns monotonic seconds and is injectable purely so tests can jump
    forward an hour instead of sleeping through one.
    """

    def __init__(self, limit: int, clock: Callable[[], float] | None = None) -> None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        self._limit = limit
        self._clock = clock or _monotonic
        self._lock = threading.Lock()
        self._count = 0
        self._window_start = self._clock()

    @property
    def limit(self) -> int:
        return self._limit

    def _roll_window_if_due(self, now: float) -> None:
        """Reset the counter if the current hour has elapsed. Caller holds the lock."""
        if now - self._window_start >= SECONDS_PER_HOUR:
            self._window_start = now
            self._count = 0

    def acquire(self) -> bool:
        """Consume one unit of budget. `False` if the hour's budget is spent.

        A refusal does not increment, so a caller that retries in a loop cannot
        push the count further past the limit.
        """
        with self._lock:
            now = self._clock()
            self._roll_window_if_due(now)
            if self._count >= self._limit:
                return False
            self._count += 1
            return True

    def remaining(self) -> int:
        """Budget left in the current window (mainly for logging and tests)."""
        with self._lock:
            self._roll_window_if_due(self._clock())
            return max(0, self._limit - self._count)


def _monotonic() -> float:
    import time

    return time.monotonic()
