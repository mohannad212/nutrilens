"""
app/db/base.py
================

WHY THIS FILE EXISTS
---------------------
SQLAlchemy needs three things to talk to Postgres: a declarative `Base`
class that every ORM model inherits from, an `Engine` that knows how to
open connections, and a `Session` factory that hands out transactional
"units of work". This file creates exactly those three things, once, so
every other file (`models.py`, `deps.py`, services) imports from here
instead of re-creating engines all over the codebase.

WHAT PROBLEM IT SOLVES
------------------------
If every file that touched the database created its own `create_engine(...)`
call, you'd get a new connection pool per file — wasteful, and impossible
to reason about. Centralising it gives one connection pool for the whole
process and one `Base.metadata` that Alembic can introspect to autogenerate
migrations.

HOW IT WORKS INTERNALLY
--------------------------
`create_engine(settings.database_url)` builds a *connection pool* — it does
NOT open a connection yet; connections are opened lazily on first use and
reused after that. `sessionmaker(bind=engine, ...)` is a factory: every
call to `SessionLocal()` produces a *new* `Session` object bound to that
engine. A `Session` is SQLAlchemy's "unit of work" — it tracks objects
you've loaded/added, and only actually talks to Postgres when you call
`.commit()`, `.flush()`, or run a query. `Base` is an empty class whose
only job is bookkeeping: every model that inherits from it registers
itself in `Base.metadata`, which is the table registry Alembic reads to
know what tables *should* exist.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- Global engine + per-request session (chosen): the standard SQLAlchemy
  2.0 pattern; `deps.py`'s `get_db()` yields a fresh session per request
  and closes it afterward, so requests never leak or share transactions.
- A single global `Session` reused across requests: dangerous — one
  request's uncommitted changes or errors would leak into the next.
- Async engine (`create_async_engine` + `AsyncSession`): valid alternative
  since FastAPI is async, but adds complexity (async ORM calls everywhere)
  for a project this size; we chose sync SQLAlchemy inside FastAPI's sync
  route handlers, which FastAPI runs in a threadpool automatically.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "What's the difference between the engine and a session?"
A: The engine owns the connection pool (low-level, shared, thread-safe).
   The session is a short-lived object that borrows a connection from the
   pool for the duration of one unit of work (one request), then returns
   it to the pool.

Q: "Why `autoflush=False` / `autocommit=False`?" (see settings below)
A: We want explicit control over when SQL is sent (`flush`) and when a
   transaction is finalised (`commit`) — implicit autocommit makes it easy
   to accidentally persist partial, inconsistent state.

Q: "What is `Base.metadata` used for?"
A: It's the catalogue of every table+column defined via the ORM models.
   Alembic's `--autogenerate` diffs this catalogue against the actual
   database schema to produce a migration script.

COMMON STUDENT MISTAKES
--------------------------
- Creating a new engine per request (huge performance cost — connection
  pooling is defeated).
- Forgetting to close/return the session, exhausting the connection pool
  under load (this is why `deps.get_db` uses a `try/finally`).
- Importing models before `Base` is defined (circular import) — that's why
  `Base` lives in its own tiny file rather than inside `models.py`.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Every ORM model (User, MealLog, ...) inherits from this."""


engine = create_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)
