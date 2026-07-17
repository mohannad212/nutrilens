"""
tests/conftest.py
====================

WHY THIS FILE EXISTS
---------------------
Pytest automatically discovers fixtures defined in any `conftest.py` and
makes them available to every test file in the same directory (and
below) without an explicit import. This file provides one fixture,
`db_session`, that every M1/M2 test uses to get a clean, isolated
database to read/write against.

WHAT PROBLEM IT SOLVES
------------------------
Tests must not depend on a real, shared Postgres instance being up and
already migrated — that's slow, and one test's leftover data could break
another test's assumptions. This fixture creates a brand-new SQLite
in-memory database, builds the schema straight from `Base.metadata`
(the same ORM models Alembic migrates in production), runs exactly one
test against it, and throws it away.

HOW IT WORKS INTERNALLY
--------------------------
`create_engine("sqlite:///:memory:")` creates a database that lives only
in process memory and vanishes when the connection closes.
`Base.metadata.create_all(engine)` issues `CREATE TABLE` for every model
directly (bypassing Alembic — fine for unit tests, since we're testing
application logic, not the migration files themselves). A fresh `Session`
is `yield`ed to the test; after the test finishes, the fixture's teardown
code (after `yield`) closes the session and disposes of the engine.

WHY THIS IMPLEMENTATION (vs alternatives)
--------------------------------------------
- SQLite in-memory (chosen) for unit tests: no external process to start,
  each test gets a truly clean database, and tests run in milliseconds —
  ideal for a graduation project's CI pipeline.
- Running these same tests against a real Postgres test database: more
  faithful to production (true ENUM types, real constraint behavior) and
  is exactly what `11-testing-qa.md`'s integration tests do — but slower
  and requires a running Postgres container. NutriLens uses SQLite for
  fast unit tests of ORM logic and reserves a real Postgres instance
  (via docker-compose) for integration tests that exercise Alembic
  migrations directly.

DISCUSSION / DEFENSE QUESTIONS
--------------------------------
Q: "Doesn't using SQLite instead of Postgres risk testing something that
    doesn't behave identically in production?"
A: Yes, for Postgres-specific features (native ENUM validation, some
   NUMERIC precision edge cases) — that's exactly why the project also has
   Postgres-backed integration tests (`11`) alongside these fast unit
   tests; this fixture is for testing model/relationship *logic*, not
   database-engine-specific behavior.

Q: "Why a new engine and tables per test instead of one shared test DB?"
A: Isolation — a previous test's leftover row could make a later test pass
   or fail for the wrong reason. Fresh schema per test trades a little
   speed for full confidence that each test is independent.

COMMON STUDENT MISTAKES
--------------------------
- Sharing one SQLite file/engine across the whole test session — leftover
  rows from test A silently affect test B's assertions.
- Forgetting `Base.metadata.create_all(engine)` — tests fail with
  "no such table" errors that have nothing to do with the actual logic
  being tested.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.deps import get_db
from app.main import create_app


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # one shared connection, so the same in-memory
        # DB is visible across the TestClient's threads
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def client(db_session):
    """
    A FastAPI TestClient wired to the SAME in-memory db_session used by
    unit tests, via FastAPI's dependency_overrides. This lets M2 tests hit
    real HTTP routes (POST /auth/register, /auth/login) while still
    running against a throwaway in-memory database — no live Postgres
    needed for these fast tests.
    """
    app = create_app()

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
