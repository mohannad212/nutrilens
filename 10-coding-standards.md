# 10 — Coding Standards & Conventions

Applies to the whole monorepo. Keep it boring and consistent — a grader and a teammate should be able
to read any file without surprises.

---

## 1. Backend (Python)

- **Python 3.11**, type hints on every function signature. `mypy` must pass (no `Any` escapes without
  a comment justifying it).
- **Formatter:** `black` (line length 100). **Linter:** `ruff` (import sort, pyflakes, pycodestyle).
- **PEP 8** naming: `snake_case` funcs/vars, `PascalCase` classes, `UPPER_SNAKE` constants.
- **Docstrings** on public functions/classes (one-line summary + args/returns for non-trivial ones).
- **Structure:** routers thin → services hold logic → integrations/ml isolated. No business logic in
  routers; no DB access in ml/ modules.
- **Errors:** raise typed exceptions from `core/errors.py`; never `raise Exception(...)`. Never leak
  stack traces to clients.
- **Secrets/logging:** never log passwords, tokens, image bytes, or full USDA payloads. Use structured
  logs with a `request_id`.
- **Config:** everything via `pydantic-settings`; no literals for hosts/keys/thresholds.

## 2. Mobile (TypeScript / React Native)

- **TypeScript strict** (`"strict": true`). No `any` unless justified with a comment.
- **ESLint** (`@typescript-eslint`, react, react-hooks, react-native) + **Prettier** (single quotes,
  trailing commas, width 100).
- **Components:** function components + hooks only. One component per file, `PascalCase` filename.
- **Server state:** React Query only; **never** store server data in `useState` long-term.
- **Types mirror the API:** `src/types/` matches `03` schemas field-for-field; when `03` changes,
  update both sides in the same PR.
- **No secrets in the app.** API base URL via config; JWT only in secure storage.
- **Styling:** theme tokens from `07 §8`; no hard-coded hex in components.

## 3. Git workflow

- **Branches:** `main` (protected, release-only) · `develop` (integration) · `feature/*`
  (e.g. `feature/portion-pipeline`, `feature/nutrition-endpoint`).
- **Model weights** are **not** committed — distributed via GitHub Releases (`01 §3`, `04 §7`).
- **Commits:** Conventional Commits — `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`.
  Example: `feat(recognise): add top-3 fallback below confidence threshold`.
- **PRs:** into `develop`; must pass CI (lint + tests) and include tests for new behaviour. Squash-merge.
- **Milestone gate:** tag `develop` at each milestone DoD (`m2-auth`, `m5-core-loop`, …).

## 4. Environment setup (document in root README too)

```bash
# Backend
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp ../.env.example ../.env            # fill USDA_API_KEY, SECRET_KEY, DATABASE_URL
docker compose up -d db               # or local postgres
alembic upgrade head
python scripts/seed_food101_classes.py
uvicorn app.main:app --reload

# Mobile
cd mobile && npm install
cp .env.example .env                  # set API_BASE_URL to your machine IP
npx expo start
```

## 5. Definition of Done (every task)

A task/PR is done only when: code matches these standards; types/lint/format pass; unit tests for the
new behaviour exist and pass; the relevant `03`/`02` contract is honoured; no secrets committed; the
milestone DoD in `09` is still satisfiable.

---

Prev: [`09-build-phases.md`](./09-build-phases.md) · Next: [`11-testing-qa.md`](./11-testing-qa.md)
