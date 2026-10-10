# vault-to-voice — instructions for Claude

LangGraph agent: Obsidian notes → blog posts in the author's voice, every paragraph cited.
Portfolio project. Spec: `docs/superpowers/specs/2026-10-10-vault-to-voice-design.md`.

## Session protocol
1. Start: read this file, `PROGRESS.md`, and the current milestone plan in `docs/superpowers/plans/`. Nothing else unless needed.
2. Work task-by-task from the plan, TDD (test first), one commit per task.
3. End: run full test suite, update `PROGRESS.md` (done / next / decisions / issues), commit, and give the user a short "what was built + which concepts" summary.

## Hard rules
- NEVER read, index, or commit the user's real Obsidian vault. Only `sample_vault/` lives in the repo; every sample note has `fictional: true`.
- NEVER write GitHub Actions workflows, Dockerfiles, or docker-compose files. Those are hands-on labs: teach step by step, user types, Claude reviews.
- Tests must run offline with no API keys (FakeLLMClient + HashEmbedder). Real-model tests use `@pytest.mark.integration`.
- Dev/testing embeddings: `EMBEDDING_PROVIDER=local` (MiniLM). OpenAI embeddings are for the user's own runs.

## Stack & commands
Python 3.11, uv, ruff, mypy, pytest, LangGraph, FAISS, FastAPI, Streamlit, Typer.
- Install: `uv sync`
- Tests: `uv run pytest` (integration: `uv run pytest -m integration`)
- Lint: `uv run ruff check . && uv run ruff format --check .`
- CLI: `uv run v2v --help`

## Conventions
- src layout: `src/vault_to_voice/`. Small focused modules; protocols at seams (VaultReader, Embedder, LLMClient, Publisher).
- All LLM calls go through `LLMClient.structured(prompt, Schema)` returning pydantic models.
- Note text sent to LLMs is wrapped as data (`guardrails.wrap_note`).
