# Progress

## Current
- Session: S4 (CI lab 1) — complete. M1 merged to `main` (b50f628). Repo public at https://github.com/yogeshmvgd/vault-to-voice ; user-written `.github/workflows/ci.yml` (checkout → setup-uv py3.11 → `uv sync --locked` → ruff check → ruff format --check → pytest) is green; CI badge in README.
- Next: **S5a — M1.5 Obsidian daily-notes format** (`docs/superpowers/plans/M1.5-obsidian-format.md`, 6 tasks), then **S5 — M2a** (`docs/superpowers/plans/M2-post-graph.md` Tasks 1–6). Branch `m2-post-graph`.

## Roadmap
| # | Session | Driver | Status |
|---|---|---|---|
| S0 | git init, CLAUDE.md, PROGRESS.md, M1 plan | Claude | done |
| S1 | M1a scaffold, config/guard, sample vault, reader, manifest | Claude | done |
| S2 | M1b embedders, FAISS store, ingest, opinion scoring | Claude | done |
| S3 | M1c style profile, idea detection, CLI | Claude | done |
| S4 | CI/CD lab 1 — GitHub Actions lint + tests | User (guided) | done |
| S5a | M1.5 Obsidian daily-notes reader, work/theme ideas, sample vault rewrite | Claude | todo |
| S5 | M2a graph state, context, interview, outline, checkpointer, Store | Claude | todo |
| S6 | M2b Send fan-out, citations, auto-revise, draft review, publish | Claude | todo |
| S7 | M3 FastAPI + SSE, Streamlit, LangSmith | Claude | todo |
| S8 | M4 critic, guardrails, mini eval | Claude | todo |
| S9 | Docker lab — Dockerfile + compose | User (guided) | todo |
| S10 | CI/CD lab 2 — image build + deploy | User (guided) | todo |
| S11 | M5 README, Mermaid, polish | Claude | todo |

## Decisions
- 2026-10-10: FAISS (not Chroma); embeddings switchable — MiniLM for dev, OpenAI for user runs.
- 2026-10-10: CI/CD and Docker are guided labs, not generated.

- 2026-10-10: Work on branch `m1a-core-data`; the final review runs once, after M1 (S3), not at the end of each session.
- 2026-10-10: Added README.md stub (hatchling needs it) and .gitignore; `docs/**` is excluded from `ruff format`, which had been rewriting the code blocks in the plan.

- 2026-10-10: M1 review fixes: duplicate-id fallback is unique; a `dirty` meta flag triggers crash rebuild; `v2v ideas` refuses an out-of-sync index; console uses backslashreplace.

- 2026-10-10: CI tests on Python 3.11 (oldest supported); local dev happened to use 3.14. User is a fresher on CI/Docker — labs teach from zero (commands, YAML, git auth).

- 2026-10-10: Real vault format = Obsidian daily notes `Daily/DD-MM-YYYY.md` with `## Games/Books/Movies` bullets linking `[[Work]]` pages; each linked bullet becomes a dated per-work Note; unlinked bullets skipped. Ideas: per-work (stable `idea_work_<id>`) + cross-work themes. Spec: `docs/superpowers/specs/2026-10-10-obsidian-daily-notes-design.md`.
- 2026-10-10: M2 plan written (`docs/superpowers/plans/M2-post-graph.md`); critic deferred to M4; draft edit re-checks citations; CLI write = approve/revise only.

## Known issues
- First CI install is slow (torch via sentence-transformers); possible later optimisation: CPU-only torch index.
- Deferred minors from M1 review: unreadable files (OSError) abort ingest; rebuild wipes opinion scores (full LLM rescoring); empty notes skipped silently; scorer error mid-run not caught per note; KMeans ConvergenceWarning on identical vectors; sentence-transformers FutureWarning (`get_embedding_dimension`).
- Git warns that LF will become CRLF on Windows. This is harmless; adding a `.gitattributes` later would make line endings explicit.
