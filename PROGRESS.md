# Progress

## Current
- Session: S2 (M1b) — complete on branch `m1a-core-data` (Tasks 6–10, 54 tests green + MiniLM integration test passed, ruff clean).
- Next: **S3 — M1c** (Task 11 style statistics onward) per `docs/superpowers/plans/M1-core-data.md`. Stay on branch `m1a-core-data` until M1 is done; whole-branch review + merge after S3.

## Roadmap
| # | Session | Driver | Status |
|---|---|---|---|
| S0 | git init, CLAUDE.md, PROGRESS.md, M1 plan | Claude | done |
| S1 | M1a scaffold, config/guard, sample vault, reader, manifest | Claude | done |
| S2 | M1b embedders, FAISS store, ingest, opinion scoring | Claude | done |
| S3 | M1c style profile, idea detection, CLI | Claude | todo |
| S4 | CI/CD lab 1 — GitHub Actions lint + tests | User (guided) | todo |
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

## Known issues
- Git warns that LF will become CRLF on Windows. This is harmless; adding a `.gitattributes` later would make line endings explicit.
