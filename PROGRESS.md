# Progress

## Current
- Session: S8 (M4, plan `docs/superpowers/plans/M4-critic-guardrails-eval.md` Tasks 1-9) — complete on branch `m4-quality` (from `m3-api-ui`): checkpoint msgpack allowlist (no more "unregistered type" warnings), thread-safe manifest (RLock), input/output guardrails, critic node (measured style score + LLM editor notes + safety notes) feeding the one auto-revise and shown in draft review / API / Streamlit, mini eval `v2v eval` (offline OfflineLLM over fixed sample-vault cases, `--real` for the user's models). 287 offline tests green (4 integration deselected; all 4 integration tests pass with MiniLM), ruff clean. Offline eval: 3/3 published, coverage 1.0, faithfulness 1.0, style 0.574 / 0.499 / 0.656 (mean 0.576).
- Session: S8.5 (M4.5, plan `docs/superpowers/plans/M4.5-ui-setup.md`) — complete on `m4-quality`: shared `setup_actions.py` (ingest/style/ideas, one code path for CLI and API), API `GET /config`, `POST /ingest`, `POST /style`, `POST /ideas/detect` with 409 guards (no setup during a post run, one setup action at a time, no post run during setup; out-of-sync index -> 409; other failures -> clean 500 message), ApiClient methods (30 min read timeout), Streamlit sidebar Setup panel. 308 offline tests green (4 integration deselected), ruff clean.
- Next: **S9 — Docker lab** (Dockerfile + compose): user-driven, guided step by step; Claude reviews.

## Roadmap
| # | Session | Driver | Status |
|---|---|---|---|
| S0 | git init, CLAUDE.md, PROGRESS.md, M1 plan | Claude | done |
| S1 | M1a scaffold, config/guard, sample vault, reader, manifest | Claude | done |
| S2 | M1b embedders, FAISS store, ingest, opinion scoring | Claude | done |
| S3 | M1c style profile, idea detection, CLI | Claude | done |
| S4 | CI/CD lab 1 — GitHub Actions lint + tests | User (guided) | done |
| S5a | M1.5 Obsidian daily-notes reader, work/theme ideas, sample vault rewrite | Claude | done |
| S5 | M2a graph state, context, interview, outline, checkpointer, Store | Claude | done |
| S6 | M2b Send fan-out, citations, auto-revise, draft review, publish | Claude | done |
| S7 | M3 FastAPI + SSE, Streamlit, LangSmith | Claude | done (manual smoke pending) |
| S8 | M4 critic, guardrails, mini eval | Claude | done (`--real` eval + manual smokes pending) |
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

- 2026-10-10: Theme ideas only when >= 6 works (real clustering); smaller vaults get work ideas only. Code fences and [[A^block]] links handled.

- 2026-10-10: S5: plan's test-fixture imports in tests/fakes.py moved to the top of the file (ruff E402).

- 2026-10-10: S6: plan code applied as written; only deviation: the Task 11 fixture imports appended to tests/fakes.py were placed at the top of the file (ruff E402).

- 2026-10-10: M3: start/resume requests are the SSE streams (one request = one graph step); GET /threads/{id}/stream = retry a stalled thread; one run per thread (409 otherwise). User-approved deviation from the spec.
- 2026-10-10: M3: ingest/style/ideas stay CLI-only (user-approved); no sse-starlette (plain StreamingResponse); UI tested with AppTest over a real ApiClient(TestClient).
- 2026-10-10: S7: plan code applied as written; work moved to new branch `m3-api-ui` (from `m2-post-graph`) as the plan specifies. No API deviations needed.

- 2026-10-10: M4: critic triggers the one auto-revise on any major note or style_score < CRITIC_MIN_STYLE (0.5); minor notes are only shown; critic re-runs after the revision and after user edits. User decision: keep the plan defaults (threshold 0.5; a low style score redrafts the whole post via the one auto-revise).
- 2026-10-10: M4: guardrails warn, never block: injection heuristics -> guard_warnings at draft review; output issues -> major "safety" critic notes. User decision: PII only warns, never blocks publish.
- 2026-10-10: M4: eval uses fixed sample-vault cases (eval/cases.json) and an extractive OfflineLLM by default; --real for the user's models. Offline numbers are a regression harness, not a quality claim.
- 2026-10-10: M4: checkpoints use an explicit msgpack allowlist (CHECKPOINT_TYPES); add new state models there (a test enforces it).
- 2026-10-10: S8: plan code applied as written; no deviations needed.

- 2026-10-10: M4.5: setup actions (ingest/style/ideas) are now in the API and Streamlit UI, reversing the M3 CLI-only decision at the user's request. CLI and API share `setup_actions.py`; the CLI keeps its behaviour.

## Known issues
- M4.5 manual UI smoke (Setup panel against the real vault/model) pending; to do by the user.
- Deferred minors from M1.5 review: duplicate entry ids across Daily/ subfolders resolved by path order; inserting a same-work bullet above renumbers -N ids; block-quoted bullets ignored; task checkboxes `[ ]` leak into entry body; links inside inline code get rendered.
- First CI install is slow (torch via sentence-transformers); possible later optimisation: CPU-only torch index.
- Deferred minors from M1 review: unreadable files (OSError) abort ingest; rebuild wipes opinion scores (full LLM rescoring); empty notes skipped silently; scorer error mid-run not caught per note; KMeans ConvergenceWarning on identical vectors; sentence-transformers FutureWarning (`get_embedding_dimension`).
- Git warns that LF will become CRLF on Windows. This is harmless; adding a `.gitattributes` later would make line endings explicit.
- Manual `v2v write` smoke with a real OpenAI key still pending (the MiniLM integration tests, incl. `test_write_on_sample_vault_with_minilm`, pass as of S8).
- M3 manual end-to-end smoke (`v2v serve` + Streamlit in a browser with a real OpenAI key, restart-mid-interview check, optional LangSmith trace check) not run in S7; to do by the user.
- `v2v eval --real` (user's LLM + embeddings, paid calls) not run in S8; to do by the user. M4 manual UI smoke (Style match metric, Editor notes, resuming a pre-M4 thread) also pending.
