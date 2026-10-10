# vault-to-voice — System Design & Build Plan

## Context
Portfolio project (job switch 2026). A LangGraph agent turns daily Obsidian notes (games / books / movies) into blog posts that keep the author's voice, with every paragraph grounded in cited notes or interview answers. It must visibly exercise the full agentic-AI curriculum (foundations, LangChain, LangGraph core, persistence/memory/streaming/RAG, observability, multi-agent, guardrails/eval). Repo ships a fictional sample vault; the real vault is never read from or committed to the repo. Project folder is currently empty: `E:\JOB SWITCH 2026\Projects\blog-generate`.

## Locked decisions
| Topic | Decision |
|---|---|
| Goal | Portfolio — polish, clear architecture, tests, demo-ability |
| LLM | Provider-agnostic via `init_chat_model` (anthropic / openai / ollama in `.env`) behind our `LLMClient` |
| Architecture | Plain-Python pipelines (ingest, ideas, style) + one LangGraph **post graph** per idea |
| Serving | FastAPI (+SSE streaming) backend, Streamlit frontend over HTTP |
| Vector store | **FAISS** (`IndexIDMap2(IndexFlatIP)`, cosine via L2-normalised vectors), persisted to `data/faiss.index`; chunk metadata/text in SQLite |
| Embeddings | `Embedder` protocol, `EMBEDDING_PROVIDER` in `.env`: `local` = sentence-transformers `all-MiniLM-L6-v2` (384-d, default for dev/Claude testing, free); `openai` = `text-embedding-3-small` (1536-d, user's own runs). Unit tests use `HashEmbedder`; MiniLM integration tests marked `@pytest.mark.integration`. Embeddings cached by content hash; provider/dim change forces index rebuild |
| Checkpointer | SQLite default; Postgres via factory (used in Docker) |
| Long-term memory | LangGraph Store: style profile + past interview answers |
| Opinion scoring | LLM per new/changed note at ingest, cached in manifest |
| Style profile | Measured stats + LLM tone summary → versioned JSON |
| Citations | Structured sentences with sources; checker = ID validity + embedding similarity + LLM judge in grey zone; one auto-revise, remaining flags shown at approval; `connective` sentences exempt |
| Approvals | approve / revise(feedback, cap 3) / edit(text) at outline and draft |
| Interview | Adaptive, one question per interrupt, min 3 / max 5 |
| Sessions | One thread per idea (`thread_id = idea_id`), many in parallel |
| Output | `outbox/<date>-<slug>.md` (clean) + `.sources.json` sidecar |
| Pulled into MVP | Send fan-out drafting, critic node, basic guardrails + mini eval |
| CI + Docker | **Not built by Claude.** Delivered as a step-by-step guided lab (`docs/learn/`) the user does themselves, Claude reviews |

## Tech stack
Python 3.11 · uv · ruff · mypy · pre-commit · pydantic-settings · LangChain core + `init_chat_model` · LangGraph (StateGraph, subgraphs, Send, interrupt/Command) · langgraph-checkpoint-sqlite / -postgres · LangGraph Store · sentence-transformers (MiniLM) + OpenAI embeddings · faiss-cpu · scikit-learn · python-frontmatter · FastAPI + uvicorn + sse-starlette · Streamlit · Typer · structlog · LangSmith (optional) · pytest. (GitHub Actions + Docker Compose: user-built via guided labs.)

## End-to-end flow

```
              ┌────────────── OFFLINE PIPELINES (plain Python, CLI + API) ──────────────┐
 Vault (.md) ─► [1] Ingest ──► FAISS index + SQLite (chunks, hash, opinion score)
                    │                         │
                    └──► [2] Style profile ◄──┘──► Store: style_profile (JSON, versioned)
                                              │
                         [3] Idea detection ◄─┘──► ideas table (title, angle, note_ids)
              └──────────────────────────────────────────────────────────────────────────┘
                                              │ user picks idea in UI
              ┌──────────── POST GRAPH (LangGraph, thread_id = idea_id) ─────────────────┐
              │ load_context → interview⟲(interrupt) → outline → outline_review(interrupt)│
              │ → Send fan-out draft_section×N → merge → check_citations → critic         │
              │ → [auto-revise once] → draft_review(interrupt) → publish                  │
              └───────────────────────────────────────────────────────────────────────────┘
                     ▲ checkpoints (SQLite/Postgres)        ▼ outbox/post.md + sources.json
   Streamlit UI ◄──HTTP/SSE──► FastAPI ──► pipelines + graph        LangSmith traces everything
```

### [1] Ingest (`v2v ingest`)
1. `VaultReader` (protocol; `FilesystemVaultReader` now, MCP later) walks `VAULT_PATH`, parses frontmatter (`id, date, type, title`); `note_id` = frontmatter id or path slug.
2. Manifest (SQLite) compares content hash → classify new / changed / deleted / unchanged.
3. New/changed: split into paragraphs → embed (configured Embedder, batched, cached by chunk hash) → L2-normalise → `index.add_with_ids` using int64 `chunk_id` from SQLite `chunks` table (`chunk_id, note_id, para_idx, text, hash`). Changed/deleted notes: `index.remove_ids(old chunk_ids)` + delete rows. `faiss.write_index` after each run. Index meta (`embedding_model`, `dim`) stored; mismatch → forced full rebuild.
4. Opinion scorer (LLM, structured) → `{score 0–1, highlights[]}` stored in manifest.
5. Guardrail: note text is always passed to LLMs inside `<note>` data tags with an instruction to never follow embedded instructions; PII scrub (emails/phones) before any LLM call.

### [2] Style profile (`v2v style`)
- `stats.py`: avg/median sentence length, paragraph length, type-token ratio, punctuation habits (em-dash, `!`, `?`, ellipsis), first-person rate, top recurring 2–4-grams.
- `profile.py`: LLM summarises tone descriptors + 3–5 exemplar sentences from notes.
- Saved as versioned JSON in Store (namespace `("style",)`) and mirrored to `data/style_profile.json`.

### [3] Idea detection (`v2v ideas`)
- Note embedding = mean of its chunk embeddings. KMeans, k chosen by silhouette over 2..min(10, n/3), fixed seed.
- Idea trigger: cluster size ≥ `MIN_NOTES` (4) AND ≥ `MIN_OPINIONATED` (3) notes with score ≥ 0.5.
- LLM labels each qualifying cluster → `Idea{idea_id=hash(sorted note_ids), title, angle, theme, note_ids, opinion_score, status}`. Stored in SQLite; ideas already published marked `done`.

### [4] Post graph (per idea)
State (`PostState`): `idea, notes, style, past_answers, qa[], outline, drafts{section_id: Section}, draft, citation_report, critic_notes, auto_revised: bool, user_rounds{outline, draft}, final_path`.

| Node | What it does |
|---|---|
| `load_context` | Fetch idea's notes (SQLite chunks; FAISS search for extra related notes), style + past answers from Store |
| `interview` subgraph | `ask_question` → `interrupt({type:"interview_question", q, grounding_note_ids})` → `record_answer` (`qa:n`) → `enough?` (min 3, max 5, LLM coverage check) |
| `make_outline` | Sections `{id, heading, point, planned_sources}` |
| `outline_review` | `interrupt({type:"outline_review", outline})`; resume `{action: approve|revise|edit, feedback?, edited?}`; revise cap 3 |
| fan-out | conditional edge returns `[Send("draft_section", {section, ctx}) ...]` — parallel |
| `draft_section` | Writes paragraphs as `Sentence{text, sources[], kind: claim|connective}` in author's style |
| `merge_draft` | Reducer merges sections in outline order |
| `check_citations` | Per claim sentence: invalid/missing IDs → flag; cosine(sentence, best chunk of cited source) ≥ 0.55 supported, ≤ 0.30 unsupported, between → batched LLM judge |
| `critic` | Scores voice vs style profile (stats-based style score + LLM notes) and substance |
| `route_after_checks` | If flags/critic issues and not `auto_revised` → `revise_sections` (Send only flagged sections) → `check_citations`; else `draft_review` |
| `draft_review` | `interrupt({type:"draft_review", draft, flags, critic_notes, style_score})`; approve / revise (→ fan-out, cap 3) / edit (edited text sourced `user`) |
| `publish` | Render clean Markdown + frontmatter → `outbox/`; write `.sources.json` (paragraph → sources, flags, scores); save Q&A to Store; mark idea `done` |

Errors: LLM calls wrapped with retry (3, exponential) + timeout; on failure the thread stays at its last checkpoint and the UI offers "retry" (re-invoke with `None`).

### [5] API (FastAPI)
`POST /ingest` · `POST /style` · `POST /ideas/detect` · `GET /ideas` · `POST /threads {idea_id}` · `GET /threads` · `GET /threads/{id}` (state + pending interrupt) · `POST /threads/{id}/resume {payload}` · `GET /threads/{id}/stream` (SSE of node updates) · `GET /health`.

### [6] UI (Streamlit)
Sidebar: vault status + buttons (ingest / style / detect), ideas list ("start post"), threads list with status. Main pane renders the pending interrupt by `type`: chat-style interview, outline editor with approve/revise/edit, draft view with unsourced sentences highlighted red + critic notes + style score, live progress via SSE, final post preview + download.

### [7] Mini eval (`v2v eval`)
Runs the full graph on the sample vault with scripted answers/auto-approve; reports citation coverage %, judge-faithfulness %, style-match score; writes `eval/report.json`. (RAGAS replaces judge later.)

### [8] Safety of the real vault
`VAULT_PATH` default `./sample_vault`; config validator rejects any path inside the repo except `sample_vault`; `.gitignore`: `data/`, `outbox/*`, `.env`, `eval/report.json`; test asserts every sample note has `fictional: true`.

## Repo layout
```
vault-to-voice/
├─ sample_vault/{games,books,movies}/*.md     (~30 fictional notes, 4 themes)
├─ src/vault_to_voice/
│  ├─ config.py  llm.py  logging.py  guardrails.py  cli.py
│  ├─ vault/     (models.py, reader.py)
│  ├─ index/     (embedder.py, faiss_store.py, manifest.py, ingest.py)
│  ├─ ideas/     (opinion.py, cluster.py, labeler.py, detector.py)
│  ├─ style/     (stats.py, profile.py)
│  ├─ graph/     (state.py, builder.py, checkpoint.py, memory.py, nodes/{context,interview,outline,draft,citations,critic,publish}.py)
│  ├─ api/       (app.py, routes.py, schemas.py)
│  ├─ ui/streamlit_app.py
│  └─ eval/run.py
├─ tests/ (fakes.py: FakeLLMClient keyed by schema, HashEmbedder; unit/, graph/, api/)
├─ docs/learn/ 01-github-actions.md, 02-docker-compose.md   (guided labs; user writes ci.yml / Dockerfile / compose)
├─ .env.example   .gitignore   LICENSE (MIT)   README.md (Mermaid)
└─ data/ outbox/ (gitignored)
```

## Extension seams (later phases)
Supervisor → wraps post graph as a subgraph · RAGAS → consumes `.sources.json` · MCP → new `VaultReader` impl · dev.to/Hashnode → `Publisher` protocol (Outbox impl now) · EC2 → compose file + Postgres checkpointer already in place.

## Milestones (each ends demoable, tests green)
1. **M1 Core data** — config, vault reader, manifest, ingest, opinion scoring, idea detection, style profile, CLI, sample vault.
2. **M2 Post graph** — state, all nodes, Send fan-out, citation checker, interrupts, SQLite checkpointer + Store, publish.
3. **M3 Serving** — FastAPI + SSE, Streamlit UI, LangSmith + structlog.
4. **M4 Quality** — critic node, guardrails, mini eval.
5. **M5 Ship** — README + Mermaid, LICENSE, Postgres checkpointer factory ready.
6. **Lab A (user)** — GitHub Actions CI, step by step: lint → tests → cache → badge. Claude explains each step and reviews.
7. **Lab B (user)** — Dockerfile → compose (api, ui, postgres) step by step. Claude explains and reviews.

## Verification
- `uv run pytest` — offline (FakeLLMClient + HashEmbedder, no OpenAI key needed): FAISS add/remove/persist round-trip, manifest incremental cases, clustering determinism, stats, citation checker thresholds, guardrail wrapping, vault-path validator, full graph run through every interrupt with `MemorySaver` + `Command(resume=…)` incl. revise/edit branches and auto-revise cap, API flow via TestClient.
- Manual: `v2v ingest && v2v style && v2v ideas` on sample vault → `v2v serve` + Streamlit → complete one post end-to-end → check `outbox/` files; kill server mid-interview, restart, resume same thread.
- `v2v eval` produces report. After the labs: CI green on GitHub; `docker compose up` runs the flow on Postgres.
