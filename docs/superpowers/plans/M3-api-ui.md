# M3 API and UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the M2 post graph over HTTP and put a UI on it: a FastAPI service that starts a run for an idea, streams graph progress and interrupts as Server-Sent Events, resumes with interview answers / review decisions, retries stalled runs, and lists ideas, threads and published posts; a thin Streamlit UI over that API (ideas list, chat-style interview, outline review with edit, draft review with flagged sentences, coverage and edit, published post with sources and download); and opt-in LangSmith tracing that is off in tests.

**Architecture:** Three layers, each testable alone. `api/service.py` (`RunService`) owns the compiled graph and turns each run into an iterator of SSE frames (`api/events.py`), with one run at a time per thread (thread_id = idea_id). `api/app.py` is a thin FastAPI shell over the service (sync endpoints, `StreamingResponse` with `text/event-stream`). The UI is split into `ui/client.py` (HTTP + SSE parsing, works with `httpx.Client` or FastAPI's `TestClient`), `ui/views.py` (pure formatting helpers), and `ui/streamlit_app.py` (widgets only). Tests drive the real API through `TestClient`, and the real Streamlit script through `AppTest` with an `ApiClient` over that `TestClient`, so the whole stack runs offline in-process. Tracing is environment-only: `observability.configure_tracing` sets the `LANGSMITH_*` variables LangChain reads, and `thread_config` labels every run.

**Tech Stack:** FastAPI 0.143 (Starlette 1.7), uvicorn, Streamlit 1.65 (`streamlit.testing.v1.AppTest`), httpx 0.28, LangGraph 1.2 (`graph.stream(..., stream_mode="updates")`, `get_state(..., subgraphs=True)`), langsmith 0.14 (already installed through LangChain), structlog, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-vault-to-voice-design.md` (sections [5] API, [6] UI, Milestone 3 "Serving", Verification). Builds on `docs/superpowers/plans/M2-post-graph.md`.

**Sessions:** All tasks = S7 (M3). End the session per the `CLAUDE.md` protocol. Work on a new branch `m3-api-ui` created from `m2-post-graph` (or from `main` once M2 is merged).

**Deliberate deviations from spec (recorded, not silent):**
- **Streaming shape.** The spec lists `POST /threads/{id}/resume` and a separate `GET /threads/{id}/stream`. Here `POST /threads` (start) and `POST /threads/{id}/resume` *are* the SSE streams: each runs the graph to its next interrupt and streams `node` / `interrupt` / `done` / `error` events. `GET /threads/{id}/stream` keeps its name but means "continue a stalled thread" (re-invoke with `None`, the spec's "retry"). One request = one step, so there is no server-side queue of pending inputs and nothing to lose if the client disconnects.
- **No `POST /ingest`, `POST /style`, `POST /ideas/detect` in M3.** Those stay CLI commands (`v2v ingest/style/ideas`); the UI sidebar tells the user to run them when there are no ideas. They are long batch jobs that need a job model of their own; the M3 brief is the post-writing loop.
- **No sse-starlette.** A plain `StreamingResponse(media_type="text/event-stream")` with a 3-line `sse()` formatter is enough for one-way events, needs no extra dependency, and works with `TestClient.stream` (probed).
- **No critic notes or style score in the draft view.** The critic is M4; the draft view shows citation coverage, flag count and red flagged sentences now, and M4 adds critic fields to the same `draft_review` payload.
- **Live progress granularity.** Progress events are per parent-graph node (`load_context`, `interview`, `make_outline`, `draft_section` x N, ...). Interview subgraph inner nodes are not streamed (`subgraphs=False`) because the UI only needs "it is working".

**Verified API facts (probed against the installed / to-be-installed versions before writing this plan):**
- `graph.stream(value, config, stream_mode="updates")` yields one dict per finished node, `{node_name: update}`; a pause yields `{"__interrupt__": (Interrupt(value=...),)}` (a tuple). Resuming inside the interview subgraph yields only `__interrupt__` until the subgraph finishes, then `{"interview": ...}`.
- A node exception raised from `graph.stream` leaves the thread at its last checkpoint: `get_state(config).next == ("make_outline",)` and `.interrupts == ()`; `graph.invoke(None, config)` then re-runs that node. `RuntimeError` is *not* retried by `RetryPolicy` (LangGraph's `default_retry_on` skips it), so tests use it to simulate a failure without retry delays.
- While the interview subgraph is paused, the parent state has no `qa` yet. `graph.get_state(config, subgraphs=True).tasks[i].state.values["qa"]` (task name `"interview"`) holds the answers so far.
- `thread_config` may carry `run_name`, `tags`, `metadata` next to `configurable`; `invoke`, `stream` and `get_state` accept it unchanged.
- FastAPI 0.143 `TestClient` subclasses `httpx2.Client` (not `httpx.Client`); it supports `.request(...)` and `.stream(method, url, json=...)` with `.iter_lines()`, which returns SSE lines with the blank separators as `""`. So `ApiClient` takes any httpx-style client object rather than type-checking for `httpx.Client`.
- A `ServiceError` raised in a sync route before the `StreamingResponse` is built goes through `@app.exception_handler` and returns proper JSON status codes (404/409); validation errors on the body return 422.
- Streamlit 1.65 `AppTest.from_file(path, default_timeout=30)`: `at.session_state["client"] = obj` before the first `at.run()` injects a client; `at.button(key=...).click().run()`, `at.text_input(key=...).input(...)`, `at.text_area(key=...).input(...)`, `at.chat_message[i].markdown[0].value`, `at.metric[i].value`, `at.sidebar.button`, `at.sidebar.error`, `at.success/.warning/.error/.info/.caption/.markdown/.title` all work; `st.status`, `st.expander`, `st.columns`, `st.download_button` and `st.rerun()` run fine inside `AppTest`. A real `ApiClient(TestClient(app))` works from the AppTest script thread.
- `st.button(..., width="stretch")` exists in 1.65 (and replaces the deprecated `use_container_width`).
- langsmith 0.14 reads `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` (with `LANGCHAIN_*` fallbacks); tracing is off unless `LANGSMITH_TRACING=true`.
- `uv add fastapi uvicorn streamlit httpx` resolves to fastapi 0.143.0, uvicorn 0.54.0, streamlit 1.65.0, httpx 0.28.1 on Python 3.11+.
- Whole plan dry-run: every file below was applied to a scratch copy of the repo at `b670b5d`; `uv run pytest` gave 232 passed (190 existing + 42 new), `ruff check` / `ruff format --check` clean, and `mypy src` reports no errors in the new modules.

## Global Constraints

- Python `>=3.11` (CI runs 3.11). ruff line length 100; `uv run ruff check . && uv run ruff format --check .` clean before every commit.
- Tests run offline with no API keys: `FakeLLMClient` + `HashEmbedder` + `InMemorySaver`/`InMemoryStore` (or SQLite under `tmp_path`), FastAPI `TestClient`, Streamlit `AppTest`. Real-model tests are `@pytest.mark.integration`.
- LangSmith tracing is opt-in (`LANGSMITH_TRACING=true` plus `LANGSMITH_API_KEY`) and forced off in every test by an autouse fixture.
- `thread_id = idea_id`. Checkpoints in `data/checkpoints.db`, Store in `data/store.db` (unchanged from M2).
- The API never reads the vault: it only uses the manifest, FAISS index, checkpoints, Store and outbox built by `v2v ingest/style/ideas`. Never read, index or commit the user's real vault.
- The UI talks to the API only over HTTP (`V2V_API_URL`, default `http://127.0.0.1:8000`); it never imports the graph.
- Never write GitHub Actions workflows, Dockerfiles, or compose files (user labs).
- One commit per task; messages end with a blank line and `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Server restarted (or killed) mid-interview → a new process on the same SQLite files shows the same pending question and earlier answers, and resuming continues the same thread without re-asking the model. Pinned in Task 5 (`test_restarted_server_resumes_the_same_thread`).
2. Double submit / two tabs on one thread (second start or resume while a run streams) → `409 already running`, never two graph runs interleaving one checkpoint; other threads are not blocked; a dropped stream releases the lock. Pinned in Task 3.
3. LLM failure mid-step → an `error` event (not a 500 or a hung stream), thread status `stalled`, and Retry continues from the failed node. Pinned in Tasks 2, 3 and 8.
4. Bad client input: unknown idea (404), wrong-state actions such as resuming a new thread or starting twice (409), resume bodies with `null`, numbers or lists (422, nothing consumed), malformed review decisions (review shown again with `error`). Pinned in Tasks 3 and 4.
5. API not running when the UI opens → the sidebar shows "API unavailable" and the page still renders; saving an unchanged draft warns instead of submitting an empty edit. Pinned in Task 8.

## File Structure

```
src/vault_to_voice/
  config.py                + api_host, api_port, langsmith_tracing, langsmith_api_key, langsmith_project
  observability.py         configure_tracing(settings) -> bool
  graph/runner.py          thread_config adds run_name / tags / metadata for traces
  cli.py                   + `serve` command (uvicorn)
  api/
    __init__.py
    events.py              sse(), parse_sse(), stream_run()  (no FastAPI import)
    service.py             RunService, ThreadView, ServiceError / NotFound / Conflict
    app.py                 create_app(service): FastAPI routes
    factory.py             build_service(settings, llm, tracing=) (SQLite wiring)
  ui/
    __init__.py
    client.py              ApiClient, ApiError
    views.py               last_event, flagged_markdown, section_texts, changed_sections,
                           edited_outline, strip_frontmatter  (no streamlit import)
    streamlit_app.py       the Streamlit page (widgets only)
tests/
  conftest.py              + autouse _no_langsmith
  fakes.py                 + sse_events, make_service
  test_observability.py  test_api_events.py  test_api_service.py  test_api_app.py
  test_api_factory.py  test_ui_client.py  test_ui_views.py  test_ui_app.py
.env.example               + API / UI / LangSmith keys
```

API surface (all JSON unless marked SSE):

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | | `{status, tracing}` |
| GET | `/ideas` | | `list[Idea]` |
| GET | `/threads` | | `list[ThreadView]` (started threads only) |
| POST | `/threads` | `{idea_id}` | SSE: run from START to the first interrupt |
| GET | `/threads/{id}` | | `ThreadView{thread_id, title, status: new/waiting/stalled/published, pending, outline, qa, final_path}` |
| POST | `/threads/{id}/resume` | `{payload: str \| dict}` | SSE: resume to the next interrupt or `done` |
| GET | `/threads/{id}/stream` | | SSE: continue a `stalled` thread |
| GET | `/threads/{id}/post` | | `{final_path, markdown, sources}` |

SSE events: `node {node}` · `interrupt {...interrupt payload from M2}` · `done {final_path}` · `error {message}`.

---

### Task 1: Dependencies, settings, opt-in LangSmith tracing, trace labels

**Files:**
- Modify: `pyproject.toml`/`uv.lock` (via `uv add`), `src/vault_to_voice/config.py`, `src/vault_to_voice/graph/runner.py`, `tests/conftest.py`, `.env.example`
- Create: `src/vault_to_voice/observability.py`
- Test: `tests/test_observability.py`

**Interfaces:**
- Consumes: `Settings` (`vault_to_voice.config`), `thread_config` (`vault_to_voice.graph.runner`).
- Produces:
  - `Settings` new fields: `api_host: str = "127.0.0.1"`, `api_port: int = 8000`, `langsmith_tracing: bool = False`, `langsmith_api_key: str = ""`, `langsmith_project: str = "vault-to-voice"` (env: `API_HOST`, `API_PORT`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`).
  - `configure_tracing(settings: Settings) -> bool` in `vault_to_voice.observability`: True only when tracing is on *and* a key is set; writes `LANGSMITH_TRACING/API_KEY/PROJECT` to `os.environ`.
  - `thread_config(thread_id) -> {"configurable": {"thread_id"}, "run_name": "post_graph", "tags": ["vault-to-voice", "post-graph"], "metadata": {"idea_id": thread_id}}`.

- [ ] **Step 1: Add dependencies**

Run: `uv add fastapi uvicorn streamlit httpx`
Expected: `pyproject.toml` gains `fastapi>=0.143.0`, `uvicorn>=0.54.0`, `streamlit>=1.65.0`, `httpx>=0.28.1`; `uv.lock` updated (CI uses `uv sync --locked`, so commit both).

- [ ] **Step 2: Write the failing tests**

`tests/test_observability.py`:
```python
import os

import pytest

from vault_to_voice.config import Settings
from vault_to_voice.graph.runner import thread_config
from vault_to_voice.observability import configure_tracing


@pytest.fixture
def clean_env(monkeypatch):
    # monkeypatch records these, so whatever configure_tracing writes is undone after the test
    for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.delenv(name, raising=False)


def make(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_tracing_is_off_by_default(clean_env):
    assert make().langsmith_tracing is False
    assert configure_tracing(make()) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_needs_an_api_key(clean_env):
    assert configure_tracing(make(langsmith_tracing=True)) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_tracing_on_sets_langsmith_env(clean_env):
    s = make(langsmith_tracing=True, langsmith_api_key="lsv2-test", langsmith_project="p1")
    assert configure_tracing(s) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2-test"
    assert os.environ["LANGSMITH_PROJECT"] == "p1"


def test_tracing_reads_env_vars(clean_env, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "k")
    s = make()
    assert s.langsmith_tracing is True and s.langsmith_api_key == "k"
    assert s.langsmith_project == "vault-to-voice"


def test_tests_never_trace():
    # the autouse conftest fixture forces tracing off for every test
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY" not in os.environ


def test_thread_config_labels_traces():
    cfg = thread_config("idea_x")
    assert cfg["configurable"] == {"thread_id": "idea_x"}
    assert cfg["run_name"] == "post_graph" and "post-graph" in cfg["tags"]
    assert cfg["metadata"] == {"idea_id": "idea_x"}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_observability.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vault_to_voice.observability'`.

- [ ] **Step 4: Implement**

In `src/vault_to_voice/config.py`, add after `category_headings: str = "games=game,books=book,movies=movie"`:
```python
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "vault-to-voice"
```

`src/vault_to_voice/observability.py`:
```python
import os

import structlog

from vault_to_voice.config import Settings

log = structlog.get_logger()


def configure_tracing(settings: Settings) -> bool:
    """Opt-in LangSmith tracing for every LangChain/LangGraph call in this process.

    Off unless LANGSMITH_TRACING=true *and* LANGSMITH_API_KEY is set. LangChain reads these
    environment variables at call time, so setting them here is all the wiring needed.
    """
    if settings.langsmith_tracing and not settings.langsmith_api_key:
        log.warning("tracing_disabled", reason="LANGSMITH_TRACING=true but no LANGSMITH_API_KEY")
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    log.info("tracing_enabled", project=settings.langsmith_project)
    return True
```

In `src/vault_to_voice/graph/runner.py`, replace `thread_config` with:
```python
def thread_config(thread_id: str) -> dict:
    """Run config for one post thread; run_name/tags/metadata label its LangSmith traces."""
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "post_graph",
        "tags": ["vault-to-voice", "post-graph"],
        "metadata": {"idea_id": thread_id},
    }
```

In `tests/conftest.py`, add after `import pytest` (before the `vault_dir` fixture):
```python


@pytest.fixture(autouse=True)
def _no_langsmith(monkeypatch):
    """Tracing is opt-in: tests never send traces, even from a shell that enables them."""
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    for name in ("LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
```

In `.env.example`, replace the block
```
# Optional LangSmith tracing (used from M3)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
```
with
```
# Optional LangSmith tracing: on only when LANGSMITH_TRACING=true AND a key is set
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=vault-to-voice

# API (`v2v serve`) and UI (`streamlit run src/vault_to_voice/ui/streamlit_app.py`)
API_HOST=127.0.0.1
API_PORT=8000
V2V_API_URL=http://127.0.0.1:8000
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_observability.py -v && uv run pytest`
Expected: 6 new tests PASS; full suite green (the existing graph/CLI tests still pass with the richer `thread_config`).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add pyproject.toml uv.lock .env.example src/vault_to_voice/config.py src/vault_to_voice/observability.py src/vault_to_voice/graph/runner.py tests/conftest.py tests/test_observability.py
git commit -m "feat: M3 deps, opt-in LangSmith tracing and run labels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: SSE framing of graph runs

**Files:**
- Create: `src/vault_to_voice/api/__init__.py` (empty), `src/vault_to_voice/api/events.py`
- Modify: `tests/fakes.py`
- Test: `tests/test_api_events.py`

**Interfaces:**
- Consumes: compiled post graph (`build_post_graph`), `thread_config`, `start_input`.
- Produces (in `vault_to_voice.api.events`, no FastAPI import so the UI can use it):
  - `sse(event: str, data: Any) -> str` — one frame `event: <e>\ndata: <json>\n\n`.
  - `parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, Any]]` — inverse of `sse`; ignores `:` comment lines; flushes a final event without a trailing blank line.
  - `stream_run(graph, config: dict, value) -> Iterator[str]` — runs `graph.stream(value, config, stream_mode="updates")`; yields `node {node}` per finished node, `interrupt <payload>` on a pause, `done {final_path}` when the graph ended, `error {message}` (`"<ExcType>: <msg>"`) on any exception (the thread stays at its checkpoint).
  - `tests/fakes.py`: `sse_events(frames: Iterable[str]) -> list[tuple[str, Any]]`.

- [ ] **Step 1: Add the test helper**

In `tests/fakes.py`, change the import lines at the top (keep imports at the top of the file; ruff E402):
```python
from collections.abc import Callable, Iterable
```
(replacing `from collections.abc import Callable`) and add, as the first `vault_to_voice` import:
```python
from vault_to_voice.api.events import parse_sse
```
Append at the end of `tests/fakes.py`:
```python


def sse_events(frames: Iterable[str]) -> list[tuple[str, Any]]:
    """Parse SSE frames (strings, as yielded by the API layer) into (event, data) pairs."""
    return list(parse_sse("".join(frames).split("\n")))
```

- [ ] **Step 2: Write the failing tests**

`tests/test_api_events.py`:
```python
from langgraph.checkpoint.memory import InMemorySaver

from tests.fakes import FakeLLMClient, graph_responses, make_deps, seed_idea, sse_events
from vault_to_voice.api.events import parse_sse, sse, stream_run
from vault_to_voice.graph.builder import build_post_graph
from vault_to_voice.graph.runner import start_input, thread_config


def test_sse_frame_round_trips():
    frame = sse("interrupt", {"type": "outline_review", "title": "Café\nmenu"})
    assert frame.startswith("event: interrupt\ndata: ") and frame.endswith("\n\n")
    assert frame.count("\n") == 3  # JSON escapes the newline, so one data line
    assert list(parse_sse(frame.split("\n"))) == [
        ("interrupt", {"type": "outline_review", "title": "Café\nmenu"})
    ]


def test_parse_sse_ignores_comments_and_flushes_the_last_event():
    lines = [": keep-alive", "event: node", 'data: {"node": "a"}', "", "event: done", "data: {}"]
    assert list(parse_sse(lines)) == [("node", {"node": "a"}), ("done", {})]


def test_stream_run_yields_nodes_then_the_interrupt(tmp_path, vault_dir):
    deps = make_deps(tmp_path, FakeLLMClient(graph_responses()))
    idea = seed_idea(deps, vault_dir)
    graph = build_post_graph(deps, checkpointer=InMemorySaver())
    events = sse_events(stream_run(graph, thread_config(idea.idea_id), start_input(idea)))
    assert events[0] == ("node", {"node": "load_context"})
    name, value = events[-1]
    assert name == "interrupt" and value["type"] == "interview_question"
    assert value["number"] == 1


class _Boom:
    def stream(self, *args, **kwargs):
        raise RuntimeError("model is down")


def test_stream_run_turns_exceptions_into_an_error_event():
    events = sse_events(stream_run(_Boom(), thread_config("t1"), None))
    assert events == [("error", {"message": "RuntimeError: model is down"})]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_events.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'vault_to_voice.api'` (raised from `tests/fakes.py`).

- [ ] **Step 4: Implement**

Create empty `src/vault_to_voice/api/__init__.py`.

`src/vault_to_voice/api/events.py`:
```python
import json
from collections.abc import Iterable, Iterator
from typing import Any

import structlog

log = structlog.get_logger()


def sse(event: str, data: Any) -> str:
    """One Server-Sent Events frame. JSON never contains raw newlines, so one data line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, Any]]:
    """Inverse of `sse` over a stream of lines (as from httpx `iter_lines`)."""
    event: str | None = None
    data: list[str] = []
    for line in lines:
        if line == "":
            if event is not None or data:
                yield event or "message", json.loads("\n".join(data)) if data else None
            event, data = None, []
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].strip())
    if event is not None or data:
        yield event or "message", json.loads("\n".join(data)) if data else None


def stream_run(graph, config: dict, value) -> Iterator[str]:
    """Run the post graph until its next interrupt (or the end), yielding SSE frames.

    Events: `node` {node} per finished node · `interrupt` {...payload} · `done` {final_path}
    · `error` {message}. On error the thread stays at its last checkpoint and can be retried.
    """
    thread_id = config["configurable"]["thread_id"]
    try:
        for chunk in graph.stream(value, config, stream_mode="updates"):
            for name, update in chunk.items():
                if name == "__interrupt__":
                    yield sse("interrupt", update[0].value)
                else:
                    yield sse("node", {"node": name})
        snapshot = graph.get_state(config)
    except Exception as exc:
        log.warning("run_failed", thread_id=thread_id, error=repr(exc))
        yield sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        return
    if not snapshot.next:
        log.info("run_done", thread_id=thread_id)
        yield sse("done", {"final_path": snapshot.values.get("final_path")})
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_api_events.py -v && uv run pytest`
Expected: 4 PASS; full suite green.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vault_to_voice/api/__init__.py src/vault_to_voice/api/events.py tests/fakes.py tests/test_api_events.py
git commit -m "feat: SSE event framing for post graph runs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Run service — thread views, start / resume / retry, one run per thread

**Files:**
- Create: `src/vault_to_voice/api/service.py`
- Modify: `tests/fakes.py`
- Test: `tests/test_api_service.py`

**Interfaces:**
- Consumes: `GraphDeps`, `build_post_graph`, `thread_config`, `start_input`, `stream_run` (Task 2), `Manifest.list_ideas/get_idea`.
- Produces (in `vault_to_voice.api.service`):
  - `ThreadStatus = Literal["new", "waiting", "stalled", "published"]`.
  - `ServiceError(Exception)` with `status_code = 400`; `NotFound` (404); `Conflict` (409).
  - `ThreadView(BaseModel)`: `thread_id: str, title: str, status: ThreadStatus, pending: dict | None, outline: dict | None, qa: list[dict], final_path: str | None`.
  - `RunService(deps: GraphDeps, checkpointer, *, tracing: bool = False)` with attributes `deps`, `tracing`, `graph`, and methods `list_ideas() -> list[Idea]`, `thread(id) -> ThreadView`, `list_threads() -> list[ThreadView]` (status != new), `post(id) -> {"final_path", "markdown", "sources"}`, `start(idea_id) -> Iterator[str]`, `resume(id, payload) -> Iterator[str]`, `retry(id) -> Iterator[str]`. Run methods validate state and take the thread's lock *eagerly* (so errors raise before any response starts) and return a `_LockedStream` that releases the lock when exhausted, failed, closed or garbage-collected.
  - Status rules: no checkpoint values → `new`; values but no `next` → `published`; `next` with an interrupt → `waiting` (pending = interrupt payload); `next` without interrupt → `stalled`.
  - `tests/fakes.py`: `make_service(tmp_path, vault_dir, **responses) -> (RunService, Idea, FakeLLMClient)`.

Note on the lock: it is a `threading.Lock` (not `RLock`) on purpose. Starlette pulls each frame of a sync iterator in a worker thread, so the thread that releases the lock may differ from the one that took it; only `Lock` allows that.

- [ ] **Step 1: Add the test helper**

In `tests/fakes.py`, add at the top (keep imports at the top; ruff E402):
```python
import datetime as dt
```
as the first line, `from langgraph.checkpoint.memory import InMemorySaver` before `from langgraph.store.memory import InMemoryStore`, and `from vault_to_voice.api.service import RunService` after `from vault_to_voice.api.events import parse_sse`. Append at the end:
```python


def make_service(tmp_path: Path, vault_dir: Path, **responses):
    """RunService over the fake graph: (service, idea, fake_llm). Offline, in-memory checkpoints."""
    fake = FakeLLMClient(graph_responses(**responses))
    deps = make_deps(tmp_path, fake)
    deps.today = lambda: dt.date(2026, 10, 10)
    idea = seed_idea(deps, vault_dir)
    return RunService(deps, InMemorySaver()), idea, fake
```

- [ ] **Step 2: Write the failing tests**

`tests/test_api_service.py`:
```python
import json
from pathlib import Path

import pytest

from tests.fakes import graph_responses, make_service, sse_events
from vault_to_voice.api.service import Conflict, NotFound
from vault_to_voice.graph.state import Outline

ANSWERS = ["The tells.", "Twenty deaths.", "Patience."]
APPROVE = {"action": "approve"}


def last(frames):
    return sse_events(frames)[-1]


def test_new_thread_view(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    view = service.thread(idea.idea_id)
    assert view.status == "new" and view.pending is None and view.title == "Fair difficulty"
    assert service.list_threads() == []
    assert [i.idea_id for i in service.list_ideas()] == [idea.idea_id]


def test_full_run_through_the_service(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    tid = idea.idea_id
    assert last(service.start(tid))[1]["type"] == "interview_question"
    service_view = service.thread(tid)
    assert service_view.status == "waiting" and service_view.pending["number"] == 1
    for answer in ANSWERS[:2]:
        last(service.resume(tid, answer))
    view = service.thread(tid)
    assert [q["answer"] for q in view.qa] == ANSWERS[:2]  # read from the paused subgraph
    assert last(service.resume(tid, ANSWERS[2]))[1]["type"] == "outline_review"
    assert service.thread(tid).outline["title"] == "Fair Bosses"
    review = last(service.resume(tid, APPROVE))[1]
    assert review["type"] == "draft_review" and review["coverage"] == 1.0
    name, data = last(service.resume(tid, APPROVE))
    assert name == "done" and Path(data["final_path"]).exists()
    view = service.thread(tid)
    assert view.status == "published" and view.final_path == data["final_path"]
    assert [t.thread_id for t in service.list_threads()] == [tid]
    post = service.post(tid)
    assert post["markdown"].startswith("---\n")
    assert post["sources"] == json.loads(
        Path(data["final_path"]).with_suffix(".sources.json").read_text(encoding="utf-8")
    )


def test_unknown_idea_is_not_found(tmp_path, vault_dir):
    service, _, _ = make_service(tmp_path, vault_dir)
    with pytest.raises(NotFound, match="unknown idea"):
        service.start("idea_nope")
    with pytest.raises(NotFound):
        service.thread("idea_nope")


def test_wrong_state_operations_conflict(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    tid = idea.idea_id
    with pytest.raises(Conflict, match="nothing is waiting"):
        service.resume(tid, "answer")
    with pytest.raises(Conflict, match="not stalled"):
        service.retry(tid)
    last(service.start(tid))
    with pytest.raises(Conflict, match="already started"):
        service.start(tid)
    with pytest.raises(NotFound, match="not published"):
        service.post(tid)


def test_one_run_at_a_time_per_thread(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    other = idea.model_copy(update={"idea_id": "idea_other", "title": "Other"})
    service.deps.manifest.save_ideas([idea, other])
    running = service.start(idea.idea_id)
    next(running)  # first frame produced; the run holds the thread's lock
    with pytest.raises(Conflict, match="already running"):
        service.start(idea.idea_id)
    second = sse_events(service.start("idea_other"))  # other threads are not blocked
    assert second[-1][0] == "interrupt"
    list(running)  # exhausting the stream releases the lock
    assert last(service.resume(idea.idea_id, "ok"))[0] == "interrupt"


def test_dropped_stream_releases_the_lock(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    stream = service.start(idea.idea_id)
    stream.close()  # e.g. the client disconnected before the first frame
    assert last(service.start(idea.idea_id))[0] == "interrupt"


def test_failed_step_stalls_then_retry_continues(tmp_path, vault_dir):
    calls = {"n": 0}
    good = graph_responses()[Outline]

    def flaky_outline(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("model is down")  # RuntimeError is not retried by LangGraph
        return good(prompt)

    service, idea, fake = make_service(tmp_path, vault_dir)
    fake.responses[Outline] = flaky_outline
    tid = idea.idea_id
    last(service.start(tid))
    last(service.resume(tid, ANSWERS[0]))
    last(service.resume(tid, ANSWERS[1]))
    assert last(service.resume(tid, ANSWERS[2])) == (
        "error",
        {"message": "RuntimeError: model is down"},
    )
    assert service.thread(tid).status == "stalled"
    assert last(service.retry(tid))[1]["type"] == "outline_review"


def test_malformed_review_decision_shows_the_review_again(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    tid = idea.idea_id
    last(service.start(tid))
    for answer in ANSWERS:
        last(service.resume(tid, answer))
    name, again = last(service.resume(tid, {"action": "publish-now"}))
    assert name == "interrupt" and again["type"] == "outline_review" and again["error"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_service.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'vault_to_voice.api.service'`.

- [ ] **Step 4: Implement**

`src/vault_to_voice/api/service.py`:
```python
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from langgraph.types import Command
from pydantic import BaseModel

from vault_to_voice.api.events import stream_run
from vault_to_voice.graph.builder import build_post_graph
from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.runner import start_input, thread_config
from vault_to_voice.ideas.detector import Idea

ThreadStatus = Literal["new", "waiting", "stalled", "published"]


class ServiceError(Exception):
    status_code = 400


class NotFound(ServiceError):
    status_code = 404


class Conflict(ServiceError):
    status_code = 409


class ThreadView(BaseModel):
    """What a client needs to render one post thread (thread_id = idea_id)."""

    thread_id: str
    title: str
    status: ThreadStatus
    pending: dict | None = None
    outline: dict | None = None
    qa: list[dict] = []
    final_path: str | None = None


_NEEDS = {
    "new": "this thread has already started; resume it or retry it instead",
    "waiting": "nothing is waiting for an answer on this thread",
    "stalled": "this thread is not stalled; there is nothing to retry",
}


class _LockedStream:
    """Iterator of SSE frames that holds the thread's run lock until it is exhausted or closed.

    A plain generator would leak the lock if the response were dropped before its first frame
    (an unstarted generator never runs its `finally`).
    """

    def __init__(self, lock: threading.Lock, frames: Iterator[str]):
        self._lock, self._frames, self._open = lock, frames, True

    def __iter__(self) -> "_LockedStream":
        return self

    def __next__(self) -> str:
        try:
            return next(self._frames)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._open:
            self._open = False
            self._lock.release()

    def __del__(self) -> None:
        self.close()


class RunService:
    """The post graph behind the HTTP API: one SSE run at a time per thread."""

    def __init__(self, deps: GraphDeps, checkpointer, *, tracing: bool = False):
        self.deps = deps
        self.tracing = tracing
        self.graph = build_post_graph(deps, checkpointer=checkpointer)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # ---- reads -------------------------------------------------------------------------
    def list_ideas(self) -> list[Idea]:
        return self.deps.manifest.list_ideas()

    def _idea(self, idea_id: str) -> Idea:
        idea = self.deps.manifest.get_idea(idea_id)
        if idea is None:
            raise NotFound(f"unknown idea {idea_id}; run `v2v ideas` to detect ideas")
        return idea

    def thread(self, thread_id: str) -> ThreadView:
        idea = self._idea(thread_id)
        # subgraphs=True: while the interview subgraph is paused, its `qa` lives in the
        # subgraph's own checkpoint, not yet in the parent state.
        snap = self.graph.get_state(thread_config(thread_id), subgraphs=True)
        values: dict[str, Any] = snap.values or {}
        qa = values.get("qa", [])
        for task in snap.tasks:
            if task.name == "interview" and task.state is not None:
                qa = task.state.values.get("qa", qa)
        pending = None
        if not values:
            status: ThreadStatus = "new"
        elif not snap.next:
            status = "published"
        elif snap.interrupts:
            status, pending = "waiting", snap.interrupts[0].value
        else:
            status = "stalled"
        outline = values.get("outline")
        return ThreadView(
            thread_id=thread_id,
            title=idea.title,
            status=status,
            pending=pending,
            outline=outline.model_dump() if outline is not None else None,
            qa=[q.model_dump() for q in qa],
            final_path=values.get("final_path"),
        )

    def list_threads(self) -> list[ThreadView]:
        views = (self.thread(i.idea_id) for i in self.list_ideas())
        return [v for v in views if v.status != "new"]

    def post(self, thread_id: str) -> dict:
        view = self.thread(thread_id)
        if view.status != "published" or not view.final_path:
            raise NotFound(f"thread {thread_id} is not published yet")
        path = Path(view.final_path)
        sidecar = path.with_suffix(".sources.json")
        if not path.exists() or not sidecar.exists():
            raise NotFound(f"published files for {thread_id} are missing from {path.parent}")
        return {
            "final_path": str(path),
            "markdown": path.read_text(encoding="utf-8"),
            "sources": json.loads(sidecar.read_text(encoding="utf-8")),
        }

    # ---- runs (each returns SSE frames) --------------------------------------------------
    def start(self, idea_id: str) -> Iterator[str]:
        idea = self._idea(idea_id)
        return self._run(idea_id, "new", start_input(idea))

    def resume(self, thread_id: str, payload: Any) -> Iterator[str]:
        return self._run(thread_id, "waiting", Command(resume=payload))

    def retry(self, thread_id: str) -> Iterator[str]:
        return self._run(thread_id, "stalled", None)

    def _lock(self, thread_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(thread_id, threading.Lock())

    def _run(self, thread_id: str, needs: ThreadStatus, value) -> Iterator[str]:
        lock = self._lock(thread_id)
        if not lock.acquire(blocking=False):
            raise Conflict(f"thread {thread_id} is already running; wait for it to finish")
        try:
            status = self.thread(thread_id).status
            if status != needs:
                raise Conflict(f"{_NEEDS[needs]} (status: {status})")
        except BaseException:
            lock.release()
            raise
        return _LockedStream(lock, stream_run(self.graph, thread_config(thread_id), value))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_api_service.py -v && uv run pytest`
Expected: 8 PASS; full suite green.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vault_to_voice/api/service.py tests/fakes.py tests/test_api_service.py
git commit -m "feat: run service with thread views and one run per thread

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: FastAPI app

**Files:**
- Create: `src/vault_to_voice/api/app.py`
- Test: `tests/test_api_app.py`

**Interfaces:**
- Consumes: `RunService`, `ServiceError`, `ThreadView` (Task 3), `parse_sse` (Task 2, in tests), `Idea`.
- Produces: `create_app(service: RunService) -> FastAPI` with the routes in the API surface table; `StartRequest{idea_id: str}`, `ResumeRequest{payload: str | dict}`, `PostOut{final_path, markdown, sources}`. `ServiceError` subclasses map to their `status_code` with `{"detail": str(exc)}`. SSE responses carry `Cache-Control: no-cache` and `X-Accel-Buffering: no` (so a reverse proxy does not buffer the stream later).

- [ ] **Step 1: Write the failing tests**

`tests/test_api_app.py`:
```python
import pytest
from fastapi.testclient import TestClient

from tests.fakes import make_service
from vault_to_voice.api.app import create_app
from vault_to_voice.api.events import parse_sse

ANSWERS = ["The tells.", "Twenty deaths.", "Patience."]
APPROVE = {"action": "approve"}


@pytest.fixture
def api(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    return TestClient(create_app(service)), idea.idea_id


def events(client, method, url, **kwargs):
    with client.stream(method, url, **kwargs) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return list(parse_sse(response.iter_lines()))


def resume(client, tid, payload):
    return events(client, "POST", f"/threads/{tid}/resume", json={"payload": payload})


def test_health_and_ideas(api):
    client, tid = api
    assert client.get("/health").json() == {"status": "ok", "tracing": False}
    ideas = client.get("/ideas").json()
    assert [i["idea_id"] for i in ideas] == [tid] and ideas[0]["status"] == "new"
    assert client.get("/threads").json() == []
    assert client.get(f"/threads/{tid}").json()["status"] == "new"


def test_full_flow_over_http(api):
    client, tid = api
    first = events(client, "POST", "/threads", json={"idea_id": tid})
    assert first[0] == ("node", {"node": "load_context"})
    assert first[-1][0] == "interrupt" and first[-1][1]["type"] == "interview_question"
    for answer in ANSWERS:
        last = resume(client, tid, answer)[-1]
    assert last[1]["type"] == "outline_review"
    view = client.get(f"/threads/{tid}").json()
    assert view["status"] == "waiting" and view["pending"]["type"] == "outline_review"
    assert len(view["qa"]) == 3
    last = resume(client, tid, APPROVE)[-1]
    assert last[1]["type"] == "draft_review" and last[1]["coverage"] == 1.0
    done = resume(client, tid, APPROVE)
    assert ("node", {"node": "publish"}) in done and done[-1][0] == "done"
    post = client.get(f"/threads/{tid}/post").json()
    assert post["final_path"] == done[-1][1]["final_path"]
    assert "# Fair Bosses" in post["markdown"] and post["sources"]["paragraphs"]
    assert [t["status"] for t in client.get("/threads").json()] == ["published"]
    assert client.get("/ideas").json()[0]["status"] == "done"


def test_errors_map_to_http_status_codes(api):
    client, tid = api
    r = client.post("/threads", json={"idea_id": "idea_nope"})
    assert r.status_code == 404 and "unknown idea" in r.json()["detail"]
    assert client.get("/threads/idea_nope").status_code == 404
    assert client.post(f"/threads/{tid}/resume", json={"payload": "x"}).status_code == 409
    assert client.get(f"/threads/{tid}/stream").status_code == 409
    assert client.get(f"/threads/{tid}/post").status_code == 404
    events(client, "POST", "/threads", json={"idea_id": tid})
    assert client.post("/threads", json={"idea_id": tid}).status_code == 409


@pytest.mark.parametrize("body", [{}, {"payload": None}, {"payload": 5}, {"payload": [1]}])
def test_invalid_resume_body_is_rejected(api, body):
    client, tid = api
    events(client, "POST", "/threads", json={"idea_id": tid})
    assert client.post(f"/threads/{tid}/resume", json=body).status_code == 422
    assert client.get(f"/threads/{tid}").json()["pending"]["number"] == 1  # nothing consumed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vault_to_voice.api.app'`.

- [ ] **Step 3: Implement**

`src/vault_to_voice/api/app.py`:
```python
from collections.abc import Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from vault_to_voice.api.service import RunService, ServiceError, ThreadView
from vault_to_voice.ideas.detector import Idea


class StartRequest(BaseModel):
    idea_id: str


class ResumeRequest(BaseModel):
    # Interview: the answer text. Reviews: {"action": "approve"|"revise"|"edit", ...}.
    payload: str | dict


class PostOut(BaseModel):
    final_path: str
    markdown: str
    sources: dict


def _sse(frames: Iterator[str]) -> StreamingResponse:
    return StreamingResponse(
        frames,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(service: RunService) -> FastAPI:
    app = FastAPI(title="vault-to-voice", version="0.1.0")
    app.state.service = service

    @app.exception_handler(ServiceError)
    async def _service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "tracing": service.tracing}

    @app.get("/ideas")
    def ideas() -> list[Idea]:
        return service.list_ideas()

    @app.get("/threads")
    def threads() -> list[ThreadView]:
        return service.list_threads()

    @app.post("/threads")
    def start(body: StartRequest) -> StreamingResponse:
        return _sse(service.start(body.idea_id))

    @app.get("/threads/{thread_id}")
    def thread(thread_id: str) -> ThreadView:
        return service.thread(thread_id)

    @app.post("/threads/{thread_id}/resume")
    def resume(thread_id: str, body: ResumeRequest) -> StreamingResponse:
        return _sse(service.resume(thread_id, body.payload))

    @app.get("/threads/{thread_id}/stream")
    def stream(thread_id: str) -> StreamingResponse:
        """Continue a stalled thread (a step failed or the server died mid-step)."""
        return _sse(service.retry(thread_id))

    @app.get("/threads/{thread_id}/post")
    def post(thread_id: str) -> PostOut:
        return PostOut(**service.post(thread_id))

    return app
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api_app.py -v && uv run pytest`
Expected: 7 PASS (4 parametrized invalid bodies); full suite green.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vault_to_voice/api/app.py tests/test_api_app.py
git commit -m "feat: FastAPI app with SSE start, resume and retry endpoints

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Production wiring and `v2v serve`

**Files:**
- Create: `src/vault_to_voice/api/factory.py`
- Modify: `src/vault_to_voice/cli.py`
- Test: `tests/test_api_factory.py`

**Interfaces:**
- Consumes: `make_checkpointer`, `make_store` (`graph/checkpoint.py`), `make_embedder`, `FaissStore`, `Manifest`, `OutboxPublisher`, `RunService`, `create_app`, `configure_tracing`, `cli.make_llm`, `cli._settings`.
- Produces:
  - `build_service(settings: Settings, llm: LLMClient, *, tracing: bool = False) -> RunService` — same wiring as `v2v write` (SQLite manifest, FAISS, SQLite checkpoints + Store under `DATA_DIR`, outbox publisher). The LLM is passed in so `api/` never imports `cli.py`.
  - `v2v serve [--host H] [--port P]` — configures tracing, builds the service, runs `uvicorn.run(app, host=..., port=...)` (defaults `API_HOST`/`API_PORT`). No `--reload` (needs an import string; not worth it here).

- [ ] **Step 1: Write the failing tests**

`tests/test_api_factory.py`:
```python
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tests.fakes import FakeLLMClient, graph_responses, seed_idea, sse_events
from vault_to_voice import cli
from vault_to_voice.api.factory import build_service
from vault_to_voice.config import Settings


def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        outbox_dir=tmp_path / "outbox",
        embedding_provider="hash",
    )


def test_restarted_server_resumes_the_same_thread(tmp_path, vault_dir):
    s = settings(tmp_path)
    first = build_service(s, FakeLLMClient(graph_responses()))
    idea = seed_idea(first.deps, vault_dir)
    sse_events(first.start(idea.idea_id))
    sse_events(first.resume(idea.idea_id, "The tells."))

    fresh_llm = FakeLLMClient(graph_responses())
    second = build_service(s, fresh_llm)  # a new process: same SQLite files, nothing in memory
    view = second.thread(idea.idea_id)
    assert view.status == "waiting" and view.pending["number"] == 2
    assert [q["answer"] for q in view.qa] == ["The tells."]
    assert fresh_llm.calls == []  # reading state never re-asks the model
    assert sse_events(second.resume(idea.idea_id, "Twenty deaths."))[-1][1]["number"] == 3


def test_serve_builds_the_app_and_runs_uvicorn(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setattr(cli, "make_llm", lambda settings: FakeLLMClient(graph_responses()))
    seen = {}

    def fake_run(app, host, port):
        seen.update(app=app, host=host, port=port)

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = CliRunner().invoke(cli.app, ["serve", "--port", "9001"])
    assert result.exit_code == 0, result.output
    assert seen["host"] == "127.0.0.1" and seen["port"] == 9001
    assert TestClient(seen["app"]).get("/health").json() == {"status": "ok", "tracing": False}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api_factory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vault_to_voice.api.factory'`.

- [ ] **Step 3: Implement**

`src/vault_to_voice/api/factory.py`:
```python
from vault_to_voice.api.service import RunService
from vault_to_voice.config import Settings
from vault_to_voice.graph.checkpoint import make_checkpointer, make_store
from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.index.embedder import make_embedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.llm import LLMClient
from vault_to_voice.publish import OutboxPublisher


def build_service(settings: Settings, llm: LLMClient, *, tracing: bool = False) -> RunService:
    """Production wiring: SQLite manifest/checkpoints/Store under DATA_DIR, outbox publisher."""
    embedder = make_embedder(settings)
    deps = GraphDeps(
        llm=llm,
        manifest=Manifest(settings.manifest_path),
        index=FaissStore(settings.faiss_path, embedder.dim),
        embedder=embedder,
        settings=settings,
        store=make_store(settings.store_path),
        publisher=OutboxPublisher(settings.outbox_dir),
    )
    return RunService(deps, make_checkpointer(settings.checkpoint_path), tracing=tracing)
```

Append to `src/vault_to_voice/cli.py` (imports are local so `v2v ingest` etc. do not pay for FastAPI/uvicorn imports):
```python


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind address (default API_HOST)."),
    port: int | None = typer.Option(None, help="Port (default API_PORT)."),
):
    """Run the HTTP API (FastAPI + SSE) that the Streamlit UI talks to."""
    import uvicorn

    from vault_to_voice.api.app import create_app
    from vault_to_voice.api.factory import build_service
    from vault_to_voice.observability import configure_tracing

    s = _settings()
    service = build_service(s, make_llm(s), tracing=configure_tracing(s))
    uvicorn.run(create_app(service), host=host or s.api_host, port=port or s.api_port)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api_factory.py -v && uv run pytest`
Expected: 2 PASS; full suite green.

- [ ] **Step 5: Smoke the real server (manual, offline-safe)**

Run: `uv run v2v serve --port 8765` in one terminal, then `curl http://127.0.0.1:8765/health` and `curl http://127.0.0.1:8765/ideas`.
Expected: `{"status":"ok","tracing":false}` and the ideas JSON from `data/manifest.db` (empty list if `v2v ideas` was never run). Stop with Ctrl+C. (Starting a thread needs a working LLM; that is the session-end smoke in Task 9.)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vault_to_voice/api/factory.py src/vault_to_voice/cli.py tests/test_api_factory.py
git commit -m "feat: production service wiring and v2v serve

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: API client for the UI

**Files:**
- Create: `src/vault_to_voice/ui/__init__.py` (empty), `src/vault_to_voice/ui/client.py`
- Test: `tests/test_ui_client.py`

**Interfaces:**
- Consumes: `parse_sse` (Task 2); the HTTP API (Task 4).
- Produces (in `vault_to_voice.ui.client`):
  - `Event = tuple[str, Any]`.
  - `ApiError(RuntimeError)` with `.status_code: int`, `.detail: str`; `str(e) == "<code>: <detail>"`.
  - `ApiClient(http)` where `http` is any httpx-style client (`httpx.Client` or FastAPI `TestClient`); `ApiClient.from_url(base_url, *, timeout=300.0)`; methods `health() -> dict`, `ideas() -> list[dict]`, `threads() -> list[dict]`, `thread(id) -> dict`, `post(id) -> dict`, `start(idea_id) -> Iterator[Event]`, `resume(id, payload) -> Iterator[Event]`, `retry(id) -> Iterator[Event]`. Stream methods are generators: HTTP errors raise `ApiError` on first iteration. Connection failures surface as `httpx.HTTPError` (the UI catches both).

- [ ] **Step 1: Write the failing tests**

`tests/test_ui_client.py`:
```python
import httpx
import pytest
from fastapi.testclient import TestClient

from tests.fakes import make_service
from vault_to_voice.api.app import create_app
from vault_to_voice.ui.client import ApiClient, ApiError

APPROVE = {"action": "approve"}


@pytest.fixture
def client(tmp_path, vault_dir):
    service, idea, _ = make_service(tmp_path, vault_dir)
    return ApiClient(TestClient(create_app(service))), idea.idea_id


def test_client_drives_a_post_to_publication(client):
    api, tid = client
    assert api.health()["status"] == "ok"
    assert [i["idea_id"] for i in api.ideas()] == [tid]
    events = list(api.start(tid))
    assert events[0] == ("node", {"node": "load_context"}) and events[-1][0] == "interrupt"
    for answer in ["a", "b", "c"]:
        last = list(api.resume(tid, answer))[-1]
    assert last[1]["type"] == "outline_review"
    assert list(api.resume(tid, APPROVE))[-1][1]["type"] == "draft_review"
    assert list(api.resume(tid, APPROVE))[-1][0] == "done"
    assert api.thread(tid)["status"] == "published"
    assert api.threads()[0]["thread_id"] == tid
    assert api.post(tid)["sources"]["paragraphs"]


def test_http_errors_raise_api_error_with_detail(client):
    api, tid = client
    with pytest.raises(ApiError) as exc:
        list(api.start("idea_nope"))
    assert exc.value.status_code == 404 and "unknown idea" in exc.value.detail
    with pytest.raises(ApiError) as exc:
        api.post(tid)
    assert exc.value.status_code == 404
    with pytest.raises(ApiError) as exc:
        list(api.retry(tid))
    assert exc.value.status_code == 409


def test_from_url_builds_an_httpx_client():
    api = ApiClient.from_url("http://127.0.0.1:9", timeout=5)
    assert isinstance(api.http, httpx.Client) and str(api.http.base_url) == "http://127.0.0.1:9"
    with pytest.raises(httpx.ConnectError):
        api.health()  # nothing listens on port 9: the UI must show this, not crash
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ui_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vault_to_voice.ui'`.

- [ ] **Step 3: Implement**

Create empty `src/vault_to_voice/ui/__init__.py`.

`src/vault_to_voice/ui/client.py`:
```python
from collections.abc import Iterator
from typing import Any

import httpx

from vault_to_voice.api.events import parse_sse

Event = tuple[str, Any]


class ApiError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class ApiClient:
    """Thin HTTP client for the vault-to-voice API. `http` is an httpx-style client
    (httpx.Client in the app, FastAPI's TestClient in tests)."""

    def __init__(self, http):
        self.http = http

    @classmethod
    def from_url(cls, base_url: str, *, timeout: float = 300.0) -> "ApiClient":
        return cls(httpx.Client(base_url=base_url, timeout=timeout))

    @staticmethod
    def _check(response) -> None:
        if response.status_code < 400:
            return
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(response.status_code, str(detail))

    def _json(self, method: str, url: str, **kwargs) -> Any:
        response = self.http.request(method, url, **kwargs)
        self._check(response)
        return response.json()

    def _events(self, method: str, url: str, **kwargs) -> Iterator[Event]:
        with self.http.stream(method, url, **kwargs) as response:
            if response.status_code >= 400:
                response.read()
                self._check(response)
            yield from parse_sse(response.iter_lines())

    def health(self) -> dict:
        return self._json("GET", "/health")

    def ideas(self) -> list[dict]:
        return self._json("GET", "/ideas")

    def threads(self) -> list[dict]:
        return self._json("GET", "/threads")

    def thread(self, thread_id: str) -> dict:
        return self._json("GET", f"/threads/{thread_id}")

    def post(self, thread_id: str) -> dict:
        return self._json("GET", f"/threads/{thread_id}/post")

    def start(self, idea_id: str) -> Iterator[Event]:
        return self._events("POST", "/threads", json={"idea_id": idea_id})

    def resume(self, thread_id: str, payload: str | dict) -> Iterator[Event]:
        return self._events("POST", f"/threads/{thread_id}/resume", json={"payload": payload})

    def retry(self, thread_id: str) -> Iterator[Event]:
        return self._events("GET", f"/threads/{thread_id}/stream")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ui_client.py -v && uv run pytest`
Expected: 3 PASS; full suite green.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vault_to_voice/ui/__init__.py src/vault_to_voice/ui/client.py tests/test_ui_client.py
git commit -m "feat: HTTP and SSE client for the UI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Pure UI view helpers

**Files:**
- Create: `src/vault_to_voice/ui/views.py`
- Test: `tests/test_ui_views.py`

**Interfaces:**
- Consumes: the M2 interrupt payload shapes — `draft_review.draft` is `[{section_id, paragraphs: [{sentences: [{text, sources, kind}]}]}]`, `flags` are `{section_id, paragraph, sentence, text, reason, score}` with **0-based** paragraph/sentence indices (as built by `check_citations`), `outline_review.outline` is `{title, sections: [{id, heading, point, planned_sources}]}`.
- Produces (in `vault_to_voice.ui.views`, no streamlit import):
  - `last_event(events, on_node=None) -> Event` — drains a run, calls `on_node(name)` per `node` event, returns the final non-node event, or `("error", {"message": ...Retry...})` if the stream ended without one.
  - `flagged_markdown(draft, flags, headings: dict[str, str]) -> str` — `### heading` per section, sentences joined by spaces, flagged ones as `:red[text] :red[_(reason)_]`; sentence text is markdown-escaped.
  - `section_texts(draft) -> dict[str, str]` — editable text per section (paragraphs separated by a blank line, which M2's `split_paragraphs` splits back).
  - `changed_sections(original, edited) -> dict[str, str]` — only changed, known sections (whitespace-only changes ignored).
  - `edited_outline(outline, title, headings, points) -> dict` — keeps ids and planned sources; raises `ValueError` on a length mismatch.
  - `strip_frontmatter(markdown) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_ui_views.py`:
```python
import pytest

from vault_to_voice.ui import views

DRAFT = [
    {
        "section_id": "s1",
        "paragraphs": [
            {
                "sentences": [
                    {"text": "Tells make it fair.", "sources": ["n0"], "kind": "claim"},
                    {"text": "Bosses [cheat] *a lot*.", "sources": [], "kind": "claim"},
                ]
            },
            {"sentences": [{"text": "So it goes.", "sources": [], "kind": "connective"}]},
        ],
    }
]
FLAGS = [
    {"section_id": "s1", "paragraph": 0, "sentence": 1, "text": "x", "reason": "missing_source"}
]


def test_last_event_reports_nodes_and_returns_the_final_event():
    seen = []
    events = [("node", {"node": "a"}), ("node", {"node": "b"}), ("interrupt", {"type": "q"})]
    assert views.last_event(events, on_node=seen.append) == ("interrupt", {"type": "q"})
    assert seen == ["a", "b"]


def test_last_event_without_a_result_is_an_error():
    name, data = views.last_event([("node", {"node": "a"})])
    assert name == "error" and "Retry" in data["message"]


def test_flagged_markdown_marks_only_flagged_sentences_and_escapes_markup():
    md = views.flagged_markdown(DRAFT, FLAGS, {"s1": "Tells"})
    assert md.startswith("### Tells\n\nTells make it fair. :red[")
    assert ":red[Bosses \\[cheat\\] \\*a lot\\*.] :red[_(missing source)_]" in md
    assert md.endswith("\n\nSo it goes.")
    assert views.flagged_markdown(DRAFT, [], {}).startswith("### s1\n\n")  # unknown heading


def test_section_texts_and_changed_sections():
    texts = views.section_texts(DRAFT)
    assert texts == {"s1": "Tells make it fair. Bosses [cheat] *a lot*.\n\nSo it goes."}
    assert views.changed_sections(texts, {"s1": texts["s1"] + "  "}) == {}
    assert views.changed_sections(texts, {"s1": "Mine.", "zz": "ignored"}) == {"s1": "Mine."}


def test_edited_outline_keeps_ids_and_sources():
    outline = {
        "title": "Old",
        "sections": [{"id": "s1", "heading": "H", "point": "P", "planned_sources": ["n0"]}],
    }
    assert views.edited_outline(outline, " New ", [" H2 "], ["P2"]) == {
        "title": "New",
        "sections": [{"id": "s1", "heading": "H2", "point": "P2", "planned_sources": ["n0"]}],
    }
    with pytest.raises(ValueError):
        views.edited_outline(outline, "New", [], [])


def test_strip_frontmatter():
    assert views.strip_frontmatter("---\ntitle: x\n---\n\n# Post\n") == "# Post\n"
    assert views.strip_frontmatter("# No front\n") == "# No front\n"
    assert views.strip_frontmatter("---\nunterminated") == "---\nunterminated"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ui_views.py -v`
Expected: FAIL with `ImportError: cannot import name 'views' from 'vault_to_voice.ui'`.

- [ ] **Step 3: Implement**

`src/vault_to_voice/ui/views.py`:
```python
"""Pure helpers for the Streamlit UI (no streamlit import, so they unit-test directly)."""

from collections.abc import Callable, Iterable
from typing import Any

Event = tuple[str, Any]


def last_event(events: Iterable[Event], on_node: Callable[[str], None] | None = None) -> Event:
    """Drain a run's events, reporting each finished node; return the final event
    (interrupt / done / error). A stream that ends without one is reported as an error."""
    last: Event | None = None
    for name, data in events:
        if name == "node":
            if on_node is not None:
                on_node(data["node"])
        else:
            last = (name, data)
    return last or ("error", {"message": "The run ended without a result; try Retry."})


def _md(text: str) -> str:
    """Escape characters Streamlit markdown would treat as markup inside a sentence."""
    for ch in "\\[]*_`#$":
        text = text.replace(ch, "\\" + ch)
    return text


def flagged_markdown(draft: list[dict], flags: list[dict], headings: dict[str, str]) -> str:
    """Draft as markdown with every flagged sentence shown in red with its reason."""
    bad = {(f["section_id"], f["paragraph"], f["sentence"]): f["reason"] for f in flags}
    parts: list[str] = []
    for section in draft:
        sid = section["section_id"]
        parts.append(f"### {headings.get(sid, sid)}")
        for pi, paragraph in enumerate(section["paragraphs"]):
            out = []
            for si, sentence in enumerate(paragraph["sentences"]):
                text = _md(sentence["text"])
                reason = bad.get((sid, pi, si))
                out.append(f":red[{text}] :red[_({reason.replace('_', ' ')})_]" if reason else text)
            parts.append(" ".join(out))
    return "\n\n".join(parts)


def section_texts(draft: list[dict]) -> dict[str, str]:
    """Editable plain text per section: sentences joined by spaces, paragraphs by a blank line."""
    return {
        s["section_id"]: "\n\n".join(
            " ".join(sent["text"] for sent in p["sentences"]) for p in s["paragraphs"]
        )
        for s in draft
    }


def changed_sections(original: dict[str, str], edited: dict[str, str]) -> dict[str, str]:
    """Only the sections whose text the author actually changed."""
    return {
        sid: text
        for sid, text in edited.items()
        if sid in original and text.strip() != original[sid].strip()
    }


def edited_outline(outline: dict, title: str, headings: list[str], points: list[str]) -> dict:
    """Outline dict with the author's title/headings/points; ids and planned sources kept."""
    sections = [
        {**section, "heading": heading.strip(), "point": point.strip()}
        for section, heading, point in zip(outline["sections"], headings, points, strict=True)
    ]
    return {"title": title.strip(), "sections": sections}


def strip_frontmatter(markdown: str) -> str:
    """Post body without its leading `---` YAML block (for preview only)."""
    if markdown.startswith("---\n"):
        end = markdown.find("\n---\n", 4)
        if end != -1:
            return markdown[end + len("\n---\n") :].lstrip("\n")
    return markdown
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ui_views.py -v && uv run pytest`
Expected: 6 PASS; full suite green.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vault_to_voice/ui/views.py tests/test_ui_views.py
git commit -m "feat: pure view helpers for the Streamlit UI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Streamlit UI

**Files:**
- Create: `src/vault_to_voice/ui/streamlit_app.py`
- Test: `tests/test_ui_app.py`

**Interfaces:**
- Consumes: `ApiClient`, `ApiError`, `Event` (Task 6), `views` (Task 7), `ThreadView` JSON (Task 3), M2 interrupt payloads.
- Produces: a Streamlit script (calls `main()` at import, as Streamlit requires) with these stable widget keys, which the tests and later milestones rely on:
  - sidebar: `idea_<idea_id>` buttons (label `"<title> · <status>"`); sets `session_state["thread_id"]`.
  - `start` (new thread) · `retry` (stalled thread).
  - interview: `answer_<n>` text area, `send_answer` button; earlier Q&A shown as `st.chat_message` pairs, current question as `**Q<n>/<max>:** <question>`.
  - outline review: `ol_title_<rounds_left>`, `ol_h_<rounds_left>_<i>`, `ol_p_<rounds_left>_<i>`, `ol_feedback_<rounds_left>`, buttons `ol_approve`, `ol_revise`, `ol_edit`. (Keys include `rounds_left` so a revised outline is not masked by the previous round's widget state.)
  - draft review: metrics `Citation coverage` and `Flagged sentences`, flagged markdown, expander with `dr_<rounds_left>_<section_id>` text areas and `dr_edit`; `dr_feedback_<rounds_left>`, buttons `dr_approve`, `dr_revise`.
  - published: success `Published: <path>`, `download` button, post body without frontmatter, `Sources` expander.
  - `session_state["client"]` (an `ApiClient`; built from `V2V_API_URL` if absent), `session_state["flash"]` (one-shot error shown on the next run).
- Every graph step goes through `drive(events)`: an `st.status` box lists finished nodes live while the SSE stream runs, then `st.rerun()` redraws from `GET /threads/{id}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_ui_app.py`:
```python
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import vault_to_voice.ui
from tests.fakes import graph_responses, make_service
from vault_to_voice.api.app import create_app
from vault_to_voice.graph.state import Outline, Paragraph, SectionText, Sentence
from vault_to_voice.ui.client import ApiClient

APP = str(Path(vault_to_voice.ui.__file__).parent / "streamlit_app.py")


def bad_section(_prompt):
    sentence = Sentence(text="Purple bananas orbit distant moons quietly.", sources=["n0"])
    return SectionText(paragraphs=[Paragraph(sentences=[sentence])])


def app_for(client) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["client"] = client  # the app uses this instead of building one
    at.run()
    assert not at.exception
    return at


def open_thread(at, tid) -> AppTest:
    at.button(key=f"idea_{tid}").click().run()
    assert not at.exception
    return at


@pytest.fixture
def ui(tmp_path, vault_dir):
    def _make(**responses):
        service, idea, fake = make_service(tmp_path, vault_dir, **responses)
        return ApiClient(TestClient(create_app(service))), idea.idea_id, fake

    return _make


def answer_interview(at):
    for n, answer in enumerate(["The tells.", "Twenty deaths.", "Patience."], 1):
        assert f"Q{n}/5" in at.chat_message[-1].markdown[0].value
        at.text_area(key=f"answer_{n}").input(answer)
        at.button(key="send_answer").click().run()
        assert not at.exception, at.exception


def test_full_post_with_outline_and_draft_edits(ui):
    client, tid, _ = ui()
    at = open_thread(app_for(client), tid)
    assert at.title[0].value == "Fair difficulty"
    at.button(key="start").click().run()
    answer_interview(at)
    at.text_input(key="ol_title_3").input("My Title")
    at.text_input(key="ol_h_3_0").input("Clear tells")
    at.button(key="ol_edit").click().run()
    assert not at.exception, at.exception
    assert at.metric[0].value == "100%" and at.metric[1].value == "0"
    assert any("### Clear tells" in m.value for m in at.markdown)
    at.text_area(key="dr_3_s2").input("My own words here.")
    at.button(key="dr_edit").click().run()
    assert any("My own words here." in m.value for m in at.markdown)
    at.button(key="dr_approve").click().run()
    assert not at.exception, at.exception
    assert at.success[0].value.startswith("Published: ")
    body = [m.value for m in at.markdown]
    assert body[0].startswith("# My Title") and "title:" not in body[0]  # frontmatter hidden
    assert "> My own words here." in body
    assert any("published" in b.label for b in at.sidebar.button)


def test_interview_shows_earlier_answers_chat_style(ui):
    client, tid, _ = ui()
    at = open_thread(app_for(client), tid)
    at.button(key="start").click().run()
    at.text_area(key="answer_1").input("The tells.")
    at.button(key="send_answer").click().run()
    texts = [m.markdown[0].value for m in at.chat_message]
    assert texts == ["What made it fair?", "The tells.", "**Q2/5:** What made it fair?"]


def test_flagged_draft_highlights_sentences_and_revise_feedback(ui):
    client, tid, fake = ui(section_fn=bad_section)
    at = open_thread(app_for(client), tid)
    at.button(key="start").click().run()
    answer_interview(at)
    at.button(key="ol_approve").click().run()
    assert at.metric[0].value == "0%" and at.metric[1].value == "2"
    assert any(":red[Purple bananas" in m.value for m in at.markdown)
    at.text_input(key="dr_feedback_3").input("cite better")
    at.button(key="dr_revise").click().run()
    assert not at.exception, at.exception
    assert "2 revisions left" in [c.value for c in at.caption]
    assert any("cite better" in prompt for schema, prompt in fake.calls if schema is SectionText)


def test_saving_unchanged_draft_warns_instead_of_submitting(ui):
    client, tid, _ = ui()
    at = open_thread(app_for(client), tid)
    at.button(key="start").click().run()
    answer_interview(at)
    at.button(key="ol_approve").click().run()
    at.button(key="dr_edit").click().run()
    assert at.warning[0].value == "No section was changed."


def test_stalled_thread_offers_retry(ui):
    calls = {"n": 0}
    good = graph_responses()[Outline]

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("model is down")
        return good(prompt)

    client, tid, fake = ui()
    fake.responses[Outline] = flaky
    at = open_thread(app_for(client), tid)
    at.button(key="start").click().run()
    answer_interview(at)
    assert "RuntimeError: model is down" in at.error[0].value
    assert at.warning[0].value.startswith("The last step did not finish")
    at.button(key="retry").click().run()
    assert not at.exception, at.exception
    assert at.text_input(key="ol_title_3").value == "Fair Bosses"


class _DownHttp:
    def request(self, *args, **kwargs):
        raise httpx.ConnectError("connection refused")

    stream = request


def test_api_down_shows_an_error_not_a_crash():
    at = app_for(ApiClient(_DownHttp()))
    assert "API unavailable" in at.sidebar.error[0].value
    assert at.info[0].value.startswith("Pick an idea")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ui_app.py -v`
Expected: all 6 FAIL, because the script `src/vault_to_voice/ui/streamlit_app.py` does not exist yet.

- [ ] **Step 3: Implement**

`src/vault_to_voice/ui/streamlit_app.py`:
```python
"""Streamlit UI over the vault-to-voice API.

Run: `uv run streamlit run src/vault_to_voice/ui/streamlit_app.py` (with `v2v serve` running).
"""

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import streamlit as st

from vault_to_voice.ui import views
from vault_to_voice.ui.client import ApiClient, ApiError, Event


def get_client() -> ApiClient:
    # Tests put a client (over FastAPI's TestClient) into session_state before the first run.
    if "client" not in st.session_state:
        url = os.environ.get("V2V_API_URL", "http://127.0.0.1:8000")
        st.session_state["client"] = ApiClient.from_url(url)
    return st.session_state["client"]


def drive(events: Iterator[Event]) -> None:
    """Run one graph step, showing live node progress from the SSE stream, then redraw."""
    with st.status("Working...", expanded=True) as box:
        try:
            name, data = views.last_event(events, on_node=lambda n: box.write(f"finished `{n}`"))
        except (ApiError, httpx.HTTPError) as exc:
            name, data = "error", {"message": str(exc)}
        box.update(label="Step finished", state="error" if name == "error" else "complete")
    if name == "error":
        st.session_state["flash"] = data["message"]
    st.rerun()


def sidebar(client: ApiClient) -> None:
    with st.sidebar:
        st.header("vault-to-voice")
        try:
            ideas = client.ideas()
            threads = {t["thread_id"]: t["status"] for t in client.threads()}
        except (ApiError, httpx.HTTPError) as exc:
            st.error(f"API unavailable: {exc}")
            return
        st.subheader("Ideas")
        if not ideas:
            st.caption("No ideas yet. Run `v2v ingest`, `v2v style`, then `v2v ideas`.")
        for idea in ideas:
            status = threads.get(idea["idea_id"], "new")
            label = f"{idea['title']} · {status}"
            if st.button(label, key=f"idea_{idea['idea_id']}", width="stretch"):
                st.session_state["thread_id"] = idea["idea_id"]


def interview(client: ApiClient, view: dict, pending: dict) -> None:
    for qa in view["qa"]:
        st.chat_message("assistant").write(qa["question"])
        st.chat_message("user").write(qa["answer"])
    st.chat_message("assistant").write(
        f"**Q{pending['number']}/{pending['max']}:** {pending['question']}"
    )
    answer = st.text_area("Your answer", key=f"answer_{pending['number']}")
    if st.button("Send answer", key="send_answer"):
        drive(client.resume(view["thread_id"], answer))


def outline_review(client: ApiClient, view: dict, pending: dict) -> None:
    outline, rounds = pending["outline"], pending["rounds_left"]
    st.subheader("Outline review")
    st.caption(f"{rounds} revisions left")
    title = st.text_input("Title", outline["title"], key=f"ol_title_{rounds}")
    headings, points = [], []
    for i, section in enumerate(outline["sections"]):
        key = f"{rounds}_{i}"  # new keys each round, so a revised outline is not masked
        headings.append(st.text_input(f"Section {i + 1}", section["heading"], key=f"ol_h_{key}"))
        points.append(st.text_area("Point", section["point"], key=f"ol_p_{key}"))
    feedback = st.text_input("Feedback for a revision", key=f"ol_feedback_{rounds}")
    approve, revise, edit = st.columns(3)
    tid = view["thread_id"]
    if approve.button("Approve", key="ol_approve"):
        drive(client.resume(tid, {"action": "approve"}))
    if revise.button("Revise", key="ol_revise"):
        drive(client.resume(tid, {"action": "revise", "feedback": feedback}))
    if edit.button("Use my edits", key="ol_edit"):
        edited = views.edited_outline(outline, title, headings, points)
        drive(client.resume(tid, {"action": "edit", "edited": edited}))


def draft_review(client: ApiClient, view: dict, pending: dict) -> None:
    rounds, flags = pending["rounds_left"], pending["flags"]
    sections = (view.get("outline") or {}).get("sections", [])
    headings = {s["id"]: s["heading"] for s in sections}
    st.subheader("Draft review")
    coverage, flagged = st.columns(2)
    coverage.metric("Citation coverage", f"{pending['coverage']:.0%}")
    flagged.metric("Flagged sentences", len(flags))
    st.markdown(views.flagged_markdown(pending["draft"], flags, headings))
    tid = view["thread_id"]
    with st.expander("Edit sections"):
        original = views.section_texts(pending["draft"])
        edited = {
            sid: st.text_area(headings.get(sid, sid), text, key=f"dr_{rounds}_{sid}", height=200)
            for sid, text in original.items()
        }
        if st.button("Save edits", key="dr_edit"):
            changed = views.changed_sections(original, edited)
            if changed:
                drive(client.resume(tid, {"action": "edit", "edited": changed}))
            st.warning("No section was changed.")
    st.caption(f"{rounds} revisions left")
    feedback = st.text_input("Feedback for a revision", key=f"dr_feedback_{rounds}")
    approve, revise = st.columns(2)
    if approve.button("Approve and publish", key="dr_approve"):
        drive(client.resume(tid, {"action": "approve"}))
    if revise.button("Revise", key="dr_revise"):
        drive(client.resume(tid, {"action": "revise", "feedback": feedback}))


def published(client: ApiClient, view: dict) -> None:
    post = client.post(view["thread_id"])
    st.success(f"Published: {post['final_path']}")
    st.download_button(
        "Download Markdown",
        post["markdown"],
        file_name=Path(post["final_path"]).name,
        mime="text/markdown",
        key="download",
    )
    st.markdown(views.strip_frontmatter(post["markdown"]))
    sources = post["sources"]
    with st.expander("Sources"):
        st.metric("Citation coverage", f"{sources['citation_coverage']:.0%}")
        for paragraph in sources["paragraphs"]:
            st.markdown(f"> {paragraph['text']}")
            st.caption("sources: " + (", ".join(paragraph["sources"]) or "none"))


PENDING = {
    "interview_question": interview,
    "outline_review": outline_review,
    "draft_review": draft_review,
}


def main() -> None:
    st.set_page_config(page_title="vault-to-voice", layout="wide")
    client = get_client()
    sidebar(client)
    flash = st.session_state.pop("flash", None)
    if flash:
        st.error(flash)
    thread_id = st.session_state.get("thread_id")
    if thread_id is None:
        st.info("Pick an idea in the sidebar to start a post.")
        return
    try:
        view = client.thread(thread_id)
    except (ApiError, httpx.HTTPError) as exc:
        st.error(str(exc))
        return
    st.title(view["title"])
    tid = view["thread_id"]
    if view["status"] == "new":
        if st.button("Start post", key="start"):
            drive(client.start(tid))
    elif view["status"] == "stalled":
        st.warning("The last step did not finish (an error, or the server stopped).")
        if st.button("Retry", key="retry"):
            drive(client.retry(tid))
    elif view["status"] == "published":
        published(client, view)
    else:
        pending = view["pending"]
        if pending.get("error"):
            st.error(pending["error"])
        PENDING[pending["type"]](client, view, pending)


main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ui_app.py -v && uv run pytest`
Expected: 6 PASS (about 7 s; each test boots the script several times); full suite green.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add src/vault_to_voice/ui/streamlit_app.py tests/test_ui_app.py
git commit -m "feat: Streamlit UI over the API

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: End-to-end smoke and session close

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1: Full verification**

Run: `uv run pytest && uv run ruff check . && uv run ruff format --check .`
Expected: about 232 passed, 3 deselected; ruff clean.

- [ ] **Step 2: Manual end-to-end (needs the user's OpenAI key in `.env`; skip if none and record it under Known issues)**

```bash
uv run v2v ingest && uv run v2v style && uv run v2v ideas
uv run v2v serve                                              # terminal 1
uv run streamlit run src/vault_to_voice/ui/streamlit_app.py   # terminal 2
```
In the browser: pick an idea, Start post, answer 3 questions, edit one outline heading and use the edits, edit one draft section and save, approve. Check `outbox/<date>-<slug>.md` and `.sources.json`. Then start a second idea, answer one question, stop `v2v serve` with Ctrl+C, start it again, reload the page: the same question 2 is shown with answer 1 above it.
Optional tracing check: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...` in `.env`, restart `v2v serve` (`/health` shows `"tracing": true`), run one step and confirm a `post_graph` trace tagged `post-graph` with `idea_id` metadata appears in the `vault-to-voice` LangSmith project.

- [ ] **Step 3: Update PROGRESS.md**

Set `## Current` to S7 complete (M3: FastAPI + SSE service, Streamlit UI, opt-in LangSmith; test count), mark S7 `done` in the roadmap, set Next to S8 (M4 critic, guardrails, mini eval). Add decisions:
- `2026-10-10: M3: start/resume requests are the SSE streams (one request = one graph step); GET /threads/{id}/stream = retry a stalled thread; one run per thread (409 otherwise).`
- `2026-10-10: M3: ingest/style/ideas stay CLI-only; no sse-starlette (plain StreamingResponse); UI tested with AppTest over a real ApiClient(TestClient).`
Add known issues: `LangGraph warns "Deserializing unregistered type ..." when reading SQLite checkpoints (future versions will block it; register allowed msgpack modules before upgrading LangGraph)`; `the manifest SQLite connection is shared across server threads without its own lock (fine for one user; revisit with Postgres in M5)`; and the result (or skip) of the manual smoke.

- [ ] **Step 4: Commit**

```bash
git add PROGRESS.md
git commit -m "docs: update progress after M3

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
