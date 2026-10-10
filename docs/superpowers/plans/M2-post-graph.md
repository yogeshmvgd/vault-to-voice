# M2 Post Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the LangGraph post graph: for one idea, load its notes, interview the author (interrupts), plan an outline (human review), draft sections in parallel (`Send`), check every claim's citations, auto-revise once, let the author review the draft, and publish a clean Markdown post plus a `.sources.json` sidecar. All runs are checkpointed in SQLite (one thread per idea) and resumable; a `v2v write <idea_id>` CLI drives it.

**Architecture:** One `StateGraph(PostState)`. Nodes are plain functions `node(state, deps)` bound with `functools.partial`, so each is unit-testable without a graph. All collaborators (LLM, manifest, FAISS, embedder, settings, Store, publisher) live in a `GraphDeps` dataclass. Nodes that call `interrupt()` are separate from nodes that call the LLM, so resuming never repeats an LLM call. The interview is a compiled subgraph. Long-term memory (style profile, past interview answers) lives in the LangGraph Store; short-term state lives in checkpoints.

**Tech Stack:** LangGraph 1.2 (`StateGraph`, subgraphs, `Send`, `interrupt`/`Command`, `RetryPolicy`), langgraph-checkpoint-sqlite 3.1 (`SqliteSaver`, `SqliteStore`), pydantic v2, existing M1 modules, Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-10-10-vault-to-voice-design.md` (sections [4] Post graph, Locked decisions: Checkpointer, Long-term memory, Citations, Approvals, Interview, Sessions, Output).

**Prerequisite:** M1.5 (`docs/superpowers/plans/M1.5-obsidian-format.md`) is complete — notes carry `work_id`/`links`, ideas carry `kind`/`work_ids`. Test fixtures here use the flat format (still supported via `VAULT_FORMAT=flat`); the sample-vault integration test uses the Obsidian format.

**Sessions:** Tasks 1–6 = S5 (M2a: state, context, interview, outline, checkpointer, Store) · Tasks 7–12 = S6 (M2b: Send fan-out, citations, auto-revise, draft review, publish, CLI). End each session per `CLAUDE.md` protocol. Work on branch `m2-post-graph`.

**Deliberate deviations from spec (recorded, not silent):**
- The `critic` node is M4 (spec milestone 4). In M2, `route_after_checks` routes on citation flags only; M4 adds the critic beside it.
- Draft `edit` sends the edited text back through `check_citations` and shows the draft again (rather than publishing immediately), so coverage numbers stay true. Outline `edit` proceeds straight to drafting.
- The `v2v write` CLI offers approve / revise only; `edit` is exercised by tests and arrives in the UI in M3.

**Verified API facts (probed against the installed versions before writing this plan):**
- `interrupt(value)` inside a node pauses the run; `graph.invoke(...)` returns a dict with `"__interrupt__": [Interrupt(value=...)]`; resume with `graph.invoke(Command(resume=x), config)`. On resume the *whole node* re-runs from its start — hence LLM calls never live in a node that also interrupts.
- An interrupt raised inside a compiled subgraph node shows up in the parent's `graph.get_state(config).interrupts`.
- A conditional edge may return a list of `Send(node, payload)`; the payload is passed to the node as-is. `Send` objects expose `.node` and `.arg`.
- `functools.partial(fn, deps=deps)` works as a node.
- `SqliteSaver(sqlite3.connect(path, check_same_thread=False))` persists checkpoints across processes. `SqliteStore` needs `sqlite3.connect(path, check_same_thread=False, isolation_level=None)` and `.setup()`; without autocommit it fails with "cannot start a transaction within a transaction".
- Pydantic models stored in state round-trip through `SqliteSaver`.
- `RetryPolicy(max_attempts=3)` is passed as `add_node(name, fn, retry_policy=...)`.

## Global Constraints

- Python `>=3.11` (CI runs 3.11). ruff line length 100; `uv run ruff check . && uv run ruff format --check .` clean before every commit.
- Tests run offline with no API keys: `FakeLLMClient` + `HashEmbedder` + `InMemorySaver`/`InMemoryStore` (or SQLite files under `tmp_path`). Real-model tests are `@pytest.mark.integration`.
- Every LLM call: `deps.llm.structured(prompt, Schema, system=DATA_INSTRUCTION)`; note and answer text inside prompts goes through `scrub_pii` + `wrap_note` (via `graph/prompts.py`).
- `thread_id = idea_id`. Checkpoints in `data/checkpoints.db`, Store in `data/store.db`.
- Interview: min 3 / max 5 questions, one per interrupt. Review rounds (revise) cap 3 for outline and for draft, counted separately.
- Citation thresholds: cosine ≥ 0.55 supported, ≤ 0.30 unsupported, between → batched LLM judge. `connective` sentences exempt; sentences sourced `user` count as supported claims.
- Auto-revise at most once per run (`auto_revised` flag).
- Output: `outbox/<YYYY-MM-DD>-<slug>.md` (frontmatter + clean Markdown, no citation markers) + `<same>.sources.json`. Never overwrite an existing post (suffix `-2`, `-3`, …).
- Never write GitHub Actions workflows, Dockerfiles, or compose files.
- One commit per task; messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. Process killed mid-run (between interrupts) → re-running `v2v write <idea_id>` resumes the same thread at the same pending question/review, without re-asking the LLM for questions already asked. Pinned in Task 6 (graph) and Task 12 (CLI).
2. Malformed resume payloads (unknown action, bare string, `edit` without/with invalid `edited`, unknown section ids, non-string text) → the same review is shown again with an `error`, never a crash. Pinned in Tasks 5 and 9.
3. LLM cites ids not in context (hallucinated note ids, `qa:9`, empty sources) or plans with unknown ids → flagged / filtered, never a crash or a `KeyError`. Pinned in Tasks 5 and 8.
4. Idea whose notes were deleted from the vault after `v2v ideas` → missing notes skipped with a warning; if none remain, a clear `ValueError` telling the user to re-run ingest/ideas. Pinned in Task 3.
5. Blank interview answers → recorded as `(skipped)`; interview still bounded by min/max. Pinned in Task 4. Duplicate post slug on the same day → `-2` suffix, nothing overwritten. Pinned in Task 10.

## File Structure

```
src/vault_to_voice/
  config.py                 + graph settings (interview/review/citation/related) and checkpoint/store paths
  index/manifest.py         + check_same_thread=False, get_idea, set_idea_status
  publish.py                RenderedPost, Publisher protocol, OutboxPublisher
  cli.py                    + style syncs Store; new `write` command
  graph/
    __init__.py
    state.py                pydantic models (NoteContext, QA, Question, CoverageCheck, Section, Outline,
                            Sentence, Paragraph, SectionText, SectionDraft, Flag, CitationReport,
                            ReviewDecision), merge_drafts reducer, PostState, DraftTask, valid_sources
    deps.py                 GraphDeps dataclass, LLM_RETRY
    memory.py               Store helpers: put_style/get_style, save_answers/past_answers
    checkpoint.py           make_checkpointer, make_store (SQLite)
    prompts.py              notes_block, qa_block, style_brief
    render.py               render_markdown
    runner.py               thread_config, start_input, run_until_interrupt, pending_interrupt
    builder.py              build_post_graph
    nodes/
      __init__.py
      context.py            load_context
      interview.py          ask_question, wait_for_answer, check_coverage, build_interview_graph
      outline.py            make_outline, outline_review, route_after_outline_review
      draft.py              fan_out_sections, draft_section, merge_draft
      citations.py          JudgeVerdict(s), check_citations
      review.py             route_after_checks, prepare_auto_revise, draft_review, route_after_draft_review
      publish.py            publish
tests/
  fakes.py                  + make_deps, NOTES, seed_idea, GOOD_SENTENCE, good_section, graph_responses
  test_graph_state.py  test_graph_memory.py  test_graph_prompts.py  test_node_context.py
  test_node_interview.py  test_node_outline.py  test_graph_builder.py  test_node_draft.py
  test_node_citations.py  test_node_review.py  test_publish.py  test_graph_e2e.py  (+ test_cli.py, test_manifest.py)
```

---

### Task 1: Dependencies, settings, state models, manifest additions

**Files:**
- Modify: `pyproject.toml`/`uv.lock` (via `uv add`), `src/vault_to_voice/config.py`, `src/vault_to_voice/index/manifest.py`, `tests/test_manifest.py`
- Create: `src/vault_to_voice/graph/__init__.py` (empty), `src/vault_to_voice/graph/nodes/__init__.py` (empty), `src/vault_to_voice/graph/state.py`
- Test: `tests/test_graph_state.py`

**Interfaces:**
- Consumes: `Idea` (`vault_to_voice.ideas.detector`).
- Produces:
  - `Settings` new fields: `interview_min_questions=3, interview_max_questions=5, max_review_rounds=3, citation_support=0.55, citation_reject=0.30, related_notes=2, related_min_score=0.5`; properties `checkpoint_path` (`data_dir/"checkpoints.db"`), `store_path` (`data_dir/"store.db"`).
  - `Manifest.get_idea(idea_id) -> Idea | None`, `Manifest.set_idea_status(idea_id, status) -> None` (raises `KeyError` if unknown). Manifest connection usable from other threads.
  - `graph/state.py`: `SourceChunk(chunk_id:int, text:str)`; `NoteContext(note_id, title, type, date: str|None, highlights: list[str]=[], chunks: list[SourceChunk], related: bool=False, work_id: str|None=None)` + property `text`; `Question(question, grounding_note_ids=[])`; `QA(qa_id, question, answer, grounding_note_ids=[])`; `CoverageCheck(enough: bool, missing: str="")`; `Section(id, heading, point, planned_sources=[])`; `Outline(title, sections: list[Section] (min 1))`; `Sentence(text, sources=[], kind: "claim"|"connective"="claim")`; `Paragraph(sentences (min 1))`; `SectionText(paragraphs: list[Paragraph] (min 1))`; `SectionDraft(section_id, paragraphs)`; `Flag(section_id, paragraph:int, sentence:int, text, reason: "missing_source"|"invalid_source"|"unsupported", score: float|None=None)`; `CitationReport(n_claims, n_supported, flags=[])` + property `coverage`; `ReviewDecision(action: "approve"|"revise"|"edit", feedback="", edited: dict|None=None)`; `merge_drafts(left, right) -> dict`; `PostState` (TypedDict, total=False); `DraftTask` (TypedDict); `valid_sources(notes, qa) -> set[str]`.

- [ ] **Step 1: Add dependencies**

Run: `uv add "langgraph>=1.2" "langgraph-checkpoint-sqlite>=3.1"`
Expected: `pyproject.toml` lists both; `uv.lock` updated (CI uses `uv sync --locked`, so both files must be committed together).

- [ ] **Step 2: Write failing tests**

`tests/test_graph_state.py`:
```python
import pytest
from pydantic import ValidationError

from vault_to_voice.config import Settings
from vault_to_voice.graph.state import (
    QA,
    CitationReport,
    NoteContext,
    Outline,
    ReviewDecision,
    Sentence,
    SourceChunk,
    merge_drafts,
    valid_sources,
)


def test_settings_graph_defaults(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert (s.interview_min_questions, s.interview_max_questions, s.max_review_rounds) == (3, 5, 3)
    assert (s.citation_support, s.citation_reject) == (0.55, 0.30)
    assert s.checkpoint_path == tmp_path / "checkpoints.db"
    assert s.store_path == tmp_path / "store.db"


def test_sentence_defaults_to_claim():
    assert Sentence(text="x").kind == "claim"


def test_outline_needs_a_section():
    with pytest.raises(ValidationError):
        Outline(title="t", sections=[])


def test_review_decision_rejects_unknown_action():
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate({"action": "publish"})


def test_merge_drafts_overrides_by_key():
    assert merge_drafts({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}
    assert merge_drafts(None, {"a": 1}) == {"a": 1}


def test_coverage():
    assert CitationReport(n_claims=4, n_supported=3).coverage == 0.75
    assert CitationReport(n_claims=0, n_supported=0).coverage == 1.0


def test_note_context_text_and_valid_sources():
    chunks = [SourceChunk(chunk_id=1, text="a"), SourceChunk(chunk_id=2, text="b")]
    note = NoteContext(note_id="n0", title="t", type="game", date=None, chunks=chunks)
    assert note.text == "a\n\nb"
    qa = [QA(qa_id="qa:1", question="q", answer="x")]
    assert valid_sources([note], qa) == {"n0", "qa:1", "user"}
```

Append to `tests/test_manifest.py`:
```python
def test_idea_status_and_get():
    from vault_to_voice.ideas.detector import Idea

    m = Manifest(":memory:")
    idea = Idea(idea_id="idea_x", title="t", angle="a", theme="th", note_ids=["a"], opinion_score=0.5)
    m.save_ideas([idea])
    m.set_idea_status("idea_x", "done")
    assert m.get_idea("idea_x").status == "done"
    assert m.get_idea("nope") is None


def test_set_status_unknown_idea_raises():
    import pytest

    with pytest.raises(KeyError):
        Manifest(":memory:").set_idea_status("nope", "done")


def test_manifest_usable_from_other_thread(tmp_path):
    import threading

    m = Manifest(tmp_path / "m.db")
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("n", m.chunk_count()))
    t.start()
    t.join()
    assert out["n"] == 0
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_graph_state.py tests/test_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError: vault_to_voice.graph`; manifest tests fail with `AttributeError` / `ProgrammingError` (thread).

- [ ] **Step 4: Implement**

`src/vault_to_voice/config.py` — add inside `Settings` after `seed: int = 42`:
```python
    interview_min_questions: int = 3
    interview_max_questions: int = 5
    max_review_rounds: int = 3
    citation_support: float = 0.55
    citation_reject: float = 0.30
    related_notes: int = 2
    related_min_score: float = 0.5
```
and after the `style_profile_path` property:
```python
    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.db"

    @property
    def store_path(self) -> Path:
        return self.data_dir / "store.db"
```

`src/vault_to_voice/index/manifest.py` — in `__init__` change the connect line to
```python
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
```
(LangGraph runs nodes in worker threads.) `get_idea` / `set_idea_status` below were already added in M1.5 Task 4 — keep them, only add their tests. For reference:
```python
    def get_idea(self, idea_id: str):
        from vault_to_voice.ideas.detector import Idea

        r = self.conn.execute("SELECT data FROM ideas WHERE idea_id = ?", (idea_id,)).fetchone()
        return Idea.model_validate_json(r[0]) if r else None

    def set_idea_status(self, idea_id: str, status: str) -> None:
        r = self.conn.execute("SELECT data FROM ideas WHERE idea_id = ?", (idea_id,)).fetchone()
        if r is None:
            raise KeyError(idea_id)
        data = json.loads(r[0])
        data["status"] = status
        self.conn.execute(
            "UPDATE ideas SET data = ?, status = ? WHERE idea_id = ?",
            (json.dumps(data), status, idea_id),
        )
        self.conn.commit()
```

`src/vault_to_voice/graph/state.py`:
```python
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from vault_to_voice.ideas.detector import Idea


class SourceChunk(BaseModel):
    chunk_id: int
    text: str


class NoteContext(BaseModel):
    note_id: str
    title: str
    type: str
    date: str | None
    highlights: list[str] = Field(default_factory=list)
    chunks: list[SourceChunk]
    related: bool = False  # pulled in by FAISS search, not part of the idea's cluster
    work_id: str | None = None  # Obsidian format: the work this note is about

    @property
    def text(self) -> str:
        return "\n\n".join(c.text for c in self.chunks)


class Question(BaseModel):
    question: str
    grounding_note_ids: list[str] = Field(default_factory=list)


class QA(BaseModel):
    qa_id: str  # "qa:1", "qa:2", ... — citable like a note id
    question: str
    answer: str
    grounding_note_ids: list[str] = Field(default_factory=list)


class CoverageCheck(BaseModel):
    enough: bool
    missing: str = ""


class Section(BaseModel):
    id: str
    heading: str
    point: str
    planned_sources: list[str] = Field(default_factory=list)


class Outline(BaseModel):
    title: str
    sections: list[Section] = Field(min_length=1)


class Sentence(BaseModel):
    text: str
    sources: list[str] = Field(default_factory=list)
    kind: Literal["claim", "connective"] = "claim"


class Paragraph(BaseModel):
    sentences: list[Sentence] = Field(min_length=1)


class SectionText(BaseModel):
    """What the LLM returns for one section."""

    paragraphs: list[Paragraph] = Field(min_length=1)


class SectionDraft(BaseModel):
    section_id: str
    paragraphs: list[Paragraph]


class Flag(BaseModel):
    section_id: str
    paragraph: int
    sentence: int
    text: str
    reason: Literal["missing_source", "invalid_source", "unsupported"]
    score: float | None = None


class CitationReport(BaseModel):
    n_claims: int
    n_supported: int
    flags: list[Flag] = Field(default_factory=list)

    @property
    def coverage(self) -> float:
        return round(self.n_supported / self.n_claims, 3) if self.n_claims else 1.0


class ReviewDecision(BaseModel):
    action: Literal["approve", "revise", "edit"]
    feedback: str = ""
    edited: dict | None = None  # outline: Outline as dict; draft: {section_id: text}


def merge_drafts(left: dict | None, right: dict | None) -> dict:
    """Reducer for parallel draft_section writes: later writes replace the same section id."""
    return {**(left or {}), **(right or {})}


class PostState(TypedDict, total=False):
    idea: Idea
    notes: list[NoteContext]
    style: dict | None
    past_answers: list[QA]
    qa: list[QA]
    pending_question: Question | None
    interview_done: bool
    outline: Outline
    last_action: str
    review_error: str | None
    feedback: str
    section_feedback: dict[str, str]
    revise_ids: list[str]
    drafts: Annotated[dict[str, SectionDraft], merge_drafts]
    draft: list[SectionDraft]
    citation_report: CitationReport
    auto_revised: bool
    user_rounds: dict[str, int]
    final_path: str | None


class DraftTask(TypedDict):
    """Payload sent to one parallel draft_section run."""

    section: Section
    outline_title: str
    notes: list[NoteContext]
    qa: list[QA]
    style: dict | None
    feedback: str


def valid_sources(notes: list[NoteContext], qa: list[QA]) -> set[str]:
    return {n.note_id for n in notes} | {q.qa_id for q in qa} | {"user"}
```

Create empty `src/vault_to_voice/graph/__init__.py` and `src/vault_to_voice/graph/nodes/__init__.py`.

- [ ] **Step 5: Run tests**

Run: `uv run pytest -v` → all PASS. `uv run ruff check . && uv run ruff format --check .` → clean.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/vault_to_voice/config.py src/vault_to_voice/index/manifest.py src/vault_to_voice/graph tests/test_graph_state.py tests/test_manifest.py
git commit -m "feat: graph state models, settings and manifest idea helpers"
```

---

### Task 2: Long-term memory (Store) and SQLite checkpointer factories

**Files:**
- Create: `src/vault_to_voice/graph/memory.py`, `src/vault_to_voice/graph/checkpoint.py`
- Modify: `src/vault_to_voice/cli.py` (`style` command also writes the profile to the Store), `tests/test_cli.py`
- Test: `tests/test_graph_memory.py`

**Interfaces:**
- Consumes: `StyleProfile` (style/profile.py), `QA` (Task 1), `Settings.store_path`/`checkpoint_path` (Task 1).
- Produces: `STYLE_NS = ("style",)`, `STYLE_KEY = "profile"`, `ANSWERS_NS = ("answers",)`; `put_style(store, profile: StyleProfile) -> None`; `get_style(store) -> dict | None`; `save_answers(store, idea_id: str, qa: list[QA]) -> None`; `past_answers(store, *, exclude_idea: str | None = None, limit: int = 10) -> list[QA]` (newest first); `make_checkpointer(path: Path) -> SqliteSaver`; `make_store(path: Path) -> SqliteStore`.

- [ ] **Step 1: Write failing tests**

`tests/test_graph_memory.py`:
```python
import datetime as dt

from langgraph.store.memory import InMemoryStore

from vault_to_voice.graph.checkpoint import make_checkpointer, make_store
from vault_to_voice.graph.memory import get_style, past_answers, put_style, save_answers
from vault_to_voice.graph.state import QA
from vault_to_voice.style.profile import StyleProfile, ToneSummary
from vault_to_voice.style.stats import compute_stats


def profile() -> StyleProfile:
    return StyleProfile(
        version=2,
        created_at=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        stats=compute_stats(["I loved it."]),
        tone=ToneSummary(tone=["wry"], voice_notes="Punchy."),
    )


def test_style_roundtrip():
    store = InMemoryStore()
    assert get_style(store) is None
    put_style(store, profile())
    style = get_style(store)
    assert style["version"] == 2 and style["tone"]["tone"] == ["wry"]


def test_past_answers_excludes_current_idea_and_limits():
    store = InMemoryStore()
    save_answers(store, "idea_a", [QA(qa_id="qa:1", question="q", answer="from a")])
    save_answers(
        store, "idea_b", [QA(qa_id=f"qa:{i}", question="q", answer=f"b{i}") for i in range(1, 4)]
    )
    assert [q.answer for q in past_answers(store, exclude_idea="idea_b")] == ["from a"]
    assert len(past_answers(store, limit=2)) == 2


def test_sqlite_store_persists_across_connections(tmp_path):
    path = tmp_path / "d" / "store.db"
    put_style(make_store(path), profile())
    assert get_style(make_store(path))["version"] == 2


def test_checkpointer_creates_parent_dir(tmp_path):
    make_checkpointer(tmp_path / "x" / "cp.db")
    assert (tmp_path / "x").is_dir()
```

Append to `tests/test_cli.py`:
```python
def test_style_also_saved_to_store(env, vault_dir, write_note):
    from vault_to_voice.config import Settings
    from vault_to_voice.graph.checkpoint import make_store
    from vault_to_voice.graph.memory import get_style

    write_note("n.md", note_text("n", "boss fight fair"))
    assert runner.invoke(cli.app, ["style", "--vault", str(vault_dir)]).exit_code == 0
    assert get_style(make_store(Settings().store_path))["version"] == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_graph_memory.py tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: vault_to_voice.graph.checkpoint`; the CLI test fails on import.

- [ ] **Step 3: Implement**

`src/vault_to_voice/graph/memory.py`:
```python
from langgraph.store.base import BaseStore

from vault_to_voice.graph.state import QA
from vault_to_voice.style.profile import StyleProfile

STYLE_NS = ("style",)
STYLE_KEY = "profile"
ANSWERS_NS = ("answers",)


def put_style(store: BaseStore, profile: StyleProfile) -> None:
    store.put(STYLE_NS, STYLE_KEY, profile.model_dump(mode="json"))


def get_style(store: BaseStore) -> dict | None:
    item = store.get(STYLE_NS, STYLE_KEY)
    return item.value if item else None


def save_answers(store: BaseStore, idea_id: str, qa: list[QA]) -> None:
    store.put(ANSWERS_NS, idea_id, {"qa": [q.model_dump() for q in qa]})


def past_answers(
    store: BaseStore, *, exclude_idea: str | None = None, limit: int = 10
) -> list[QA]:
    items = sorted(store.search(ANSWERS_NS, limit=100), key=lambda i: i.updated_at, reverse=True)
    answers = [
        QA.model_validate(q) for item in items if item.key != exclude_idea for q in item.value["qa"]
    ]
    return answers[:limit]
```

`src/vault_to_voice/graph/checkpoint.py`:
```python
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore


def make_checkpointer(path: Path) -> SqliteSaver:
    """Short-term memory: one checkpoint history per thread (thread_id = idea_id)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def make_store(path: Path) -> SqliteStore:
    """Long-term memory shared by all threads (style profile, past interview answers)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # SqliteStore manages its own transactions, so the connection must be in autocommit mode.
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn)
    store.setup()
    return store
```

`src/vault_to_voice/cli.py` — add imports `from vault_to_voice.graph.checkpoint import make_store` and `from vault_to_voice.graph.memory import put_style`; in `style`, right after `profile = save_profile(...)`:
```python
    put_style(make_store(s.store_path), profile)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v` → all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/vault_to_voice/graph/memory.py src/vault_to_voice/graph/checkpoint.py src/vault_to_voice/cli.py tests/test_graph_memory.py tests/test_cli.py
git commit -m "feat: LangGraph Store memory and SQLite checkpointer factories"
```

---

### Task 3: GraphDeps, prompt helpers, test fixtures, `load_context`

**Files:**
- Create: `src/vault_to_voice/graph/deps.py`, `src/vault_to_voice/graph/prompts.py`, `src/vault_to_voice/graph/nodes/context.py`
- Modify: `tests/fakes.py`
- Test: `tests/test_graph_prompts.py`, `tests/test_node_context.py`

**Interfaces:**
- Consumes: `Manifest`, `FaissStore`, `Embedder`, `LLMClient`, `Settings`, `ingest`, `FilesystemVaultReader`, `Idea`, `make_idea_id`, guardrails; Task 1 models; Task 2 memory helpers.
- Produces:
  - `GraphDeps(llm, manifest, index: FaissStore, embedder, settings, store: BaseStore, today: Callable[[], date] = date.today)` (dataclass); `LLM_RETRY = RetryPolicy(max_attempts=3)`.
  - `notes_block(notes: list[NoteContext]) -> str`, `qa_block(qa: list[QA]) -> str`, `style_brief(style: dict | None) -> str`.
  - `load_context(state, deps) -> dict` with keys `notes, style, past_answers, qa, user_rounds, auto_revised`; idea notes sorted by `(date or "", note_id)`, related notes appended after; raises `ValueError` mentioning `v2v ingest` when no idea note is indexed.
  - tests/fakes.py: `make_deps(tmp_path, llm, *, dim=256, **settings_overrides) -> GraphDeps`; `NOTES: dict[str, str]`; `seed_idea(deps, vault_dir, note_ids=("n0","n1","n2","n3")) -> Idea` (writes all `NOTES`, ingests them, saves the idea).

- [ ] **Step 1: Write failing tests**

Append to `tests/fakes.py` (keep the existing `FakeLLMClient`):
```python
from pathlib import Path

from langgraph.store.memory import InMemoryStore

from vault_to_voice.config import Settings
from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.ideas.detector import Idea, make_idea_id
from vault_to_voice.index.embedder import HashEmbedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.ingest import ingest
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.vault.reader import FilesystemVaultReader

NOTES = {
    "n0": "Boss fights feel fair when every attack has a clear tell.\n\n"
    "I died twenty times and learned the pattern.",
    "n1": "Fair difficulty means every death teaches you something.\n\n"
    "The tide mechanic shows the danger before it arrives.",
    "n2": "Cheap bosses spam attacks with no wind up.\n\n"
    "I quit the game when the camera lost lock on.",
    "n3": "Hard games respect the player when the rules are honest.\n\n"
    "Patience beats button mashing in the late game.",
    "x0": "Baking bread on a rainy afternoon is calming.\n\nThe oven timer never rushes me.",
}


def make_deps(tmp_path: Path, llm, *, dim: int = 256, **settings_overrides) -> GraphDeps:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        outbox_dir=tmp_path / "outbox",
        embedding_provider="hash",
        **settings_overrides,
    )
    return GraphDeps(
        llm=llm,
        manifest=Manifest(":memory:"),
        index=FaissStore(tmp_path / "f.index", dim),
        embedder=HashEmbedder(dim),
        settings=settings,
        store=InMemoryStore(),
    )


def seed_idea(deps: GraphDeps, vault_dir: Path, note_ids=("n0", "n1", "n2", "n3")) -> Idea:
    for nid, body in NOTES.items():
        (vault_dir / f"{nid}.md").write_text(
            f"---\nid: {nid}\ndate: 2026-01-01\ntype: game\ntitle: Note {nid}\n---\n{body}\n",
            encoding="utf-8",
        )
    ingest(FilesystemVaultReader(vault_dir), deps.manifest, deps.index, deps.embedder)
    ids = sorted(note_ids)
    idea = Idea(
        idea_id=make_idea_id(ids),
        title="Fair difficulty",
        angle="Hard is fine if it is honest.",
        theme="difficulty",
        note_ids=ids,
        opinion_score=0.8,
    )
    deps.manifest.save_ideas([idea])
    return idea
```

`tests/test_graph_prompts.py`:
```python
from vault_to_voice.graph.prompts import notes_block, qa_block, style_brief
from vault_to_voice.graph.state import QA, NoteContext, SourceChunk


def test_notes_block_wraps_and_scrubs():
    note = NoteContext(
        note_id="n0",
        title="T",
        type="game",
        date="2026-01-01",
        chunks=[SourceChunk(chunk_id=1, text="mail a@b.com")],
    )
    out = notes_block([note])
    assert out.startswith('<note id="n0">') and "[EMAIL]" in out and "a@b.com" not in out


def test_qa_block_uses_qa_ids():
    out = qa_block([QA(qa_id="qa:1", question="Why?", answer="Because.")])
    assert '<note id="qa:1">' in out and "Because." in out


def test_style_brief_without_and_with_profile():
    assert "no style profile" in style_brief(None).lower()
    brief = style_brief(
        {
            "tone": {"tone": ["wry"], "voice_notes": "Punchy.", "exemplars": ["The fight is fair."]},
            "stats": {"avg_sentence_len": 11.2, "first_person_rate": 0.4,
                      "top_phrases": ["the thing is"]},
        }
    )
    assert "wry" in brief and "The fight is fair." in brief and "the thing is" in brief
```

`tests/test_node_context.py`:
```python
import pytest

from tests.fakes import FakeLLMClient, make_deps, seed_idea
from vault_to_voice.graph.memory import STYLE_KEY, STYLE_NS, save_answers
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.state import QA


def core_ids(out):
    return [n.note_id for n in out["notes"] if not n.related]


def test_loads_idea_notes_with_chunks(tmp_path, vault_dir):
    deps = make_deps(tmp_path, FakeLLMClient({}))
    idea = seed_idea(deps, vault_dir)
    out = load_context({"idea": idea}, deps)
    assert core_ids(out) == ["n0", "n1", "n2", "n3"]
    first = out["notes"][0]
    assert len(first.chunks) == 2 and "clear tell" in first.chunks[0].text
    assert out["qa"] == [] and out["user_rounds"] == {"outline": 0, "draft": 0}
    assert out["auto_revised"] is False


def test_related_notes_found_via_faiss(tmp_path, vault_dir):
    deps = make_deps(tmp_path, FakeLLMClient({}), related_min_score=-1.0)
    idea = seed_idea(deps, vault_dir, note_ids=("n0", "n1", "n2"))
    related = [n.note_id for n in load_context({"idea": idea}, deps)["notes"] if n.related]
    assert 1 <= len(related) <= 2 and set(related) <= {"n3", "x0"}


def test_missing_notes_skipped(tmp_path, vault_dir):
    deps = make_deps(tmp_path, FakeLLMClient({}))
    idea = seed_idea(deps, vault_dir)
    deps.manifest.delete_note("n2")
    assert core_ids(load_context({"idea": idea}, deps)) == ["n0", "n1", "n3"]


def test_no_notes_left_raises(tmp_path, vault_dir):
    deps = make_deps(tmp_path, FakeLLMClient({}))
    idea = seed_idea(deps, vault_dir)
    for nid in idea.note_ids:
        deps.manifest.delete_note(nid)
    with pytest.raises(ValueError, match="v2v ingest"):
        load_context({"idea": idea}, deps)


def test_notes_sorted_by_date_pages_first(tmp_path):
    from vault_to_voice.ideas.detector import Idea
    from vault_to_voice.vault.models import Note

    deps = make_deps(tmp_path, FakeLLMClient({}))
    for nid, date in [("w@2026-03-01", "2026-03-01"), ("w", None), ("w@2026-01-05", "2026-01-05")]:
        deps.manifest.upsert_note(Note(id=nid, title="W", type="game", body="b", path="x.md",
                                       hash="h", work_id="w",
                                       date=None if date is None else date))
        deps.manifest.add_chunks(nid, ["p"])
    idea = Idea(idea_id="idea_work_w", title="t", angle="a", theme="x",
                note_ids=["w", "w@2026-01-05", "w@2026-03-01"], opinion_score=0.5, kind="work")
    out = load_context({"idea": idea}, deps)
    assert core_ids(out) == ["w", "w@2026-01-05", "w@2026-03-01"]
    assert out["notes"][0].work_id == "w"


def test_style_and_past_answers_from_store(tmp_path, vault_dir):
    deps = make_deps(tmp_path, FakeLLMClient({}))
    idea = seed_idea(deps, vault_dir)
    deps.store.put(STYLE_NS, STYLE_KEY, {"version": 1})
    save_answers(deps.store, "idea_other", [QA(qa_id="qa:1", question="q", answer="old")])
    save_answers(deps.store, idea.idea_id, [QA(qa_id="qa:1", question="q", answer="mine")])
    out = load_context({"idea": idea}, deps)
    assert out["style"] == {"version": 1}
    assert [q.answer for q in out["past_answers"]] == ["old"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_graph_prompts.py tests/test_node_context.py -v`
Expected: FAIL — `ModuleNotFoundError: vault_to_voice.graph.deps` (collection error in `tests/fakes.py`).

- [ ] **Step 3: Implement**

`src/vault_to_voice/graph/deps.py`:
```python
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field

from langgraph.store.base import BaseStore
from langgraph.types import RetryPolicy

from vault_to_voice.config import Settings
from vault_to_voice.index.embedder import Embedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.llm import LLMClient

# Retry LLM-calling nodes on transient errors (exponential backoff, LangGraph defaults).
LLM_RETRY = RetryPolicy(max_attempts=3)


@dataclass
class GraphDeps:
    """Everything the post graph's nodes need, injected once at build time."""

    llm: LLMClient
    manifest: Manifest
    index: FaissStore
    embedder: Embedder
    settings: Settings
    store: BaseStore
    today: Callable[[], dt.date] = field(default=dt.date.today)
```

`src/vault_to_voice/graph/prompts.py`:
```python
from vault_to_voice.graph.state import QA, NoteContext
from vault_to_voice.guardrails import scrub_pii, wrap_note


def notes_block(notes: list[NoteContext]) -> str:
    return "\n".join(
        wrap_note(n.note_id, scrub_pii(f"{n.title} ({n.type}, {n.date or 'undated'})\n{n.text}"))
        for n in notes
    )


def qa_block(qa: list[QA]) -> str:
    return "\n".join(
        wrap_note(q.qa_id, scrub_pii(f"Q: {q.question}\nA: {q.answer}")) for q in qa
    )


def style_brief(style: dict | None) -> str:
    if not style:
        return "No style profile yet: write in a natural, first-person blog voice."
    tone = style.get("tone", {})
    stats = style.get("stats", {})
    lines = [f"Tone: {', '.join(tone.get('tone', []))}", f"Voice: {tone.get('voice_notes', '')}"]
    if stats:
        lines.append(
            f"Average sentence length: {stats.get('avg_sentence_len')} words; "
            f"first-person rate: {stats.get('first_person_rate')}"
        )
    if stats.get("top_phrases"):
        lines.append("Recurring phrases: " + ", ".join(stats["top_phrases"][:8]))
    if tone.get("exemplars"):
        lines.append("Example sentences:\n" + "\n".join(f"- {e}" for e in tone["exemplars"]))
    return "\n".join(lines)
```

`src/vault_to_voice/graph/nodes/context.py`:
```python
import numpy as np
import structlog

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.memory import get_style, past_answers
from vault_to_voice.graph.state import NoteContext, PostState, SourceChunk

log = structlog.get_logger()


def load_context(state: PostState, deps: GraphDeps) -> dict:
    idea = state["idea"]
    notes = _note_contexts(deps, idea.note_ids, related=False)
    if not notes:
        raise ValueError(
            f"None of the notes for {idea.idea_id} are indexed any more. "
            "Run `v2v ingest` and `v2v ideas` again."
        )
    # Chronological order (undated work pages first) so drafts can follow time.
    notes.sort(key=lambda n: (n.date or "", n.note_id))
    related = _related_note_ids(deps, [n.note_id for n in notes])
    notes += _note_contexts(deps, related, related=True)
    return {
        "notes": notes,
        "style": get_style(deps.store),
        "past_answers": past_answers(deps.store, exclude_idea=idea.idea_id),
        "qa": [],
        "user_rounds": {"outline": 0, "draft": 0},
        "auto_revised": False,
    }


def _note_contexts(deps: GraphDeps, note_ids: list[str], *, related: bool) -> list[NoteContext]:
    out = []
    for nid in note_ids:
        row = deps.manifest.get_note(nid)
        chunk_ids = deps.manifest.chunk_ids(nid)
        if row is None or not chunk_ids:
            log.warning("context_note_missing", note_id=nid)
            continue
        chunks = [
            SourceChunk(chunk_id=c.chunk_id, text=c.text)
            for c in deps.manifest.get_chunks(chunk_ids)
        ]
        out.append(
            NoteContext(
                note_id=nid,
                title=row.title,
                type=row.type,
                date=row.date,
                highlights=row.highlights,
                chunks=chunks,
                related=related,
                work_id=row.work_id,
            )
        )
    return out


def _related_note_ids(deps: GraphDeps, note_ids: list[str]) -> list[str]:
    """Extra notes close to the idea's centroid in embedding space (RAG beyond the cluster)."""
    k = deps.settings.related_notes
    chunk_ids = [c for nid in note_ids for c in deps.manifest.chunk_ids(nid)]
    if k <= 0 or not chunk_ids or len(deps.index) == 0:
        return []
    try:
        vectors = deps.index.get_vectors(chunk_ids)
    except RuntimeError:  # index out of sync with the manifest
        log.warning("context_index_out_of_sync")
        return []
    query = vectors.mean(axis=0)
    norm = np.linalg.norm(query)
    if norm == 0:
        return []
    hits = deps.index.search(query / norm, k=min(len(deps.index), 50))
    owner = {c.chunk_id: c.note_id for c in deps.manifest.get_chunks([i for i, _ in hits])}
    found: list[str] = []
    for chunk_id, score in hits:
        nid = owner.get(chunk_id)
        if nid is None or nid in note_ids or nid in found:
            continue
        if score < deps.settings.related_min_score:
            break  # hits are sorted by score
        found.append(nid)
        if len(found) == k:
            break
    return found
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v` → all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/vault_to_voice/graph/deps.py src/vault_to_voice/graph/prompts.py src/vault_to_voice/graph/nodes/context.py tests/fakes.py tests/test_graph_prompts.py tests/test_node_context.py
git commit -m "feat: graph deps, prompt helpers and load_context node"
```

---

### Task 4: Interview subgraph

**Files:**
- Create: `src/vault_to_voice/graph/nodes/interview.py`
- Test: `tests/test_node_interview.py`

**Interfaces:**
- Consumes: `GraphDeps`, `LLM_RETRY` (Task 3), `notes_block`, `qa_block` (Task 3), `Question`, `QA`, `CoverageCheck`, `PostState` (Task 1), `DATA_INSTRUCTION`.
- Produces: `ask_question(state, deps) -> {"pending_question": Question}` (grounding ids filtered to context notes); `wait_for_answer(state, deps) -> {"qa": [...], "pending_question": None}` — interrupts with `{"type": "interview_question", "question", "grounding_note_ids", "number", "max"}`, resume value = answer string; `check_coverage(state, deps) -> {"interview_done": bool}`; `build_interview_graph(deps) -> StateGraph` (uncompiled).

- [ ] **Step 1: Write failing tests**

`tests/test_node_interview.py`:
```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.fakes import FakeLLMClient, make_deps, seed_idea
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.nodes.interview import build_interview_graph
from vault_to_voice.graph.state import CoverageCheck, Question

CFG = {"configurable": {"thread_id": "t"}}


def run_interview(tmp_path, vault_dir, coverage, answers):
    count = {"n": 0}

    def ask(_prompt):
        count["n"] += 1
        return Question(question=f"Question {count['n']}?", grounding_note_ids=["n0", "bogus"])

    fake = FakeLLMClient(
        {Question: ask, CoverageCheck: lambda _: CoverageCheck(enough=coverage)}
    )
    deps = make_deps(tmp_path, fake)
    idea = seed_idea(deps, vault_dir)
    state = {"idea": idea, **load_context({"idea": idea}, deps)}
    app = build_interview_graph(deps).compile(checkpointer=InMemorySaver())
    result = app.invoke(state, CFG)
    asked = []
    for answer in answers:
        if "__interrupt__" not in result:
            break
        asked.append(result["__interrupt__"][0].value)
        result = app.invoke(Command(resume=answer), CFG)
    return result, asked, fake


def calls(fake, schema):
    return [prompt for s, prompt in fake.calls if s is schema]


def test_min_three_questions_even_if_covered(tmp_path, vault_dir):
    result, _, fake = run_interview(tmp_path, vault_dir, True, ["a"] * 6)
    assert len(result["qa"]) == 3 and "__interrupt__" not in result
    assert len(calls(fake, CoverageCheck)) == 1  # only consulted once the minimum is reached


def test_max_five_questions(tmp_path, vault_dir):
    result, _, _ = run_interview(tmp_path, vault_dir, False, ["a"] * 9)
    assert len(result["qa"]) == 5


def test_stops_between_min_and_max_when_covered(tmp_path, vault_dir):
    count = {"n": 0}

    def coverage(_prompt):
        count["n"] += 1
        return CoverageCheck(enough=count["n"] >= 2)  # not enough at 3, enough at 4

    fake = FakeLLMClient({Question: lambda _: Question(question="Q?"), CoverageCheck: coverage})
    deps = make_deps(tmp_path, fake)
    idea = seed_idea(deps, vault_dir)
    app = build_interview_graph(deps).compile(checkpointer=InMemorySaver())
    result = app.invoke({"idea": idea, **load_context({"idea": idea}, deps)}, CFG)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume="a"), CFG)
    assert len(result["qa"]) == 4


def test_question_payload_qa_ids_and_no_reask_on_resume(tmp_path, vault_dir):
    result, asked, fake = run_interview(tmp_path, vault_dir, True, ["x", "y", "z"])
    assert asked[0] == {
        "type": "interview_question",
        "question": "Question 1?",
        "grounding_note_ids": ["n0"],
        "number": 1,
        "max": 5,
    }
    assert [q.qa_id for q in result["qa"]] == ["qa:1", "qa:2", "qa:3"]
    assert len(calls(fake, Question)) == 3  # resuming re-runs wait_for_answer, not ask_question


def test_blank_answer_recorded_as_skipped(tmp_path, vault_dir):
    result, _, _ = run_interview(tmp_path, vault_dir, True, ["   ", "b", "c"])
    assert result["qa"][0].answer == "(skipped)"


def test_prompt_includes_notes_and_previous_answers(tmp_path, vault_dir):
    _, _, fake = run_interview(tmp_path, vault_dir, True, ["I loved the tells", "b", "c"])
    second = calls(fake, Question)[1]
    assert '<note id="n0">' in second and "I loved the tells" in second
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_node_interview.py -v`
Expected: FAIL — `ModuleNotFoundError: vault_to_voice.graph.nodes.interview`.

- [ ] **Step 3: Implement**

`src/vault_to_voice/graph/nodes/interview.py`:
```python
from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from vault_to_voice.graph.deps import LLM_RETRY, GraphDeps
from vault_to_voice.graph.prompts import notes_block, qa_block
from vault_to_voice.graph.state import QA, CoverageCheck, PostState, Question
from vault_to_voice.guardrails import DATA_INSTRUCTION

QUESTION_PROMPT = (
    "You are interviewing the author to gather material for a blog post.\n"
    "Post idea: {title} — {angle}\n"
    "Ask ONE short, specific question that draws out a personal opinion, memory or example "
    "that the notes and previous answers do not already cover. Never repeat a question.\n"
    "grounding_note_ids: ids of the notes your question builds on.\n"
)
COVERAGE_PROMPT = (
    "Decide whether this interview has gathered enough personal material (opinions, memories, "
    "concrete examples) to write the post. enough: true or false. missing: what is still "
    "missing, if anything.\n"
)


def ask_question(state: PostState, deps: GraphDeps) -> dict:
    idea = state["idea"]
    parts = [
        QUESTION_PROMPT.format(title=idea.title, angle=idea.angle),
        "Notes:",
        notes_block(state["notes"]),
    ]
    if state.get("qa"):
        parts += ["Answers so far:", qa_block(state["qa"])]
    if state.get("past_answers"):
        parts += ["Already answered in earlier interviews (do not ask again):",
                  qa_block(state["past_answers"])]
    question = deps.llm.structured("\n\n".join(parts), Question, system=DATA_INSTRUCTION)
    known = {n.note_id for n in state["notes"]}
    grounding = [i for i in question.grounding_note_ids if i in known]
    return {"pending_question": question.model_copy(update={"grounding_note_ids": grounding})}


def wait_for_answer(state: PostState, deps: GraphDeps) -> dict:
    # No LLM call here: on resume LangGraph re-runs this whole node, and interrupt() returns
    # the resume value instead of pausing.
    question = state["pending_question"]
    qa = state.get("qa", [])
    number = len(qa) + 1
    answer = interrupt(
        {
            "type": "interview_question",
            "question": question.question,
            "grounding_note_ids": question.grounding_note_ids,
            "number": number,
            "max": deps.settings.interview_max_questions,
        }
    )
    text = str(answer).strip() or "(skipped)"
    new = QA(
        qa_id=f"qa:{number}",
        question=question.question,
        answer=text,
        grounding_note_ids=question.grounding_note_ids,
    )
    return {"qa": [*qa, new], "pending_question": None}


def check_coverage(state: PostState, deps: GraphDeps) -> dict:
    n = len(state.get("qa", []))
    if n < deps.settings.interview_min_questions:
        return {"interview_done": False}
    if n >= deps.settings.interview_max_questions:
        return {"interview_done": True}
    idea = state["idea"]
    prompt = "\n\n".join(
        [COVERAGE_PROMPT, f"Post idea: {idea.title} — {idea.angle}", qa_block(state["qa"])]
    )
    check = deps.llm.structured(prompt, CoverageCheck, system=DATA_INSTRUCTION)
    return {"interview_done": check.enough}


def build_interview_graph(deps: GraphDeps) -> StateGraph:
    g = StateGraph(PostState)
    g.add_node("ask_question", partial(ask_question, deps=deps), retry_policy=LLM_RETRY)
    g.add_node("wait_for_answer", partial(wait_for_answer, deps=deps))
    g.add_node("check_coverage", partial(check_coverage, deps=deps), retry_policy=LLM_RETRY)
    g.add_edge(START, "ask_question")
    g.add_edge("ask_question", "wait_for_answer")
    g.add_edge("wait_for_answer", "check_coverage")
    g.add_conditional_edges(
        "check_coverage",
        lambda s: END if s["interview_done"] else "ask_question",
        ["ask_question", END],
    )
    return g
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v` → all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/vault_to_voice/graph/nodes/interview.py tests/test_node_interview.py
git commit -m "feat: adaptive interview subgraph with interrupts"
```

---

### Task 5: Outline and outline review

**Files:**
- Create: `src/vault_to_voice/graph/nodes/outline.py`
- Test: `tests/test_node_outline.py`

**Interfaces:**
- Consumes: Task 1 models (`Outline`, `ReviewDecision`, `valid_sources`), Task 3 helpers.
- Produces: `make_outline(state, deps) -> {"outline", "review_error": None}` (section ids renumbered `s1..sN`, `planned_sources` filtered to valid ids); `outline_review(state, deps) -> dict` — interrupts with `{"type": "outline_review", "outline": dict, "rounds_left": int, "error": str|None}`; resume value `{"action": "approve"|"revise"|"edit", "feedback"?: str, "edited"?: Outline-dict}`; sets `last_action` ∈ {`approve`, `revise`, `retry`}; on approve/edit also `revise_ids` (all section ids), `feedback=""`, `section_feedback={}`; `route_after_outline_review(state) -> "make_outline" | "outline_review" | "dispatch"`.

- [ ] **Step 1: Write failing tests**

`tests/test_node_outline.py`:
```python
from unittest.mock import patch

import pytest

from tests.fakes import FakeLLMClient, make_deps, seed_idea
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.nodes.outline import (
    make_outline,
    outline_review,
    route_after_outline_review,
)
from vault_to_voice.graph.state import QA, Outline, Section


def llm_outline(_prompt):
    return Outline(
        title="Fair Bosses",
        sections=[
            Section(id="a", heading="Tells", point="Tells make it fair.",
                    planned_sources=["n0", "ghost"]),
            Section(id="b", heading="Learning", point="Death teaches.", planned_sources=["qa:1"]),
        ],
    )


@pytest.fixture
def ctx(tmp_path, vault_dir):
    fake = FakeLLMClient({Outline: llm_outline})
    deps = make_deps(tmp_path, fake)
    idea = seed_idea(deps, vault_dir)
    state = {
        "idea": idea,
        **load_context({"idea": idea}, deps),
        "qa": [QA(qa_id="qa:1", question="q", answer="I learned")],
    }
    return deps, state, fake


@pytest.fixture
def reviewed(ctx):
    deps, state, _ = ctx
    return deps, {**state, **make_outline(state, deps)}


def review(state, deps, payload):
    # interrupt() only works inside a running graph; patch it to unit-test the node.
    with patch("vault_to_voice.graph.nodes.outline.interrupt", return_value=payload):
        return outline_review(state, deps)


def test_make_outline_normalises_ids_and_sources(ctx):
    deps, state, _ = ctx
    out = make_outline(state, deps)["outline"]
    assert [s.id for s in out.sections] == ["s1", "s2"]
    assert out.sections[0].planned_sources == ["n0"]
    assert out.sections[1].planned_sources == ["qa:1"]


def test_feedback_and_previous_outline_reach_prompt(reviewed):
    deps, state = reviewed
    make_outline({**state, "feedback": "make it funnier"}, deps)
    prompt = deps.llm.calls[-1][1]
    assert "make it funnier" in prompt and "Fair Bosses" in prompt


def test_interrupt_payload(reviewed):
    deps, state = reviewed
    with patch(
        "vault_to_voice.graph.nodes.outline.interrupt", return_value={"action": "approve"}
    ) as fake_interrupt:
        outline_review(state, deps)
    sent = fake_interrupt.call_args.args[0]
    assert sent["type"] == "outline_review" and sent["rounds_left"] == 3
    assert sent["outline"]["title"] == "Fair Bosses" and sent["error"] is None


def test_approve_routes_to_drafting(reviewed):
    deps, state = reviewed
    upd = review(state, deps, {"action": "approve"})
    assert route_after_outline_review({**state, **upd}) == "dispatch"
    assert upd["revise_ids"] == ["s1", "s2"] and upd["section_feedback"] == {}


def test_revise_counts_rounds_and_caps(reviewed):
    deps, state = reviewed
    upd = review(state, deps, {"action": "revise", "feedback": "shorter"})
    assert route_after_outline_review({**state, **upd}) == "make_outline"
    assert upd["feedback"] == "shorter" and upd["user_rounds"]["outline"] == 1
    capped = {**state, "user_rounds": {"outline": 3, "draft": 0}}
    upd = review(capped, deps, {"action": "revise", "feedback": "again"})
    assert route_after_outline_review({**capped, **upd}) == "outline_review"
    assert "limit" in upd["review_error"]


def test_edit_replaces_outline(reviewed):
    deps, state = reviewed
    edited = {
        "title": "Mine",
        "sections": [{"id": "z", "heading": "Only", "point": "p", "planned_sources": ["n1"]}],
    }
    upd = review(state, deps, {"action": "edit", "edited": edited})
    assert upd["outline"].title == "Mine" and [s.id for s in upd["outline"].sections] == ["s1"]
    assert route_after_outline_review({**state, **upd}) == "dispatch"


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "publish"},
        "approve",
        {"action": "edit"},
        {"action": "edit", "edited": {"title": "x", "sections": []}},
    ],
)
def test_bad_payloads_reask_with_error(reviewed, payload):
    deps, state = reviewed
    upd = review(state, deps, payload)
    assert route_after_outline_review({**state, **upd}) == "outline_review"
    assert upd["review_error"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_node_outline.py -v`
Expected: FAIL — `ModuleNotFoundError: vault_to_voice.graph.nodes.outline`.

- [ ] **Step 3: Implement**

`src/vault_to_voice/graph/nodes/outline.py`:
```python
from langgraph.types import interrupt
from pydantic import ValidationError

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.prompts import notes_block, qa_block, style_brief
from vault_to_voice.graph.state import Outline, PostState, ReviewDecision, valid_sources
from vault_to_voice.guardrails import DATA_INSTRUCTION

OUTLINE_PROMPT = (
    "Plan a blog post in the author's voice.\n"
    "title: the post title. sections: 3-6 sections, each with heading, point (one sentence: "
    "what the section argues) and planned_sources (ids of the notes or interview answers below, "
    "e.g. a note id or 'qa:2', that support it). Use only ids that appear below.\n"
)


def _normalise(outline: Outline, valid: set[str]) -> Outline:
    sections = [
        s.model_copy(
            update={"id": f"s{i}", "planned_sources": [x for x in s.planned_sources if x in valid]}
        )
        for i, s in enumerate(outline.sections, 1)
    ]
    return outline.model_copy(update={"sections": sections})


def _error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    return f"Invalid decision: {first['msg']} at {'.'.join(map(str, first['loc'])) or 'payload'}"


def make_outline(state: PostState, deps: GraphDeps) -> dict:
    idea = state["idea"]
    qa = state.get("qa", [])
    parts = [
        OUTLINE_PROMPT,
        f"Post idea: {idea.title} — {idea.angle}",
        "Voice:",
        style_brief(state.get("style")),
        "Notes:",
        notes_block(state["notes"]),
        "Interview answers:",
        qa_block(qa),
    ]
    if state.get("feedback") and state.get("outline"):
        parts += [
            "Previous outline:",
            state["outline"].model_dump_json(),
            "Revise it using this feedback from the author:",
            state["feedback"],
        ]
    outline = deps.llm.structured("\n\n".join(parts), Outline, system=DATA_INSTRUCTION)
    return {"outline": _normalise(outline, valid_sources(state["notes"], qa)), "review_error": None}


def _approved(outline: Outline) -> dict:
    return {
        "outline": outline,
        "last_action": "approve",
        "review_error": None,
        "revise_ids": [s.id for s in outline.sections],
        "feedback": "",
        "section_feedback": {},
    }


def _retry(message: str) -> dict:
    return {"last_action": "retry", "review_error": message}


def outline_review(state: PostState, deps: GraphDeps) -> dict:
    rounds = state.get("user_rounds") or {"outline": 0, "draft": 0}
    left = deps.settings.max_review_rounds - rounds["outline"]
    payload = interrupt(
        {
            "type": "outline_review",
            "outline": state["outline"].model_dump(),
            "rounds_left": left,
            "error": state.get("review_error"),
        }
    )
    try:
        decision = ReviewDecision.model_validate(payload)
    except ValidationError as exc:
        return _retry(_error(exc))
    if decision.action == "approve":
        return _approved(state["outline"])
    if decision.action == "revise":
        if left <= 0:
            return _retry("Revision limit reached — approve or edit the outline.")
        return {
            "last_action": "revise",
            "feedback": decision.feedback,
            "review_error": None,
            "user_rounds": {**rounds, "outline": rounds["outline"] + 1},
        }
    try:
        edited = Outline.model_validate(decision.edited)
    except ValidationError as exc:
        return _retry(_error(exc))
    return _approved(_normalise(edited, valid_sources(state["notes"], state.get("qa", []))))


def route_after_outline_review(state: PostState) -> str:
    return {"revise": "make_outline", "retry": "outline_review"}.get(
        state["last_action"], "dispatch"
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v` → all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/vault_to_voice/graph/nodes/outline.py tests/test_node_outline.py
git commit -m "feat: outline node and human outline review"
```

---

### Task 6: Graph builder (through outline review) and runner helpers

**Files:**
- Create: `src/vault_to_voice/graph/builder.py`, `src/vault_to_voice/graph/runner.py`
- Test: `tests/test_graph_builder.py`

**Interfaces:**
- Consumes: Tasks 2–5.
- Produces: `build_post_graph(deps, *, checkpointer=None) -> CompiledStateGraph` (M2a: `outline_review` "dispatch" route goes to `END`; Task 11 replaces it with drafting); `thread_config(thread_id) -> dict`; `start_input(idea) -> dict`; `run_until_interrupt(graph, config, value) -> dict | None` (returns the pending interrupt's value, or `None` when the run finished); `pending_interrupt(graph, config) -> dict | None`.

- [ ] **Step 1: Write failing tests**

`tests/test_graph_builder.py`:
```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.fakes import FakeLLMClient, make_deps, seed_idea
from vault_to_voice.graph.builder import build_post_graph
from vault_to_voice.graph.checkpoint import make_checkpointer
from vault_to_voice.graph.runner import (
    pending_interrupt,
    run_until_interrupt,
    start_input,
    thread_config,
)
from vault_to_voice.graph.state import CoverageCheck, Outline, Question, Section


def fake_llm():
    count = {"n": 0}

    def ask(_prompt):
        count["n"] += 1
        return Question(question=f"Q{count['n']}?")

    return FakeLLMClient(
        {
            Question: ask,
            CoverageCheck: lambda _: CoverageCheck(enough=True),
            Outline: lambda _: Outline(
                title="Fair Bosses", sections=[Section(id="x", heading="H", point="P")]
            ),
        }
    )


def test_interview_then_outline_review_and_revise_loop(tmp_path, vault_dir):
    deps = make_deps(tmp_path, fake_llm())
    idea = seed_idea(deps, vault_dir)
    graph = build_post_graph(deps, checkpointer=InMemorySaver())
    cfg = thread_config(idea.idea_id)
    value = run_until_interrupt(graph, cfg, start_input(idea))
    questions = 0
    while value["type"] == "interview_question":
        questions += 1
        value = run_until_interrupt(graph, cfg, Command(resume=f"answer {questions}"))
    assert questions == 3 and value["type"] == "outline_review"
    assert pending_interrupt(graph, cfg)["type"] == "outline_review"
    value = run_until_interrupt(graph, cfg, Command(resume={"action": "revise", "feedback": "x"}))
    assert value["type"] == "outline_review" and value["rounds_left"] == 2
    assert sum(1 for s, _ in deps.llm.calls if s is Outline) == 2


def test_resume_after_restart_with_sqlite(tmp_path, vault_dir):
    deps = make_deps(tmp_path, fake_llm())
    idea = seed_idea(deps, vault_dir)
    path = tmp_path / "cp.db"
    cfg = thread_config(idea.idea_id)
    first = build_post_graph(deps, checkpointer=make_checkpointer(path))
    run_until_interrupt(first, cfg, start_input(idea))
    run_until_interrupt(first, cfg, Command(resume="one"))
    restarted = build_post_graph(deps, checkpointer=make_checkpointer(path))  # new process
    assert pending_interrupt(restarted, cfg)["number"] == 2
    assert run_until_interrupt(restarted, cfg, Command(resume="two"))["number"] == 3
    assert sum(1 for s, _ in deps.llm.calls if s is Question) == 3  # nothing re-asked
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_graph_builder.py -v`
Expected: FAIL — `ModuleNotFoundError: vault_to_voice.graph.builder`.

- [ ] **Step 3: Implement**

`src/vault_to_voice/graph/runner.py`:
```python
from vault_to_voice.ideas.detector import Idea


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def start_input(idea: Idea) -> dict:
    return {"idea": idea}


def run_until_interrupt(graph, config: dict, value) -> dict | None:
    """Invoke (start, resume via Command, or retry via None); return the next interrupt or None."""
    result = graph.invoke(value, config)
    pending = result.get("__interrupt__") if isinstance(result, dict) else None
    return pending[0].value if pending else None


def pending_interrupt(graph, config: dict) -> dict | None:
    snapshot = graph.get_state(config)
    return snapshot.interrupts[0].value if snapshot.interrupts else None
```

`src/vault_to_voice/graph/builder.py`:
```python
from functools import partial

from langgraph.graph import END, START, StateGraph

from vault_to_voice.graph.deps import LLM_RETRY, GraphDeps
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.nodes.interview import build_interview_graph
from vault_to_voice.graph.nodes.outline import (
    make_outline,
    outline_review,
    route_after_outline_review,
)
from vault_to_voice.graph.state import PostState


def build_post_graph(deps: GraphDeps, *, checkpointer=None):
    g = StateGraph(PostState)
    g.add_node("load_context", partial(load_context, deps=deps))
    g.add_node("interview", build_interview_graph(deps).compile())  # subgraph as a node
    g.add_node("make_outline", partial(make_outline, deps=deps), retry_policy=LLM_RETRY)
    g.add_node("outline_review", partial(outline_review, deps=deps))

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "interview")
    g.add_edge("interview", "make_outline")
    g.add_edge("make_outline", "outline_review")
    g.add_conditional_edges(
        "outline_review",
        route_after_outline_review,
        {"make_outline": "make_outline", "outline_review": "outline_review", "dispatch": END},
    )
    return g.compile(checkpointer=checkpointer, store=deps.store)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v` → all PASS; ruff clean.

- [ ] **Step 5: Commit and close S5**

```bash
git add src/vault_to_voice/graph/builder.py src/vault_to_voice/graph/runner.py tests/test_graph_builder.py
git commit -m "feat: post graph through outline review with resumable SQLite threads"
```
Update `PROGRESS.md` (S5 done, next S6 Task 7), commit `docs: update progress after S5`.

---

### Task 7: Parallel section drafting (`Send` fan-out), merge, render

**Files:**
- Create: `src/vault_to_voice/graph/nodes/draft.py`, `src/vault_to_voice/graph/render.py`
- Test: `tests/test_node_draft.py`

**Interfaces:**
- Consumes: Task 1 models (`DraftTask`, `SectionText`, `SectionDraft`), Task 3 helpers.
- Produces: `fan_out_sections(state) -> list[Send]` (one `Send("draft_section", DraftTask)` per id in `state["revise_ids"]`, feedback = `state["feedback"]` and `state["section_feedback"][id]` joined by newline, empties dropped); `draft_section(task: DraftTask, deps) -> {"drafts": {section_id: SectionDraft}}`; `merge_draft(state) -> {"draft": list[SectionDraft]}` in outline order; `render_markdown(outline, draft) -> str`.

- [ ] **Step 1: Write failing tests**

`tests/test_node_draft.py`:
```python
import pytest

from tests.fakes import FakeLLMClient, make_deps, seed_idea
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.nodes.draft import draft_section, fan_out_sections, merge_draft
from vault_to_voice.graph.render import render_markdown
from vault_to_voice.graph.state import (
    QA,
    Outline,
    Paragraph,
    Section,
    SectionDraft,
    SectionText,
    Sentence,
)

OUTLINE = Outline(
    title="Fair Bosses",
    sections=[
        Section(id="s1", heading="Tells", point="Tells make it fair.", planned_sources=["n0"]),
        Section(id="s2", heading="Learning", point="Death teaches.", planned_sources=["qa:1"]),
    ],
)


def sd(section_id, *texts):
    sentences = [Sentence(text=t, sources=["n0"]) for t in texts]
    return SectionDraft(section_id=section_id, paragraphs=[Paragraph(sentences=sentences)])


@pytest.fixture
def ready(tmp_path, vault_dir):
    text = SectionText(
        paragraphs=[Paragraph(sentences=[Sentence(text="Every attack has a tell.", sources=["n0"])])]
    )
    fake = FakeLLMClient({SectionText: lambda _: text})
    deps = make_deps(tmp_path, fake)
    idea = seed_idea(deps, vault_dir)
    state = {
        **load_context({"idea": idea}, deps),
        "idea": idea,
        "qa": [QA(qa_id="qa:1", question="q", answer="I learned")],
        "outline": OUTLINE,
        "revise_ids": ["s1", "s2"],
        "feedback": "",
        "section_feedback": {"s2": "fix s2"},
    }
    return deps, state, fake


def test_fan_out_sends_one_task_per_revise_id(ready):
    _, state, _ = ready
    sends = fan_out_sections(state)
    assert [s.node for s in sends] == ["draft_section", "draft_section"]
    assert [s.arg["section"].id for s in sends] == ["s1", "s2"]
    assert sends[0].arg["feedback"] == "" and sends[1].arg["feedback"] == "fix s2"
    assert sends[0].arg["outline_title"] == "Fair Bosses"


def test_fan_out_only_flagged_sections(ready):
    _, state, _ = ready
    sends = fan_out_sections({**state, "revise_ids": ["s2"], "feedback": "warmer"})
    assert [s.arg["section"].id for s in sends] == ["s2"]
    assert sends[0].arg["feedback"] == "warmer\nfix s2"


def test_draft_section_returns_keyed_draft(ready):
    deps, state, fake = ready
    out = draft_section(fan_out_sections(state)[0].arg, deps)
    assert list(out["drafts"]) == ["s1"] and out["drafts"]["s1"].section_id == "s1"
    prompt = fake.calls[-1][1]
    assert "Tells" in prompt and '<note id="n0">' in prompt and '<note id="qa:1">' in prompt


def test_merge_follows_outline_order_and_ignores_stale():
    drafts = {"s2": sd("s2", "B."), "old": sd("old", "X."), "s1": sd("s1", "A.")}
    merged = merge_draft({"outline": OUTLINE, "drafts": drafts})["draft"]
    assert [d.section_id for d in merged] == ["s1", "s2"]


def test_render_markdown():
    draft = [sd("s1", "A b.", "C d."), sd("s2", "E f.")]
    assert render_markdown(OUTLINE, draft) == (
        "# Fair Bosses\n\n## Tells\n\nA b. C d.\n\n## Learning\n\nE f.\n"
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_node_draft.py -v`
Expected: FAIL — `ModuleNotFoundError: vault_to_voice.graph.nodes.draft`.

- [ ] **Step 3: Implement**

`src/vault_to_voice/graph/render.py`:
```python
from vault_to_voice.graph.state import Outline, SectionDraft


def render_markdown(outline: Outline, draft: list[SectionDraft]) -> str:
    """Clean Markdown: headings and prose only, no citation markers."""
    headings = {s.id: s.heading for s in outline.sections}
    parts = [f"# {outline.title}"]
    for section in draft:
        parts.append(f"## {headings.get(section.section_id, section.section_id)}")
        parts += [" ".join(s.text for s in p.sentences) for p in section.paragraphs]
    return "\n\n".join(parts) + "\n"
```

`src/vault_to_voice/graph/nodes/draft.py`:
```python
from langgraph.types import Send

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.prompts import notes_block, qa_block, style_brief
from vault_to_voice.graph.state import DraftTask, PostState, SectionDraft, SectionText
from vault_to_voice.guardrails import DATA_INSTRUCTION

DRAFT_PROMPT = (
    "Write one section of a blog post in the author's voice (see Voice).\n"
    "Return paragraphs made of sentences; each sentence has text, kind and sources.\n"
    "kind='claim' for any sentence stating a fact, opinion, memory or example: its sources MUST "
    "list the ids (note ids or 'qa:N') whose text supports it. kind='connective' only for short "
    "transitions with no content of their own; they have no sources.\n"
    "Never cite an id that does not appear below. Do not invent experiences.\n"
)


def fan_out_sections(state: PostState) -> list[Send]:
    """Conditional edge: one parallel draft_section run per section that needs (re)drafting."""
    sections = {s.id: s for s in state["outline"].sections}
    section_feedback = state.get("section_feedback") or {}
    sends = []
    for sid in state["revise_ids"]:
        feedback = "\n".join(f for f in [state.get("feedback"), section_feedback.get(sid)] if f)
        task: DraftTask = {
            "section": sections[sid],
            "outline_title": state["outline"].title,
            "notes": state["notes"],
            "qa": state.get("qa", []),
            "style": state.get("style"),
            "feedback": feedback,
        }
        sends.append(Send("draft_section", task))
    return sends


def draft_section(task: DraftTask, deps: GraphDeps) -> dict:
    section = task["section"]
    parts = [
        DRAFT_PROMPT,
        f"Post title: {task['outline_title']}",
        f"Section: {section.heading}\nPoint: {section.point}\n"
        f"Planned sources: {', '.join(section.planned_sources) or '(none)'}",
        "Voice:",
        style_brief(task["style"]),
        "Notes:",
        notes_block(task["notes"]),
        "Interview answers:",
        qa_block(task["qa"]),
    ]
    if task["feedback"]:
        parts += ["Fix these problems from the previous version:", task["feedback"]]
    text = deps.llm.structured("\n\n".join(parts), SectionText, system=DATA_INSTRUCTION)
    return {"drafts": {section.id: SectionDraft(section_id=section.id, paragraphs=text.paragraphs)}}


def merge_draft(state: PostState) -> dict:
    drafts = state.get("drafts") or {}
    return {"draft": [drafts[s.id] for s in state["outline"].sections if s.id in drafts]}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v` → all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/vault_to_voice/graph/nodes/draft.py src/vault_to_voice/graph/render.py tests/test_node_draft.py
git commit -m "feat: parallel section drafting with Send fan-out"
```

---

### Task 8: Citation checker

**Files:**
- Create: `src/vault_to_voice/graph/nodes/citations.py`
- Test: `tests/test_node_citations.py`

**Interfaces:**
- Consumes: Task 1 models (`Flag`, `CitationReport`, `valid_sources`), `GraphDeps` (embedder, settings, llm), guardrails.
- Produces: `JudgeVerdict(index: int, supported: bool)`, `JudgeVerdicts(verdicts: list[JudgeVerdict])`; `check_citations(state, deps) -> {"citation_report": CitationReport}`. Rules per claim sentence (connective skipped): no sources → `missing_source`; any source not in `valid_sources` → `invalid_source`; `"user"` in sources → supported; else score = max cosine(sentence, each text of each cited source — note chunks or QA answer): ≥ `citation_support` supported, ≤ `citation_reject` → `unsupported` (with score), else one batched LLM judge call for all grey-zone sentences (missing verdict = unsupported).

- [ ] **Step 1: Write failing tests**

`tests/test_node_citations.py`:
```python
import pytest

from tests.fakes import FakeLLMClient, make_deps, seed_idea
from vault_to_voice.graph.nodes.citations import JudgeVerdict, JudgeVerdicts, check_citations
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.state import QA, Paragraph, SectionDraft, Sentence

N0_FIRST = "Boss fights feel fair when every attack has a clear tell."
GREY = "Boss fights feel fair when the camera spins wildly around."  # ~half the words of N0_FIRST


def setup(tmp_path, vault_dir, verdict=True):
    fake = FakeLLMClient(
        {JudgeVerdicts: lambda _: JudgeVerdicts(verdicts=[JudgeVerdict(index=0, supported=verdict)])}
    )
    deps = make_deps(tmp_path, fake)
    idea = seed_idea(deps, vault_dir)
    qa = [QA(qa_id="qa:1", question="q", answer="I died twenty times on the bell tower.")]
    return deps, {**load_context({"idea": idea}, deps), "idea": idea, "qa": qa}, fake


def check(deps, state, *sentences):
    draft = [SectionDraft(section_id="s1", paragraphs=[Paragraph(sentences=list(sentences))])]
    return check_citations({**state, "draft": draft}, deps)["citation_report"]


def test_exact_note_text_supported(tmp_path, vault_dir):
    deps, state, fake = setup(tmp_path, vault_dir)
    report = check(deps, state, Sentence(text=N0_FIRST, sources=["n0"]))
    assert (report.n_claims, report.n_supported, report.flags) == (1, 1, [])
    assert fake.calls == []  # no judge needed


def test_qa_source_supported(tmp_path, vault_dir):
    deps, state, _ = setup(tmp_path, vault_dir)
    report = check(deps, state, Sentence(text="I died twenty times on the bell tower.",
                                         sources=["qa:1"]))
    assert report.n_supported == 1


@pytest.mark.parametrize(
    ("sources", "reason"),
    [([], "missing_source"), (["ghost"], "invalid_source"), (["qa:9"], "invalid_source")],
)
def test_missing_and_invalid_sources_flagged(tmp_path, vault_dir, sources, reason):
    deps, state, _ = setup(tmp_path, vault_dir)
    report = check(deps, state, Sentence(text="Some claim.", sources=sources))
    assert [f.reason for f in report.flags] == [reason]
    assert (report.flags[0].section_id, report.flags[0].paragraph, report.flags[0].sentence) == (
        "s1", 0, 0)


def test_unrelated_text_unsupported(tmp_path, vault_dir):
    deps, state, _ = setup(tmp_path, vault_dir)
    report = check(deps, state, Sentence(text="Purple bananas orbit distant moons quietly.",
                                         sources=["n0"]))
    assert [f.reason for f in report.flags] == ["unsupported"]
    assert report.flags[0].score <= 0.30


def test_connective_exempt_and_user_supported(tmp_path, vault_dir):
    deps, state, _ = setup(tmp_path, vault_dir)
    report = check(
        deps,
        state,
        Sentence(text="So.", kind="connective"),
        Sentence(text="My own edit.", sources=["user"]),
    )
    assert (report.n_claims, report.n_supported, report.flags) == (1, 1, [])


@pytest.mark.parametrize(("verdict", "supported"), [(True, 1), (False, 0)])
def test_grey_zone_goes_to_llm_judge(tmp_path, vault_dir, verdict, supported):
    deps, state, fake = setup(tmp_path, vault_dir, verdict)
    report = check(deps, state, Sentence(text=GREY, sources=["n0"]))
    assert report.n_supported == supported
    judge_prompts = [p for s, p in fake.calls if s is JudgeVerdicts]
    assert len(judge_prompts) == 1 and GREY in judge_prompts[0]
    assert '<note id="n0">' in judge_prompts[0]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_node_citations.py -v`
Expected: FAIL — `ModuleNotFoundError: vault_to_voice.graph.nodes.citations`.

- [ ] **Step 3: Implement**

`src/vault_to_voice/graph/nodes/citations.py`:
```python
from pydantic import BaseModel

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.state import CitationReport, Flag, PostState, Sentence, valid_sources
from vault_to_voice.guardrails import DATA_INSTRUCTION, scrub_pii, wrap_note

JUDGE_PROMPT = (
    "For each numbered claim, decide whether its cited source texts support it. A claim is "
    "supported if the sources state it or clearly imply it. Return one verdict per claim index.\n"
)


class JudgeVerdict(BaseModel):
    index: int
    supported: bool


class JudgeVerdicts(BaseModel):
    verdicts: list[JudgeVerdict]


def _support(deps: GraphDeps, text: str, source_texts: list[str]) -> float:
    vectors = deps.embedder.embed([scrub_pii(t) for t in [text, *source_texts]])
    return round(float((vectors[1:] @ vectors[0]).max()), 3)


def _judge(
    deps: GraphDeps, grey: list[tuple[dict, Sentence, float]], texts: dict[str, list[str]]
) -> dict[int, bool]:
    items = []
    for i, (_, sentence, _) in enumerate(grey):
        sources = "\n".join(
            wrap_note(src, scrub_pii("\n".join(texts[src]))) for src in sentence.sources
        )
        items.append(f"[{i}] claim: {sentence.text}\nsources:\n{sources}")
    out = deps.llm.structured(
        JUDGE_PROMPT + "\n\n".join(items), JudgeVerdicts, system=DATA_INSTRUCTION
    )
    return {v.index: v.supported for v in out.verdicts}


def check_citations(state: PostState, deps: GraphDeps) -> dict:
    qa = state.get("qa", [])
    valid = valid_sources(state["notes"], qa)
    texts = {n.note_id: [c.text for c in n.chunks] for n in state["notes"]}
    texts.update({q.qa_id: [q.answer] for q in qa})
    settings = deps.settings

    flags: list[Flag] = []
    grey: list[tuple[dict, Sentence, float]] = []
    n_claims = n_supported = 0
    for section in state["draft"]:
        for pi, paragraph in enumerate(section.paragraphs):
            for si, sentence in enumerate(paragraph.sentences):
                if sentence.kind == "connective":
                    continue
                n_claims += 1
                loc = {"section_id": section.section_id, "paragraph": pi, "sentence": si,
                       "text": sentence.text}
                if not sentence.sources:
                    flags.append(Flag(**loc, reason="missing_source"))
                elif any(src not in valid for src in sentence.sources):
                    flags.append(Flag(**loc, reason="invalid_source"))
                elif "user" in sentence.sources:
                    n_supported += 1
                else:
                    cited = [t for src in sentence.sources for t in texts[src]]
                    score = _support(deps, sentence.text, cited)
                    if score >= settings.citation_support:
                        n_supported += 1
                    elif score <= settings.citation_reject:
                        flags.append(Flag(**loc, reason="unsupported", score=score))
                    else:
                        grey.append((loc, sentence, score))
    if grey:
        verdicts = _judge(deps, grey, texts)
        for i, (loc, _, score) in enumerate(grey):
            if verdicts.get(i, False):
                n_supported += 1
            else:
                flags.append(Flag(**loc, reason="unsupported", score=score))
    report = CitationReport(n_claims=n_claims, n_supported=n_supported, flags=flags)
    return {"citation_report": report}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v` → all PASS; ruff clean. If `test_grey_zone_goes_to_llm_judge` lands outside the grey zone because of a `HashEmbedder` bucket collision, adjust `GREY`'s non-shared words (keep exactly 5 shared with `N0_FIRST`) — record a `Ruling:`.

- [ ] **Step 5: Commit**

```bash
git add src/vault_to_voice/graph/nodes/citations.py tests/test_node_citations.py
git commit -m "feat: citation checker with similarity thresholds and LLM judge"
```

---

### Task 9: Auto-revise routing and human draft review

**Files:**
- Create: `src/vault_to_voice/graph/nodes/review.py`
- Test: `tests/test_node_review.py`

**Interfaces:**
- Consumes: Task 1 models, `render_markdown` (Task 7), `split_paragraphs` (index/chunking.py), `split_sentences` (style/stats.py).
- Produces: `route_after_checks(state) -> "prepare_auto_revise" | "draft_review"`; `prepare_auto_revise(state) -> {"revise_ids", "section_feedback", "feedback": "", "auto_revised": True}`; `draft_review(state, deps) -> dict` — interrupts with `{"type": "draft_review", "markdown", "draft": [dict], "flags": [dict], "coverage": float, "rounds_left": int, "error": str|None}`; resume `{"action": "approve"|"revise"|"edit", "feedback"?, "edited"?: {section_id: text}}`; sets `last_action` ∈ {`publish`, `dispatch`, `check`, `retry`}; `route_after_draft_review(state) -> "publish" | "dispatch" | "check_citations" | "draft_review"`.

- [ ] **Step 1: Write failing tests**

`tests/test_node_review.py`:
```python
from unittest.mock import patch

import pytest

from tests.fakes import FakeLLMClient, make_deps
from vault_to_voice.graph.nodes.review import (
    draft_review,
    prepare_auto_revise,
    route_after_checks,
    route_after_draft_review,
)
from vault_to_voice.graph.state import (
    CitationReport,
    Flag,
    Outline,
    Paragraph,
    Section,
    SectionDraft,
    Sentence,
)


def flagged():
    flag = Flag(section_id="s2", paragraph=0, sentence=0, text="Bad claim.", reason="unsupported",
                score=0.1)
    return CitationReport(n_claims=2, n_supported=1, flags=[flag])


def sd(section_id, text):
    return SectionDraft(
        section_id=section_id,
        paragraphs=[Paragraph(sentences=[Sentence(text=text, sources=["n0"])])],
    )


@pytest.fixture
def state():
    outline = Outline(
        title="Fair Bosses",
        sections=[Section(id="s1", heading="Tells", point="p"),
                  Section(id="s2", heading="Learning", point="q")],
    )
    return {
        "outline": outline,
        "draft": [sd("s1", "Good."), sd("s2", "Bad claim.")],
        "citation_report": flagged(),
        "user_rounds": {"outline": 1, "draft": 0},
    }


@pytest.fixture
def deps(tmp_path):
    return make_deps(tmp_path, FakeLLMClient({}))


def review(state, deps, payload):
    with patch("vault_to_voice.graph.nodes.review.interrupt", return_value=payload):
        return draft_review(state, deps)


def test_route_after_checks():
    clean = CitationReport(n_claims=1, n_supported=1)
    assert route_after_checks({"citation_report": flagged(), "auto_revised": False}) == (
        "prepare_auto_revise")
    assert route_after_checks({"citation_report": flagged(), "auto_revised": True}) == (
        "draft_review")
    assert route_after_checks({"citation_report": clean, "auto_revised": False}) == "draft_review"


def test_prepare_auto_revise_targets_flagged_sections_once():
    upd = prepare_auto_revise({"citation_report": flagged()})
    assert upd["revise_ids"] == ["s2"] and upd["auto_revised"] is True
    assert "Bad claim." in upd["section_feedback"]["s2"] and upd["feedback"] == ""


def test_payload(state, deps):
    with patch(
        "vault_to_voice.graph.nodes.review.interrupt", return_value={"action": "approve"}
    ) as fake_interrupt:
        draft_review(state, deps)
    sent = fake_interrupt.call_args.args[0]
    assert sent["type"] == "draft_review" and "## Learning\n\nBad claim." in sent["markdown"]
    assert sent["flags"][0]["text"] == "Bad claim." and sent["coverage"] == 0.5
    assert sent["rounds_left"] == 3 and sent["error"] is None


def test_approve_publishes(state, deps):
    upd = review(state, deps, {"action": "approve"})
    assert route_after_draft_review({**state, **upd}) == "publish"


def test_revise_redrafts_all_sections(state, deps):
    upd = review(state, deps, {"action": "revise", "feedback": "warmer"})
    assert route_after_draft_review({**state, **upd}) == "dispatch"
    assert upd["revise_ids"] == ["s1", "s2"] and upd["feedback"] == "warmer"
    assert upd["section_feedback"] == {} and upd["user_rounds"] == {"outline": 1, "draft": 1}


def test_revise_cap(state, deps):
    capped = {**state, "user_rounds": {"outline": 0, "draft": 3}}
    upd = review(capped, deps, {"action": "revise", "feedback": "again"})
    assert route_after_draft_review({**capped, **upd}) == "draft_review"
    assert "limit" in upd["review_error"]


def test_edit_replaces_section_with_user_sources_and_rechecks(state, deps):
    upd = review(state, deps, {"action": "edit",
                               "edited": {"s2": "New first. New second.\n\nAnother para."}})
    assert route_after_draft_review({**state, **upd}) == "check_citations"
    new = upd["drafts"]["s2"]
    assert [s.text for s in new.paragraphs[0].sentences] == ["New first.", "New second."]
    assert all(s.sources == ["user"] for p in new.paragraphs for s in p.sentences)
    assert [d.section_id for d in upd["draft"]] == ["s1", "s2"] and upd["draft"][1] == new


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "nope"},
        {"action": "edit"},
        {"action": "edit", "edited": {"s9": "text"}},
        {"action": "edit", "edited": {"s1": 5}},
        {"action": "edit", "edited": {"s1": "   "}},
    ],
)
def test_bad_payloads_reask_with_error(state, deps, payload):
    upd = review(state, deps, payload)
    assert route_after_draft_review({**state, **upd}) == "draft_review" and upd["review_error"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_node_review.py -v`
Expected: FAIL — `ModuleNotFoundError: vault_to_voice.graph.nodes.review`.

- [ ] **Step 3: Implement**

`src/vault_to_voice/graph/nodes/review.py`:
```python
from collections import defaultdict

from langgraph.types import interrupt
from pydantic import ValidationError

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.render import render_markdown
from vault_to_voice.graph.state import Paragraph, PostState, ReviewDecision, SectionDraft, Sentence
from vault_to_voice.index.chunking import split_paragraphs
from vault_to_voice.style.stats import split_sentences


def route_after_checks(state: PostState) -> str:
    if state["citation_report"].flags and not state.get("auto_revised"):
        return "prepare_auto_revise"
    return "draft_review"


def prepare_auto_revise(state: PostState) -> dict:
    problems: dict[str, list[str]] = defaultdict(list)
    for flag in state["citation_report"].flags:
        problems[flag.section_id].append(f'- "{flag.text}" ({flag.reason.replace("_", " ")})')
    feedback = {
        sid: "These sentences are not supported by their cited sources. Cite the right note or "
        "answer ids, rewrite them so the sources support them, or drop them:\n" + "\n".join(lines)
        for sid, lines in problems.items()
    }
    return {
        "revise_ids": sorted(problems),
        "section_feedback": feedback,
        "feedback": "",
        "auto_revised": True,
    }


def _retry(message: str) -> dict:
    return {"last_action": "retry", "review_error": message}


def _user_section(section_id: str, text: str) -> SectionDraft:
    paragraphs = [
        Paragraph(sentences=[Sentence(text=s, sources=["user"]) for s in split_sentences(p)])
        for p in split_paragraphs(text)
    ]
    return SectionDraft(section_id=section_id, paragraphs=paragraphs)


def draft_review(state: PostState, deps: GraphDeps) -> dict:
    rounds = state.get("user_rounds") or {"outline": 0, "draft": 0}
    left = deps.settings.max_review_rounds - rounds["draft"]
    report = state["citation_report"]
    payload = interrupt(
        {
            "type": "draft_review",
            "markdown": render_markdown(state["outline"], state["draft"]),
            "draft": [d.model_dump() for d in state["draft"]],
            "flags": [f.model_dump() for f in report.flags],
            "coverage": report.coverage,
            "rounds_left": left,
            "error": state.get("review_error"),
        }
    )
    try:
        decision = ReviewDecision.model_validate(payload)
    except ValidationError as exc:
        return _retry(f"Invalid decision: {exc.errors()[0]['msg']}")
    if decision.action == "approve":
        return {"last_action": "publish", "review_error": None}
    if decision.action == "revise":
        if left <= 0:
            return _retry("Revision limit reached — approve or edit the draft.")
        return {
            "last_action": "dispatch",
            "review_error": None,
            "revise_ids": [s.id for s in state["outline"].sections],
            "feedback": decision.feedback,
            "section_feedback": {},
            "user_rounds": {**rounds, "draft": rounds["draft"] + 1},
        }
    edited = decision.edited or {}
    known = {d.section_id for d in state["draft"]}
    if (
        not edited
        or not set(edited) <= known
        or not all(isinstance(v, str) and v.strip() for v in edited.values())
    ):
        return _retry("Edit needs {section_id: text} for existing sections, with non-empty text.")
    new = {sid: _user_section(sid, text) for sid, text in edited.items()}
    return {
        "drafts": new,
        "draft": [new.get(d.section_id, d) for d in state["draft"]],
        "last_action": "check",
        "review_error": None,
    }


def route_after_draft_review(state: PostState) -> str:
    return {
        "publish": "publish",
        "dispatch": "dispatch",
        "check": "check_citations",
    }.get(state["last_action"], "draft_review")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v` → all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/vault_to_voice/graph/nodes/review.py tests/test_node_review.py
git commit -m "feat: one-shot auto-revise and human draft review"
```

---

### Task 10: Publisher and publish node

**Files:**
- Create: `src/vault_to_voice/publish.py`, `src/vault_to_voice/graph/nodes/publish.py`
- Modify: `src/vault_to_voice/graph/deps.py` (add `publisher`), `tests/fakes.py` (`make_deps` passes an `OutboxPublisher`)
- Test: `tests/test_publish.py`

**Interfaces:**
- Consumes: `render_markdown` (Task 7), `save_answers` (Task 2), `Manifest.set_idea_status` (Task 1), `slugify` (vault/reader.py).
- Produces: `RenderedPost(slug: str, date: date, markdown: str, sources: dict)`; `Publisher` protocol `publish(post) -> Path`; `OutboxPublisher(outbox_dir)` writing `<date>-<slug>.md` + `<date>-<slug>.sources.json`, suffixing `-2`, `-3`… instead of overwriting; `GraphDeps.publisher: Publisher | None = None`; `publish(state, deps) -> {"final_path": str}` (also saves Q&A to the Store and marks the idea `done`).

- [ ] **Step 1: Write failing tests**

In `tests/fakes.py`, add `from vault_to_voice.publish import OutboxPublisher` and pass `publisher=OutboxPublisher(tmp_path / "outbox"),` to the `GraphDeps(...)` call in `make_deps`.

`tests/test_publish.py`:
```python
import datetime as dt
import json
from pathlib import Path

from tests.fakes import FakeLLMClient, make_deps, seed_idea
from vault_to_voice.graph.memory import past_answers
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.nodes.publish import publish
from vault_to_voice.graph.state import (
    QA,
    CitationReport,
    Outline,
    Paragraph,
    Section,
    SectionDraft,
    Sentence,
)
from vault_to_voice.publish import OutboxPublisher, RenderedPost


def test_outbox_publisher_writes_both_files_and_never_overwrites(tmp_path):
    publisher = OutboxPublisher(tmp_path / "out")
    post = RenderedPost(slug="fair-bosses", date=dt.date(2026, 10, 10), markdown="# x\n",
                        sources={"a": 1})
    first, second = publisher.publish(post), publisher.publish(post)
    assert first.name == "2026-10-10-fair-bosses.md"
    assert second.name == "2026-10-10-fair-bosses-2.md"
    assert json.loads(first.with_suffix(".sources.json").read_text(encoding="utf-8")) == {"a": 1}


def test_publish_node(tmp_path, vault_dir):
    deps = make_deps(tmp_path, FakeLLMClient({}))
    deps.today = lambda: dt.date(2026, 10, 10)
    idea = seed_idea(deps, vault_dir)
    outline = Outline(title="Fair Bosses", sections=[Section(id="s1", heading="Tells", point="p")])
    sentences = [Sentence(text="Every attack has a tell.", sources=["n0"]),
                 Sentence(text="So.", kind="connective")]
    state = {
        **load_context({"idea": idea}, deps),
        "idea": idea,
        "outline": outline,
        "draft": [SectionDraft(section_id="s1", paragraphs=[Paragraph(sentences=sentences)])],
        "qa": [QA(qa_id="qa:1", question="q", answer="a")],
        "citation_report": CitationReport(n_claims=1, n_supported=1),
    }
    path = Path(publish(state, deps)["final_path"])
    assert path.name == "2026-10-10-fair-bosses.md"
    text = path.read_text(encoding="utf-8")
    _, front, body = text.split("---\n", 2)
    assert 'title: "Fair Bosses"' in front and f"idea_id: {idea.idea_id}" in front
    assert body == "\n# Fair Bosses\n\n## Tells\n\nEvery attack has a tell. So.\n"
    side = json.loads(path.with_suffix(".sources.json").read_text(encoding="utf-8"))
    assert side["paragraphs"][0]["sources"] == ["n0"] and side["citation_coverage"] == 1.0
    assert side["interview"][0]["answer"] == "a"
    assert side["source_details"]["n0"] == {"title": "Note n0", "date": "2026-01-01",
                                            "work_id": None}
    assert deps.manifest.get_idea(idea.idea_id).status == "done"
    assert [q.answer for q in past_answers(deps.store)] == ["a"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_publish.py -v`
Expected: FAIL — `ModuleNotFoundError: vault_to_voice.publish` (collection error in `tests/fakes.py`).

- [ ] **Step 3: Implement**

`src/vault_to_voice/publish.py`:
```python
import datetime as dt
import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class RenderedPost(BaseModel):
    slug: str
    date: dt.date
    markdown: str
    sources: dict


class Publisher(Protocol):
    def publish(self, post: RenderedPost) -> Path: ...


class OutboxPublisher:
    """Writes posts to a local folder. Later publishers (dev.to, Hashnode) share the protocol."""

    def __init__(self, outbox_dir: Path):
        self.outbox_dir = Path(outbox_dir)

    def publish(self, post: RenderedPost) -> Path:
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{post.date.isoformat()}-{post.slug}"
        path = self.outbox_dir / f"{stem}.md"
        n = 2
        while path.exists():
            path = self.outbox_dir / f"{stem}-{n}.md"
            n += 1
        path.write_text(post.markdown, encoding="utf-8")
        path.with_suffix(".sources.json").write_text(
            json.dumps(post.sources, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path
```

`src/vault_to_voice/graph/deps.py` — add `from vault_to_voice.publish import Publisher` and, between `store` and `today`, the field:
```python
    publisher: Publisher | None = None
```

`src/vault_to_voice/graph/nodes/publish.py`:
```python
import json

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.memory import save_answers
from vault_to_voice.graph.render import render_markdown
from vault_to_voice.graph.state import PostState
from vault_to_voice.publish import RenderedPost
from vault_to_voice.vault.reader import slugify


def publish(state: PostState, deps: GraphDeps) -> dict:
    if deps.publisher is None:
        raise RuntimeError("GraphDeps.publisher is not configured")
    idea, outline, draft = state["idea"], state["outline"], state["draft"]
    report = state["citation_report"]
    qa = state.get("qa", [])
    today = deps.today()
    front = "\n".join(
        [
            "---",
            f"title: {json.dumps(outline.title)}",
            f"date: {today.isoformat()}",
            f"idea_id: {idea.idea_id}",
            f"theme: {json.dumps(idea.theme)}",
            f"notes: {json.dumps(idea.note_ids)}",
            "---",
            "",
        ]
    )
    headings = {s.id: s.heading for s in outline.sections}
    sources = {
        "idea_id": idea.idea_id,
        "title": outline.title,
        "date": today.isoformat(),
        "citation_coverage": report.coverage,
        "flags": [f.model_dump() for f in report.flags],
        "paragraphs": [
            {
                "section_id": section.section_id,
                "heading": headings.get(section.section_id),
                "text": " ".join(s.text for s in p.sentences),
                "sources": sorted({src for s in p.sentences for src in s.sources}),
                "sentences": [s.model_dump() for s in p.sentences],
            }
            for section in draft
            for p in section.paragraphs
        ],
        "interview": [q.model_dump() for q in qa],
        "source_details": {
            n.note_id: {"title": n.title, "date": n.date, "work_id": n.work_id}
            for n in state["notes"]
        },
    }
    post = RenderedPost(
        slug=slugify(outline.title) or idea.idea_id,
        date=today,
        markdown=front + "\n" + render_markdown(outline, draft),
        sources=sources,
    )
    path = deps.publisher.publish(post)
    save_answers(deps.store, idea.idea_id, qa)
    deps.manifest.set_idea_status(idea.idea_id, "done")
    return {"final_path": str(path)}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v` → all PASS; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/vault_to_voice/publish.py src/vault_to_voice/graph/nodes/publish.py src/vault_to_voice/graph/deps.py tests/fakes.py tests/test_publish.py
git commit -m "feat: outbox publisher and publish node with sources sidecar"
```

---

### Task 11: Full graph wiring and end-to-end tests

**Files:**
- Modify: `src/vault_to_voice/graph/builder.py`, `tests/fakes.py` (shared graph responses)
- Test: `tests/test_graph_e2e.py`

**Interfaces:**
- Consumes: everything above.
- Produces: final `build_post_graph` topology:
  `START → load_context → interview → make_outline → outline_review →(route) make_outline | outline_review | dispatch`;
  `dispatch →(fan_out_sections: Send×N) draft_section → merge_draft → check_citations →(route_after_checks) prepare_auto_revise | draft_review`;
  `prepare_auto_revise → dispatch`; `draft_review →(route) dispatch | check_citations | draft_review | publish`; `publish → END`.
  tests/fakes.py: `GOOD_SENTENCE` (= first paragraph of `NOTES["n0"]`), `good_section(prompt) -> SectionText`, `graph_responses(section_fn=good_section) -> dict` (Question, CoverageCheck(enough=True), two-section Outline, SectionText).

- [ ] **Step 1: Write failing tests**

Append to `tests/fakes.py`:
```python
from vault_to_voice.graph.state import (
    CoverageCheck,
    Outline,
    Paragraph,
    Question,
    Section,
    SectionText,
    Sentence,
)

GOOD_SENTENCE = NOTES["n0"].split("\n\n")[0]


def good_section(_prompt) -> SectionText:
    return SectionText(
        paragraphs=[
            Paragraph(
                sentences=[
                    Sentence(text=GOOD_SENTENCE, sources=["n0"]),
                    Sentence(text="That is the deal.", kind="connective"),
                ]
            )
        ]
    )


def graph_responses(section_fn=good_section) -> dict:
    return {
        Question: lambda _: Question(question="What made it fair?", grounding_note_ids=["n0"]),
        CoverageCheck: lambda _: CoverageCheck(enough=True),
        Outline: lambda _: Outline(
            title="Fair Bosses",
            sections=[
                Section(id="a", heading="Tells", point="Tells make it fair.",
                        planned_sources=["n0"]),
                Section(id="b", heading="Learning", point="Death teaches.",
                        planned_sources=["qa:1"]),
            ],
        ),
        SectionText: section_fn,
    }
```

`tests/test_graph_e2e.py`:
```python
import datetime as dt
import json
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.fakes import FakeLLMClient, graph_responses, make_deps, seed_idea
from vault_to_voice.graph.builder import build_post_graph
from vault_to_voice.graph.runner import run_until_interrupt, start_input, thread_config
from vault_to_voice.graph.state import Outline, Paragraph, SectionText, Sentence

ANSWERS = ["The tells.", "Twenty deaths.", "Patience."]
APPROVE = {"action": "approve"}


def bad_section(_prompt):
    sentence = Sentence(text="Purple bananas orbit distant moons quietly.", sources=["n0"])
    return SectionText(paragraphs=[Paragraph(sentences=[sentence])])


def make_graph(tmp_path, vault_dir, **responses):
    fake = FakeLLMClient(graph_responses(**responses))
    deps = make_deps(tmp_path, fake)
    deps.today = lambda: dt.date(2026, 10, 10)
    idea = seed_idea(deps, vault_dir)
    return build_post_graph(deps, checkpointer=InMemorySaver()), deps, idea, fake


def drive(graph, idea, decisions):
    """Start the thread, feed each decision to the next interrupt; return (seen, last, values)."""
    cfg = thread_config(idea.idea_id)
    seen = []
    value = run_until_interrupt(graph, cfg, start_input(idea))
    for decision in decisions:
        if value is None:
            break
        seen.append(value)
        value = run_until_interrupt(graph, cfg, Command(resume=decision))
    return seen, value, graph.get_state(cfg).values


def count(fake, schema):
    return sum(1 for s, _ in fake.calls if s is schema)


def test_happy_path_publishes(tmp_path, vault_dir):
    graph, deps, idea, fake = make_graph(tmp_path, vault_dir)
    seen, last, values = drive(graph, idea, [*ANSWERS, APPROVE, APPROVE])
    assert [s["type"] for s in seen] == ["interview_question"] * 3 + ["outline_review",
                                                                     "draft_review"]
    assert seen[-1]["flags"] == [] and seen[-1]["coverage"] == 1.0
    assert last is None and Path(values["final_path"]).exists()
    assert count(fake, SectionText) == 2  # one parallel draft per section
    assert deps.manifest.get_idea(idea.idea_id).status == "done"


def test_auto_revise_runs_once_then_shows_flags(tmp_path, vault_dir):
    graph, _, idea, fake = make_graph(tmp_path, vault_dir, section_fn=bad_section)
    _, last, _ = drive(graph, idea, [*ANSWERS, APPROVE])
    assert last["type"] == "draft_review" and len(last["flags"]) == 2
    assert count(fake, SectionText) == 4  # 2 drafts + 2 auto-revisions, never a second round


def test_user_revise_redrafts_and_is_capped(tmp_path, vault_dir):
    graph, _, idea, fake = make_graph(tmp_path, vault_dir)
    revise = {"action": "revise", "feedback": "warmer"}
    _, last, _ = drive(graph, idea, [*ANSWERS, APPROVE, revise, revise, revise, revise])
    assert last["type"] == "draft_review" and last["rounds_left"] == 0
    assert "limit" in last["error"]
    assert count(fake, SectionText) == 2 + 3 * 2


def test_outline_revise_then_draft_edit_then_publish(tmp_path, vault_dir):
    graph, _, idea, fake = make_graph(tmp_path, vault_dir)
    decisions = [
        *ANSWERS,
        {"action": "revise", "feedback": "tighter"},
        APPROVE,
        {"action": "edit", "edited": {"s2": "My own words here."}},
        APPROVE,
    ]
    seen, last, values = drive(graph, idea, decisions)
    assert count(fake, Outline) == 2 and last is None
    assert [s["type"] for s in seen][-2:] == ["draft_review", "draft_review"]
    side = json.loads(Path(values["final_path"]).with_suffix(".sources.json").read_text(
        encoding="utf-8"))
    assert side["paragraphs"][-1]["sources"] == ["user"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_graph_e2e.py -v`
Expected: FAIL — runs end after outline approval (`dispatch` still maps to `END`), so draft-review assertions fail.

- [ ] **Step 3: Implement**

Replace `src/vault_to_voice/graph/builder.py` with:
```python
from functools import partial

from langgraph.graph import END, START, StateGraph

from vault_to_voice.graph.deps import LLM_RETRY, GraphDeps
from vault_to_voice.graph.nodes.citations import check_citations
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.nodes.draft import draft_section, fan_out_sections, merge_draft
from vault_to_voice.graph.nodes.interview import build_interview_graph
from vault_to_voice.graph.nodes.outline import (
    make_outline,
    outline_review,
    route_after_outline_review,
)
from vault_to_voice.graph.nodes.publish import publish
from vault_to_voice.graph.nodes.review import (
    draft_review,
    prepare_auto_revise,
    route_after_checks,
    route_after_draft_review,
)
from vault_to_voice.graph.state import PostState


def _dispatch(state: PostState) -> dict:
    """Join point before the Send fan-out (outline approval, auto-revise and user revise)."""
    return {"review_error": None}


def build_post_graph(deps: GraphDeps, *, checkpointer=None):
    g = StateGraph(PostState)
    g.add_node("load_context", partial(load_context, deps=deps))
    g.add_node("interview", build_interview_graph(deps).compile())  # subgraph as a node
    g.add_node("make_outline", partial(make_outline, deps=deps), retry_policy=LLM_RETRY)
    g.add_node("outline_review", partial(outline_review, deps=deps))
    g.add_node("dispatch", _dispatch)
    g.add_node("draft_section", partial(draft_section, deps=deps), retry_policy=LLM_RETRY)
    g.add_node("merge_draft", merge_draft)
    g.add_node("check_citations", partial(check_citations, deps=deps), retry_policy=LLM_RETRY)
    g.add_node("prepare_auto_revise", prepare_auto_revise)
    g.add_node("draft_review", partial(draft_review, deps=deps))
    g.add_node("publish", partial(publish, deps=deps))

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "interview")
    g.add_edge("interview", "make_outline")
    g.add_edge("make_outline", "outline_review")
    g.add_conditional_edges(
        "outline_review", route_after_outline_review, ["make_outline", "outline_review", "dispatch"]
    )
    g.add_conditional_edges("dispatch", fan_out_sections, ["draft_section"])
    g.add_edge("draft_section", "merge_draft")
    g.add_edge("merge_draft", "check_citations")
    g.add_conditional_edges(
        "check_citations", route_after_checks, ["prepare_auto_revise", "draft_review"]
    )
    g.add_edge("prepare_auto_revise", "dispatch")
    g.add_conditional_edges(
        "draft_review",
        route_after_draft_review,
        ["dispatch", "check_citations", "draft_review", "publish"],
    )
    g.add_edge("publish", END)
    return g.compile(checkpointer=checkpointer, store=deps.store)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v` → all PASS (including Task 6's builder tests); ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/vault_to_voice/graph/builder.py tests/fakes.py tests/test_graph_e2e.py
git commit -m "feat: full post graph with fan-out, citation checks, auto-revise and publish"
```

---

### Task 12: `v2v write` CLI and sample-vault integration test

**Files:**
- Modify: `src/vault_to_voice/cli.py`, `tests/test_cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `v2v write IDEA_ID` — builds deps + SQLite checkpointer/Store; starts the thread, or resumes it if it has a pending step ("resuming saved session"), or reports "already published: <path>"; prompts by interrupt type (interview answer; outline/draft: `approve`/`revise` + feedback); prints `published: <path>`. Unknown idea → exit 1.

- [ ] **Step 1: Write failing tests**

In `tests/test_cli.py`:
- extend `fake_llm()` so its dict also contains the graph responses: `return FakeLLMClient({..existing three.., **graph_responses()})` (import `graph_responses` from `tests.fakes`);
- in the `env` fixture add `monkeypatch.setenv("OUTBOX_DIR", str(tmp_path / "outbox"))`;
- append:

```python
def _ready_idea(vault_dir, write_note):
    from tests.fakes import NOTES
    from vault_to_voice.config import Settings
    from vault_to_voice.index.manifest import Manifest

    for nid, body in NOTES.items():
        write_note(f"{nid}.md", note_text(nid, body))
    assert runner.invoke(cli.app, ["ingest", "--vault", str(vault_dir)]).exit_code == 0
    assert runner.invoke(cli.app, ["ideas"]).exit_code == 0
    return Manifest(Settings().manifest_path).list_ideas()[0].idea_id


def test_write_full_session(env, vault_dir, write_note):
    idea_id = _ready_idea(vault_dir, write_note)
    r = runner.invoke(cli.app, ["write", idea_id], input="a1\na2\na3\napprove\napprove\n")
    assert r.exit_code == 0, r.output
    assert "Q1/5" in r.output and "Outline: Fair Bosses" in r.output and "published:" in r.output
    assert list((env / "outbox").glob("*.md"))
    again = runner.invoke(cli.app, ["write", idea_id])
    assert again.exit_code == 0 and "already published" in again.output


def test_write_resumes_after_interruption(env, vault_dir, write_note):
    idea_id = _ready_idea(vault_dir, write_note)
    first = runner.invoke(cli.app, ["write", idea_id], input="a1\n")  # input ends at question 2
    assert first.exit_code != 0
    second = runner.invoke(cli.app, ["write", idea_id], input="a2\na3\napprove\napprove\n")
    assert second.exit_code == 0, second.output
    assert "resuming saved session" in second.output and "Q2/5" in second.output
    assert "published:" in second.output


def test_write_unknown_idea(env):
    r = runner.invoke(cli.app, ["write", "idea_nope"])
    assert r.exit_code == 1 and "unknown idea" in r.output


@pytest.mark.integration
def test_write_on_sample_vault_with_minilm(monkeypatch, tmp_path):
    import json
    import re

    from vault_to_voice.config import Settings
    from vault_to_voice.graph.state import Paragraph, SectionText, Sentence
    from vault_to_voice.index.manifest import Manifest

    def cite_first_note(prompt):
        note_id, text = re.search(r'<note id="([^"]+)">\n[^\n]*\n([^\n]+)', prompt).groups()
        return SectionText(paragraphs=[Paragraph(sentences=[Sentence(text=text,
                                                                     sources=[note_id])])])

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUTBOX_DIR", str(tmp_path / "outbox"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    llm = fake_llm()
    llm.responses[SectionText] = cite_first_note
    monkeypatch.setattr(cli, "make_llm", lambda settings: llm)
    assert runner.invoke(cli.app, ["ingest"]).exit_code == 0
    assert runner.invoke(cli.app, ["ideas"]).exit_code == 0
    idea_id = Manifest(Settings().manifest_path).list_ideas()[0].idea_id
    r = runner.invoke(cli.app, ["write", idea_id], input="a1\na2\na3\napprove\napprove\n")
    assert r.exit_code == 0, r.output
    post = next((tmp_path / "outbox").glob("*.md"))
    side = json.loads(post.with_suffix(".sources.json").read_text(encoding="utf-8"))
    assert side["citation_coverage"] == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `No such command 'write'`.

- [ ] **Step 3: Implement**

`src/vault_to_voice/cli.py` — add imports:
```python
import click
from langgraph.types import Command

from vault_to_voice.graph.builder import build_post_graph
from vault_to_voice.graph.checkpoint import make_checkpointer, make_store
from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.runner import (
    pending_interrupt,
    run_until_interrupt,
    start_input,
    thread_config,
)
from vault_to_voice.publish import OutboxPublisher
```
(`make_store` is already imported since Task 2 — keep one import.) Append:
```python
def _ask_user(value: dict):
    kind = value["type"]
    if kind == "interview_question":
        typer.echo(f"\nQ{value['number']}/{value['max']}: {value['question']}")
        return typer.prompt("answer", default="", show_default=False)
    if value.get("error"):
        typer.echo(f"! {value['error']}")
    if kind == "outline_review":
        outline = value["outline"]
        typer.echo(f"\nOutline: {outline['title']}")
        for section in outline["sections"]:
            typer.echo(f"  {section['id']}. {section['heading']} - {section['point']}")
    else:
        typer.echo("\n" + value["markdown"])
        typer.echo(f"citation coverage: {value['coverage']:.0%}")
        for flag in value["flags"]:
            typer.echo(f"  ! [{flag['reason']}] {flag['text']}")
    action = typer.prompt(
        f"approve / revise ({value['rounds_left']} revisions left)",
        type=click.Choice(["approve", "revise"]),
        default="approve",
    )
    if action == "revise":
        return {"action": "revise", "feedback": typer.prompt("feedback")}
    return {"action": "approve"}


@app.command()
def write(idea_id: str = typer.Argument(..., help="Idea id from `v2v ideas`.")):
    """Write a post for an idea: interview, outline, draft, publish. Safe to stop and re-run."""
    s = _settings()
    manifest = Manifest(s.manifest_path)
    idea = manifest.get_idea(idea_id)
    if idea is None:
        typer.echo(f"unknown idea {idea_id}; run `v2v ideas` to list ideas")
        raise typer.Exit(1)
    embedder = make_embedder(s)
    deps = GraphDeps(
        llm=make_llm(s),
        manifest=manifest,
        index=FaissStore(s.faiss_path, embedder.dim),
        embedder=embedder,
        settings=s,
        store=make_store(s.store_path),
        publisher=OutboxPublisher(s.outbox_dir),
    )
    graph = build_post_graph(deps, checkpointer=make_checkpointer(s.checkpoint_path))
    config = thread_config(idea_id)
    snapshot = graph.get_state(config)
    if snapshot.values.get("final_path") and not snapshot.next:
        typer.echo(f"already published: {snapshot.values['final_path']}")
        return
    if snapshot.next:
        typer.echo("resuming saved session")
        value = pending_interrupt(graph, config)
        if value is None:  # stopped mid-step (crash): re-run that step
            value = run_until_interrupt(graph, config, None)
    else:
        value = run_until_interrupt(graph, config, start_input(idea))
    while value is not None:
        value = run_until_interrupt(graph, config, Command(resume=_ask_user(value)))
    typer.echo(f"published: {graph.get_state(config).values['final_path']}")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -v` → all PASS; ruff clean.
Then: `uv run pytest -m integration -v` → PASS (MiniLM, offline fake LLM).
Manual smoke (needs `OPENAI_API_KEY`): `uv run v2v ingest && uv run v2v style && uv run v2v ideas`, then `uv run v2v write <idea_id>`; press Ctrl+C during the interview, re-run the same command and confirm it resumes; finish and open the file in `outbox/`.

- [ ] **Step 5: Commit and close M2**

```bash
git add src/vault_to_voice/cli.py tests/test_cli.py
git commit -m "feat: v2v write CLI driving the resumable post graph"
```
Update `PROGRESS.md` (M2 done; next S7 M3 serving), commit `docs: update progress after M2`.
