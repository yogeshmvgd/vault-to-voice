import dataclasses
import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

from langgraph.types import Command
from pydantic import BaseModel

from vault_to_voice import setup_actions
from vault_to_voice.api.events import stream_run
from vault_to_voice.graph.builder import build_post_graph
from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.runner import start_input, thread_config
from vault_to_voice.ideas.detector import Idea, IndexOutOfSyncError

ThreadStatus = Literal["new", "waiting", "stalled", "published"]


class ServiceError(Exception):
    status_code = 400


class NotFound(ServiceError):
    status_code = 404


class Conflict(ServiceError):
    status_code = 409


class BadRequest(ServiceError):
    status_code = 400


class ActionFailed(ServiceError):
    status_code = 500


class ThreadView(BaseModel):
    """What a client needs to render one post thread (thread_id = idea_id)."""

    thread_id: str
    title: str
    status: ThreadStatus
    pending: dict | None = None
    outline: dict | None = None
    qa: list[dict] = []
    final_path: str | None = None
    style_score: float | None = None  # latest critic run (None before the first draft)
    critic_notes: list[dict] = []
    warnings: list[str] = []  # input guardrail: instruction-like text found in the notes


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
        self._maintenance = threading.Lock()  # one setup action at a time, never during runs

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
        critic = values.get("critic_report")
        return ThreadView(
            thread_id=thread_id,
            title=idea.title,
            status=status,
            pending=pending,
            outline=outline.model_dump() if outline is not None else None,
            qa=[q.model_dump() for q in qa],
            final_path=values.get("final_path"),
            style_score=critic.style_score if critic is not None else None,
            critic_notes=[n.model_dump() for n in critic.notes] if critic is not None else [],
            warnings=values.get("guard_warnings", []),
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
            if self._maintenance.locked():
                raise Conflict("a setup action is running; wait for it to finish")
            status = self.thread(thread_id).status
            if status != needs:
                raise Conflict(f"{_NEEDS[needs]} (status: {status})")
        except BaseException:
            lock.release()
            raise
        return _LockedStream(lock, stream_run(self.graph, thread_config(thread_id), value))

    # ---- setup actions (same code path as the CLI) ------------------------------------------
    def config(self) -> dict:
        """Non-secret settings for display. Never add keys here."""
        s = self.deps.settings
        return {
            "vault_path": str(s.vault_path),
            "vault_format": s.vault_format,
            "llm_model": s.llm_model,
            "embedding_provider": s.embedding_provider,
            "data_dir": str(s.data_dir),
            "outbox_dir": str(s.outbox_dir),
        }

    def _setup(self, name: str, action: Callable[[], Any]) -> Any:
        if not self._maintenance.acquire(blocking=False):
            raise Conflict("another setup action is running; wait for it to finish")
        try:
            with self._locks_guard:
                running = any(lock.locked() for lock in self._locks.values())
            if running:
                raise Conflict("a post is being written; wait for that step to finish")
            try:
                return action()
            except ServiceError:
                raise
            except IndexOutOfSyncError as exc:
                raise Conflict(str(exc)) from exc
            except setup_actions.NoNotesError as exc:
                raise BadRequest(str(exc)) from exc
            except Exception as exc:  # shown to the user as a message, never a traceback
                raise ActionFailed(f"{name} failed: {exc}") from exc
        finally:
            self._maintenance.release()

    def ingest(self, *, no_score: bool = False) -> dict:
        d = self.deps

        def run() -> dict:
            result = setup_actions.ingest_vault(
                d.settings, d.manifest, d.index, d.embedder, None if no_score else d.llm
            )
            out = dataclasses.asdict(result.report)
            out["stats"] = dataclasses.asdict(result.stats) if result.stats else None
            return out

        return self._setup("index vault", run)

    def build_style(self) -> dict:
        d = self.deps

        def run() -> dict:
            profile = setup_actions.build_style(d.settings, d.llm, d.store)
            return {
                "version": profile.version,
                "path": str(d.settings.style_profile_path),
                "tone": profile.tone.tone,
                "voice_notes": profile.tone.voice_notes,
            }

        return self._setup("build style profile", run)

    def detect_ideas(self) -> list[Idea]:
        d = self.deps
        return self._setup(
            "find ideas", lambda: setup_actions.find_ideas(d.settings, d.manifest, d.index, d.llm)
        )
