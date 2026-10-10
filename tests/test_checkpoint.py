import datetime as dt
import logging
import types
import typing

from langgraph.checkpoint.serde import jsonplus
from langgraph.types import Command
from pydantic import BaseModel

from tests.fakes import FakeLLMClient, graph_responses, make_deps, seed_idea
from vault_to_voice.graph.builder import build_post_graph
from vault_to_voice.graph.checkpoint import (
    CHECKPOINT_TYPES,
    checkpoint_serde,
    make_checkpointer,
    memory_checkpointer,
)
from vault_to_voice.graph.runner import run_until_interrupt, start_input, thread_config
from vault_to_voice.graph.state import CitationReport, Flag, PostState, SectionDraft

SERDE_LOGGER = "langgraph.checkpoint.serde.jsonplus"


def _models(tp, seen: set[type]) -> None:
    """Collect every pydantic model reachable from a type annotation."""
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        if tp in seen:
            return
        seen.add(tp)
        for field in tp.model_fields.values():
            _models(field.annotation, seen)
        return
    for arg in typing.get_args(tp):
        _models(arg, seen)
    if isinstance(tp, types.UnionType):
        for arg in tp.__args__:
            _models(arg, seen)


def test_every_state_model_is_on_the_allowlist():
    found: set[type] = set()
    for hint in typing.get_type_hints(PostState, include_extras=True).values():
        _models(hint, found)
    missing = {f"{t.__module__}.{t.__name__}" for t in found - set(CHECKPOINT_TYPES)}
    assert not missing, f"add to CHECKPOINT_TYPES: {sorted(missing)}"


def test_serde_round_trips_models_without_warnings(caplog):
    report = CitationReport(
        n_claims=1,
        n_supported=0,
        flags=[Flag(section_id="s1", paragraph=0, sentence=0, text="x", reason="unsupported")],
    )
    serde = checkpoint_serde()
    with caplog.at_level(logging.WARNING, logger=SERDE_LOGGER):
        back = serde.loads_typed(serde.dumps_typed(report))
    assert back == report and isinstance(back, CitationReport)
    assert not [r for r in caplog.records if r.name == SERDE_LOGGER]


def test_sqlite_checkpoints_reload_without_unregistered_type_warnings(
    tmp_path, vault_dir, caplog, monkeypatch
):
    # LangGraph warns once per type per process; start from a clean slate.
    monkeypatch.setattr(jsonplus, "_warned_unregistered_types", set())
    deps = make_deps(tmp_path, FakeLLMClient(graph_responses()))
    deps.today = lambda: dt.date(2026, 10, 10)
    idea = seed_idea(deps, vault_dir)
    db = tmp_path / "checkpoints.db"
    graph = build_post_graph(deps, checkpointer=make_checkpointer(db))
    cfg = thread_config(idea.idea_id)
    value = run_until_interrupt(graph, cfg, start_input(idea))
    for decision in ["The tells.", "Twenty deaths.", "Patience.", {"action": "approve"}]:
        value = run_until_interrupt(graph, cfg, Command(resume=decision))
    assert value["type"] == "draft_review"

    reopened = build_post_graph(deps, checkpointer=make_checkpointer(db))  # a fresh process
    with caplog.at_level(logging.WARNING, logger=SERDE_LOGGER):
        values = reopened.get_state(cfg).values
    assert not [r for r in caplog.records if r.name == SERDE_LOGGER], caplog.text
    assert isinstance(values["draft"][0], SectionDraft)
    assert isinstance(values["citation_report"], CitationReport)


def test_memory_checkpointer_uses_the_allowlist(tmp_path, vault_dir, caplog, monkeypatch):
    monkeypatch.setattr(jsonplus, "_warned_unregistered_types", set())
    deps = make_deps(tmp_path, FakeLLMClient(graph_responses()))
    idea = seed_idea(deps, vault_dir)
    graph = build_post_graph(deps, checkpointer=memory_checkpointer())
    cfg = thread_config(idea.idea_id)
    with caplog.at_level(logging.WARNING, logger=SERDE_LOGGER):
        run_until_interrupt(graph, cfg, start_input(idea))
        run_until_interrupt(graph, cfg, Command(resume="The tells."))
        graph.get_state(cfg)
    assert not [r for r in caplog.records if r.name == SERDE_LOGGER], caplog.text
