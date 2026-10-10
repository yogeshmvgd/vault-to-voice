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
