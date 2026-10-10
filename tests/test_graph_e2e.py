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
    assert [s["type"] for s in seen] == ["interview_question"] * 3 + [
        "outline_review",
        "draft_review",
    ]
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
    side = json.loads(
        Path(values["final_path"]).with_suffix(".sources.json").read_text(encoding="utf-8")
    )
    assert side["paragraphs"][-1]["sources"] == ["user"]
