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
            Section(
                id="a",
                heading="Tells",
                point="Tells make it fair.",
                planned_sources=["n0", "ghost"],
            ),
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
