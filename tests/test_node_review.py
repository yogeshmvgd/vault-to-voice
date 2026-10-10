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
    flag = Flag(
        section_id="s2", paragraph=0, sentence=0, text="Bad claim.", reason="unsupported", score=0.1
    )
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
        sections=[
            Section(id="s1", heading="Tells", point="p"),
            Section(id="s2", heading="Learning", point="q"),
        ],
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
        "prepare_auto_revise"
    )
    assert route_after_checks({"citation_report": flagged(), "auto_revised": True}) == (
        "draft_review"
    )
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
    upd = review(
        state, deps, {"action": "edit", "edited": {"s2": "New first. New second.\n\nAnother para."}}
    )
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
