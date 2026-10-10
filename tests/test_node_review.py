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
    CriticNote,
    CriticReport,
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


def major(section_id="s1", note="Too formal for this author."):
    return CriticNote(section_id=section_id, kind="voice", severity="major", note=note)


def test_route_after_checks_also_revises_for_the_critic():
    clean = CitationReport(n_claims=1, n_supported=1)
    minor = CriticReport(notes=[CriticNote(section_id="s1", kind="clarity", note="ok")])
    for critic, expected in [
        (CriticReport(notes=[major()]), "prepare_auto_revise"),
        (CriticReport(style_score=0.1, style_ok=False), "prepare_auto_revise"),
        (minor, "draft_review"),
    ]:
        state = {"citation_report": clean, "critic_report": critic, "auto_revised": False}
        assert route_after_checks(state) == expected
    done = {"citation_report": clean, "critic_report": CriticReport(notes=[major()])}
    assert route_after_checks({**done, "auto_revised": True}) == "draft_review"


def test_prepare_auto_revise_merges_citation_and_critic_feedback(state):
    critic = CriticReport(style_score=0.3, style_ok=False, notes=[major("s2", "Say more.")])
    upd = prepare_auto_revise({**state, "critic_report": critic})
    assert upd["revise_ids"] == ["s1", "s2"]  # low style score: every section is redrafted
    s2 = upd["section_feedback"]["s2"]
    assert "Bad claim." in s2 and "- [voice] Say more." in s2 and "style match 0.3" in s2
    assert "Bad claim." not in upd["section_feedback"]["s1"]


def test_prepare_auto_revise_ignores_minor_notes():
    minor = CriticReport(notes=[CriticNote(section_id="s1", kind="clarity", note="meh")])
    upd = prepare_auto_revise({"citation_report": flagged(), "critic_report": minor})
    assert upd["revise_ids"] == ["s2"]


def test_payload_carries_critic_notes_style_score_and_warnings(state, deps):
    critic = CriticReport(style_score=0.72, notes=[major()])
    full = {**state, "critic_report": critic, "guard_warnings": ["Note n1 contains ..."]}
    with patch(
        "vault_to_voice.graph.nodes.review.interrupt", return_value={"action": "approve"}
    ) as fake_interrupt:
        draft_review(full, deps)
    sent = fake_interrupt.call_args.args[0]
    assert sent["style_score"] == 0.72 and sent["critic_notes"][0]["note"] == major().note
    assert sent["warnings"] == ["Note n1 contains ..."]


def test_payload_without_a_critic_report(state, deps):
    with patch(
        "vault_to_voice.graph.nodes.review.interrupt", return_value={"action": "approve"}
    ) as fake_interrupt:
        draft_review(state, deps)
    sent = fake_interrupt.call_args.args[0]
    assert sent["critic_notes"] == [] and sent["style_score"] is None and sent["warnings"] == []
