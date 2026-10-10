import pytest

from vault_to_voice.eval.offline import OfflineLLM
from vault_to_voice.graph.nodes.citations import JudgeVerdicts
from vault_to_voice.graph.nodes.critic import CriticReview
from vault_to_voice.graph.prompts import notes_block, qa_block
from vault_to_voice.graph.state import QA, NoteContext, Outline, SectionText, SourceChunk
from vault_to_voice.ideas.opinion import OpinionScore

NOTES = [
    NoteContext(
        note_id=f"n{i}",
        title=f"Note {i}",
        type="game",
        date="2026-01-01",
        chunks=[SourceChunk(chunk_id=i, text=f"Sentence {i} is the first. Then more.")],
    )
    for i in range(3)
]
QAS = [QA(qa_id="qa:1", question="Why?", answer="Because the tells are fair.")]


def test_outline_splits_notes_into_sections_and_uses_the_idea_title():
    prompt = f"Post idea: Fair Bosses — angle\n\nNotes:\n\n{notes_block(NOTES)}\n\n{qa_block(QAS)}"
    outline = OfflineLLM().structured(prompt, Outline)
    assert outline.title == "Fair Bosses"
    assert [s.planned_sources for s in outline.sections] == [["n0", "n1", "qa:1"], ["n2"]]


def test_section_copies_the_first_sentence_of_each_planned_source():
    prompt = f"Planned sources: n1, qa:1\n\n{notes_block(NOTES)}\n\n{qa_block(QAS)}"
    text = OfflineLLM().structured(prompt, SectionText)
    sentences = text.paragraphs[0].sentences
    assert [(s.text, s.sources) for s in sentences[:2]] == [
        ("Sentence 1 is the first.", ["n1"]),
        ("Because the tells are fair.", ["qa:1"]),
    ]
    assert sentences[-1].kind == "connective"


def test_section_without_usable_planned_sources_falls_back_to_the_first_note():
    prompt = f"Planned sources: (none)\n\n{notes_block(NOTES)}"
    sentences = OfflineLLM().structured(prompt, SectionText).paragraphs[0].sentences
    assert sentences[0].sources == ["n0"]


def test_judge_supports_only_verbatim_claims():
    note = notes_block(NOTES[:1])
    prompt = (
        f"Judge.\n[0] claim: Sentence 0 is the first.\nsources:\n{note}\n\n"
        f"[1] claim: Purple bananas.\nsources:\n{note}"
    )
    out = OfflineLLM().structured(prompt, JudgeVerdicts)
    assert [(v.index, v.supported) for v in out.verdicts] == [(0, True), (1, False)]


def test_critic_has_no_notes_and_unknown_schemas_fail_loudly():
    llm = OfflineLLM()
    assert llm.structured("draft", CriticReview).notes == []
    with pytest.raises(KeyError, match="OpinionScore"):
        llm.structured("x", OpinionScore)
    assert llm.calls == ["CriticReview", "OpinionScore"]
