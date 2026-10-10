import pytest

from tests.fakes import FakeLLMClient, make_deps
from vault_to_voice.graph.nodes.critic import CriticReview, critic, safety_notes, style_score
from vault_to_voice.graph.state import (
    CriticNote,
    CriticReport,
    Outline,
    Paragraph,
    Section,
    SectionDraft,
    Sentence,
)
from vault_to_voice.style.stats import compute_stats

AUTHOR = (
    "I died twenty times. I loved it. Honestly? The tells were clear.\n\n"
    "I kept going. I learned the pattern. I won."
)


def sd(sid: str, text: str) -> SectionDraft:
    sentences = [Sentence(text=t.strip().rstrip(".") + ".") for t in text.split(". ")]
    return SectionDraft(section_id=sid, paragraphs=[Paragraph(sentences=sentences)])


def style_of(text: str) -> dict:
    return {"stats": compute_stats([text]).model_dump(), "tone": {"tone": ["blunt"]}}


def state(draft, style=None):
    outline = Outline(
        title="T", sections=[Section(id=d.section_id, heading="H", point="p") for d in draft]
    )
    return {"outline": outline, "draft": draft, "style": style}


def test_style_score_is_none_without_a_profile_or_draft():
    assert style_score([sd("s1", "I won")], None) is None
    assert style_score([], style_of(AUTHOR)) is None


def test_style_score_prefers_the_authors_shape():
    style = style_of(AUTHOR)
    close = [sd("s1", "I died ten times. I loved it. I won")]
    far = [
        sd(
            "s1",
            "The game, which was released in the spring to considerable acclaim among critics "
            "and players alike, presents a sequence of encounters whose difficulty escalates "
            "gradually across many hours of careful and deliberate play",
        )
    ]
    near, distant = style_score(close, style), style_score(far, style)
    assert 0.0 <= distant < near <= 1.0


def test_safety_notes_flag_publishing_problems():
    notes = safety_notes([sd("s1", "Mail me at jo@example.com"), sd("s2", "All clean here")])
    assert [(n.section_id, n.kind, n.severity) for n in notes] == [("s1", "safety", "major")]
    assert "email address" in notes[0].note


def test_critic_node_keeps_known_sections_and_scores_style(tmp_path):
    review = CriticReview(
        notes=[
            CriticNote(section_id="s1", kind="voice", severity="major", note="Too formal."),
            CriticNote(section_id="s9", kind="clarity", note="Unknown section."),
            CriticNote(section_id="s1", kind="safety", severity="major", note="LLM cannot."),
        ]
    )
    fake = FakeLLMClient({CriticReview: review})
    deps = make_deps(tmp_path, fake)
    report = critic(state([sd("s1", "I won")], style_of(AUTHOR)), deps)["critic_report"]
    assert [n.note for n in report.notes] == ["Too formal."]
    assert report.style_score is not None and report.needs_revision
    schema, prompt = fake.calls[0]
    assert schema is CriticReview and '<note id="s1">' in prompt
    assert "Measured style match: " in prompt


def test_critic_prompt_never_carries_pii(tmp_path):
    fake = FakeLLMClient({CriticReview: CriticReview()})
    deps = make_deps(tmp_path, fake)
    report = critic(state([sd("s1", "Mail me at jo@example.com")]), deps)["critic_report"]
    assert "jo@example.com" not in fake.calls[0][1]
    assert [n.kind for n in report.notes] == ["safety"]  # caught by the output guardrail


def test_low_style_score_marks_style_not_ok(tmp_path):
    deps = make_deps(tmp_path, FakeLLMClient({CriticReview: CriticReview()}), critic_min_style=1.1)
    report = critic(state([sd("s1", "I won")], style_of(AUTHOR)), deps)["critic_report"]
    assert report.style_ok is False and report.needs_revision and report.notes == []


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        (CriticReport(), False),
        (CriticReport(notes=[CriticNote(section_id="s1", kind="voice", note="meh")]), False),
        (
            CriticReport(
                notes=[CriticNote(section_id="s1", kind="voice", severity="major", note="x")]
            ),
            True,
        ),
        (CriticReport(style_score=0.2, style_ok=False), True),
    ],
)
def test_needs_revision(report, expected):
    assert report.needs_revision is expected
