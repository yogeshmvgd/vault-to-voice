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
        paragraphs=[
            Paragraph(sentences=[Sentence(text="Every attack has a tell.", sources=["n0"])])
        ]
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
