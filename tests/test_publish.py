import datetime as dt
import json
from pathlib import Path

from tests.fakes import FakeLLMClient, make_deps, seed_idea
from vault_to_voice.graph.memory import past_answers
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.nodes.publish import publish
from vault_to_voice.graph.state import (
    QA,
    CitationReport,
    Outline,
    Paragraph,
    Section,
    SectionDraft,
    Sentence,
)
from vault_to_voice.publish import OutboxPublisher, RenderedPost


def test_outbox_publisher_writes_both_files_and_never_overwrites(tmp_path):
    publisher = OutboxPublisher(tmp_path / "out")
    post = RenderedPost(
        slug="fair-bosses", date=dt.date(2026, 10, 10), markdown="# x\n", sources={"a": 1}
    )
    first, second = publisher.publish(post), publisher.publish(post)
    assert first.name == "2026-10-10-fair-bosses.md"
    assert second.name == "2026-10-10-fair-bosses-2.md"
    assert json.loads(first.with_suffix(".sources.json").read_text(encoding="utf-8")) == {"a": 1}


def test_publish_node(tmp_path, vault_dir):
    deps = make_deps(tmp_path, FakeLLMClient({}))
    deps.today = lambda: dt.date(2026, 10, 10)
    idea = seed_idea(deps, vault_dir)
    outline = Outline(title="Fair Bosses", sections=[Section(id="s1", heading="Tells", point="p")])
    sentences = [
        Sentence(text="Every attack has a tell.", sources=["n0"]),
        Sentence(text="So.", kind="connective"),
    ]
    state = {
        **load_context({"idea": idea}, deps),
        "idea": idea,
        "outline": outline,
        "draft": [SectionDraft(section_id="s1", paragraphs=[Paragraph(sentences=sentences)])],
        "qa": [QA(qa_id="qa:1", question="q", answer="a")],
        "citation_report": CitationReport(n_claims=1, n_supported=1),
    }
    path = Path(publish(state, deps)["final_path"])
    assert path.name == "2026-10-10-fair-bosses.md"
    text = path.read_text(encoding="utf-8")
    _, front, body = text.split("---\n", 2)
    assert 'title: "Fair Bosses"' in front and f"idea_id: {idea.idea_id}" in front
    assert body == "\n# Fair Bosses\n\n## Tells\n\nEvery attack has a tell. So.\n"
    side = json.loads(path.with_suffix(".sources.json").read_text(encoding="utf-8"))
    assert side["paragraphs"][0]["sources"] == ["n0"] and side["citation_coverage"] == 1.0
    assert side["interview"][0]["answer"] == "a"
    assert side["source_details"]["n0"] == {
        "title": "Note n0",
        "date": "2026-01-01",
        "work_id": None,
    }
    assert deps.manifest.get_idea(idea.idea_id).status == "done"
    assert [q.answer for q in past_answers(deps.store)] == ["a"]
