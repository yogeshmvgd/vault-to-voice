import pytest

from vault_to_voice.ui import views

DRAFT = [
    {
        "section_id": "s1",
        "paragraphs": [
            {
                "sentences": [
                    {"text": "Tells make it fair.", "sources": ["n0"], "kind": "claim"},
                    {"text": "Bosses [cheat] *a lot*.", "sources": [], "kind": "claim"},
                ]
            },
            {"sentences": [{"text": "So it goes.", "sources": [], "kind": "connective"}]},
        ],
    }
]
FLAGS = [
    {"section_id": "s1", "paragraph": 0, "sentence": 1, "text": "x", "reason": "missing_source"}
]


def test_last_event_reports_nodes_and_returns_the_final_event():
    seen = []
    events = [("node", {"node": "a"}), ("node", {"node": "b"}), ("interrupt", {"type": "q"})]
    assert views.last_event(events, on_node=seen.append) == ("interrupt", {"type": "q"})
    assert seen == ["a", "b"]


def test_last_event_without_a_result_is_an_error():
    name, data = views.last_event([("node", {"node": "a"})])
    assert name == "error" and "Retry" in data["message"]


def test_flagged_markdown_marks_only_flagged_sentences_and_escapes_markup():
    md = views.flagged_markdown(DRAFT, FLAGS, {"s1": "Tells"})
    assert md.startswith("### Tells\n\nTells make it fair. :red[")
    assert ":red[Bosses \\[cheat\\] \\*a lot\\*.] :red[_(missing source)_]" in md
    assert md.endswith("\n\nSo it goes.")
    assert views.flagged_markdown(DRAFT, [], {}).startswith("### s1\n\n")  # unknown heading


def test_section_texts_and_changed_sections():
    texts = views.section_texts(DRAFT)
    assert texts == {"s1": "Tells make it fair. Bosses [cheat] *a lot*.\n\nSo it goes."}
    assert views.changed_sections(texts, {"s1": texts["s1"] + "  "}) == {}
    assert views.changed_sections(texts, {"s1": "Mine.", "zz": "ignored"}) == {"s1": "Mine."}


def test_edited_outline_keeps_ids_and_sources():
    outline = {
        "title": "Old",
        "sections": [{"id": "s1", "heading": "H", "point": "P", "planned_sources": ["n0"]}],
    }
    assert views.edited_outline(outline, " New ", [" H2 "], ["P2"]) == {
        "title": "New",
        "sections": [{"id": "s1", "heading": "H2", "point": "P2", "planned_sources": ["n0"]}],
    }
    with pytest.raises(ValueError):
        views.edited_outline(outline, "New", [], [])


def test_strip_frontmatter():
    assert views.strip_frontmatter("---\ntitle: x\n---\n\n# Post\n") == "# Post\n"
    assert views.strip_frontmatter("# No front\n") == "# No front\n"
    assert views.strip_frontmatter("---\nunterminated") == "---\nunterminated"


def test_critic_markdown_puts_major_notes_first_and_escapes():
    notes = [
        {"section_id": "s1", "kind": "clarity", "severity": "minor", "note": "Name *it*."},
        {"section_id": "s2", "kind": "voice", "severity": "major", "note": "Too formal."},
    ]
    out = views.critic_markdown(notes, {"s1": "Tells", "s2": "Learning"})
    assert out.split("\n") == [
        "- **major - voice** (Learning): Too formal.",
        "- clarity (Tells): Name \\*it\\*.",
    ]


@pytest.mark.parametrize(("score", "label"), [(None, "n/a"), (0.734, "73%"), (1.0, "100%")])
def test_style_label(score, label):
    assert views.style_label(score) == label
