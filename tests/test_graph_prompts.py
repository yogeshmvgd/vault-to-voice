from vault_to_voice.graph.prompts import notes_block, qa_block, style_brief
from vault_to_voice.graph.state import QA, NoteContext, SourceChunk


def test_notes_block_wraps_and_scrubs():
    note = NoteContext(
        note_id="n0",
        title="T",
        type="game",
        date="2026-01-01",
        chunks=[SourceChunk(chunk_id=1, text="mail a@b.com")],
    )
    out = notes_block([note])
    assert out.startswith('<note id="n0">') and "[EMAIL]" in out and "a@b.com" not in out


def test_qa_block_uses_qa_ids():
    out = qa_block([QA(qa_id="qa:1", question="Why?", answer="Because.")])
    assert '<note id="qa:1">' in out and "Because." in out


def test_style_brief_without_and_with_profile():
    assert "no style profile" in style_brief(None).lower()
    brief = style_brief(
        {
            "tone": {
                "tone": ["wry"],
                "voice_notes": "Punchy.",
                "exemplars": ["The fight is fair."],
            },
            "stats": {
                "avg_sentence_len": 11.2,
                "first_person_rate": 0.4,
                "top_phrases": ["the thing is"],
            },
        }
    )
    assert "wry" in brief and "The fight is fair." in brief and "the thing is" in brief
