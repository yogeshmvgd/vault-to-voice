import pytest
from pydantic import ValidationError

from vault_to_voice.config import Settings
from vault_to_voice.graph.state import (
    QA,
    CitationReport,
    NoteContext,
    Outline,
    ReviewDecision,
    Sentence,
    SourceChunk,
    merge_drafts,
    valid_sources,
)


def test_settings_graph_defaults(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert (s.interview_min_questions, s.interview_max_questions, s.max_review_rounds) == (3, 5, 3)
    assert (s.citation_support, s.citation_reject) == (0.55, 0.30)
    assert s.checkpoint_path == tmp_path / "checkpoints.db"
    assert s.store_path == tmp_path / "store.db"


def test_sentence_defaults_to_claim():
    assert Sentence(text="x").kind == "claim"


def test_outline_needs_a_section():
    with pytest.raises(ValidationError):
        Outline(title="t", sections=[])


def test_review_decision_rejects_unknown_action():
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate({"action": "publish"})


def test_merge_drafts_overrides_by_key():
    assert merge_drafts({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}
    assert merge_drafts(None, {"a": 1}) == {"a": 1}


def test_coverage():
    assert CitationReport(n_claims=4, n_supported=3).coverage == 0.75
    assert CitationReport(n_claims=0, n_supported=0).coverage == 1.0


def test_note_context_text_and_valid_sources():
    chunks = [SourceChunk(chunk_id=1, text="a"), SourceChunk(chunk_id=2, text="b")]
    note = NoteContext(note_id="n0", title="t", type="game", date=None, chunks=chunks)
    assert note.text == "a\n\nb"
    qa = [QA(qa_id="qa:1", question="q", answer="x")]
    assert valid_sources([note], qa) == {"n0", "qa:1", "user"}
