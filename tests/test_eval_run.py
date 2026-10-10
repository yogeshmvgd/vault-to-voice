import json

import pytest
from typer.testing import CliRunner

from tests.fakes import FakeLLMClient
from vault_to_voice import cli
from vault_to_voice.config import Settings
from vault_to_voice.eval.offline import OfflineLLM
from vault_to_voice.eval.run import (
    CaseResult,
    EvalCase,
    EvalReport,
    faithfulness,
    load_cases,
    run_eval,
    summarise,
    write_report,
)
from vault_to_voice.graph.nodes.citations import JudgeVerdict, JudgeVerdicts
from vault_to_voice.graph.state import (
    QA,
    NoteContext,
    Paragraph,
    SectionDraft,
    Sentence,
    SourceChunk,
)
from vault_to_voice.index.embedder import HashEmbedder, LocalEmbedder
from vault_to_voice.vault.factory import make_reader


def test_every_case_uses_real_sample_vault_notes():
    ids = {n.id for n in make_reader(Settings(_env_file=None)).iter_notes()}
    cases = load_cases()
    assert len(cases) >= 3 and len({c.case_id for c in cases}) == len(cases)
    for case in cases:
        assert set(case.note_ids) <= ids, case.case_id
        assert len(case.answers) >= 3


def draft(*sentences):
    return [SectionDraft(section_id="s1", paragraphs=[Paragraph(sentences=list(sentences))])]


NOTE = NoteContext(
    note_id="n0", title="T", type="game", date=None, chunks=[SourceChunk(chunk_id=1, text="Fair.")]
)


def test_faithfulness_scores_every_claim_and_skips_the_authors_own_words():
    judge = FakeLLMClient(
        {
            JudgeVerdicts: JudgeVerdicts(
                verdicts=[
                    JudgeVerdict(index=0, supported=True),
                    JudgeVerdict(index=1, supported=False),
                ]
            )
        }
    )
    sentences = draft(
        Sentence(text="Fair.", sources=["n0"]),
        Sentence(text="Answered.", sources=["qa:1"]),
        Sentence(text="No source."),  # counts against faithfulness, never judged
        Sentence(text="Mine.", sources=["user"]),  # the author's edit: not scored
        Sentence(text="So.", kind="connective"),
    )
    qa = [QA(qa_id="qa:1", question="q", answer="a")]
    assert faithfulness(judge, sentences, [NOTE], qa) == round(1 / 3, 3)
    prompt = judge.calls[0][1]
    assert "[0] claim: Fair." in prompt and "[1] claim: Answered." in prompt
    assert "No source." not in prompt and "Mine." not in prompt


def test_faithfulness_edge_cases():
    nobody = FakeLLMClient({})  # must not be called
    assert faithfulness(nobody, draft(Sentence(text="So.", kind="connective")), [NOTE], []) is None
    assert faithfulness(nobody, draft(Sentence(text="Unsourced.")), [NOTE], []) == 0.0


def test_offline_eval_on_the_sample_vault(tmp_path):
    report = run_eval(OfflineLLM(), HashEmbedder(256), workdir=tmp_path)
    assert report.mode == "offline" and report.embedder == "hash:256"
    assert [r.error for r in report.cases] == [None] * len(report.cases)
    for r in report.cases:
        assert r.published and r.n_claims > 0
        assert r.citation_coverage == 1.0 and r.faithfulness == 1.0
        assert r.style_score is not None and 0.0 <= r.style_score <= 1.0
    assert report.summary["published_rate"] == 1.0 and report.summary["errors"] == 0.0
    assert list((tmp_path / "outbox").glob("*.sources.json"))  # posts really were published


def test_a_broken_case_is_reported_without_hiding_the_others(tmp_path):
    good = load_cases()[0]
    broken = EvalCase(
        case_id="missing", title="t", angle="a", theme="x", note_ids=["no-such-note"], answers=["a"]
    )
    report = run_eval(OfflineLLM(), HashEmbedder(256), workdir=tmp_path, cases=[broken, good])
    assert report.cases[0].error.startswith("ValueError: None of the notes")
    assert report.cases[1].published and report.summary["errors"] == 1.0


def test_summarise_ignores_missing_values():
    results = [
        CaseResult(case_id="a", published=True, citation_coverage=1.0, style_score=None),
        CaseResult(case_id="b", published=False, citation_coverage=0.5, style_score=0.4),
    ]
    summary = summarise(results)
    assert summary["published_rate"] == 0.5 and summary["citation_coverage"] == 0.75
    assert summary["style_score"] == 0.4 and summary["faithfulness"] is None


def test_write_report_round_trips(tmp_path):
    report = run_eval(OfflineLLM(), HashEmbedder(256), workdir=tmp_path, cases=load_cases()[:1])
    path = write_report(report, tmp_path / "eval" / "report.json")
    assert EvalReport.model_validate_json(path.read_text(encoding="utf-8")) == report


def test_cli_eval_writes_the_report_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "make_llm", lambda s: pytest.fail("offline eval must not build an LLM")
    )
    out = tmp_path / "report.json"
    result = CliRunner().invoke(cli.app, ["eval", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "fair-difficulty: coverage=1.0" in result.output and "summary: " in result.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "offline" and len(data["cases"]) == len(load_cases())


@pytest.mark.integration
def test_offline_eval_with_minilm_embeddings(tmp_path):
    embedder = LocalEmbedder(Settings(_env_file=None).local_embedding_model)
    report = run_eval(OfflineLLM(), embedder, workdir=tmp_path)
    assert report.summary["errors"] == 0.0 and report.summary["published_rate"] == 1.0
    assert report.summary["citation_coverage"] >= 0.9
