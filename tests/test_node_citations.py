import pytest

from tests.fakes import FakeLLMClient, make_deps, seed_idea
from vault_to_voice.graph.nodes.citations import JudgeVerdict, JudgeVerdicts, check_citations
from vault_to_voice.graph.nodes.context import load_context
from vault_to_voice.graph.state import QA, Paragraph, SectionDraft, Sentence

N0_FIRST = "Boss fights feel fair when every attack has a clear tell."
GREY = "Boss fights feel fair when the camera spins wildly around."  # ~half the words of N0_FIRST


def setup(tmp_path, vault_dir, verdict=True):
    fake = FakeLLMClient(
        {
            JudgeVerdicts: lambda _: JudgeVerdicts(
                verdicts=[JudgeVerdict(index=0, supported=verdict)]
            )
        }
    )
    deps = make_deps(tmp_path, fake)
    idea = seed_idea(deps, vault_dir)
    qa = [QA(qa_id="qa:1", question="q", answer="I died twenty times on the bell tower.")]
    return deps, {**load_context({"idea": idea}, deps), "idea": idea, "qa": qa}, fake


def check(deps, state, *sentences):
    draft = [SectionDraft(section_id="s1", paragraphs=[Paragraph(sentences=list(sentences))])]
    return check_citations({**state, "draft": draft}, deps)["citation_report"]


def test_exact_note_text_supported(tmp_path, vault_dir):
    deps, state, fake = setup(tmp_path, vault_dir)
    report = check(deps, state, Sentence(text=N0_FIRST, sources=["n0"]))
    assert (report.n_claims, report.n_supported, report.flags) == (1, 1, [])
    assert fake.calls == []  # no judge needed


def test_qa_source_supported(tmp_path, vault_dir):
    deps, state, _ = setup(tmp_path, vault_dir)
    report = check(
        deps, state, Sentence(text="I died twenty times on the bell tower.", sources=["qa:1"])
    )
    assert report.n_supported == 1


@pytest.mark.parametrize(
    ("sources", "reason"),
    [([], "missing_source"), (["ghost"], "invalid_source"), (["qa:9"], "invalid_source")],
)
def test_missing_and_invalid_sources_flagged(tmp_path, vault_dir, sources, reason):
    deps, state, _ = setup(tmp_path, vault_dir)
    report = check(deps, state, Sentence(text="Some claim.", sources=sources))
    assert [f.reason for f in report.flags] == [reason]
    assert (report.flags[0].section_id, report.flags[0].paragraph, report.flags[0].sentence) == (
        "s1",
        0,
        0,
    )


def test_unrelated_text_unsupported(tmp_path, vault_dir):
    deps, state, _ = setup(tmp_path, vault_dir)
    report = check(
        deps, state, Sentence(text="Purple bananas orbit distant moons quietly.", sources=["n0"])
    )
    assert [f.reason for f in report.flags] == ["unsupported"]
    assert report.flags[0].score <= 0.30


def test_connective_exempt_and_user_supported(tmp_path, vault_dir):
    deps, state, _ = setup(tmp_path, vault_dir)
    report = check(
        deps,
        state,
        Sentence(text="So.", kind="connective"),
        Sentence(text="My own edit.", sources=["user"]),
    )
    assert (report.n_claims, report.n_supported, report.flags) == (1, 1, [])


@pytest.mark.parametrize(("verdict", "supported"), [(True, 1), (False, 0)])
def test_grey_zone_goes_to_llm_judge(tmp_path, vault_dir, verdict, supported):
    deps, state, fake = setup(tmp_path, vault_dir, verdict)
    report = check(deps, state, Sentence(text=GREY, sources=["n0"]))
    assert report.n_supported == supported
    judge_prompts = [p for s, p in fake.calls if s is JudgeVerdicts]
    assert len(judge_prompts) == 1 and GREY in judge_prompts[0]
    assert '<note id="n0">' in judge_prompts[0]
