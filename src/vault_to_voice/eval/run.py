"""Mini eval: run the whole post graph on the fictional sample vault and score the posts.

Scripted interview answers, auto-approve at both reviews. Metrics per case: citation coverage
(the graph's own checker), judge faithfulness (an LLM judge over every cited claim), style
match (the critic's measured score), flags and critic notes. Writes eval/report.json.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Literal

from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from vault_to_voice.config import REPO_ROOT, SAMPLE_VAULT, Settings
from vault_to_voice.graph.builder import build_post_graph
from vault_to_voice.graph.checkpoint import memory_checkpointer
from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.memory import put_style
from vault_to_voice.graph.nodes.citations import JUDGE_PROMPT, JudgeVerdicts
from vault_to_voice.graph.runner import run_until_interrupt, start_input, thread_config
from vault_to_voice.graph.state import QA, NoteContext, SectionDraft
from vault_to_voice.guardrails import DATA_INSTRUCTION, scrub_pii, wrap_note
from vault_to_voice.ideas.detector import Idea, make_idea_id
from vault_to_voice.index.embedder import Embedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.ingest import ingest
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.llm import LLMClient
from vault_to_voice.publish import OutboxPublisher
from vault_to_voice.style.profile import build_style_profile
from vault_to_voice.vault.factory import make_reader

CASES_PATH = Path(__file__).with_name("cases.json")
DEFAULT_REPORT = REPO_ROOT / "eval" / "report.json"
MAX_STEPS = 25  # interrupts per case; a run that needs more is reported as an error
APPROVE = {"action": "approve"}


class EvalCase(BaseModel):
    case_id: str
    title: str
    angle: str
    theme: str
    note_ids: list[str] = Field(min_length=1)
    answers: list[str] = Field(min_length=1)

    def idea(self) -> Idea:
        ids = sorted(self.note_ids)
        return Idea(
            idea_id=make_idea_id(ids),
            title=self.title,
            angle=self.angle,
            theme=self.theme,
            note_ids=ids,
            opinion_score=1.0,
        )


class CaseResult(BaseModel):
    case_id: str
    published: bool = False
    n_claims: int = 0
    citation_coverage: float | None = None
    faithfulness: float | None = None
    style_score: float | None = None
    flags: int = 0
    critic_notes: int = 0
    auto_revised: bool = False
    guard_warnings: int = 0
    error: str | None = None


class EvalReport(BaseModel):
    mode: Literal["offline", "real"]
    created_at: datetime
    embedder: str
    cases: list[CaseResult]
    summary: dict[str, float | None]


def load_cases(path: Path = CASES_PATH) -> list[EvalCase]:
    return [EvalCase.model_validate(c) for c in json.loads(Path(path).read_text("utf-8"))]


def faithfulness(
    llm: LLMClient, draft: list[SectionDraft], notes: list[NoteContext], qa: list[QA]
) -> float | None:
    """Share of cited claims an LLM judge finds supported by their sources (all claims, not
    only the checker's grey zone). Claims the author wrote ('user') are not scored; claims
    without a known source count as unsupported. None when there is nothing to score."""
    texts = {n.note_id: n.text for n in notes} | {q.qa_id: q.answer for q in qa}
    claims = [
        s
        for section in draft
        for p in section.paragraphs
        for s in p.sentences
        if s.kind == "claim" and "user" not in s.sources
    ]
    judged = [s for s in claims if s.sources and all(src in texts for src in s.sources)]
    if not claims:
        return None
    if not judged:
        return 0.0
    items = []
    for i, sentence in enumerate(judged):
        sources = "\n".join(wrap_note(src, scrub_pii(texts[src])) for src in sentence.sources)
        items.append(f"[{i}] claim: {sentence.text}\nsources:\n{sources}")
    out = llm.structured(JUDGE_PROMPT + "\n\n".join(items), JudgeVerdicts, system=DATA_INSTRUCTION)
    supported = {v.index for v in out.verdicts if v.supported}
    return round(len(supported & set(range(len(judged)))) / len(claims), 3)


def _settings(workdir: Path, vault: Path) -> Settings:
    # No .env and fixed vault format: an eval run must not depend on the user's settings.
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        vault_path=vault,
        data_dir=workdir / "data",
        outbox_dir=workdir / "outbox",
        embedding_provider="hash",
        vault_format="obsidian",
        daily_folder="Daily",
        daily_date_format="%d-%m-%Y",
        category_headings="games=game,books=book,movies=movie",
    )


def _drive(graph, case: EvalCase) -> None:
    cfg = thread_config(case.idea().idea_id)
    value = run_until_interrupt(graph, cfg, start_input(case.idea()))
    for _ in range(MAX_STEPS):
        if value is None:
            return
        if value["type"] == "interview_question":
            decision = case.answers[min(value["number"], len(case.answers)) - 1]
        else:
            decision = APPROVE
        value = run_until_interrupt(graph, cfg, Command(resume=decision))
    raise RuntimeError(f"no result after {MAX_STEPS} steps")


def _score(graph, llm: LLMClient, case: EvalCase) -> CaseResult:
    values = graph.get_state(thread_config(case.idea().idea_id)).values
    report, critic = values["citation_report"], values.get("critic_report")
    return CaseResult(
        case_id=case.case_id,
        published=bool(values.get("final_path")),
        n_claims=report.n_claims,
        citation_coverage=report.coverage,
        faithfulness=faithfulness(llm, values["draft"], values["notes"], values.get("qa", [])),
        style_score=critic.style_score if critic else None,
        flags=len(report.flags),
        critic_notes=len(critic.notes) if critic else 0,
        auto_revised=bool(values.get("auto_revised")),
        guard_warnings=len(values.get("guard_warnings", [])),
    )


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(mean(known), 3) if known else None


def summarise(results: list[CaseResult]) -> dict[str, float | None]:
    return {
        "cases": float(len(results)),
        "published_rate": _mean([float(r.published) for r in results]),
        "citation_coverage": _mean([r.citation_coverage for r in results]),
        "faithfulness": _mean([r.faithfulness for r in results]),
        "style_score": _mean([r.style_score for r in results]),
        "errors": float(sum(r.error is not None for r in results)),
    }


def run_eval(
    llm: LLMClient,
    embedder: Embedder,
    *,
    workdir: Path,
    mode: Literal["offline", "real"] = "offline",
    cases: list[EvalCase] | None = None,
    vault: Path = SAMPLE_VAULT,
) -> EvalReport:
    """Ingest the sample vault into `workdir`, build a style profile, run every case."""
    cases = load_cases() if cases is None else cases
    settings = _settings(Path(workdir), vault)
    manifest = Manifest(":memory:")
    index = FaissStore(settings.faiss_path, embedder.dim)
    ingest(make_reader(settings), manifest, index, embedder)
    store = InMemoryStore()
    notes = list(make_reader(settings).iter_notes())
    put_style(store, build_style_profile(notes, llm, seed=settings.seed))
    manifest.save_ideas([c.idea() for c in cases])
    deps = GraphDeps(
        llm=llm,
        manifest=manifest,
        index=index,
        embedder=embedder,
        settings=settings,
        store=store,
        publisher=OutboxPublisher(settings.outbox_dir),
    )
    graph = build_post_graph(deps, checkpointer=memory_checkpointer())
    results = []
    for case in cases:
        try:
            _drive(graph, case)
            results.append(_score(graph, llm, case))
        except Exception as exc:  # one broken case must not hide the others' scores
            results.append(CaseResult(case_id=case.case_id, error=f"{type(exc).__name__}: {exc}"))
    manifest.close()
    return EvalReport(
        mode=mode,
        created_at=datetime.now(UTC),
        embedder=f"{embedder.name}:{embedder.dim}",
        cases=results,
        summary=summarise(results),
    )


def write_report(report: EvalReport, path: Path = DEFAULT_REPORT) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path
