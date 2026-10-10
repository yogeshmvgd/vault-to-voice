import sys
import tempfile
from pathlib import Path

import click
import typer
from langgraph.types import Command
from pydantic import ValidationError

from vault_to_voice.config import Settings
from vault_to_voice.graph.builder import build_post_graph
from vault_to_voice.graph.checkpoint import make_checkpointer, make_store
from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.memory import put_style
from vault_to_voice.graph.runner import (
    pending_interrupt,
    run_until_interrupt,
    start_input,
    thread_config,
)
from vault_to_voice.ideas.detector import IndexOutOfSyncError, detect_ideas
from vault_to_voice.ideas.opinion import OpinionScorer
from vault_to_voice.index.embedder import make_embedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.ingest import ingest as run_ingest
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.llm import LangChainLLMClient, LLMClient
from vault_to_voice.publish import OutboxPublisher
from vault_to_voice.style.profile import build_style_profile, save_profile
from vault_to_voice.vault.factory import make_reader

app = typer.Typer(
    help="vault-to-voice: turn your notes into posts in your own voice.", no_args_is_help=True
)

VaultOpt = typer.Option(None, "--vault", help="Vault path (overrides VAULT_PATH).")
EvalOutOpt = typer.Option(None, "--out", help="Report path (default eval/report.json).")


def _safe_console() -> None:
    """Avoid UnicodeEncodeError on legacy consoles (e.g. Windows cp1252) for non-ASCII paths."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


@app.callback()
def _main() -> None:
    _safe_console()


def make_llm(settings: Settings) -> LLMClient:
    return LangChainLLMClient(
        settings.llm_model, timeout=settings.llm_timeout, max_retries=settings.llm_max_retries
    )


def _settings(vault: Path | None = None) -> Settings:
    try:
        return Settings(vault_path=vault) if vault else Settings()
    except ValidationError as exc:
        typer.echo(str(exc))
        raise typer.Exit(2) from exc


@app.command()
def ingest(
    vault: Path | None = VaultOpt,
    no_score: bool = typer.Option(False, "--no-score", help="Skip LLM opinion scoring."),
):
    """Index new/changed notes into FAISS and score them for opinion."""
    s = _settings(vault)
    embedder = make_embedder(s)
    scorer = None if no_score else OpinionScorer(make_llm(s))
    reader = make_reader(s)
    r = run_ingest(
        reader,
        Manifest(s.manifest_path),
        FaissStore(s.faiss_path, embedder.dim),
        embedder,
        scorer,
    )
    typer.echo(
        f"new={r.new} changed={r.changed} deleted={r.deleted} "
        f"unchanged={r.unchanged} rebuilt={r.rebuilt} scored={r.scored}"
    )
    stats = getattr(reader, "stats", None)
    if stats is not None:
        typer.echo(
            f"daily={stats.daily_files} entries={stats.entries} pages={stats.pages} "
            f"unlinked_skipped={stats.unlinked_skipped} empty_skipped={stats.empty_skipped}"
        )


@app.command()
def style(vault: Path | None = VaultOpt):
    """Build the style profile (stats + LLM tone) from your notes."""
    s = _settings(vault)
    notes = list(make_reader(s).iter_notes())
    if not notes:
        typer.echo("no notes found")
        raise typer.Exit(1)
    profile = save_profile(
        build_style_profile(notes, make_llm(s), seed=s.seed), s.style_profile_path
    )
    put_style(make_store(s.store_path), profile)
    typer.echo(f"style profile v{profile.version} saved to {s.style_profile_path}")
    typer.echo(f"tone: {', '.join(profile.tone.tone)}")


@app.command()
def ideas():
    """Cluster indexed notes and list post ideas."""
    s = _settings()
    embedder = make_embedder(s)
    try:
        found = detect_ideas(
            Manifest(s.manifest_path),
            FaissStore(s.faiss_path, embedder.dim),
            make_llm(s),
            min_notes=s.min_notes,
            min_opinionated=s.min_opinionated,
            opinion_threshold=s.opinion_threshold,
            seed=s.seed,
        )
    except IndexOutOfSyncError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from exc
    if not found:
        typer.echo("no ideas yet")
    for idea in found:
        typer.echo(
            f"{idea.idea_id}  [{idea.kind}] {idea.title}  "
            f"({len(idea.note_ids)} notes, opinion {idea.opinion_score})"
        )


def _ask_user(value: dict):
    kind = value["type"]
    if kind == "interview_question":
        typer.echo(f"\nQ{value['number']}/{value['max']}: {value['question']}")
        return typer.prompt("answer", default="", show_default=False)
    if value.get("error"):
        typer.echo(f"! {value['error']}")
    if kind == "outline_review":
        outline = value["outline"]
        typer.echo(f"\nOutline: {outline['title']}")
        for section in outline["sections"]:
            typer.echo(f"  {section['id']}. {section['heading']} - {section['point']}")
    else:
        typer.echo("\n" + value["markdown"])
        typer.echo(f"citation coverage: {value['coverage']:.0%}")
        for flag in value["flags"]:
            typer.echo(f"  ! [{flag['reason']}] {flag['text']}")
    action = typer.prompt(
        f"approve / revise ({value['rounds_left']} revisions left)",
        type=click.Choice(["approve", "revise"]),
        default="approve",
    )
    if action == "revise":
        return {"action": "revise", "feedback": typer.prompt("feedback")}
    return {"action": "approve"}


@app.command()
def write(idea_id: str = typer.Argument(..., help="Idea id from `v2v ideas`.")):
    """Write a post for an idea: interview, outline, draft, publish. Safe to stop and re-run."""
    s = _settings()
    manifest = Manifest(s.manifest_path)
    idea = manifest.get_idea(idea_id)
    if idea is None:
        typer.echo(f"unknown idea {idea_id}; run `v2v ideas` to list ideas")
        raise typer.Exit(1)
    embedder = make_embedder(s)
    deps = GraphDeps(
        llm=make_llm(s),
        manifest=manifest,
        index=FaissStore(s.faiss_path, embedder.dim),
        embedder=embedder,
        settings=s,
        store=make_store(s.store_path),
        publisher=OutboxPublisher(s.outbox_dir),
    )
    graph = build_post_graph(deps, checkpointer=make_checkpointer(s.checkpoint_path))
    config = thread_config(idea_id)
    snapshot = graph.get_state(config)
    if snapshot.values.get("final_path") and not snapshot.next:
        typer.echo(f"already published: {snapshot.values['final_path']}")
        return
    if snapshot.next:
        typer.echo("resuming saved session")
        value = pending_interrupt(graph, config)
        if value is None:  # stopped mid-step (crash): re-run that step
            value = run_until_interrupt(graph, config, None)
    else:
        value = run_until_interrupt(graph, config, start_input(idea))
    while value is not None:
        value = run_until_interrupt(graph, config, Command(resume=_ask_user(value)))
    typer.echo(f"published: {graph.get_state(config).values['final_path']}")


@app.command("eval")
def eval_(
    real: bool = typer.Option(
        False, "--real", help="Use the configured LLM and embeddings (makes paid API calls)."
    ),
    out: Path | None = EvalOutOpt,
):
    """Mini eval: write posts for the sample-vault cases and score them (offline by default)."""
    from vault_to_voice.eval.offline import OfflineLLM
    from vault_to_voice.eval.run import DEFAULT_REPORT, run_eval, write_report
    from vault_to_voice.index.embedder import HashEmbedder

    if real:
        s = _settings()
        llm, embedder = make_llm(s), make_embedder(s)
    else:
        llm, embedder = OfflineLLM(), HashEmbedder(256)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workdir:
        report = run_eval(llm, embedder, workdir=Path(workdir), mode="real" if real else "offline")
    path = write_report(report, out or DEFAULT_REPORT)
    for r in report.cases:
        if r.error:
            typer.echo(f"{r.case_id}: ERROR {r.error}")
            continue
        typer.echo(
            f"{r.case_id}: coverage={r.citation_coverage} faithfulness={r.faithfulness} "
            f"style={r.style_score} flags={r.flags} critic_notes={r.critic_notes}"
        )
    typer.echo("summary: " + " ".join(f"{k}={v}" for k, v in report.summary.items()))
    typer.echo(f"report written to {path}")
    if report.summary["errors"]:
        raise typer.Exit(1)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Bind address (default API_HOST)."),
    port: int | None = typer.Option(None, help="Port (default API_PORT)."),
):
    """Run the HTTP API (FastAPI + SSE) that the Streamlit UI talks to."""
    import uvicorn

    from vault_to_voice.api.app import create_app
    from vault_to_voice.api.factory import build_service
    from vault_to_voice.observability import configure_tracing

    s = _settings()
    service = build_service(s, make_llm(s), tracing=configure_tracing(s))
    uvicorn.run(create_app(service), host=host or s.api_host, port=port or s.api_port)
