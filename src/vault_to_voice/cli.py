import sys
from pathlib import Path

import typer
from pydantic import ValidationError

from vault_to_voice.config import Settings
from vault_to_voice.ideas.detector import IndexOutOfSyncError, detect_ideas
from vault_to_voice.ideas.opinion import OpinionScorer
from vault_to_voice.index.embedder import make_embedder
from vault_to_voice.index.faiss_store import FaissStore
from vault_to_voice.index.ingest import ingest as run_ingest
from vault_to_voice.index.manifest import Manifest
from vault_to_voice.llm import LangChainLLMClient, LLMClient
from vault_to_voice.style.profile import build_style_profile, save_profile
from vault_to_voice.vault.reader import FilesystemVaultReader

app = typer.Typer(
    help="vault-to-voice: turn your notes into posts in your own voice.", no_args_is_help=True
)

VaultOpt = typer.Option(None, "--vault", help="Vault path (overrides VAULT_PATH).")


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
    r = run_ingest(
        FilesystemVaultReader(s.vault_path),
        Manifest(s.manifest_path),
        FaissStore(s.faiss_path, embedder.dim),
        embedder,
        scorer,
    )
    typer.echo(
        f"new={r.new} changed={r.changed} deleted={r.deleted} "
        f"unchanged={r.unchanged} rebuilt={r.rebuilt} scored={r.scored}"
    )


@app.command()
def style(vault: Path | None = VaultOpt):
    """Build the style profile (stats + LLM tone) from your notes."""
    s = _settings(vault)
    notes = list(FilesystemVaultReader(s.vault_path).iter_notes())
    if not notes:
        typer.echo("no notes found")
        raise typer.Exit(1)
    profile = save_profile(
        build_style_profile(notes, make_llm(s), seed=s.seed), s.style_profile_path
    )
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
            f"{idea.idea_id}  {idea.title}  "
            f"({len(idea.note_ids)} notes, opinion {idea.opinion_score})"
        )
