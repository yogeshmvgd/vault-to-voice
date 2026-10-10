import pytest
from typer.testing import CliRunner

from tests.conftest import note_text
from tests.fakes import FakeLLMClient
from vault_to_voice import cli
from vault_to_voice.config import SAMPLE_VAULT
from vault_to_voice.ideas.labeler import ClusterLabel
from vault_to_voice.ideas.opinion import OpinionScore
from vault_to_voice.style.profile import ToneSummary

runner = CliRunner()


def fake_llm():
    return FakeLLMClient(
        {
            OpinionScore: lambda _: OpinionScore(score=0.8, highlights=["strong view"]),
            ToneSummary: lambda _: ToneSummary(tone=["wry"], voice_notes="Punchy.", exemplars=[]),
            ClusterLabel: lambda _: ClusterLabel(title="Idea", angle="Angle.", theme="theme"),
        }
    )


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("VAULT_FORMAT", "flat")
    monkeypatch.setenv("MIN_NOTES", "2")
    monkeypatch.setenv("MIN_OPINIONATED", "2")
    monkeypatch.setattr(cli, "make_llm", lambda settings: fake_llm())
    return tmp_path


def test_full_cli_flow(env, vault_dir, write_note):
    for i in range(6):
        write_note(f"n{i}.md", note_text(f"n{i}", f"boss fight fair difficulty note {i}"))
    r = runner.invoke(cli.app, ["ingest", "--vault", str(vault_dir)])
    assert r.exit_code == 0, r.output
    assert "new=6" in r.output and "scored=6" in r.output
    r = runner.invoke(cli.app, ["style", "--vault", str(vault_dir)])
    assert r.exit_code == 0 and "style profile v1" in r.output
    r = runner.invoke(cli.app, ["ideas"])
    assert r.exit_code == 0 and "idea_" in r.output


def test_vault_inside_repo_rejected(env):
    r = runner.invoke(cli.app, ["ingest", "--vault", str(SAMPLE_VAULT.parent / "vault")])
    assert r.exit_code == 2 and "inside the repo" in r.output


@pytest.mark.integration
def test_sample_vault_with_minilm_yields_ideas(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(cli, "make_llm", lambda settings: fake_llm())
    assert runner.invoke(cli.app, ["ingest"]).exit_code == 0
    r = runner.invoke(cli.app, ["ideas"])
    assert r.exit_code == 0 and "[work]" in r.output and "[theme]" in r.output, r.output


def test_safe_console_survives_cp1252(monkeypatch):
    import io
    import sys

    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    cli._safe_console()
    print("本")  # must not raise UnicodeEncodeError


def test_ideas_before_ingest_in_sync_message(env):
    from vault_to_voice.index.manifest import Manifest
    from vault_to_voice.vault.models import Note

    m = Manifest(env / "data" / "manifest.db")
    m.upsert_note(Note(id="a", title="a", body="b", path="a.md", hash="h"))
    m.add_chunks("a", ["p"])
    m.close()
    r = runner.invoke(cli.app, ["ideas"])
    assert r.exit_code == 1 and "v2v ingest" in r.output


def test_ingest_obsidian_vault_reports_entries(env, monkeypatch, vault_dir, write_note):
    monkeypatch.setenv("VAULT_FORMAT", "obsidian")
    write_note("Daily/10-10-2026.md", "## Games\n- Played [[Hades]], loved it.\n- Too tired.\n")
    write_note("Games/Hades.md", "Best roguelike dialogue I know.\n")
    r = runner.invoke(cli.app, ["ingest", "--vault", str(vault_dir)])
    assert r.exit_code == 0, r.output
    assert "new=2" in r.output
    assert "daily=1 entries=1 pages=1 unlinked_skipped=1 empty_skipped=0" in r.output


def test_ideas_show_kind(env, vault_dir, write_note):
    for i in range(6):
        write_note(f"n{i}.md", note_text(f"n{i}", f"boss fight fair difficulty note {i}"))
    runner.invoke(cli.app, ["ingest", "--vault", str(vault_dir)])
    r = runner.invoke(cli.app, ["ideas"])
    assert r.exit_code == 0 and "[theme]" in r.output
