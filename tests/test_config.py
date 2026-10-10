import pytest
from pydantic import ValidationError

from vault_to_voice.config import REPO_ROOT, SAMPLE_VAULT, Settings


def test_default_vault_is_sample_vault():
    assert Settings(_env_file=None).vault_path == SAMPLE_VAULT.resolve()


def test_sample_vault_subfolder_allowed():
    s = Settings(_env_file=None, vault_path=SAMPLE_VAULT / "games")
    assert s.vault_path == (SAMPLE_VAULT / "games").resolve()


def test_path_outside_repo_allowed(tmp_path):
    assert Settings(_env_file=None, vault_path=tmp_path).vault_path == tmp_path.resolve()


@pytest.mark.parametrize("inside", ["my-notes", "vault", "."])
def test_path_inside_repo_rejected(inside):
    with pytest.raises(ValidationError, match="inside the repo"):
        Settings(_env_file=None, vault_path=REPO_ROOT / inside)


def test_vault_path_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VAULT_PATH", str(tmp_path))
    assert Settings(_env_file=None).vault_path == tmp_path.resolve()


def test_derived_paths(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert s.manifest_path == tmp_path / "manifest.db"
    assert s.faiss_path == tmp_path / "faiss.index"
    assert s.style_profile_path == tmp_path / "style_profile.json"


def test_obsidian_defaults_and_categories():
    s = Settings(_env_file=None)
    assert (s.vault_format, s.daily_folder, s.daily_date_format) == (
        "obsidian",
        "Daily",
        "%d-%m-%Y",
    )
    assert s.categories == {"games": "game", "books": "book", "movies": "movie"}
    custom = Settings(_env_file=None, category_headings="Gaming=game, Reading = book")
    assert custom.categories == {"gaming": "game", "reading": "book"}


def test_invalid_category_type_rejected():
    with pytest.raises(ValidationError, match="category"):
        Settings(_env_file=None, category_headings="games=videogame")


def test_provider_keys_from_env_file_exported(tmp_path, monkeypatch):
    from vault_to_voice.config import export_provider_keys

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-from-file\nANTHROPIC_API_KEY=\n", encoding="utf-8")
    s = Settings(_env_file=env)
    assert "sk-from-file" not in repr(s)
    export_provider_keys(s)
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-from-file"
    assert "ANTHROPIC_API_KEY" not in os.environ  # blank keys are not exported


def test_real_env_var_wins_over_env_file(tmp_path, monkeypatch):
    import os

    from vault_to_voice.config import export_provider_keys

    monkeypatch.setenv("OPENAI_API_KEY", "sk-shell")
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-file\n", encoding="utf-8")
    export_provider_keys(Settings(_env_file=env))
    assert os.environ["OPENAI_API_KEY"] == "sk-shell"
