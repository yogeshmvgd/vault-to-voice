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
