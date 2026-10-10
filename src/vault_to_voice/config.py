from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_VAULT = REPO_ROOT / "sample_vault"


class VaultPathError(ValueError):
    pass


def check_vault_path(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    resolved = Path(path).expanduser().resolve()
    root = repo_root.resolve()
    sample = (root / "sample_vault").resolve()
    if resolved.is_relative_to(root) and not resolved.is_relative_to(sample):
        raise VaultPathError(
            f"VAULT_PATH {resolved} is inside the repo. Only sample_vault/ may live in the "
            f"repo; point VAULT_PATH at your vault outside {root}."
        )
    return resolved


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", validate_default=True)

    vault_path: Path = SAMPLE_VAULT
    data_dir: Path = REPO_ROOT / "data"
    outbox_dir: Path = REPO_ROOT / "outbox"

    llm_model: str = "openai:gpt-4o-mini"
    llm_timeout: float = 60.0
    llm_max_retries: int = 3

    embedding_provider: Literal["local", "openai", "hash"] = "local"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    min_notes: int = 4
    min_opinionated: int = 3
    opinion_threshold: float = 0.5
    seed: int = 42

    @field_validator("vault_path")
    @classmethod
    def _vault_outside_repo(cls, v: Path) -> Path:
        return check_vault_path(v)

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.db"

    @property
    def faiss_path(self) -> Path:
        return self.data_dir / "faiss.index"

    @property
    def style_profile_path(self) -> Path:
        return self.data_dir / "style_profile.json"
