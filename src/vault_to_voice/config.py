from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
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


_NOTE_TYPES = {"game", "book", "movie", "other"}


def _parse_categories(raw: str) -> dict[str, str]:
    out = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        heading, _, note_type = pair.partition("=")
        note_type = note_type.strip().lower()
        if note_type not in _NOTE_TYPES:
            raise ValueError(f"category heading {heading!r} maps to unknown type {note_type!r}")
        out[heading.strip().lower()] = note_type
    return out


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
    interview_min_questions: int = 3
    interview_max_questions: int = 5
    max_review_rounds: int = 3
    citation_support: float = 0.55
    citation_reject: float = 0.30
    related_notes: int = 2
    related_min_score: float = 0.5
    vault_format: Literal["obsidian", "flat"] = "obsidian"
    daily_folder: str = "Daily"
    daily_date_format: str = "%d-%m-%Y"
    category_headings: str = "games=game,books=book,movies=movie"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "vault-to-voice"

    @property
    def categories(self) -> dict[str, str]:
        return _parse_categories(self.category_headings)

    @model_validator(mode="after")
    def _check_categories(self):
        _parse_categories(self.category_headings)
        return self

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

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.db"

    @property
    def store_path(self) -> Path:
        return self.data_dir / "store.db"
