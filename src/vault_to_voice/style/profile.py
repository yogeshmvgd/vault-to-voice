import random
import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from vault_to_voice.guardrails import DATA_INSTRUCTION, scrub_pii, wrap_note
from vault_to_voice.llm import LLMClient
from vault_to_voice.style.stats import StyleStats, compute_stats
from vault_to_voice.vault.models import Note

TONE_PROMPT = (
    "Describe the writing voice of the author of these notes.\n"
    "tone: 3-6 adjectives. voice_notes: 2-3 sentences on how they write (rhythm, humour, how "
    "they express opinions). exemplars: 3-5 sentences copied VERBATIM from the notes that best "
    "show the voice.\n\n"
)


class ToneSummary(BaseModel):
    tone: list[str] = Field(min_length=1, max_length=6)
    voice_notes: str
    exemplars: list[str] = Field(default_factory=list)


class StyleProfile(BaseModel):
    version: int
    created_at: datetime
    stats: StyleStats
    tone: ToneSummary


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def build_style_profile(
    notes: list[Note], llm: LLMClient, *, sample_size: int = 15, seed: int = 42
) -> StyleProfile:
    stats = compute_stats([n.body for n in notes])
    rng = random.Random(seed)
    sample = notes if len(notes) <= sample_size else rng.sample(notes, sample_size)
    prompt = TONE_PROMPT + "\n".join(wrap_note(n.id, scrub_pii(n.body)) for n in sample)
    tone = llm.structured(prompt, ToneSummary, system=DATA_INSTRUCTION)
    corpus = _norm(" ".join(n.body for n in notes))
    verbatim = [e for e in tone.exemplars if _norm(e) in corpus]  # drop hallucinated exemplars
    tone = tone.model_copy(update={"exemplars": verbatim})
    return StyleProfile(version=1, created_at=datetime.now(UTC), stats=stats, tone=tone)


def load_profile(path: Path) -> StyleProfile | None:
    path = Path(path)
    if not path.exists():
        return None
    return StyleProfile.model_validate_json(path.read_text(encoding="utf-8"))


def save_profile(profile: StyleProfile, path: Path) -> StyleProfile:
    path = Path(path)
    existing = load_profile(path)
    if existing is not None:
        profile = profile.model_copy(update={"version": existing.version + 1})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return profile
