from pydantic import BaseModel, Field

from vault_to_voice.guardrails import DATA_INSTRUCTION, scrub_pii, wrap_note
from vault_to_voice.llm import LLMClient
from vault_to_voice.vault.models import Note

OPINION_PROMPT = (
    "Score how much this note expresses the author's personal opinion versus plain summary.\n"
    "score: 0.0 = pure plot/feature summary, 1.0 = strongly opinionated "
    "(judgements, feelings, arguments).\n"
    "highlights: up to 5 short phrases capturing the author's opinions (empty if none).\n\n"
)


class OpinionScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    highlights: list[str] = Field(default_factory=list, max_length=5)


class OpinionScorer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def score(self, note: Note) -> OpinionScore:
        prompt = OPINION_PROMPT + wrap_note(note.id, scrub_pii(note.body))
        return self.llm.structured(prompt, OpinionScore, system=DATA_INSTRUCTION)
