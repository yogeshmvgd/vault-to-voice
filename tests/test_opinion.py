from tests.fakes import FakeLLMClient
from vault_to_voice.guardrails import DATA_INSTRUCTION
from vault_to_voice.ideas.opinion import OpinionScore, OpinionScorer
from vault_to_voice.vault.models import Note


def test_scores_note_with_wrapped_scrubbed_text():
    fake = FakeLLMClient({OpinionScore: OpinionScore(score=0.9, highlights=["fair fights"])})
    note = Note(id="n1", title="t", body="Loved it. mail a@b.com", path="n1.md", hash="h")
    result = OpinionScorer(fake).score(note)
    assert result.score == 0.9
    schema, prompt = fake.calls[0]
    assert '<note id="n1">' in prompt and "[EMAIL]" in prompt and "a@b.com" not in prompt


def test_system_prompt_is_data_instruction():
    seen = {}

    class Spy(FakeLLMClient):
        def structured(self, prompt, schema, *, system=None):
            seen["system"] = system
            return super().structured(prompt, schema, system=system)

    note = Note(id="n", title="t", body="b", path="n.md", hash="h")
    OpinionScorer(Spy({OpinionScore: OpinionScore(score=0.1)})).score(note)
    assert seen["system"] == DATA_INSTRUCTION
