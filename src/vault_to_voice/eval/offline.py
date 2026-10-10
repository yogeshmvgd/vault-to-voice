"""A deterministic, extractive stand-in for the LLM so `v2v eval` runs offline.

It reads the same prompts the real model gets and answers from the note text inside them:
drafts copy the first sentence of each planned source (so citations are real), the judge
supports a claim only if it appears verbatim in its sources, and the critic has no notes.
It measures the pipeline (retrieval, citation checks, style score, publishing), not writing.
"""

import re
from collections.abc import Callable
from typing import Any

from vault_to_voice.graph.nodes.citations import JudgeVerdict, JudgeVerdicts
from vault_to_voice.graph.nodes.critic import CriticReview
from vault_to_voice.graph.state import (
    CoverageCheck,
    Outline,
    Paragraph,
    Question,
    Section,
    SectionText,
    Sentence,
)
from vault_to_voice.style.profile import ToneSummary
from vault_to_voice.style.stats import split_sentences

_BLOCK = re.compile(r'<note id="([^"]+)">\n(.*?)\n</note>', re.DOTALL)
_ITEM = re.compile(r"^\[(\d+)\] claim: (.*)$", re.MULTILINE)
_IDEA = re.compile(r"^Post idea: (.*?) — ", re.MULTILINE)
_PLANNED = re.compile(r"^Planned sources: (.*)$", re.MULTILINE)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _blocks(prompt: str) -> dict[str, str]:
    """id -> body for every <note> block (note header line / 'Q:' line removed)."""
    out = {}
    for block_id, body in _BLOCK.findall(prompt):
        if block_id.startswith("qa:"):
            out[block_id] = body.split("\nA: ", 1)[-1]
        else:
            out[block_id] = body.split("\n", 1)[-1]
    return out


def _first_sentence(text: str) -> str:
    sentences = split_sentences(text)
    return sentences[0] if sentences else text.strip()


class OfflineLLM:
    """LLMClient for offline evaluation (see module docstring)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def structured(self, prompt: str, schema: type, *, system: str | None = None) -> Any:
        self.calls.append(schema.__name__)
        handlers: dict[type, Callable[[str], Any]] = {
            Question: self._question,
            CoverageCheck: lambda _: CoverageCheck(enough=True),
            Outline: self._outline,
            SectionText: self._section,
            JudgeVerdicts: self._judge,
            CriticReview: lambda _: CriticReview(),
            ToneSummary: lambda _: ToneSummary(
                tone=["candid"], voice_notes="Offline placeholder tone.", exemplars=[]
            ),
        }
        handler = handlers.get(schema)
        if handler is None:
            raise KeyError(f"OfflineLLM has no answer for {schema.__name__}")
        return handler(prompt)

    def _question(self, _prompt: str) -> Question:
        return Question(question="What stayed with you most, and why?")

    def _outline(self, prompt: str) -> Outline:
        match = _IDEA.search(prompt)
        ids = list(_blocks(prompt))
        notes = [i for i in ids if not i.startswith("qa:")]
        answers = [i for i in ids if i.startswith("qa:")]
        half = max(1, (len(notes) + 1) // 2)
        groups = [notes[:half], notes[half:]] if notes[half:] else [notes[:half]]
        sections = [
            Section(
                id=f"s{n}",
                heading=f"Part {n}",
                point="What the notes say.",
                planned_sources=group[:2] + answers[n - 1 : n],
            )
            for n, group in enumerate(groups, 1)
        ]
        return Outline(title=match.group(1) if match else "Untitled", sections=sections)

    def _section(self, prompt: str) -> SectionText:
        blocks = _blocks(prompt)
        match = _PLANNED.search(prompt)
        planned = [s.strip() for s in match.group(1).split(",")] if match else []
        sources = [s for s in planned if s in blocks] or list(blocks)[:1]
        claims = [Sentence(text=_first_sentence(blocks[s]), sources=[s]) for s in sources]
        connective = Sentence(text="That stayed with me.", kind="connective")
        return SectionText(paragraphs=[Paragraph(sentences=[*claims, connective])])

    def _judge(self, prompt: str) -> JudgeVerdicts:
        items = list(_ITEM.finditer(prompt))
        verdicts = []
        for k, item in enumerate(items):
            end = items[k + 1].start() if k + 1 < len(items) else len(prompt)
            sources = " ".join(_blocks(prompt[item.end() : end]).values())
            supported = _norm(item.group(2)) in _norm(sources)
            verdicts.append(JudgeVerdict(index=int(item.group(1)), supported=supported))
        return JudgeVerdicts(verdicts=verdicts)
