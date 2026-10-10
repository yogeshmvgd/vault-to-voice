from statistics import mean

from pydantic import BaseModel, Field

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.prompts import style_brief
from vault_to_voice.graph.state import CriticNote, CriticReport, PostState, SectionDraft
from vault_to_voice.guardrails import DATA_INSTRUCTION, output_issues, scrub_pii, wrap_note
from vault_to_voice.style.stats import compute_stats

CRITIC_PROMPT = (
    "You are the editor of this blog. Compare the draft with the author's voice (see Voice) "
    "and judge its substance. Return notes; each has section_id (one of the ids below), kind "
    "(voice: does not sound like the author; substance: vague, generic or repetitive; clarity: "
    "hard to follow), severity ('major' only for a problem worth rewriting the section, else "
    "'minor') and note (one sentence: what to change). Return no notes for a good draft.\n"
)
# Measured voice features compared with the style profile (see style/stats.py).
SHAPE_FEATURES = ("avg_sentence_len", "avg_paragraph_len", "first_person_rate")
PUNCT_FEATURES = ("em_dash", "exclamation", "question")


class CriticReview(BaseModel):
    """What the LLM returns."""

    notes: list[CriticNote] = Field(default_factory=list)


def section_text(section: SectionDraft) -> str:
    return "\n\n".join(" ".join(s.text for s in p.sentences) for p in section.paragraphs)


def _closeness(a: float, b: float) -> float:
    """1.0 when equal, falling linearly to 0.0 when one value is zero and the other is not."""
    hi = max(abs(a), abs(b))
    return 1.0 if hi == 0 else 1.0 - abs(a - b) / hi


def style_score(draft: list[SectionDraft], style: dict | None) -> float | None:
    """0-1 similarity of the draft's measured stats to the author's style profile.

    Mean of four parts: sentence length, paragraph length, first-person rate and punctuation
    habits (em dash / ! / ? per 1000 words, averaged). None without a profile or a draft.
    """
    want = (style or {}).get("stats")
    text = "\n\n".join(section_text(s) for s in draft)
    if not want or not text.strip():
        return None
    got = compute_stats([text])
    parts = [_closeness(getattr(got, k), float(want.get(k, 0.0))) for k in SHAPE_FEATURES]
    punct = want.get("punctuation", {})
    parts.append(
        mean(_closeness(got.punctuation[k], float(punct.get(k, 0.0))) for k in PUNCT_FEATURES)
    )
    return round(mean(parts), 3)


def safety_notes(draft: list[SectionDraft]) -> list[CriticNote]:
    """Deterministic output guardrail: one major note per section with publishing problems."""
    notes = []
    for section in draft:
        issues = output_issues(section_text(section))
        if issues:
            notes.append(
                CriticNote(
                    section_id=section.section_id,
                    kind="safety",
                    severity="major",
                    note="Remove this before publishing: " + "; ".join(issues) + ".",
                )
            )
    return notes


def critic(state: PostState, deps: GraphDeps) -> dict:
    draft, style = state["draft"], state.get("style")
    score = style_score(draft, style)
    headings = {s.id: s.heading for s in state["outline"].sections}
    sections = "\n".join(
        wrap_note(d.section_id, scrub_pii(f"{headings.get(d.section_id, '')}\n{section_text(d)}"))
        for d in draft
    )
    measured = "no style profile" if score is None else f"{score:.2f} (1.0 = identical)"
    prompt = "\n\n".join(
        [
            CRITIC_PROMPT,
            "Voice:",
            style_brief(style),
            f"Measured style match: {measured}",
            "Draft sections:",
            sections,
        ]
    )
    review = deps.llm.structured(prompt, CriticReview, system=DATA_INSTRUCTION)
    known = {d.section_id for d in draft}
    notes = [n for n in review.notes if n.section_id in known and n.kind != "safety"]
    report = CriticReport(
        style_score=score,
        style_ok=score is None or score >= deps.settings.critic_min_style,
        notes=notes + safety_notes(draft),
    )
    return {"critic_report": report}
