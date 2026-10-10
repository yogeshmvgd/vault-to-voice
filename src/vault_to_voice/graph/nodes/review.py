from collections import defaultdict

from langgraph.types import interrupt
from pydantic import ValidationError

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.render import render_markdown
from vault_to_voice.graph.state import Paragraph, PostState, ReviewDecision, SectionDraft, Sentence
from vault_to_voice.index.chunking import split_paragraphs
from vault_to_voice.style.stats import split_sentences

CITATION_HEADER = (
    "These sentences are not supported by their cited sources. Cite the right note or "
    "answer ids, rewrite them so the sources support them, or drop them:\n"
)
CRITIC_HEADER = "The editor found these problems; fix them:\n"
VOICE_LINE = (
    "- [voice] The draft does not sound like the author (style match {score}): follow the "
    "Voice section for sentence length, first person and punctuation habits."
)


def route_after_checks(state: PostState) -> str:
    critic = state.get("critic_report")
    problems = state["citation_report"].flags or (critic is not None and critic.needs_revision)
    if problems and not state.get("auto_revised"):
        return "prepare_auto_revise"
    return "draft_review"


def prepare_auto_revise(state: PostState) -> dict:
    """One automatic revision of the sections with citation flags or major critic notes."""
    cited: dict[str, list[str]] = defaultdict(list)
    for flag in state["citation_report"].flags:
        cited[flag.section_id].append(f'- "{flag.text}" ({flag.reason.replace("_", " ")})')
    edits: dict[str, list[str]] = defaultdict(list)
    critic = state.get("critic_report")
    if critic is not None:
        for note in critic.notes:
            if note.severity == "major":
                edits[note.section_id].append(f"- [{note.kind}] {note.note}")
        if not critic.style_ok:
            for section in state.get("draft", []):
                edits[section.section_id].append(VOICE_LINE.format(score=critic.style_score))
    feedback = {}
    for sid in sorted(set(cited) | set(edits)):
        parts = []
        if cited[sid]:
            parts.append(CITATION_HEADER + "\n".join(cited[sid]))
        if edits[sid]:
            parts.append(CRITIC_HEADER + "\n".join(edits[sid]))
        feedback[sid] = "\n\n".join(parts)
    return {
        "revise_ids": sorted(feedback),
        "section_feedback": feedback,
        "feedback": "",
        "auto_revised": True,
    }


def _retry(message: str) -> dict:
    return {"last_action": "retry", "review_error": message}


def _user_section(section_id: str, text: str) -> SectionDraft:
    paragraphs = [
        Paragraph(sentences=[Sentence(text=s, sources=["user"]) for s in split_sentences(p)])
        for p in split_paragraphs(text)
    ]
    return SectionDraft(section_id=section_id, paragraphs=paragraphs)


def draft_review(state: PostState, deps: GraphDeps) -> dict:
    rounds = state.get("user_rounds") or {"outline": 0, "draft": 0}
    left = deps.settings.max_review_rounds - rounds["draft"]
    report = state["citation_report"]
    critic = state.get("critic_report")
    payload = interrupt(
        {
            "type": "draft_review",
            "markdown": render_markdown(state["outline"], state["draft"]),
            "draft": [d.model_dump() for d in state["draft"]],
            "flags": [f.model_dump() for f in report.flags],
            "coverage": report.coverage,
            "critic_notes": [n.model_dump() for n in critic.notes] if critic else [],
            "style_score": critic.style_score if critic else None,
            "warnings": state.get("guard_warnings", []),
            "rounds_left": left,
            "error": state.get("review_error"),
        }
    )
    try:
        decision = ReviewDecision.model_validate(payload)
    except ValidationError as exc:
        return _retry(f"Invalid decision: {exc.errors()[0]['msg']}")
    if decision.action == "approve":
        return {"last_action": "publish", "review_error": None}
    if decision.action == "revise":
        if left <= 0:
            return _retry("Revision limit reached — approve or edit the draft.")
        return {
            "last_action": "dispatch",
            "review_error": None,
            "revise_ids": [s.id for s in state["outline"].sections],
            "feedback": decision.feedback,
            "section_feedback": {},
            "user_rounds": {**rounds, "draft": rounds["draft"] + 1},
        }
    edited = decision.edited or {}
    known = {d.section_id for d in state["draft"]}
    if (
        not edited
        or not set(edited) <= known
        or not all(isinstance(v, str) and v.strip() for v in edited.values())
    ):
        return _retry("Edit needs {section_id: text} for existing sections, with non-empty text.")
    new = {sid: _user_section(sid, text) for sid, text in edited.items()}
    return {
        "drafts": new,
        "draft": [new.get(d.section_id, d) for d in state["draft"]],
        "last_action": "check",
        "review_error": None,
    }


def route_after_draft_review(state: PostState) -> str:
    return {
        "publish": "publish",
        "dispatch": "dispatch",
        "check": "check_citations",
    }.get(state["last_action"], "draft_review")
