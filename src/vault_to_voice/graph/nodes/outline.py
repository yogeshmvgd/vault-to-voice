from langgraph.types import interrupt
from pydantic import ValidationError

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.prompts import notes_block, qa_block, style_brief
from vault_to_voice.graph.state import Outline, PostState, ReviewDecision, valid_sources
from vault_to_voice.guardrails import DATA_INSTRUCTION

OUTLINE_PROMPT = (
    "Plan a blog post in the author's voice.\n"
    "title: the post title. sections: 3-6 sections, each with heading, point (one sentence: "
    "what the section argues) and planned_sources (ids of the notes or interview answers below, "
    "e.g. a note id or 'qa:2', that support it). Use only ids that appear below.\n"
)


def _normalise(outline: Outline, valid: set[str]) -> Outline:
    sections = [
        s.model_copy(
            update={"id": f"s{i}", "planned_sources": [x for x in s.planned_sources if x in valid]}
        )
        for i, s in enumerate(outline.sections, 1)
    ]
    return outline.model_copy(update={"sections": sections})


def _error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    return f"Invalid decision: {first['msg']} at {'.'.join(map(str, first['loc'])) or 'payload'}"


def make_outline(state: PostState, deps: GraphDeps) -> dict:
    idea = state["idea"]
    qa = state.get("qa", [])
    parts = [
        OUTLINE_PROMPT,
        f"Post idea: {idea.title} — {idea.angle}",
        "Voice:",
        style_brief(state.get("style")),
        "Notes:",
        notes_block(state["notes"]),
        "Interview answers:",
        qa_block(qa),
    ]
    if state.get("feedback") and state.get("outline"):
        parts += [
            "Previous outline:",
            state["outline"].model_dump_json(),
            "Revise it using this feedback from the author:",
            state["feedback"],
        ]
    outline = deps.llm.structured("\n\n".join(parts), Outline, system=DATA_INSTRUCTION)
    return {"outline": _normalise(outline, valid_sources(state["notes"], qa)), "review_error": None}


def _approved(outline: Outline) -> dict:
    return {
        "outline": outline,
        "last_action": "approve",
        "review_error": None,
        "revise_ids": [s.id for s in outline.sections],
        "feedback": "",
        "section_feedback": {},
    }


def _retry(message: str) -> dict:
    return {"last_action": "retry", "review_error": message}


def outline_review(state: PostState, deps: GraphDeps) -> dict:
    rounds = state.get("user_rounds") or {"outline": 0, "draft": 0}
    left = deps.settings.max_review_rounds - rounds["outline"]
    payload = interrupt(
        {
            "type": "outline_review",
            "outline": state["outline"].model_dump(),
            "rounds_left": left,
            "error": state.get("review_error"),
        }
    )
    try:
        decision = ReviewDecision.model_validate(payload)
    except ValidationError as exc:
        return _retry(_error(exc))
    if decision.action == "approve":
        return _approved(state["outline"])
    if decision.action == "revise":
        if left <= 0:
            return _retry("Revision limit reached — approve or edit the outline.")
        return {
            "last_action": "revise",
            "feedback": decision.feedback,
            "review_error": None,
            "user_rounds": {**rounds, "outline": rounds["outline"] + 1},
        }
    try:
        edited = Outline.model_validate(decision.edited)
    except ValidationError as exc:
        return _retry(_error(exc))
    return _approved(_normalise(edited, valid_sources(state["notes"], state.get("qa", []))))


def route_after_outline_review(state: PostState) -> str:
    return {"revise": "make_outline", "retry": "outline_review"}.get(
        state["last_action"], "dispatch"
    )
