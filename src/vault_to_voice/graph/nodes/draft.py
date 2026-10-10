from langgraph.types import Send

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.prompts import notes_block, qa_block, style_brief
from vault_to_voice.graph.state import DraftTask, PostState, SectionDraft, SectionText
from vault_to_voice.guardrails import DATA_INSTRUCTION

DRAFT_PROMPT = (
    "Write one section of a blog post in the author's voice (see Voice).\n"
    "Return paragraphs made of sentences; each sentence has text, kind and sources.\n"
    "kind='claim' for any sentence stating a fact, opinion, memory or example: its sources MUST "
    "list the ids (note ids or 'qa:N') whose text supports it. kind='connective' only for short "
    "transitions with no content of their own; they have no sources.\n"
    "Never cite an id that does not appear below. Do not invent experiences.\n"
)


def fan_out_sections(state: PostState) -> list[Send]:
    """Conditional edge: one parallel draft_section run per section that needs (re)drafting."""
    sections = {s.id: s for s in state["outline"].sections}
    section_feedback = state.get("section_feedback") or {}
    sends = []
    for sid in state["revise_ids"]:
        feedback = "\n".join(f for f in [state.get("feedback"), section_feedback.get(sid)] if f)
        task: DraftTask = {
            "section": sections[sid],
            "outline_title": state["outline"].title,
            "notes": state["notes"],
            "qa": state.get("qa", []),
            "style": state.get("style"),
            "feedback": feedback,
        }
        sends.append(Send("draft_section", task))
    return sends


def draft_section(task: DraftTask, deps: GraphDeps) -> dict:
    section = task["section"]
    parts = [
        DRAFT_PROMPT,
        f"Post title: {task['outline_title']}",
        f"Section: {section.heading}\nPoint: {section.point}\n"
        f"Planned sources: {', '.join(section.planned_sources) or '(none)'}",
        "Voice:",
        style_brief(task["style"]),
        "Notes:",
        notes_block(task["notes"]),
        "Interview answers:",
        qa_block(task["qa"]),
    ]
    if task["feedback"]:
        parts += ["Fix these problems from the previous version:", task["feedback"]]
    text = deps.llm.structured("\n\n".join(parts), SectionText, system=DATA_INSTRUCTION)
    return {"drafts": {section.id: SectionDraft(section_id=section.id, paragraphs=text.paragraphs)}}


def merge_draft(state: PostState) -> dict:
    drafts = state.get("drafts") or {}
    return {"draft": [drafts[s.id] for s in state["outline"].sections if s.id in drafts]}
