import json

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.memory import save_answers
from vault_to_voice.graph.render import render_markdown
from vault_to_voice.graph.state import PostState
from vault_to_voice.publish import RenderedPost
from vault_to_voice.vault.reader import slugify


def publish(state: PostState, deps: GraphDeps) -> dict:
    if deps.publisher is None:
        raise RuntimeError("GraphDeps.publisher is not configured")
    idea, outline, draft = state["idea"], state["outline"], state["draft"]
    report = state["citation_report"]
    critic = state.get("critic_report")
    qa = state.get("qa", [])
    today = deps.today()
    front = "\n".join(
        [
            "---",
            f"title: {json.dumps(outline.title)}",
            f"date: {today.isoformat()}",
            f"idea_id: {idea.idea_id}",
            f"theme: {json.dumps(idea.theme)}",
            f"notes: {json.dumps(idea.note_ids)}",
            "---",
            "",
        ]
    )
    headings = {s.id: s.heading for s in outline.sections}
    sources = {
        "idea_id": idea.idea_id,
        "title": outline.title,
        "date": today.isoformat(),
        "citation_coverage": report.coverage,
        "flags": [f.model_dump() for f in report.flags],
        "style_score": critic.style_score if critic else None,
        "critic_notes": [n.model_dump() for n in critic.notes] if critic else [],
        "guard_warnings": state.get("guard_warnings", []),
        "paragraphs": [
            {
                "section_id": section.section_id,
                "heading": headings.get(section.section_id),
                "text": " ".join(s.text for s in p.sentences),
                "sources": sorted({src for s in p.sentences for src in s.sources}),
                "sentences": [s.model_dump() for s in p.sentences],
            }
            for section in draft
            for p in section.paragraphs
        ],
        "interview": [q.model_dump() for q in qa],
        "source_details": {
            n.note_id: {"title": n.title, "date": n.date, "work_id": n.work_id}
            for n in state["notes"]
        },
    }
    post = RenderedPost(
        slug=slugify(outline.title) or idea.idea_id,
        date=today,
        markdown=front + "\n" + render_markdown(outline, draft),
        sources=sources,
    )
    path = deps.publisher.publish(post)
    save_answers(deps.store, idea.idea_id, qa)
    deps.manifest.set_idea_status(idea.idea_id, "done")
    return {"final_path": str(path)}
