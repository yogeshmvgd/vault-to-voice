from vault_to_voice.graph.state import Outline, SectionDraft


def render_markdown(outline: Outline, draft: list[SectionDraft]) -> str:
    """Clean Markdown: headings and prose only, no citation markers."""
    headings = {s.id: s.heading for s in outline.sections}
    parts = [f"# {outline.title}"]
    for section in draft:
        parts.append(f"## {headings.get(section.section_id, section.section_id)}")
        parts += [" ".join(s.text for s in p.sentences) for p in section.paragraphs]
    return "\n\n".join(parts) + "\n"
