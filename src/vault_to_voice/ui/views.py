"""Pure helpers for the Streamlit UI (no streamlit import, so they unit-test directly)."""

from collections.abc import Callable, Iterable
from typing import Any

Event = tuple[str, Any]


def last_event(events: Iterable[Event], on_node: Callable[[str], None] | None = None) -> Event:
    """Drain a run's events, reporting each finished node; return the final event
    (interrupt / done / error). A stream that ends without one is reported as an error."""
    last: Event | None = None
    for name, data in events:
        if name == "node":
            if on_node is not None:
                on_node(data["node"])
        else:
            last = (name, data)
    return last or ("error", {"message": "The run ended without a result; try Retry."})


def _md(text: str) -> str:
    """Escape characters Streamlit markdown would treat as markup inside a sentence."""
    for ch in "\\[]*_`#$":
        text = text.replace(ch, "\\" + ch)
    return text


def flagged_markdown(draft: list[dict], flags: list[dict], headings: dict[str, str]) -> str:
    """Draft as markdown with every flagged sentence shown in red with its reason."""
    bad = {(f["section_id"], f["paragraph"], f["sentence"]): f["reason"] for f in flags}
    parts: list[str] = []
    for section in draft:
        sid = section["section_id"]
        parts.append(f"### {headings.get(sid, sid)}")
        for pi, paragraph in enumerate(section["paragraphs"]):
            out = []
            for si, sentence in enumerate(paragraph["sentences"]):
                text = _md(sentence["text"])
                reason = bad.get((sid, pi, si))
                out.append(f":red[{text}] :red[_({reason.replace('_', ' ')})_]" if reason else text)
            parts.append(" ".join(out))
    return "\n\n".join(parts)


def section_texts(draft: list[dict]) -> dict[str, str]:
    """Editable plain text per section: sentences joined by spaces, paragraphs by a blank line."""
    return {
        s["section_id"]: "\n\n".join(
            " ".join(sent["text"] for sent in p["sentences"]) for p in s["paragraphs"]
        )
        for s in draft
    }


def changed_sections(original: dict[str, str], edited: dict[str, str]) -> dict[str, str]:
    """Only the sections whose text the author actually changed."""
    return {
        sid: text
        for sid, text in edited.items()
        if sid in original and text.strip() != original[sid].strip()
    }


def edited_outline(outline: dict, title: str, headings: list[str], points: list[str]) -> dict:
    """Outline dict with the author's title/headings/points; ids and planned sources kept."""
    sections = [
        {**section, "heading": heading.strip(), "point": point.strip()}
        for section, heading, point in zip(outline["sections"], headings, points, strict=True)
    ]
    return {"title": title.strip(), "sections": sections}


def strip_frontmatter(markdown: str) -> str:
    """Post body without its leading `---` YAML block (for preview only)."""
    if markdown.startswith("---\n"):
        end = markdown.find("\n---\n", 4)
        if end != -1:
            return markdown[end + len("\n---\n") :].lstrip("\n")
    return markdown
