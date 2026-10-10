from vault_to_voice.graph.state import QA, NoteContext
from vault_to_voice.guardrails import scrub_pii, wrap_note


def notes_block(notes: list[NoteContext]) -> str:
    return "\n".join(
        wrap_note(n.note_id, scrub_pii(f"{n.title} ({n.type}, {n.date or 'undated'})\n{n.text}"))
        for n in notes
    )


def qa_block(qa: list[QA]) -> str:
    return "\n".join(wrap_note(q.qa_id, scrub_pii(f"Q: {q.question}\nA: {q.answer}")) for q in qa)


def style_brief(style: dict | None) -> str:
    if not style:
        return "No style profile yet: write in a natural, first-person blog voice."
    tone = style.get("tone", {})
    stats = style.get("stats", {})
    lines = [f"Tone: {', '.join(tone.get('tone', []))}", f"Voice: {tone.get('voice_notes', '')}"]
    if stats:
        lines.append(
            f"Average sentence length: {stats.get('avg_sentence_len')} words; "
            f"first-person rate: {stats.get('first_person_rate')}"
        )
    if stats.get("top_phrases"):
        lines.append("Recurring phrases: " + ", ".join(stats["top_phrases"][:8]))
    if tone.get("exemplars"):
        lines.append("Example sentences:\n" + "\n".join(f"- {e}" for e in tone["exemplars"]))
    return "\n".join(lines)
