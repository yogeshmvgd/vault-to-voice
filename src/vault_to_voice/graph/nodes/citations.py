from pydantic import BaseModel

from vault_to_voice.graph.deps import GraphDeps
from vault_to_voice.graph.state import CitationReport, Flag, PostState, Sentence, valid_sources
from vault_to_voice.guardrails import DATA_INSTRUCTION, scrub_pii, wrap_note

JUDGE_PROMPT = (
    "For each numbered claim, decide whether its cited source texts support it. A claim is "
    "supported if the sources state it or clearly imply it. Return one verdict per claim index.\n"
)


class JudgeVerdict(BaseModel):
    index: int
    supported: bool


class JudgeVerdicts(BaseModel):
    verdicts: list[JudgeVerdict]


def _support(deps: GraphDeps, text: str, source_texts: list[str]) -> float:
    vectors = deps.embedder.embed([scrub_pii(t) for t in [text, *source_texts]])
    return round(float((vectors[1:] @ vectors[0]).max()), 3)


def _judge(
    deps: GraphDeps, grey: list[tuple[dict, Sentence, float]], texts: dict[str, list[str]]
) -> dict[int, bool]:
    items = []
    for i, (_, sentence, _) in enumerate(grey):
        sources = "\n".join(
            wrap_note(src, scrub_pii("\n".join(texts[src]))) for src in sentence.sources
        )
        items.append(f"[{i}] claim: {sentence.text}\nsources:\n{sources}")
    out = deps.llm.structured(
        JUDGE_PROMPT + "\n\n".join(items), JudgeVerdicts, system=DATA_INSTRUCTION
    )
    return {v.index: v.supported for v in out.verdicts}


def check_citations(state: PostState, deps: GraphDeps) -> dict:
    qa = state.get("qa", [])
    valid = valid_sources(state["notes"], qa)
    texts = {n.note_id: [c.text for c in n.chunks] for n in state["notes"]}
    texts.update({q.qa_id: [q.answer] for q in qa})
    settings = deps.settings

    flags: list[Flag] = []
    grey: list[tuple[dict, Sentence, float]] = []
    n_claims = n_supported = 0
    for section in state["draft"]:
        for pi, paragraph in enumerate(section.paragraphs):
            for si, sentence in enumerate(paragraph.sentences):
                if sentence.kind == "connective":
                    continue
                n_claims += 1
                loc = {
                    "section_id": section.section_id,
                    "paragraph": pi,
                    "sentence": si,
                    "text": sentence.text,
                }
                if not sentence.sources:
                    flags.append(Flag(**loc, reason="missing_source"))
                elif any(src not in valid for src in sentence.sources):
                    flags.append(Flag(**loc, reason="invalid_source"))
                elif "user" in sentence.sources:
                    n_supported += 1
                else:
                    cited = [t for src in sentence.sources for t in texts[src]]
                    score = _support(deps, sentence.text, cited)
                    if score >= settings.citation_support:
                        n_supported += 1
                    elif score <= settings.citation_reject:
                        flags.append(Flag(**loc, reason="unsupported", score=score))
                    else:
                        grey.append((loc, sentence, score))
    if grey:
        verdicts = _judge(deps, grey, texts)
        for i, (loc, _, score) in enumerate(grey):
            if verdicts.get(i, False):
                n_supported += 1
            else:
                flags.append(Flag(**loc, reason="unsupported", score=score))
    report = CitationReport(n_claims=n_claims, n_supported=n_supported, flags=flags)
    return {"citation_report": report}
