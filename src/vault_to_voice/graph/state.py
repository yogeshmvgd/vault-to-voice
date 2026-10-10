from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from vault_to_voice.ideas.detector import Idea


class SourceChunk(BaseModel):
    chunk_id: int
    text: str


class NoteContext(BaseModel):
    note_id: str
    title: str
    type: str
    date: str | None
    highlights: list[str] = Field(default_factory=list)
    chunks: list[SourceChunk]
    related: bool = False  # pulled in by FAISS search, not part of the idea's cluster
    work_id: str | None = None  # Obsidian format: the work this note is about

    @property
    def text(self) -> str:
        return "\n\n".join(c.text for c in self.chunks)


class Question(BaseModel):
    question: str
    grounding_note_ids: list[str] = Field(default_factory=list)


class QA(BaseModel):
    qa_id: str  # "qa:1", "qa:2", ... — citable like a note id
    question: str
    answer: str
    grounding_note_ids: list[str] = Field(default_factory=list)


class CoverageCheck(BaseModel):
    enough: bool
    missing: str = ""


class Section(BaseModel):
    id: str
    heading: str
    point: str
    planned_sources: list[str] = Field(default_factory=list)


class Outline(BaseModel):
    title: str
    sections: list[Section] = Field(min_length=1)


class Sentence(BaseModel):
    text: str
    sources: list[str] = Field(default_factory=list)
    kind: Literal["claim", "connective"] = "claim"


class Paragraph(BaseModel):
    sentences: list[Sentence] = Field(min_length=1)


class SectionText(BaseModel):
    """What the LLM returns for one section."""

    paragraphs: list[Paragraph] = Field(min_length=1)


class SectionDraft(BaseModel):
    section_id: str
    paragraphs: list[Paragraph]


class Flag(BaseModel):
    section_id: str
    paragraph: int
    sentence: int
    text: str
    reason: Literal["missing_source", "invalid_source", "unsupported"]
    score: float | None = None


class CitationReport(BaseModel):
    n_claims: int
    n_supported: int
    flags: list[Flag] = Field(default_factory=list)

    @property
    def coverage(self) -> float:
        return round(self.n_supported / self.n_claims, 3) if self.n_claims else 1.0


class CriticNote(BaseModel):
    section_id: str
    kind: Literal["voice", "substance", "clarity", "safety"]
    severity: Literal["minor", "major"] = "minor"
    note: str


class CriticReport(BaseModel):
    style_score: float | None = None  # 0-1 match with the style profile; None = no profile
    style_ok: bool = True  # style_score >= Settings.critic_min_style (or no profile)
    notes: list[CriticNote] = Field(default_factory=list)

    @property
    def needs_revision(self) -> bool:
        return not self.style_ok or any(n.severity == "major" for n in self.notes)


class ReviewDecision(BaseModel):
    action: Literal["approve", "revise", "edit"]
    feedback: str = ""
    edited: dict | None = None  # outline: Outline as dict; draft: {section_id: text}


def merge_drafts(left: dict | None, right: dict | None) -> dict:
    """Reducer for parallel draft_section writes: later writes replace the same section id."""
    return {**(left or {}), **(right or {})}


class PostState(TypedDict, total=False):
    idea: Idea
    notes: list[NoteContext]
    style: dict | None
    past_answers: list[QA]
    qa: list[QA]
    pending_question: Question | None
    interview_done: bool
    outline: Outline
    last_action: str
    review_error: str | None
    feedback: str
    section_feedback: dict[str, str]
    revise_ids: list[str]
    drafts: Annotated[dict[str, SectionDraft], merge_drafts]
    draft: list[SectionDraft]
    citation_report: CitationReport
    critic_report: CriticReport
    guard_warnings: list[str]
    auto_revised: bool
    user_rounds: dict[str, int]
    final_path: str | None


class DraftTask(TypedDict):
    """Payload sent to one parallel draft_section run."""

    section: Section
    outline_title: str
    notes: list[NoteContext]
    qa: list[QA]
    style: dict | None
    feedback: str


def valid_sources(notes: list[NoteContext], qa: list[QA]) -> set[str]:
    return {n.note_id for n in notes} | {q.qa_id for q in qa} | {"user"}
