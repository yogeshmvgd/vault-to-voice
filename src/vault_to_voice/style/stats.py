import re
from collections import Counter
from statistics import mean, median

from pydantic import BaseModel

from vault_to_voice.index.chunking import split_paragraphs

_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+")
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"}
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "it",
    "is",
    "was",
    "that",
    "this",
    "for",
    "with",
    "as",
    "at",
    "be",
    "but",
    "by",
    "i",
}
PUNCT = {
    "em_dash": ("—", "--"),
    "exclamation": ("!",),
    "question": ("?",),
    "ellipsis": ("...", "…"),
    "semicolon": (";",),
    "parenthesis": ("(",),
}


class StyleStats(BaseModel):
    n_notes: int
    n_sentences: int
    n_words: int
    avg_sentence_len: float
    median_sentence_len: float
    avg_paragraph_len: float
    type_token_ratio: float
    first_person_rate: float
    punctuation: dict[str, float]
    top_phrases: list[str]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(re.sub(r"\s+", " ", text)) if s.strip()]


def top_phrases(texts: list[str], *, min_count: int = 3, min_docs: int = 2, limit: int = 15):
    total: Counter[str] = Counter()
    docs: Counter[str] = Counter()
    for text in texts:
        toks = [w.lower() for w in _WORD.findall(text)]
        grams = Counter(
            " ".join(toks[i : i + n]) for n in (2, 3, 4) for i in range(len(toks) - n + 1)
        )
        total.update(grams)
        docs.update(grams.keys())
    keep = [
        g
        for g, c in total.items()
        if c >= min_count and docs[g] >= min_docs and not all(w in STOPWORDS for w in g.split())
    ]
    keep.sort(key=lambda g: (-total[g], -len(g.split()), g))
    return keep[:limit]


def compute_stats(texts: list[str]) -> StyleStats:
    paragraphs = [p for t in texts for p in split_paragraphs(t)]
    sentences = [s for p in paragraphs for s in split_sentences(p)]
    words_per_sentence = [len(_WORD.findall(s)) for s in sentences]
    words = [w.lower() for s in sentences for w in _WORD.findall(s)]
    n_words = len(words)
    joined = "\n".join(texts)

    def per_k(count: int) -> float:
        return round(1000 * count / n_words, 2) if n_words else 0.0

    first_person = sum(1 for s in sentences if FIRST_PERSON & {w.lower() for w in _WORD.findall(s)})
    return StyleStats(
        n_notes=len(texts),
        n_sentences=len(sentences),
        n_words=n_words,
        avg_sentence_len=round(mean(words_per_sentence), 2) if sentences else 0.0,
        median_sentence_len=float(median(words_per_sentence)) if sentences else 0.0,
        avg_paragraph_len=round(len(sentences) / len(paragraphs), 2) if paragraphs else 0.0,
        type_token_ratio=round(len(set(words)) / n_words, 3) if n_words else 0.0,
        first_person_rate=round(first_person / len(sentences), 3) if sentences else 0.0,
        punctuation={k: per_k(sum(joined.count(sym) for sym in syms)) for k, syms in PUNCT.items()},
        top_phrases=top_phrases(texts),
    )
