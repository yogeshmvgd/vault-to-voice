import re


def split_paragraphs(body: str) -> list[str]:
    paragraphs = []
    for block in re.split(r"\n\s*\n", body):
        lines = [ln.strip() for ln in block.strip().splitlines()]
        lines = [re.sub(r"^#{1,6}\s+", "", ln) for ln in lines if ln]
        text = " ".join(lines).strip()
        if text:
            paragraphs.append(text)
    return paragraphs
