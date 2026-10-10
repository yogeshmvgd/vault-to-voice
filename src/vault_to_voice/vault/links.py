import hashlib
import re

from vault_to_voice.vault.reader import slugify

WIKILINK = re.compile(r"(!?)\[\[([^\[\]]+?)\]\]")
INLINE_CODE = re.compile(r"`[^`\n]*`")


def parse_link(inner: str) -> tuple[str, str]:
    """'Folder/Target#heading|alias' -> ('Target', 'alias' or 'Target')."""
    target_part, _, alias = inner.partition("|")
    target = target_part.split("#", 1)[0].strip().rsplit("/", 1)[-1].strip()
    if target.lower().endswith(".md"):
        target = target[:-3]
    return target, alias.strip() or target


def find_links(text: str) -> list[str]:
    targets = []
    for m in WIKILINK.finditer(INLINE_CODE.sub("", text)):
        target, _ = parse_link(m.group(2))
        if not m.group(1) and target:
            targets.append(target)
    return targets


def render_links(text: str) -> str:
    return WIKILINK.sub(lambda m: "" if m.group(1) else parse_link(m.group(2))[1], text)


def work_id_for(target: str) -> str:
    return slugify(target) or "work-" + hashlib.sha1(target.encode()).hexdigest()[:10]
