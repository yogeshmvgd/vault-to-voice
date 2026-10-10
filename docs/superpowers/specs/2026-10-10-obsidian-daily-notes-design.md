# Obsidian daily-notes vault format — Design

Extends `2026-10-10-vault-to-voice-design.md` (sections [1] Ingest, [3] Idea detection, [4] Post graph). Status: approved in brainstorming 2026-10-10; implementation slots in **before M2** as milestone **M1.5**.

## Context
The author writes in Obsidian with **one daily note per day**, sectioned by category, and links each entry to a **page per work** (`[[Witcher 3]]` → `Witcher 3.md`). M1 assumed one file = one note about one thing. The system must split daily notes into per-work, dated entries and group them by the linked work, while everything downstream (chunking, embeddings, FAISS, opinion scoring, M2 graph, citations) keeps working on `Note` objects.

## Decisions (from brainstorming)
| Topic | Decision |
|---|---|
| Where opinions live | Both: dated reactions in daily entries + settled views on the work page (option B). System supports either habit. |
| Daily layout | Category headings (`## Games`, `## Books`, `## Movies`) with bullets that link works (option A). |
| Entry unit | Each top-level bullet + its indented continuation lines. First link = the work; all links recorded. |
| Unlinked bullets | Ignored; counted in the ingest report. |
| Approach | New `ObsidianVaultReader` behind the existing `VaultReader` protocol; entries and pages become `Note`s (approach 1). |
| Daily file name | `Daily/DD-MM-YYYY.md` (`DAILY_DATE_FORMAT = %d-%m-%Y`). |

## Data model
`Note` gains two fields; nothing else changes.
```
id       entry: "<work_id>@<YYYY-MM-DD>" (second entry same work same day: "...-2")
         page:  frontmatter id, else work_id
title    work title (page file stem, or link text if the page does not exist)
type     entry: from its category heading; page: frontmatter type → folder (Games/Books/Movies) → "other"
date     entry: parsed from daily file name; page: frontmatter date or None
body     entry: bullet + continuation lines with [[...]] rendered as plain text; page: page body
path     vault-relative posix path of the source file
hash     sha256 of the entry text (entries) / raw file (pages)
work_id  NEW: slugify(link target or page stem), e.g. "witcher-3"; None for flat-format notes
links    NEW: work_ids of every work linked in the note (deduplicated, order of appearance)
```
Entry ids use ISO dates internally regardless of the file-name format. Manifest `notes` table gains `work_id TEXT` and `links TEXT (JSON)`. A `schema_version` meta key; a mismatch forces a full re-ingest (the index is derived data).

## Parsing rules (`ObsidianVaultReader`)
- **Classification:** a file under `DAILY_FOLDER` (default `Daily`) whose stem parses with `DAILY_DATE_FORMAT` is a daily note; every other `.md` is a work page. A file in `Daily/` whose name does not parse → warning, treated as a work page. Hidden directories are skipped (as in M1).
- **Categories:** a heading of any level whose text matches a key of `CATEGORY_HEADINGS` (case-insensitive; default `games=game,books=book,movies=movie`) opens that category; any other heading closes it.
- **Entries:** inside a category, each top-level list item (`-`, `*`, `+`, `1.`) starts an entry; indented lines and lazy continuation lines belong to it. Text outside list items is ignored.
- **Links:** `[[Target]]`, `[[Target|alias]]`, `[[Target#heading]]`, `[[Folder/Target]]` → target = last path segment before `#`/`|`. Embeds `![[...]]` and links inside inline code are ignored. Rendered body text uses the alias if present, else the target.
- **Work resolution:** by file stem anywhere in the vault (case-insensitive), like Obsidian; a link to a missing page is valid (`work_id` and title from link text).
- **Skips (counted, never crash):** unlinked bullet; bullet whose text is only links; empty work page (only frontmatter/headings). Malformed YAML / non-UTF-8 / empty files keep M1 behaviour.
- **Report:** the reader exposes counts (`daily_files`, `entries`, `pages`, `unlinked_skipped`, `empty_skipped`) surfaced by `v2v ingest`.

## Settings
`VAULT_FORMAT` = `obsidian` (default) | `flat` (M1 reader, kept for compatibility and tests) · `DAILY_FOLDER=Daily` · `DAILY_DATE_FORMAT=%d-%m-%Y` · `CATEGORY_HEADINGS=games=game,books=book,movies=movie`. `make_reader(settings)` picks the implementation.

## Idea detection
Two idea kinds; `Idea` gains `kind: "work" | "theme"` and `work_ids: list[str]`.
- **Work ideas:** per `work_id` with ≥ `MIN_NOTES` notes (entries + page) and ≥ `MIN_OPINIONATED` opinionated. Stable id `idea_work_<work_id>`; new entries grow `note_ids` of the same idea (re-labelled only when it is new). Title/angle from the LLM labeller.
- **Theme ideas:** cluster **works** (work vector = L2-normalised mean of its notes' vectors; flat-format notes without `work_id` count as their own work). A cluster qualifies with ≥ 2 distinct works plus the existing note thresholds over all member notes. Id = `"idea_" + sha1(sorted work_ids)[:10]`.
- `done` ideas are never deleted (unchanged rule). `v2v ideas` prints `[work]` / `[theme]`.

## M2 post graph impact
- `load_context` orders the idea's notes by date (undated pages first) so drafts can follow time.
- `.sources.json` paragraphs gain per-source `date` and `work_id`.
- M2 plan edits: fixtures use the new format; `Idea.kind`; `load_context` ordering. Graph topology unchanged.

## Sample vault & author setup
- `sample_vault/` rewritten in the new format: ~25 daily notes (`Daily/05-01-2026.md` … `Daily/28-06-2026.md`) + work pages in `Games/`, `Books/`, `Movies/`; same fictional works, voice and four themes; includes multi-link bullets, unlinked bullets and an empty hub page. Every file keeps `fictional: true`.
- `docs/obsidian-setup.md`: the daily template (below) and click-by-click Obsidian settings (Daily notes: folder `Daily`, format `DD-MM-YYYY`, template `Templates/Daily.md`; Templates: folder `Templates`).

```markdown
---
date: {{date:DD-MM-YYYY}}
---
## Games
- 

## Books
- 

## Movies
- 

## Other
- 
```
`## Other` is ignored by the tool.

## Testing
Offline unit tests for: daily-file classification and date parsing (incl. wrong names), category scoping, bullet/continuation splitting, all link syntaxes, embeds/code ignored, missing pages, same-day duplicates, skip counters, page type inference, stable work-idea ids across added entries, theme clustering over works, sample-vault shape. Integration (MiniLM) test: sample vault yields ≥ 1 work idea and ≥ 1 theme idea.
