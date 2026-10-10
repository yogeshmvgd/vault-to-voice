# Setting up your Obsidian vault for vault-to-voice

vault-to-voice reads **daily notes** split into category sections, where each bullet links the
game, book or movie it is about. It never modifies your vault.

## 1. Folder layout
```
YourVault/
├─ Daily/        one note per day, named DD-MM-YYYY.md (e.g. 10-10-2026.md)
├─ Games/        one page per game   (Witcher 3.md)
├─ Books/        one page per book   (Dune.md)
├─ Movies/       one page per movie
└─ Templates/    Daily.md (the template below)
```

## 2. The daily template
Create `Templates/Daily.md`:
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
`## Other` is for everything else in your day — vault-to-voice ignores it.

## 3. Obsidian settings (click by click)
1. **Settings → Core plugins** → turn on **Daily notes** and **Templates**.
2. **Settings → Templates** → *Template folder location* = `Templates`.
3. **Settings → Daily notes**:
   - *Date format* = `DD-MM-YYYY`
   - *New file location* = `Daily`
   - *Template file location* = `Templates/Daily`
4. Click the calendar icon in the left ribbon (or run "Open today's daily note") — today's note
   opens pre-filled.

## 4. Writing an entry
- One bullet per thing you played/read/watched: `- Played [[Witcher 3]] — the Bloody Baron …`
- Type `[[` and start the title; Obsidian autocompletes existing pages.
- Indented sub-bullets belong to the bullet above them.
- A bullet without a `[[link]]` is skipped (the ingest report counts these).
- Linking two works in one bullet is fine; it is filed under the first link.
- New work? Click the link once to create the page, then move it into `Games/`, `Books/` or
  `Movies/`. Add a few lines there when you have a settled opinion.

## 5. Point vault-to-voice at your vault
In `.env` (outside the repo — the tool refuses vault paths inside it):
```dotenv
VAULT_PATH=C:/Users/you/Documents/MyVault
VAULT_FORMAT=obsidian
DAILY_FOLDER=Daily
DAILY_DATE_FORMAT=%d-%m-%Y
CATEGORY_HEADINGS=games=game,books=book,movies=movie
```
Then: `uv run v2v ingest`, `uv run v2v style`, `uv run v2v ideas`.
