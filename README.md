# sell-signal

A local-first resale triage tool for laptop now, phone next.

Target workflow:
1. upload one or more item photos
2. identify the item(s) with a strong hosted multimodal model
3. research rough new/used market value
4. show a compact table for sell / inspect / skip prioritization

Initial architecture choices:
- new codebase instead of extending `what-have-we-got`
- provider abstraction for hosted multimodal models
- default priority on remote smart models, not local Ollama
- simple FastAPI backend + server-rendered HTML first
- API-first shape so a mobile web UI can reuse the same backend later

## Planned provider strategy

Primary:
- hosted multimodal model via OpenAI-compatible API

Optional later:
- Hermes bridge mode
- marketplace-specific research adapters
- shared/multi-user deployment

## Google Sheet save flow

Set these env vars before starting the web app:

```bash
export SELL_SIGNAL_GOOGLE_SHEET_ID="your-sheet-id"
export SELL_SIGNAL_GOOGLE_SHEET_TAB="SellSignal"
export SELL_SIGNAL_GOOGLE_SHEETS_COMMAND="python ${HERMES_HOME:-$HOME/.hermes}/skills/productivity/google-workspace/scripts/google_api.py"
```

Behavior:
- final result tables show a Save to Google Sheet button
- clicking Save appends the current rows into the dedicated sheet tab
- the sheet is then rewritten in sorted order so the highest-priority items stay on top
- current sort order is priority score desc, used median desc, then saved timestamp
- if the Google Sheets command points at a profile skill script, sell-signal also loads that profile's `.env` and sets `HERMES_HOME` to the profile root before invoking it

Expected sheet columns:
- Saved At
- Name
- Category
- Confidence
- Seen In
- Used Median
- New Median
- Priority
- Priority Score
- Why
- Evidence
- Submission Mode
- Provider
- Model

## Local dev

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
ruff check src tests
```
