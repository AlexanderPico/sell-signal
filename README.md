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

## Run the web interface

The web app serves on `http://127.0.0.1:8011` by default.

Basic startup:

```bash
cd /Users/aimee/.openclaw/git/AlexanderPico/sell-signal
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
.venv/bin/sell-signal-web
```

The app auto-loads a repo-local `.env` file on startup when present. Explicit exported environment variables still override `.env` values.

Equivalent explicit uvicorn command:

```bash
cd /Users/aimee/.openclaw/git/AlexanderPico/sell-signal
. .venv/bin/activate
python -m uvicorn sell_signal.web:app --host 127.0.0.1 --port 8011
```

Run in the background and detach from the shell:

```bash
cd /Users/aimee/.openclaw/git/AlexanderPico/sell-signal
.venv/bin/sell-signal-web > /tmp/sell-signal-web.log 2>&1 &
disown
```

Or with the explicit uvicorn form:

```bash
cd /Users/aimee/.openclaw/git/AlexanderPico/sell-signal
.venv/bin/python -m uvicorn sell_signal.web:app --host 127.0.0.1 --port 8011 > /tmp/sell-signal-web.log 2>&1 &
disown
```

These forms do not depend on the current shell having the virtualenv activated.

Provider-related environment variables are optional but available when needed. The default Hermes bridge route uses Gemini Flash through the Nous provider for low-cost item triage:

```bash
export SELL_SIGNAL_PROVIDER=hermes_bridge
export SELL_SIGNAL_MODEL=google/gemini-3-flash-preview
export SELL_SIGNAL_HERMES_PROVIDER=nous
export SELL_SIGNAL_HERMES_COMMAND=hermes
```

Do not route OpenAI/GPT models through the Nous provider. If you deliberately need an OpenAI model, use the already-paid Codex-backed provider explicitly:

```bash
export SELL_SIGNAL_MODEL=gpt-5.4
export SELL_SIGNAL_HERMES_PROVIDER=openai-codex
```

For an OpenAI-compatible backend instead of the default Hermes bridge mode:

```bash
export SELL_SIGNAL_PROVIDER=openai_compatible
export SELL_SIGNAL_API_BASE_URL="https://your-api-base/v1"
export SELL_SIGNAL_API_KEY="your-api-key"
export SELL_SIGNAL_MODEL="your-model"
```

## Local dev

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
ruff check src tests
```

These same checks run in the repo's generic GitHub Actions CI workflow on pushes and pull requests.
