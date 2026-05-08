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

## Local dev

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
ruff check src tests
```
