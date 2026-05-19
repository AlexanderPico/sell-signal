from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = 'sell-signal'
    provider_mode: str = 'hermes_bridge'
    model: str = 'gpt-5.4'
    api_base_url: str = ''
    api_key: str = ''
    hermes_command: str = 'hermes'
    hermes_provider: str = ''
    hermes_timeout_seconds: int = 300
    google_sheet_id: str = ''
    google_sheet_tab: str = 'SellSignal'
    google_sheets_command: str = ''
    google_sheets_timeout_seconds: int = 60


def _repo_env_path() -> Path:
    return Path(__file__).resolve().parents[2] / '.env'


def _load_repo_env() -> None:
    env_path = _repo_env_path()
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def get_settings() -> Settings:
    _load_repo_env()
    return Settings(
        app_name='sell-signal',
        provider_mode=os.getenv('SELL_SIGNAL_PROVIDER', 'hermes_bridge'),
        model=os.getenv('SELL_SIGNAL_MODEL', 'gpt-5.4'),
        api_base_url=os.getenv('SELL_SIGNAL_API_BASE_URL', ''),
        api_key=os.getenv('SELL_SIGNAL_API_KEY', ''),
        hermes_command=os.getenv('SELL_SIGNAL_HERMES_COMMAND', 'hermes'),
        hermes_provider=os.getenv('SELL_SIGNAL_HERMES_PROVIDER', ''),
        hermes_timeout_seconds=int(os.getenv('SELL_SIGNAL_HERMES_TIMEOUT', '300')),
        google_sheet_id=os.getenv('SELL_SIGNAL_GOOGLE_SHEET_ID', ''),
        google_sheet_tab=os.getenv('SELL_SIGNAL_GOOGLE_SHEET_TAB', 'SellSignal'),
        google_sheets_command=os.getenv('SELL_SIGNAL_GOOGLE_SHEETS_COMMAND', ''),
        google_sheets_timeout_seconds=int(os.getenv('SELL_SIGNAL_GOOGLE_SHEETS_TIMEOUT', '60')),
    )
