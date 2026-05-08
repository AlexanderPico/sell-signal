from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = "sell-signal"
    provider_mode: str = os.getenv("SELL_SIGNAL_PROVIDER", "hermes_bridge")
    model: str = os.getenv("SELL_SIGNAL_MODEL", "gpt-5.4")
    api_base_url: str = os.getenv("SELL_SIGNAL_API_BASE_URL", "")
    api_key: str = os.getenv("SELL_SIGNAL_API_KEY", "")
    hermes_command: str = os.getenv("SELL_SIGNAL_HERMES_COMMAND", "hermes")
    hermes_provider: str = os.getenv("SELL_SIGNAL_HERMES_PROVIDER", "")
    hermes_timeout_seconds: int = int(os.getenv("SELL_SIGNAL_HERMES_TIMEOUT", "300"))


def get_settings() -> Settings:
    return Settings()
