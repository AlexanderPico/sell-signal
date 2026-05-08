from __future__ import annotations

import subprocess

import pytest

from sell_signal.config import Settings
from sell_signal.provider import SmartProvider


def test_parse_json_payload_skips_session_prefix() -> None:
    payload = SmartProvider._parse_json_payload(
        "session_id: 20260508_124547_0a18ea\n"
        '{"name":"Example","category":"book"}'
    )
    assert payload == {"name": "Example", "category": "book"}


def test_analyze_text_uses_price_research(monkeypatch) -> None:
    provider = SmartProvider(Settings())
    calls: list[tuple[str, str | None]] = []

    def fake_run(prompt: str, *, image_path=None, toolsets=None):
        calls.append((prompt, toolsets))
        if "resale market ranges" in prompt:
            return {
                "used_low": 10,
                "used_high": 30,
                "used_median": 20,
                "new_low": 25,
                "new_high": 40,
                "new_median": 32,
                "currency": "USD",
                "evidence": ["source a", "source b"],
            }
        return [
            {
                "name": "Example Book",
                "category": "Book",
                "confidence": 0.9,
                "condition_guess": "used",
                "identifiers": {"isbn": "123"},
                "notes": "ok",
            }
        ]

    monkeypatch.setattr(provider, "_run_json_query", fake_run)
    result = provider.analyze_text("Example Book by Somebody")
    assert len(result.items) == 1
    assert result.items[0].item.category == "book"
    assert result.items[0].pricing.used_median == 20
    assert any(toolsets == "web" for _, toolsets in calls)


def test_run_json_query_surfaces_hermes_stderr(monkeypatch) -> None:
    provider = SmartProvider(Settings())

    def fake_subprocess_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=['hermes', 'chat'],
            stderr='temporary provider outage\n',
            output='',
        )

    monkeypatch.setattr('sell_signal.provider.subprocess.run', fake_subprocess_run)

    with pytest.raises(RuntimeError, match='temporary provider outage'):
        provider._run_json_query('ping')
