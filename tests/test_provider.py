from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sell_signal.config import Settings, get_settings
from sell_signal.provider import SmartProvider
from sell_signal.schema import IdentifiedItem, PriceBand, PrioritizedItem


def test_default_settings_use_gemini_on_nous(monkeypatch) -> None:
    monkeypatch.delenv('SELL_SIGNAL_MODEL', raising=False)
    monkeypatch.delenv('SELL_SIGNAL_HERMES_PROVIDER', raising=False)

    settings = get_settings()

    assert settings.model == 'google/gemini-3-flash-preview'
    assert settings.hermes_provider == 'nous'


def test_run_json_query_passes_explicit_nous_gemini_provider(monkeypatch) -> None:
    provider = SmartProvider(Settings())
    captured: dict[str, object] = {}

    def fake_subprocess_run(command, **kwargs):
        captured['command'] = command
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='[{"name":"Example","category":"book"}]',
            stderr='',
        )

    monkeypatch.setattr('sell_signal.provider.subprocess.run', fake_subprocess_run)

    provider._run_json_query('ping')

    command = captured['command']
    assert isinstance(command, list)
    assert command[command.index('-m') + 1] == 'google/gemini-3-flash-preview'
    assert command[command.index('--provider') + 1] == 'nous'


def test_run_json_query_refuses_openai_model_on_nous() -> None:
    provider = SmartProvider(
        Settings(model='gpt-5.4', hermes_provider='nous')
    )

    with pytest.raises(RuntimeError, match='Refusing to route OpenAI/GPT model'):
        provider._run_json_query('ping')


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


def test_analyze_images_dedupes_and_keeps_partial_failures(monkeypatch) -> None:
    provider = SmartProvider(Settings())

    def fake_run(prompt: str, *, image_path=None, toolsets=None):
        if toolsets == 'web':
            return {
                'used_low': 10,
                'used_high': 30,
                'used_median': 20,
                'new_low': 25,
                'new_high': 40,
                'new_median': 32,
                'currency': 'USD',
                'evidence': ['source a'],
            }
        assert image_path is not None
        if image_path.name == 'broken.jpg':
            raise RuntimeError('vision backend timeout')
        if image_path.name == 'front.jpg':
            return [{'name': 'Example Book', 'category': 'Book', 'confidence': 0.95}]
        if image_path.name == 'spine.jpg':
            return [{'name': 'example book', 'category': 'book', 'confidence': 0.91}]
        raise AssertionError(f'unexpected image path: {image_path}')

    monkeypatch.setattr(provider, '_run_json_query', fake_run)

    result = provider.analyze_images([
        Path('front.jpg'),
        Path('broken.jpg'),
        Path('spine.jpg'),
    ])

    assert len(result.items) == 1
    assert result.items[0].item.name == 'Example Book'
    assert result.items[0].source_images == ['front.jpg', 'spine.jpg']
    assert result.warnings == ['broken.jpg: vision backend timeout']


def test_analyze_images_retries_grouped_book_lot_as_individual_items(monkeypatch) -> None:
    provider = SmartProvider(Settings())
    prompts: list[str] = []

    def fake_run(prompt: str, *, image_path=None, toolsets=None):
        prompts.append(prompt)
        if toolsets == 'web':
            return {
                'used_low': 10,
                'used_high': 30,
                'used_median': 20,
                'new_low': 25,
                'new_high': 40,
                'new_median': 32,
                'currency': 'USD',
                'evidence': ['source a'],
            }
        assert image_path is not None
        if 'Do NOT return a single lot' in prompt:
            return [
                {'name': 'Sapiens', 'category': 'book', 'confidence': 0.96},
                {'name': 'Educated', 'category': 'book', 'confidence': 0.94},
            ]
        if 'Focus only on shelves of books, dvds, blu-rays, or similar spine-out media.' in prompt:
            return []
        return [
            {
                'name': 'assorted nonfiction books lot',
                'category': 'book',
                'confidence': 0.89,
                'notes': 'many books on a shelf',
            }
        ]

    monkeypatch.setattr(provider, '_run_json_query', fake_run)

    result = provider.analyze_images([Path('shelf.jpg')])

    assert [item.item.name for item in result.items] == ['Sapiens', 'Educated']
    assert all('lot' not in item.item.name.lower() for item in result.items)
    assert any('Do NOT return a single lot' in prompt for prompt in prompts)


def test_analyze_images_uses_media_shelf_fallback_when_generic_pass_finds_nothing(
    monkeypatch,
) -> None:
    provider = SmartProvider(Settings())
    prompts: list[str] = []

    def fake_run(prompt: str, *, image_path=None, toolsets=None):
        prompts.append(prompt)
        if toolsets == 'web':
            return {
                'used_low': 12,
                'used_high': 28,
                'used_median': 20,
                'new_low': 20,
                'new_high': 35,
                'new_median': 26,
                'currency': 'USD',
                'evidence': ['source a'],
            }
        assert image_path is not None
        if 'Focus only on shelves of books, dvds, blu-rays, or similar spine-out media.' in prompt:
            return [
                {'name': 'Sapiens', 'category': 'book', 'confidence': 0.97},
                {'name': 'Planet Earth II', 'category': 'dvd', 'confidence': 0.95},
            ]
        return []

    monkeypatch.setattr(provider, '_run_json_query', fake_run)

    result = provider.analyze_images([Path('media-shelf.jpg')])

    assert {item.item.name for item in result.items} == {'Sapiens', 'Planet Earth II'}
    assert {item.item.category for item in result.items} == {'book', 'dvd'}
    assert any(
        'Focus only on shelves of books, dvds, blu-rays, or similar spine-out media.'
        in prompt
        for prompt in prompts
    )


def test_analyze_images_prefers_media_shelf_items_over_room_decor(monkeypatch) -> None:
    provider = SmartProvider(Settings())

    def fake_run(prompt: str, *, image_path=None, toolsets=None):
        if toolsets == 'web':
            return {
                'used_low': 15,
                'used_high': 45,
                'used_median': 25,
                'new_low': 25,
                'new_high': 60,
                'new_median': 40,
                'currency': 'USD',
                'evidence': ['source a'],
            }
        assert image_path is not None
        if 'Focus only on shelves of books, dvds, blu-rays, or similar spine-out media.' in prompt:
            return [
                {'name': 'Sapiens', 'category': 'book', 'confidence': 0.97},
                {'name': 'Planet Earth II', 'category': 'dvd', 'confidence': 0.95},
                {'name': 'The Wire: Season 1', 'category': 'dvd', 'confidence': 0.94},
            ]
        return [
            {'name': 'tan upholstered armchair', 'category': 'furniture', 'confidence': 0.96},
            {'name': 'large framed wall art', 'category': 'art', 'confidence': 0.83},
        ]

    monkeypatch.setattr(provider, '_run_json_query', fake_run)

    result = provider.analyze_images([Path('living-room-bookshelf.jpg')])

    assert {item.item.name for item in result.items} == {
        'Sapiens',
        'Planet Earth II',
        'The Wire: Season 1',
    }
    assert all(item.item.category in {'book', 'dvd'} for item in result.items)


def test_analyze_images_retries_media_shelf_by_section_when_full_pass_is_empty(monkeypatch) -> None:
    provider = SmartProvider(Settings())
    prompts: list[str] = []

    def fake_run(prompt: str, *, image_path=None, toolsets=None):
        prompts.append(prompt)
        if toolsets == 'web':
            return {
                'used_low': 15,
                'used_high': 45,
                'used_median': 25,
                'new_low': 25,
                'new_high': 60,
                'new_median': 40,
                'currency': 'USD',
                'evidence': ['source a'],
            }
        assert image_path is not None
        if 'Focus only on this section of the image: left third.' in prompt:
            return [{'name': 'Sapiens', 'category': 'book', 'confidence': 0.97}]
        if 'Focus only on this section of the image: center third.' in prompt:
            return [{'name': 'Planet Earth II', 'category': 'dvd', 'confidence': 0.95}]
        if 'Focus only on this section of the image: right third.' in prompt:
            return []
        if 'Focus only on shelves of books, dvds, blu-rays, or similar spine-out media.' in prompt:
            return []
        return []

    monkeypatch.setattr(provider, '_run_json_query', fake_run)

    result = provider.analyze_images([Path('dense-media-shelf.jpg')])

    assert {item.item.name for item in result.items} == {'Sapiens', 'Planet Earth II'}
    assert any(
        'Focus only on this section of the image: left third.' in prompt
        for prompt in prompts
    )
    assert any(
        'Focus only on this section of the image: center third.' in prompt
        for prompt in prompts
    )


def test_analyze_images_skips_invalid_media_candidates_instead_of_failing(monkeypatch) -> None:
    provider = SmartProvider(Settings())

    def fake_run(prompt: str, *, image_path=None, toolsets=None):
        if toolsets == 'web':
            return {
                'used_low': 15,
                'used_high': 45,
                'used_median': 25,
                'new_low': 25,
                'new_high': 60,
                'new_median': 40,
                'currency': 'USD',
                'evidence': ['source a'],
            }
        assert image_path is not None
        if 'Focus only on shelves of books, dvds, blu-rays, or similar spine-out media.' in prompt:
            return [
                {'name': None, 'category': 'book', 'confidence': 0.91},
                {'name': 'Sapiens', 'category': 'book', 'confidence': 0.97},
            ]
        return []

    monkeypatch.setattr(provider, '_run_json_query', fake_run)

    result = provider.analyze_images([Path('messy-media-shelf.jpg')])

    assert [item.item.name for item in result.items] == ['Sapiens']


def test_analyze_images_dedupes_media_shelf_candidates_before_price_research(monkeypatch) -> None:
    provider = SmartProvider(Settings())
    priced_names: list[str] = []

    def fake_run(prompt: str, *, image_path=None, toolsets=None):
        if toolsets == 'web':
            item_name = 'Sapiens' if 'Sapiens' in prompt else 'Educated'
            priced_names.append(item_name)
            return {
                'used_low': 15,
                'used_high': 45,
                'used_median': 25,
                'new_low': 25,
                'new_high': 60,
                'new_median': 40,
                'currency': 'USD',
                'evidence': ['source a'],
            }
        assert image_path is not None
        if 'Focus only on shelves of books, dvds, blu-rays, or similar spine-out media.' in prompt:
            return [
                {'name': 'Sapiens', 'category': 'book', 'confidence': 0.96},
                {'name': 'sapiens', 'category': 'book', 'confidence': 0.94},
                {'name': 'Educated', 'category': 'book', 'confidence': 0.92},
            ]
        return []

    monkeypatch.setattr(provider, '_run_json_query', fake_run)

    result = provider.analyze_images([Path('duplicate-media-shelf.jpg')])

    assert {item.item.name for item in result.items} == {'Sapiens', 'Educated'}
    assert priced_names.count('Sapiens') == 1
    assert priced_names.count('Educated') == 1


def test_analyze_images_dedupes_punctuation_variants_before_price_research(monkeypatch) -> None:
    provider = SmartProvider(Settings())
    priced_names: list[str] = []

    def fake_run(prompt: str, *, image_path=None, toolsets=None):
        if toolsets == 'web':
            item_name = 'ACTA PVBLICA' if 'ACTA' in prompt else 'Memoirs'
            priced_names.append(item_name)
            return {
                'used_low': 15,
                'used_high': 45,
                'used_median': 25,
                'new_low': 25,
                'new_high': 60,
                'new_median': 40,
                'currency': 'USD',
                'evidence': ['source a'],
            }
        assert image_path is not None
        if 'Focus only on shelves of books, dvds, blu-rays, or similar spine-out media.' in prompt:
            return [
                {'name': 'ACTA. PVBLICA', 'category': 'book', 'confidence': 0.93},
                {'name': 'ACTA PVBLICA', 'category': 'book', 'confidence': 0.97},
                {'name': 'Memoirs', 'category': 'book', 'confidence': 0.91},
            ]
        return []

    monkeypatch.setattr(provider, '_run_json_query', fake_run)

    result = provider.analyze_images([Path('punctuation-media-shelf.jpg')])

    assert {item.item.name for item in result.items} == {'ACTA PVBLICA', 'Memoirs'}
    assert priced_names.count('ACTA PVBLICA') == 1
    assert priced_names.count('Memoirs') == 1


def test_merge_duplicate_items_combines_source_images_for_punctuation_variants() -> None:
    provider = SmartProvider(Settings())
    acta_with_period = PrioritizedItem(
        item=IdentifiedItem(name='ACTA. PVBLICA', category='book', confidence=0.91),
        pricing=PriceBand(used_median=18.0, new_median=25.0, evidence=['source a']),
        priority_score=55.0,
        priority_label='inspect',
        why=['spine partly obscured'],
        source_images=['left.jpg'],
    )
    acta_without_period = PrioritizedItem(
        item=IdentifiedItem(name='ACTA PVBLICA', category='book', confidence=0.97),
        pricing=PriceBand(used_median=20.0, new_median=28.0, evidence=['source a', 'source b']),
        priority_score=62.0,
        priority_label='sell',
        why=['clearer title read'],
        source_images=['right.jpg'],
    )

    merged = provider._merge_duplicate_items([acta_with_period, acta_without_period])

    assert len(merged) == 1
    assert merged[0].item.name == 'ACTA PVBLICA'
    assert merged[0].source_images == ['left.jpg', 'right.jpg']
    assert merged[0].item.confidence == 0.97
    assert merged[0].priority_label == 'sell'
    assert merged[0].pricing.evidence == ['source a', 'source b']
