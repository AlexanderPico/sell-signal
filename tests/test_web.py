from io import BytesIO

from fastapi.testclient import TestClient

from sell_signal.schema import AnalysisResult, IdentifiedItem, PriceBand, PrioritizedItem
from sell_signal.web import app

client = TestClient(app)


class FakeProvider:
    def __init__(self, settings) -> None:
        self.settings = settings

    def analyze_text(self, text: str) -> AnalysisResult:
        return AnalysisResult(
            items=[
                PrioritizedItem(
                    item=IdentifiedItem(name=text, category='book', confidence=0.95),
                    pricing=PriceBand(used_median=42.0, new_median=55.0, evidence=['source a']),
                    priority_score=61.0,
                    priority_label='sell',
                    why=['meaningful resale value'],
                )
            ],
            provider='hermes_bridge',
            model='gpt-5.4',
        )

    def analyze_images(self, image_paths) -> AnalysisResult:
        return AnalysisResult(
            items=[
                PrioritizedItem(
                    item=IdentifiedItem(name='Uploaded Book', category='book', confidence=0.99),
                    pricing=PriceBand(used_median=30.0, new_median=44.0, evidence=['source b']),
                    priority_score=49.8,
                    priority_label='inspect',
                    why=['high identification confidence'],
                )
            ],
            provider='hermes_bridge',
            model='gpt-5.4',
        )


def test_healthz() -> None:
    response = client.get('/healthz')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_index_includes_progress_status_hook() -> None:
    response = client.get('/')
    assert response.status_code == 200
    assert 'id="analyze-form"' in response.text
    assert 'id="submission-status"' in response.text
    assert 'This can take 10 to 30 seconds.' in response.text


def test_text_analysis_renders_table(monkeypatch) -> None:
    import sell_signal.web as web

    monkeypatch.setattr(web, 'SmartProvider', FakeProvider)
    response = client.post(
        '/analyze',
        data={'text_input': 'The Manga Guide to Relativity by Hideo Nitta paperback'},
    )
    assert response.status_code == 200
    assert 'Prioritized items' in response.text
    assert 'The Manga Guide to Relativity by Hideo Nitta paperback' in response.text
    assert 'sell (61.0)' in response.text


def test_upload_analysis_renders_table(monkeypatch) -> None:
    import sell_signal.web as web

    monkeypatch.setattr(web, 'SmartProvider', FakeProvider)
    response = client.post(
        '/analyze',
        files={'files': ('sample.jpg', BytesIO(b'fake-image'), 'image/jpeg')},
    )
    assert response.status_code == 200
    assert 'Uploaded Book' in response.text
    assert 'inspect (49.8)' in response.text


class ExplodingProvider:
    def __init__(self, settings) -> None:
        self.settings = settings

    def analyze_text(self, text: str) -> AnalysisResult:
        raise RuntimeError('temporary provider outage')

    def analyze_images(self, image_paths) -> AnalysisResult:
        raise RuntimeError('image analysis timeout')


class EmptyProvider:
    def __init__(self, settings) -> None:
        self.settings = settings

    def analyze_text(self, text: str) -> AnalysisResult:
        return AnalysisResult(items=[], provider='hermes_bridge', model='gpt-5.4')

    def analyze_images(self, image_paths) -> AnalysisResult:
        return AnalysisResult(items=[], provider='hermes_bridge', model='gpt-5.4')


def test_text_analysis_failure_shows_error_and_preserves_input(monkeypatch) -> None:
    import sell_signal.web as web

    monkeypatch.setattr(web, 'SmartProvider', ExplodingProvider)
    response = client.post(
        '/analyze',
        data={'text_input': 'Broken Example Title'},
    )
    assert response.status_code == 200
    assert 'Analysis failed: temporary provider outage' in response.text
    assert 'Broken Example Title' in response.text


def test_upload_analysis_failure_shows_error(monkeypatch) -> None:
    import sell_signal.web as web

    monkeypatch.setattr(web, 'SmartProvider', ExplodingProvider)
    response = client.post(
        '/analyze',
        files={'files': ('sample.jpg', BytesIO(b'fake-image'), 'image/jpeg')},
    )
    assert response.status_code == 200
    assert 'Analysis failed: image analysis timeout' in response.text


def test_empty_results_show_retry_guidance(monkeypatch) -> None:
    import sell_signal.web as web

    monkeypatch.setattr(web, 'SmartProvider', EmptyProvider)
    response = client.post(
        '/analyze',
        files={'files': ('sample.jpg', BytesIO(b'fake-image'), 'image/jpeg')},
    )
    assert response.status_code == 200
    assert 'No resale items identified.' in response.text
    assert 'Try a clearer photo or add a short text description.' in response.text
