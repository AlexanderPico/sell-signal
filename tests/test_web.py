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


class FakeMultiImageProvider:
    def __init__(self, settings) -> None:
        self.settings = settings

    def analyze_text(self, text: str, progress_callback=None) -> AnalysisResult:
        raise AssertionError('not used in this test')

    def analyze_images(self, image_paths, progress_callback=None) -> AnalysisResult:
        if progress_callback is not None:
            progress_callback('identify', 'Identified 1 distinct book across 3 uploads')
            progress_callback('price_research', 'Researched 1 item price from 1 candidate')
            progress_callback('rank', 'Ranked 1 item by resale priority')
        return AnalysisResult(
            items=[
                PrioritizedItem(
                    item=IdentifiedItem(name='Merged Book', category='book', confidence=0.98),
                    pricing=PriceBand(used_median=35.0, new_median=50.0, evidence=['source c']),
                    priority_score=58.0,
                    priority_label='sell',
                    why=['seen clearly in multiple photos'],
                    source_images=['front.jpg', 'spine.jpg'],
                )
            ],
            provider='hermes_bridge',
            model='gpt-5.4',
            warnings=['blurred.jpg: vision backend timeout'],
        )


class FakeProgressProvider:
    def __init__(self, settings) -> None:
        self.settings = settings

    def analyze_text(self, text: str, progress_callback=None, item_callback=None) -> AnalysisResult:
        if progress_callback is not None:
            progress_callback('identify', 'Found 2 candidate books from text input')
            progress_callback('price_research', 'Researched prices for 2 candidate books')
            progress_callback('rank', 'Ranked 2 books by resale priority')
        return AnalysisResult(
            items=[
                PrioritizedItem(
                    item=IdentifiedItem(name='Sapiens', category='book', confidence=0.97),
                    pricing=PriceBand(used_median=18.0, new_median=25.0, evidence=['source d']),
                    priority_score=72.0,
                    priority_label='sell',
                    why=['strong resale demand'],
                ),
                PrioritizedItem(
                    item=IdentifiedItem(name='Educated', category='book', confidence=0.95),
                    pricing=PriceBand(used_median=12.0, new_median=20.0, evidence=['source e']),
                    priority_score=54.0,
                    priority_label='inspect',
                    why=['solid used comps'],
                ),
            ],
            provider='hermes_bridge',
            model='gpt-5.4',
        )


class FakeStreamingProvider:
    def __init__(self, settings) -> None:
        self.settings = settings

    def analyze_text(self, text: str, progress_callback=None, item_callback=None) -> AnalysisResult:
        first = PrioritizedItem(
            item=IdentifiedItem(name='Sapiens', category='book', confidence=0.97),
            pricing=PriceBand(used_median=18.0, new_median=25.0, evidence=['source d']),
            priority_score=72.0,
            priority_label='sell',
            why=['strong resale demand'],
        )
        if progress_callback is not None:
            progress_callback('identify', 'Found 2 candidate books from text input')
            progress_callback('price_research', 'Researching market prices for Sapiens')
        if item_callback is not None:
            item_callback(first)
        second = PrioritizedItem(
            item=IdentifiedItem(name='Educated', category='book', confidence=0.95),
            pricing=PriceBand(used_median=12.0, new_median=20.0, evidence=['source e']),
            priority_score=54.0,
            priority_label='inspect',
            why=['solid used comps'],
        )
        return AnalysisResult(
            items=[first, second],
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
    assert 'id="submission-steps"' in response.text
    assert 'Preparing request' in response.text
    assert 'Identifying resale items' in response.text
    assert 'Researching market prices' in response.text
    assert 'Ranking resale priority' in response.text
    assert 'This can take 10 to 30 seconds.' in response.text


def test_text_analysis_renders_table_and_summary(monkeypatch) -> None:
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
    assert 'Text prompt submitted' in response.text
    assert '1 item prioritized' in response.text
    assert (
        'Top recommendation: sell — The Manga Guide to Relativity by Hideo Nitta paperback'
        in response.text
    )


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


def test_multi_image_analysis_shows_sources_warnings_and_summary(monkeypatch) -> None:
    import sell_signal.web as web

    monkeypatch.setattr(web, 'SmartProvider', FakeMultiImageProvider)
    response = client.post(
        '/analyze',
        files=[
            ('files', ('front.jpg', BytesIO(b'front'), 'image/jpeg')),
            ('files', ('spine.jpg', BytesIO(b'spine'), 'image/jpeg')),
            ('files', ('blurred.jpg', BytesIO(b'blurred'), 'image/jpeg')),
        ],
    )
    assert response.status_code == 200
    assert 'Merged Book' in response.text
    assert 'Seen in' in response.text
    assert 'front.jpg, spine.jpg' in response.text
    assert 'Some images could not be analyzed.' in response.text
    assert 'blurred.jpg: vision backend timeout' in response.text
    assert '3 upload(s) processed' in response.text
    assert '1 item prioritized' in response.text
    assert 'Top recommendation: sell — Merged Book' in response.text


def test_async_analysis_status_reports_real_progress_and_results(monkeypatch) -> None:
    import sell_signal.web as web

    monkeypatch.setattr(web, 'SmartProvider', FakeProgressProvider)
    start = client.post(
        '/analyze/start',
        data={'text_input': 'Sapiens and Educated'},
    )
    assert start.status_code == 200
    payload = start.json()
    assert payload['job_id']

    status = client.get(f"/analyze/status/{payload['job_id']}")
    assert status.status_code == 200
    data = status.json()
    assert data['status'] == 'completed'
    assert data['current_step'] == 'rank'
    assert data['current_message'] == 'Ranked 2 books by resale priority'
    assert data['result_summary'] == [
        'Text prompt submitted',
        '2 items prioritized',
        'Top recommendation: sell — Sapiens',
    ]
    assert [event['message'] for event in data['events']] == [
        'Queued analysis request',
        'Read text input and queued analysis',
        'Found 2 candidate books from text input',
        'Researched prices for 2 candidate books',
        'Ranked 2 books by resale priority',
    ]
    assert 'Sapiens' in data['result_html']
    assert 'Educated' in data['result_html']


def test_status_renders_partial_results_while_job_is_running(monkeypatch) -> None:
    import sell_signal.web as web

    monkeypatch.setattr(web, 'SmartProvider', FakeStreamingProvider)
    submission = web._build_submission_meta(text_input='Sapiens and Educated')
    job_id = web._create_job(submission=submission, text_input='Sapiens and Educated')
    web._record_job_progress(job_id, 'upload', 'Read text input and queued analysis')
    web._run_analysis_job(
        job_id=job_id,
        submission=submission,
        text_input='Sapiens and Educated',
        image_paths=None,
        temp_dir=None,
    )
    web._analysis_jobs[job_id]['status'] = 'running'
    web._analysis_jobs[job_id]['result'] = None
    web._analysis_jobs[job_id]['partial_result'] = {
        'items': [web._analysis_jobs[job_id]['partial_result']['items'][0]],
        'provider': 'hermes_bridge',
        'model': 'gpt-5.4',
        'warnings': [],
    }
    web._analysis_jobs[job_id]['current_step'] = 'price_research'
    web._analysis_jobs[job_id]['current_message'] = 'Researching market prices for Educated'

    status = client.get(f'/analyze/status/{job_id}')

    assert status.status_code == 200
    data = status.json()
    assert data['status'] == 'running'
    assert data['result'] is None
    assert data['partial_result']['items'][0]['item']['name'] == 'Sapiens'
    assert data['partial_summary'] == [
        'Text prompt submitted',
        '1 item priced so far',
        'Final ranking will update when all pricing finishes',
    ]
    assert 'Working draft while pricing continues.' in data['result_html']
    assert 'Sapiens' in data['result_html']
    assert 'Educated' not in data['result_html']


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
