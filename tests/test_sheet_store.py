from sell_signal.config import Settings
from sell_signal.schema import AnalysisResult, IdentifiedItem, PriceBand, PrioritizedItem
from sell_signal.sheet_store import GoogleSheetStore


def test_profile_env_is_loaded_and_sets_hermes_home_for_google_command(
    tmp_path,
    monkeypatch,
) -> None:
    profile_root = tmp_path / 'profile'
    profile_root.mkdir()
    env_file = profile_root / '.env'
    env_file.write_text('GOOGLE_CLIENT_ID=abc123\n')
    script_path = (
        profile_root
        / 'skills'
        / 'productivity'
        / 'google-workspace'
        / 'scripts'
        / 'google_api.py'
    )
    script_path.parent.mkdir(parents=True)
    script_path.write_text('# stub\n')
    calls: list[dict] = []

    def fake_run(command, **kwargs):
        calls.append({'command': command, 'env': kwargs.get('env')})

        class Result:
            stdout = '[]'

        return Result()

    monkeypatch.setattr('sell_signal.sheet_store.subprocess.run', fake_run)

    store = GoogleSheetStore(
        Settings(
            google_sheet_id='sheet-123',
            google_sheets_command=f'python {script_path}',
        )
    )

    assert store._load_rows() == []
    assert calls[0]['env']['HERMES_HOME'] == str(profile_root)
    assert calls[0]['env']['GOOGLE_CLIENT_ID'] == 'abc123'


class RecordingSheetStore(GoogleSheetStore):
    def __init__(self) -> None:
        super().__init__(
            Settings(
                google_sheet_id='sheet-123',
                google_sheet_tab='SellSignal',
                google_sheets_command='python /tmp/google_api.py',
            )
        )
        self.loaded_rows = [
            self.headers,
            [
                '2026-05-08T09:00:00+00:00',
                'Existing Book',
                'book',
                0.88,
                'text input',
                14.0,
                22.0,
                'inspect',
                45.0,
                'older comp',
                'source x',
                'text',
                'hermes_bridge',
                'gpt-5.4',
            ],
        ]
        self.written_rows = None

    def _load_rows(self):
        return self.loaded_rows

    def _write_rows(self, rows):
        self.written_rows = rows

    def _saved_at_timestamp(self) -> str:
        return '2026-05-08T10:00:00+00:00'


def test_save_result_appends_rows_and_resorts_collection() -> None:
    store = RecordingSheetStore()
    result = AnalysisResult(
        items=[
            PrioritizedItem(
                item=IdentifiedItem(name='High Value Book', category='book', confidence=0.97),
                pricing=PriceBand(used_median=30.0, new_median=42.0, evidence=['source a']),
                priority_score=80.0,
                priority_label='sell',
                why=['strong resale demand'],
                source_images=['shelf.jpg'],
            ),
            PrioritizedItem(
                item=IdentifiedItem(name='Mid Value Book', category='book', confidence=0.91),
                pricing=PriceBand(used_median=18.0, new_median=28.0, evidence=['source b']),
                priority_score=55.0,
                priority_label='inspect',
                why=['worth bundling later'],
            ),
        ],
        provider='hermes_bridge',
        model='gpt-5.4',
    )

    save_result = store.save_result(result, {'mode': 'images'})

    assert save_result.saved_row_count == 2
    assert save_result.total_row_count == 3
    assert store.written_rows == [
        store.headers,
        [
            '2026-05-08T10:00:00+00:00',
            'High Value Book',
            'book',
            0.97,
            'shelf.jpg',
            30.0,
            42.0,
            'sell',
            80.0,
            'strong resale demand',
            'source a',
            'images',
            'hermes_bridge',
            'gpt-5.4',
        ],
        [
            '2026-05-08T10:00:00+00:00',
            'Mid Value Book',
            'book',
            0.91,
            'text input',
            18.0,
            28.0,
            'inspect',
            55.0,
            'worth bundling later',
            'source b',
            'images',
            'hermes_bridge',
            'gpt-5.4',
        ],
        [
            '2026-05-08T09:00:00+00:00',
            'Existing Book',
            'book',
            0.88,
            'text input',
            14.0,
            22.0,
            'inspect',
            45.0,
            'older comp',
            'source x',
            'text',
            'hermes_bridge',
            'gpt-5.4',
        ],
    ]


def test_is_configured_requires_sheet_id_and_command() -> None:
    assert GoogleSheetStore(Settings()).is_configured() is False
    assert GoogleSheetStore(
        Settings(
            google_sheet_id='sheet-123',
            google_sheets_command='python /tmp/google_api.py',
        )
    ).is_configured() is True
