from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sell_signal.config import Settings
from sell_signal.schema import AnalysisResult


@dataclass(frozen=True)
class SheetSaveResult:
    saved_row_count: int
    total_row_count: int
    sheet_id: str
    worksheet_name: str


class GoogleSheetStore:
    headers = [
        'Saved At',
        'Name',
        'Category',
        'Confidence',
        'Seen In',
        'Used Median',
        'New Median',
        'Priority',
        'Priority Score',
        'Why',
        'Evidence',
        'Submission Mode',
        'Provider',
        'Model',
    ]

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_configured(self) -> bool:
        return bool(self.settings.google_sheet_id and self.settings.google_sheets_command.strip())

    def save_result(
        self,
        result: AnalysisResult,
        submission: dict[str, Any],
    ) -> SheetSaveResult:
        if not self.is_configured():
            raise RuntimeError('Google Sheets export is not configured.')
        new_rows = self._build_rows(result, submission)
        existing_rows = self._load_rows()
        all_rows = self._merge_rows(existing_rows, new_rows)
        self._write_rows(all_rows)
        return SheetSaveResult(
            saved_row_count=len(new_rows),
            total_row_count=len(all_rows) - 1,
            sheet_id=self.settings.google_sheet_id,
            worksheet_name=self.settings.google_sheet_tab,
        )

    def _build_rows(
        self,
        result: AnalysisResult,
        submission: dict[str, Any],
    ) -> list[list[Any]]:
        saved_at = self._saved_at_timestamp()
        submission_mode = str(submission.get('mode', 'unknown'))
        rows: list[list[Any]] = []
        for row in result.items:
            rows.append(
                [
                    saved_at,
                    row.item.name,
                    row.item.category,
                    row.item.confidence,
                    ', '.join(row.source_images) if row.source_images else 'text input',
                    row.pricing.used_median,
                    row.pricing.new_median,
                    row.priority_label,
                    row.priority_score,
                    '; '.join(row.why),
                    '; '.join(row.pricing.evidence),
                    submission_mode,
                    result.provider,
                    result.model,
                ]
            )
        return rows

    def _merge_rows(
        self,
        existing_rows: list[list[Any]],
        new_rows: list[list[Any]],
    ) -> list[list[Any]]:
        if existing_rows and existing_rows[0] == self.headers:
            data_rows = existing_rows[1:]
        else:
            data_rows = existing_rows
        merged_rows = [*data_rows, *new_rows]
        merged_rows.sort(key=self._sort_key)
        return [self.headers, *merged_rows]

    def _sort_key(self, row: list[Any]) -> tuple[float, float, str, str]:
        priority_score = self._as_float(row[8])
        used_median = self._as_float(row[5])
        saved_at = str(row[0])
        name = str(row[1]).lower()
        return (-priority_score, -used_median, saved_at, name)

    @staticmethod
    def _as_float(value: Any) -> float:
        if value in ('', None):
            return -1.0
        return float(value)

    def _load_rows(self) -> list[list[Any]]:
        payload = self._run_command(
            [
                'sheets',
                'get',
                self.settings.google_sheet_id,
                self._sheet_range(),
            ]
        )
        if not isinstance(payload, list):
            raise RuntimeError('Google Sheets read returned unexpected payload.')
        return [list(row) for row in payload]

    def _write_rows(self, rows: list[list[Any]]) -> None:
        self._run_command(
            [
                'sheets',
                'update',
                self.settings.google_sheet_id,
                self._sheet_write_range(len(rows)),
                '--values',
                json.dumps(rows),
            ],
            expect_json=False,
        )

    def _run_command(self, args: list[str], *, expect_json: bool = True) -> Any:
        command = [*shlex.split(self.settings.google_sheets_command), *args]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=self.settings.google_sheets_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Google Sheets request timed out.') from exc
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or '').strip()
            raise RuntimeError(details or 'Google Sheets command failed.') from exc
        if not expect_json:
            return None
        return self._parse_json_output(result.stdout)

    def _sheet_range(self) -> str:
        return f'{self.settings.google_sheet_tab}!A:{self._column_name(len(self.headers))}'

    def _sheet_write_range(self, row_count: int) -> str:
        return (
            f'{self.settings.google_sheet_tab}!A1:'
            f'{self._column_name(len(self.headers))}{row_count}'
        )

    @staticmethod
    def _column_name(index: int) -> str:
        value = index
        name = ''
        while value > 0:
            value, remainder = divmod(value - 1, 26)
            name = chr(65 + remainder) + name
        return name

    @staticmethod
    def _parse_json_output(output: str) -> Any:
        cleaned = output.strip()
        if not cleaned:
            return []
        return json.loads(cleaned)

    @staticmethod
    def _saved_at_timestamp() -> str:
        return datetime.now(UTC).replace(microsecond=0).isoformat()
