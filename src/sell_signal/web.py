from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from sell_signal.config import get_settings
from sell_signal.provider import SmartProvider
from sell_signal.schema import AnalysisResult
from sell_signal.sheet_store import GoogleSheetStore

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="sell-signal")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

STEP_LABELS = {
    'queued': 'Preparing request',
    'upload': 'Saving uploads',
    'identify': 'Identifying resale items',
    'price_research': 'Researching market prices',
    'rank': 'Ranking resale priority',
}

_analysis_jobs: dict[str, dict[str, Any]] = {}
_analysis_jobs_lock = Lock()


class SaveResultsPayload(BaseModel):
    result: AnalysisResult
    submission: dict[str, Any]


def _sheet_store() -> GoogleSheetStore:
    return GoogleSheetStore(get_settings())


def _sheet_export_enabled() -> bool:
    return _sheet_store().is_configured()


def _build_save_payload(
    result: dict | None,
    submission: dict[str, Any],
) -> dict[str, Any] | None:
    if not result or not result.get('items'):
        return None
    return {
        'result': result,
        'submission': submission,
    }


def _progress_steps(mode: str, upload_count: int = 0) -> list[dict[str, str]]:
    intake_label = (
        f'Saving {upload_count} upload(s)'
        if mode == 'images'
        else 'Reading text input'
    )
    return [
        {'id': 'queued', 'label': 'Preparing request'},
        {'id': 'upload', 'label': intake_label},
        {'id': 'identify', 'label': 'Identifying resale items'},
        {'id': 'price_research', 'label': 'Researching market prices'},
        {'id': 'rank', 'label': 'Ranking resale priority'},
    ]


def _build_submission_meta(
    *,
    text_input: str = '',
    upload_filenames: list[str] | None = None,
) -> dict[str, Any]:
    upload_filenames = upload_filenames or []
    mode = 'images' if upload_filenames else 'text'
    if not text_input and not upload_filenames:
        mode = 'idle'
    return {
        'mode': mode,
        'upload_count': len(upload_filenames),
        'upload_filenames': upload_filenames,
        'text_length': len(text_input.strip()),
        'progress_steps': _progress_steps(mode, len(upload_filenames)),
    }


def _build_result_summary(
    result: dict | None,
    submission: dict[str, Any],
) -> list[str]:
    if not result:
        return []

    items = result.get('items', [])
    warnings = result.get('warnings', [])
    summary: list[str] = []
    if submission['mode'] == 'images':
        summary.append(f"{submission['upload_count']} upload(s) processed")
    elif submission['mode'] == 'text':
        summary.append('Text prompt submitted')

    item_count = len(items)
    summary.append(f"{item_count} item{'s' if item_count != 1 else ''} prioritized")

    if warnings:
        summary.append(f"{len(warnings)} upload warning{'s' if len(warnings) != 1 else ''}")

    if items:
        top_item = items[0]
        summary.append(
            f"Top recommendation: {top_item['priority_label']} — {top_item['item']['name']}"
        )

    return summary


def _build_partial_summary(
    partial_result: dict | None,
    submission: dict[str, Any],
) -> list[str]:
    if not partial_result:
        return []

    items = partial_result.get('items', [])
    if not items:
        return []

    summary: list[str] = []
    if submission['mode'] == 'images':
        summary.append(f"{submission['upload_count']} upload(s) processed")
    elif submission['mode'] == 'text':
        summary.append('Text prompt submitted')

    item_count = len(items)
    summary.append(f"{item_count} item{'s' if item_count != 1 else ''} priced so far")
    summary.append('Final ranking will update when all pricing finishes')
    return summary


def _render_result_section(
    *,
    result: dict | None = None,
    submission: dict[str, Any] | None = None,
    result_summary: list[str] | None = None,
    result_pending: bool = False,
) -> str:
    submission = submission or _build_submission_meta()
    return templates.env.get_template('_result_section.html').render(
        result=result,
        result_summary=result_summary or _build_result_summary(result, submission),
        result_pending=result_pending,
        save_payload=_build_save_payload(result, submission),
        sheet_export_enabled=_sheet_export_enabled(),
    )


def _render_index(
    request: Request,
    *,
    result: dict | None = None,
    error: str | None = None,
    text_input: str = '',
    submission: dict[str, Any] | None = None,
) -> HTMLResponse:
    submission = submission or _build_submission_meta()
    return templates.TemplateResponse(
        request,
        'index.html',
        {
            'result': result,
            'error': error,
            'text_input': text_input,
            'submission': submission,
            'result_summary': _build_result_summary(result, submission),
            'save_payload': _build_save_payload(result, submission),
            'sheet_export_enabled': _sheet_export_enabled(),
        },
    )


def _create_job(*, submission: dict[str, Any], text_input: str) -> str:
    job_id = uuid4().hex
    record = {
        'status': 'queued',
        'submission': submission,
        'text_input': text_input,
        'events': [
            {
                'step': 'queued',
                'label': STEP_LABELS['queued'],
                'message': 'Queued analysis request',
            }
        ],
        'current_step': 'queued',
        'current_message': 'Queued analysis request',
        'result': None,
        'partial_result': None,
        'error': None,
    }
    with _analysis_jobs_lock:
        _analysis_jobs[job_id] = record
    return job_id


def _record_job_progress(job_id: str, step: str, message: str) -> None:
    with _analysis_jobs_lock:
        job = _analysis_jobs[job_id]
        job['status'] = 'running'
        job['current_step'] = step
        job['current_message'] = message
        job['events'].append(
            {
                'step': step,
                'label': STEP_LABELS.get(step, step.replace('_', ' ').title()),
                'message': message,
            }
        )


def _complete_job(job_id: str, result: dict[str, Any]) -> None:
    with _analysis_jobs_lock:
        job = _analysis_jobs[job_id]
        job['status'] = 'completed'
        job['result'] = result
        job['partial_result'] = result
        job['error'] = None


def _record_partial_result(job_id: str, item: dict[str, Any]) -> None:
    with _analysis_jobs_lock:
        job = _analysis_jobs[job_id]
        partial_result = job['partial_result']
        if partial_result is None:
            partial_result = {
                'items': [],
                'provider': get_settings().provider_mode,
                'model': get_settings().model,
                'warnings': [],
            }
            job['partial_result'] = partial_result
        partial_result['items'].append(item)


def _fail_job(job_id: str, error: str) -> None:
    with _analysis_jobs_lock:
        job = _analysis_jobs[job_id]
        job['status'] = 'failed'
        job['error'] = error
        job['current_step'] = 'failed'
        job['current_message'] = error


def _get_job(job_id: str) -> dict[str, Any]:
    with _analysis_jobs_lock:
        job = _analysis_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Unknown analysis job')
        return {
            'status': job['status'],
            'submission': dict(job['submission']),
            'text_input': job['text_input'],
            'events': [dict(event) for event in job['events']],
            'current_step': job['current_step'],
            'current_message': job['current_message'],
            'result': job['result'],
            'partial_result': job['partial_result'],
            'error': job['error'],
        }


def _copy_uploads(uploads: list[UploadFile]) -> tuple[list[Path], Path]:
    temp_dir = Path(tempfile.mkdtemp(prefix='sell-signal-uploads-'))
    image_paths: list[Path] = []
    for upload in uploads:
        destination = temp_dir / Path(upload.filename or 'upload.bin').name
        with destination.open('wb') as handle:
            handle.write(upload.file.read())
        image_paths.append(destination)
    return image_paths, temp_dir


def _run_analysis_job(
    *,
    job_id: str,
    submission: dict[str, Any],
    text_input: str,
    image_paths: list[Path] | None = None,
    temp_dir: Path | None = None,
) -> None:
    provider = SmartProvider(get_settings())

    def progress_callback(step: str, message: str) -> None:
        _record_job_progress(job_id, step, message)

    def item_callback(item) -> None:
        _record_partial_result(job_id, item.model_dump())

    try:
        if image_paths:
            result = provider.analyze_images(
                image_paths,
                progress_callback=progress_callback,
                item_callback=item_callback,
            )
        else:
            result = provider.analyze_text(
                text_input,
                progress_callback=progress_callback,
                item_callback=item_callback,
            )
        _complete_job(job_id, result.model_dump())
    except Exception as exc:
        _fail_job(job_id, f'Analysis failed: {exc}')
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.get('/healthz')
def healthz() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/', response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _render_index(request)


@app.post('/analyze/start')
async def analyze_start(
    background_tasks: BackgroundTasks,
    text_input: str = Form(''),
    files: list[UploadFile] = File(default_factory=list),
) -> JSONResponse:
    text_input = text_input.strip()
    uploads = [upload for upload in files if upload.filename]
    submission = _build_submission_meta(
        text_input=text_input,
        upload_filenames=[Path(upload.filename or 'upload.bin').name for upload in uploads],
    )

    if not text_input and not uploads:
        raise HTTPException(status_code=400, detail='Enter text or upload one or more images.')

    if text_input and uploads:
        raise HTTPException(
            status_code=400,
            detail='Use either text input or image upload for now, not both.',
        )

    job_id = _create_job(submission=submission, text_input=text_input)

    if uploads:
        image_paths, temp_dir = _copy_uploads(uploads)
        _record_job_progress(
            job_id,
            'upload',
            f'Saved {len(image_paths)} upload(s) for analysis',
        )
        background_tasks.add_task(
            _run_analysis_job,
            job_id=job_id,
            submission=submission,
            text_input='',
            image_paths=image_paths,
            temp_dir=temp_dir,
        )
    else:
        _record_job_progress(job_id, 'upload', 'Read text input and queued analysis')
        background_tasks.add_task(
            _run_analysis_job,
            job_id=job_id,
            submission=submission,
            text_input=text_input,
            image_paths=None,
            temp_dir=None,
        )

    return JSONResponse({'job_id': job_id})


@app.get('/analyze/status/{job_id}')
def analyze_status(job_id: str) -> JSONResponse:
    job = _get_job(job_id)
    result = job['result']
    partial_result = job['partial_result']
    display_result = result or partial_result
    display_summary = (
        _build_result_summary(result, job['submission'])
        if result
        else _build_partial_summary(partial_result, job['submission'])
    )
    return JSONResponse(
        {
            'status': job['status'],
            'current_step': job['current_step'],
            'current_message': job['current_message'],
            'events': job['events'],
            'result': result,
            'partial_result': partial_result,
            'error': job['error'],
            'result_summary': _build_result_summary(result, job['submission']),
            'partial_summary': _build_partial_summary(partial_result, job['submission']),
            'result_html': _render_result_section(
                result=display_result,
                submission=job['submission'],
                result_summary=display_summary,
                result_pending=result is None and partial_result is not None,
            ),
        }
    )


@app.post('/results/save')
def save_results(payload: SaveResultsPayload) -> JSONResponse:
    store = _sheet_store()
    if not store.is_configured():
        raise HTTPException(status_code=400, detail='Google Sheets export is not configured.')
    if not payload.result.items:
        raise HTTPException(status_code=400, detail='No result rows available to save.')
    try:
        save_result = store.save_result(payload.result, payload.submission)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(save_result, dict):
        response_payload = dict(save_result)
    else:
        response_payload = {
            'saved_row_count': save_result.saved_row_count,
            'total_row_count': save_result.total_row_count,
            'sheet_id': save_result.sheet_id,
            'worksheet_name': save_result.worksheet_name,
        }
    row_label = 'row' if response_payload['saved_row_count'] == 1 else 'rows'
    response_payload['message'] = (
        f"Saved {response_payload['saved_row_count']} {row_label} to Google Sheet "
        'tab '
        f"{response_payload['worksheet_name']} "
        f"({response_payload['total_row_count']} total rows)."
    )
    return JSONResponse(response_payload)


@app.post('/analyze', response_class=HTMLResponse)
async def analyze(
    request: Request,
    text_input: str = Form(''),
    files: list[UploadFile] = File(default_factory=list),
) -> HTMLResponse:
    provider = SmartProvider(get_settings())
    text_input = text_input.strip()
    uploads = [upload for upload in files if upload.filename]
    submission = _build_submission_meta(
        text_input=text_input,
        upload_filenames=[Path(upload.filename or 'upload.bin').name for upload in uploads],
    )

    if not text_input and not uploads:
        return _render_index(
            request,
            error='Enter text or upload one or more images.',
            submission=submission,
        )

    if text_input and uploads:
        return _render_index(
            request,
            error='Use either text input or image upload for now, not both.',
            text_input=text_input,
            submission=submission,
        )

    if text_input:
        try:
            result = provider.analyze_text(text_input)
        except Exception as exc:
            return _render_index(
                request,
                error=f'Analysis failed: {exc}',
                text_input=text_input,
                submission=submission,
            )
        return _render_index(
            request,
            result=result.model_dump(),
            text_input=text_input,
            submission=submission,
        )

    temp_dir = Path(tempfile.mkdtemp(prefix='sell-signal-uploads-'))
    try:
        image_paths: list[Path] = []
        for upload in uploads:
            destination = temp_dir / Path(upload.filename or 'upload.bin').name
            with destination.open('wb') as handle:
                handle.write(await upload.read())
            image_paths.append(destination)
        try:
            result = provider.analyze_images(image_paths)
        except Exception as exc:
            return _render_index(
                request,
                error=f'Analysis failed: {exc}',
                submission=submission,
            )
        return _render_index(
            request,
            result=result.model_dump(),
            submission=submission,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run() -> None:
    uvicorn.run('sell_signal.web:app', host='127.0.0.1', port=8011, reload=False)
