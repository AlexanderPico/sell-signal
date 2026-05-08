from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sell_signal.config import get_settings
from sell_signal.provider import SmartProvider

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="sell-signal")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _progress_steps(mode: str, upload_count: int = 0) -> list[str]:
    intake_step = (
        f"Saving {upload_count} upload(s)"
        if mode == "images"
        else "Reading text input"
    )
    return [
        "Preparing request",
        intake_step,
        "Identifying resale items",
        "Researching market prices",
        "Ranking resale priority",
    ]


def _build_submission_meta(
    *,
    text_input: str = "",
    upload_filenames: list[str] | None = None,
) -> dict[str, Any]:
    upload_filenames = upload_filenames or []
    mode = "images" if upload_filenames else "text"
    if not text_input and not upload_filenames:
        mode = "idle"
    return {
        "mode": mode,
        "upload_count": len(upload_filenames),
        "upload_filenames": upload_filenames,
        "text_length": len(text_input.strip()),
        "progress_steps": _progress_steps(mode, len(upload_filenames)),
    }


def _build_result_summary(
    result: dict | None,
    submission: dict[str, Any],
) -> list[str]:
    if not result:
        return []

    items = result.get("items", [])
    warnings = result.get("warnings", [])
    summary: list[str] = []
    if submission["mode"] == "images":
        summary.append(f"{submission['upload_count']} upload(s) processed")
    elif submission["mode"] == "text":
        summary.append("Text prompt submitted")

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


def _render_index(
    request: Request,
    *,
    result: dict | None = None,
    error: str | None = None,
    text_input: str = "",
    submission: dict[str, Any] | None = None,
) -> HTMLResponse:
    submission = submission or _build_submission_meta()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "result": result,
            "error": error,
            "text_input": text_input,
            "submission": submission,
            "result_summary": _build_result_summary(result, submission),
        },
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return _render_index(request)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    text_input: str = Form(""),
    files: list[UploadFile] = File(default_factory=list),
) -> HTMLResponse:
    provider = SmartProvider(get_settings())
    text_input = text_input.strip()
    uploads = [upload for upload in files if upload.filename]
    submission = _build_submission_meta(
        text_input=text_input,
        upload_filenames=[Path(upload.filename or "upload.bin").name for upload in uploads],
    )

    if not text_input and not uploads:
        return _render_index(
            request,
            error="Enter text or upload one or more images.",
            submission=submission,
        )

    if text_input and uploads:
        return _render_index(
            request,
            error="Use either text input or image upload for now, not both.",
            text_input=text_input,
            submission=submission,
        )

    if text_input:
        try:
            result = provider.analyze_text(text_input)
        except Exception as exc:
            return _render_index(
                request,
                error=f"Analysis failed: {exc}",
                text_input=text_input,
                submission=submission,
            )
        return _render_index(
            request,
            result=result.model_dump(),
            text_input=text_input,
            submission=submission,
        )

    temp_dir = Path(tempfile.mkdtemp(prefix="sell-signal-uploads-"))
    try:
        image_paths: list[Path] = []
        for upload in uploads:
            destination = temp_dir / Path(upload.filename or "upload.bin").name
            with destination.open("wb") as handle:
                handle.write(await upload.read())
            image_paths.append(destination)
        try:
            result = provider.analyze_images(image_paths)
        except Exception as exc:
            return _render_index(
                request,
                error=f"Analysis failed: {exc}",
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
    uvicorn.run("sell_signal.web:app", host="127.0.0.1", port=8011, reload=False)
