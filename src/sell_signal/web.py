from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

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


def _render_index(
    request: Request,
    *,
    result: dict | None = None,
    error: str | None = None,
    text_input: str = "",
) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"result": result, "error": error, "text_input": text_input},
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

    if not text_input and not uploads:
        return _render_index(
            request,
            error="Enter text or upload one or more images.",
        )

    if text_input and uploads:
        return _render_index(
            request,
            error="Use either text input or image upload for now, not both.",
            text_input=text_input,
        )

    if text_input:
        try:
            result = provider.analyze_text(text_input)
        except Exception as exc:
            return _render_index(
                request,
                error=f"Analysis failed: {exc}",
                text_input=text_input,
            )
        return _render_index(
            request,
            result=result.model_dump(),
            text_input=text_input,
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
            )
        return _render_index(request, result=result.model_dump())
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run() -> None:
    uvicorn.run("sell_signal.web:app", host="127.0.0.1", port=8011, reload=False)
