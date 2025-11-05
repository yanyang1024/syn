from __future__ import annotations

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pdf2zh_next.config import ConfigManager
from pdf2zh_next.config.cli_env_model import CLIEnvSettingsModel
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.high_level import TranslationError, do_translate_async_stream


logger = logging.getLogger(__name__)


# Thresholds for large-file preprocessing
MAX_BYTES_FOR_DIRECT = 30 * 1024 * 1024  # 30MB
MAX_PAGES_FOR_DIRECT = 1000
DEFAULT_CHUNK_PAGES = 200


class TranslationRequest(BaseModel):
    pdf_path: str = Field(..., description="Absolute path to source PDF file")
    source_lang: str = Field(..., description="Source language code, e.g. en, auto")
    target_lang: str = Field(..., description="Target language code, e.g. zh")
    output_dir: str = Field(..., description="Directory to save translated results")
    pages: str | None = Field(None, description="Page ranges, e.g. '1-3,5'")
    chunk_pages: int | None = Field(
        None, description="Override pages per chunk when splitting large PDF"
    )


class ChunkResult(BaseModel):
    index: int
    mono_pdf: str | None
    dual_pdf: str | None


class TranslationResponse(BaseModel):
    large_file_mode: bool
    final_mono_pdf: str | None
    final_dual_pdf: str | None
    chunk_outputs: list[ChunkResult] = Field(default_factory=list)
    output_dir: str
    work_dir: str | None


config_manager = ConfigManager()
try:
    BASE_CLI_SETTINGS: CLIEnvSettingsModel = config_manager.initialize_cli_config()
except Exception as exc:  # pragma: no cover - defensive fallback
    logger.warning("Falling back to default CLI settings: %s", exc)
    BASE_CLI_SETTINGS = CLIEnvSettingsModel()


def _build_settings(
    base_settings: CLIEnvSettingsModel,
    *,
    source_code: str,
    target_code: str,
    file_path: Path,
    output_dir: Path,
    page_spec: str | None,
) -> SettingsModel:
    """Clone base CLI settings and apply parameters."""
    settings_copy = base_settings.clone()
    settings_copy.basic.gui = False
    settings_copy.basic.debug = False
    settings_copy.basic.input_files = {str(file_path)}
    settings_copy.translation.output = str(output_dir)
    settings_copy.translation.lang_in = source_code
    settings_copy.translation.lang_out = target_code
    settings_copy.pdf.pages = page_spec
    settings_copy.report_interval = 0.2

    # Validate and convert to runtime SettingsModel
    settings_copy.validate_settings()
    model = settings_copy.to_settings_model()
    model.validate_settings()
    # high_level.do_translate_async_stream accepts the file path separately
    model.basic.input_files = set()
    return model


def _get_pdf_info(pdf_path: Path) -> tuple[int, int]:
    """Return (size_bytes, num_pages) for the PDF."""
    try:
        size = pdf_path.stat().st_size
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Cannot stat file: {e}") from e

    try:
        with fitz.open(str(pdf_path)) as doc:
            pages = len(doc)
    except Exception as e:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"Cannot open PDF: {e}") from e
    return size, pages


def _split_pdf(pdf_path: Path, work_dir: Path, chunk_pages: int) -> list[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    base = pdf_path.stem
    with fitz.open(str(pdf_path)) as doc:
        total = len(doc)
        start = 0
        index = 0
        while start < total:
            end = min(start + chunk_pages, total)
            part = fitz.open()
            part.insert_pdf(doc, from_page=start, to_page=end - 1)
            out_path = work_dir / f"{base}_part_{index:04d}_{start+1:04d}-{end:04d}.pdf"
            part.save(str(out_path))
            part.close()
            parts.append(out_path)
            index += 1
            start = end
    return parts


def _merge_pdfs(inputs: list[Path], output_path: Path) -> Path:
    if not inputs:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = fitz.open()
    try:
        for path in inputs:
            if path is None:
                continue
            if not Path(path).exists():
                continue
            with fitz.open(str(path)) as doc:
                out.insert_pdf(doc)
        out.save(str(output_path))
    finally:
        out.close()
    return output_path


@dataclass
class _SingleTranslateResult:
    mono: Path | None
    dual: Path | None


async def _translate_single(settings: SettingsModel, file_path: Path) -> _SingleTranslateResult:
    mono: Path | None = None
    dual: Path | None = None
    async for event in do_translate_async_stream(settings, file_path):
        etype = event.get("type")
        if etype == "finish":
            result = event["translate_result"]
            mono = result.mono_pdf_path
            dual = result.dual_pdf_path
            break
        if etype == "error":
            message = event.get("error", "Unknown error")
            details = event.get("details", "")
            raise HTTPException(status_code=500, detail=f"Translation error: {message}\n{details}")
    return _SingleTranslateResult(mono=mono, dual=dual)


def _is_large_file(size_bytes: int, pages: int) -> bool:
    return size_bytes > MAX_BYTES_FOR_DIRECT or pages > MAX_PAGES_FOR_DIRECT


app = FastAPI(title="PDF2ZH Translation API", version="1.0.0")


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return {"status": "ok"}


@app.post("/translate", response_model=TranslationResponse)
async def translate(req: TranslationRequest) -> TranslationResponse:
    pdf_path = Path(req.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")

    output_dir = Path(req.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    size, pages = _get_pdf_info(pdf_path)

    # If page ranges specified by client, we don't split; translate requested subset.
    should_split = req.pages is None and _is_large_file(size, pages)

    # Prepare a base settings template
    base_settings = _build_settings(
        BASE_CLI_SETTINGS,
        source_code=req.source_lang,
        target_code=req.target_lang,
        file_path=pdf_path,
        output_dir=output_dir,
        page_spec=req.pages,
    )

    if not should_split:
        # Direct translation
        res = await _translate_single(base_settings, pdf_path)
        return TranslationResponse(
            large_file_mode=False,
            final_mono_pdf=str(res.mono) if res.mono else None,
            final_dual_pdf=str(res.dual) if res.dual else None,
            chunk_outputs=[],
            output_dir=str(output_dir),
            work_dir=None,
        )

    # Large file mode: split -> parallel translate -> merge
    chunk_pages = max(req.chunk_pages or DEFAULT_CHUNK_PAGES, 50)
    work_dir = output_dir / f"work_{uuid.uuid4().hex[:8]}"
    parts = _split_pdf(pdf_path, work_dir / "parts", chunk_pages)
    if not parts:
        raise HTTPException(status_code=500, detail="Split produced no parts")

    # Translate each part in parallel. Each part uses its own sub-output dir.
    sem = asyncio.Semaphore(min(os.cpu_count() or 2, 8))

    async def run_part(index: int, part_path: Path) -> tuple[int, _SingleTranslateResult]:
        async with sem:
            part_out = work_dir / f"out_{index:04d}"
            settings = base_settings.clone()
            settings.translation.output = str(part_out)
            # Translate full part; ensure pages=None here
            settings.pdf.pages = None
            settings.validate_settings()
            return index, await _translate_single(settings, part_path)

    tasks = [run_part(i, p) for i, p in enumerate(parts)]
    results: list[tuple[int, _SingleTranslateResult]] = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])

    mono_parts: list[Path] = []
    dual_parts: list[Path] = []
    chunk_outputs: list[ChunkResult] = []
    for i, r in results:
        if r.mono:
            mono_parts.append(Path(r.mono))
        if r.dual:
            dual_parts.append(Path(r.dual))
        chunk_outputs.append(
            ChunkResult(index=i, mono_pdf=str(r.mono) if r.mono else None, dual_pdf=str(r.dual) if r.dual else None)
        )

    final_mono: str | None = None
    final_dual: str | None = None
    if mono_parts:
        final_mono_path = output_dir / f"{pdf_path.stem}_mono_merged.pdf"
        _merge_pdfs(mono_parts, final_mono_path)
        final_mono = str(final_mono_path)
    if dual_parts:
        final_dual_path = output_dir / f"{pdf_path.stem}_dual_merged.pdf"
        _merge_pdfs(dual_parts, final_dual_path)
        final_dual = str(final_dual_path)

    return TranslationResponse(
        large_file_mode=True,
        final_mono_pdf=final_mono,
        final_dual_pdf=final_dual,
        chunk_outputs=chunk_outputs,
        output_dir=str(output_dir),
        work_dir=str(work_dir),
    )


if __name__ == "__main__":
    # uvicorn pdf2zh_next.api_server:app --host 0.0.0.0 --port 8000
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run("pdf2zh_next.api_server:app", host="0.0.0.0", port=8000, reload=False)

