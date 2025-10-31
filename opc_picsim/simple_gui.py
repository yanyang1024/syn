"""Simplified Gradio interface for document translation.

This module provides an end-user friendly wrapper around the existing
translation pipeline in ``pdf2zh_next``.  It focuses on the most common
workflow: upload a PDF, select languages, choose the pages to translate,
and download the results.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any
from typing import Deque

import gradio as gr

from pdf2zh_next.config import ConfigManager
from pdf2zh_next.config.cli_env_model import CLIEnvSettingsModel
from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.high_level import TranslationError
from pdf2zh_next.high_level import do_translate_async_stream

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 30 * 1024 * 1024  # 30 MB

# Limited set of language options for a non-technical audience.
LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("Auto Detect", "auto"),
    ("English", "en"),
    ("Simplified Chinese", "zh"),
    ("Traditional Chinese", "zh-TW"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("French", "fr"),
    ("German", "de"),
    ("Spanish", "es"),
    ("Portuguese", "pt"),
]
LANGUAGE_MAP = {label: code for label, code in LANGUAGE_OPTIONS}
LANGUAGE_LABEL_BY_CODE = {code: label for label, code in LANGUAGE_OPTIONS}

PAGE_OPTIONS = ["All", "First", "First 5 pages", "Others"]

config_manager = ConfigManager()
try:
    BASE_CLI_SETTINGS = config_manager.initialize_cli_config()
except Exception as exc:  # pragma: no cover - defensive fallback
    logger.warning("Falling back to default CLI settings: %s", exc)
    BASE_CLI_SETTINGS = CLIEnvSettingsModel()


def _code_to_label(code: str, fallback: str) -> str:
    return LANGUAGE_LABEL_BY_CODE.get(code, fallback)


DEFAULT_SOURCE_LANGUAGE = _code_to_label(
    BASE_CLI_SETTINGS.translation.lang_in, LANGUAGE_OPTIONS[0][0]
)
DEFAULT_TARGET_LANGUAGE = _code_to_label(
    BASE_CLI_SETTINGS.translation.lang_out, LANGUAGE_OPTIONS[2][0]
)


def _format_size(size: int) -> str:
    """Return a human readable size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _resolve_pages(selection: str, custom_input: str | None) -> str | None:
    """Map UI selection to backend page specification."""
    selection = selection or "All"
    selection = selection.strip()
    if selection == "All":
        return None
    if selection == "First":
        return "1"
    if selection == "First 5 pages":
        return "1-5"
    if selection == "Others":
        if not custom_input or not custom_input.strip():
            raise ValueError("请在“自定义页码”中输入需要翻译的页码。")
        return custom_input.strip()
    return None


def _build_settings(
    base_settings: CLIEnvSettingsModel,
    source_label: str,
    target_label: str,
    file_path: Path,
    output_dir: Path,
    page_selection: str,
    custom_pages: str | None,
) -> SettingsModel:
    """Clone base CLI settings and apply UI selections."""
    settings_copy = base_settings.clone()
    settings_copy.basic.gui = False
    settings_copy.basic.debug = False
    settings_copy.basic.input_files = {str(file_path)}
    settings_copy.translation.output = str(output_dir)
    settings_copy.translation.lang_in = LANGUAGE_MAP.get(
        source_label, settings_copy.translation.lang_in
    )
    settings_copy.translation.lang_out = LANGUAGE_MAP.get(
        target_label, settings_copy.translation.lang_out
    )
    settings_copy.pdf.pages = _resolve_pages(page_selection, custom_pages)
    settings_copy.report_interval = 0.2

    settings_copy.validate_settings()
    settings_model = settings_copy.to_settings_model()
    settings_model.validate_settings()
    settings_model.basic.input_files = set()
    return settings_model


def _is_likely_scanned_pdf(
    file_path: Path,
    *,
    max_samples: int = 8,
    min_chars_per_page: int = 20,
    empty_page_ratio: float = 0.8,
) -> bool:
    """Heuristically treat PDFs without extractable text as scanned copies."""
    try:
        import fitz  # type: ignore[import]
    except ImportError:
        logger.warning("PyMuPDF is unavailable, skip scanned PDF detection.")
        return False

    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Unable to open %s for scanned detection: %s", file_path, exc)
        return False

    try:
        page_count = len(doc)
        if page_count == 0:
            return True

        sample_count = min(page_count, max_samples)
        step = max(1, page_count // sample_count)
        sampled = 0
        empty_like_pages = 0

        for page_index in range(0, page_count, step):
            if sampled >= sample_count:
                break
            page = doc.load_page(page_index)
            text = page.get_text("text").strip()
            if len(text) < min_chars_per_page:
                empty_like_pages += 1
            sampled += 1

        if sampled == 0:
            return True
        return (empty_like_pages / sampled) >= empty_page_ratio
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Scanned detection failed on %s: %s", file_path, exc)
        return False
    finally:
        doc.close()


def _create_placeholder_pdf(path: Path) -> Path:
    """Create a minimal blank PDF to satisfy download expectations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fitz  # type: ignore[import]
    except ImportError:
        path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")
        return path

    try:
        doc = fitz.open()
        doc.new_page()
        doc.save(str(path))
        doc.close()
        return path
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to create placeholder PDF at %s: %s", path, exc)
        path.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")
        return path


def _append_preview(
    buffer: Deque[str], message: str, limit: int = 30
) -> tuple[str, Deque[str]]:
    """Append a message to the preview buffer and join into a single string."""
    buffer.append(message)
    while len(buffer) > limit:
        buffer.popleft()
    return "\n".join(buffer), buffer


def _format_progress(event: dict[str, Any]) -> str:
    """Create a human friendly progress string from a translation event."""
    stage = event.get("stage", "处理中")
    percent = event.get("overall_progress", 0.0)
    part_index = event.get("part_index")
    total_parts = event.get("total_parts")
    stage_current = event.get("stage_current")
    stage_total = event.get("stage_total")

    suffix_parts = []
    if part_index and total_parts:
        suffix_parts.append(f"第{part_index}/{total_parts}部分")
    if stage_current is not None and stage_total:
        suffix_parts.append(f"步骤 {stage_current}/{stage_total}")
    suffix = f"（{'，'.join(suffix_parts)}）" if suffix_parts else ""
    return f"{stage} - {percent:.1f}% {suffix}".strip()


async def translate_document(
    pdf_file: str | None,
    source_label: str,
    target_label: str,
    page_range: str,
    page_input: str | None,
    state: dict,
    file_size_limit: int = MAX_FILE_SIZE,
):
    """Handle the translation lifecycle and stream updates to the UI."""
    if state is None:
        state = {}

    task = asyncio.current_task()
    state["current_task"] = task

    preview_buffer: Deque[str] = deque()
    preview_text, preview_buffer = _append_preview(preview_buffer, "准备翻译任务...")

    yield (
        preview_text,
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
    )

    try:
        if not pdf_file:
            raise gr.Error("请先上传需要翻译的 PDF 文件。")

        source_path = Path(pdf_file)
        if not source_path.exists():
            raise gr.Error("无法读取上传的文件，请重新上传。")

        size = source_path.stat().st_size
        if size > file_size_limit:
            max_size_mb = file_size_limit / (1024 * 1024)
            raise gr.Error(f"文件大小超过限制（最大 {max_size_mb:.0f} MB）。")

        preview_text, preview_buffer = _append_preview(
            preview_buffer,
            f"已选择文件：{source_path.name}（{_format_size(size)}）",
        )
        yield (
            preview_text,
            gr.update(),
            gr.update(),
            gr.update(),
        )

        session_id = str(uuid.uuid4())
        state["session_id"] = session_id
        output_dir = Path("pdf2zh_files") / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        local_copy = output_dir / source_path.name
        shutil.copy(source_path, local_copy)

        preview_text, preview_buffer = _append_preview(
            preview_buffer, "正在检测是否为影印版 PDF..."
        )
        yield (
            preview_text,
            gr.update(),
            gr.update(),
            gr.update(),
        )

        if _is_likely_scanned_pdf(local_copy):
            preview_text, preview_buffer = _append_preview(
                preview_buffer, "检测到影印版 PDF，已跳过翻译并生成空结果。"
            )
            empty_mono = _create_placeholder_pdf(output_dir / "empty_mono.pdf")
            empty_dual = _create_placeholder_pdf(output_dir / "empty_dual.pdf")
            yield (
                preview_text,
                gr.update(value="### 翻译结果", visible=True),
                gr.update(value=str(empty_mono), visible=True),
                gr.update(value=str(empty_dual), visible=True),
            )
            return

        try:
            settings = _build_settings(
                BASE_CLI_SETTINGS,
                source_label,
                target_label,
                local_copy,
                output_dir,
                page_range,
                page_input,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

        preview_text, preview_buffer = _append_preview(
            preview_buffer, "开始翻译，请稍候..."
        )
        yield (
            preview_text,
            gr.update(),
            gr.update(),
            gr.update(),
        )

        last_percent = -1.0
        async for event in do_translate_async_stream(settings, local_copy):
            event_type = event.get("type")
            if event_type in {"progress_start", "progress_update"}:
                percent = event.get("overall_progress", 0.0)
                if percent - last_percent < 1.0 and event_type != "progress_start":
                    continue
                last_percent = percent
                message = _format_progress(event)
                preview_text, preview_buffer = _append_preview(preview_buffer, message)
                yield (
                    preview_text,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
            elif event_type == "progress_end":
                message = f"完成：{event.get('stage', '当前阶段')}"
                preview_text, preview_buffer = _append_preview(preview_buffer, message)
                yield (
                    preview_text,
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
            elif event_type == "finish":
                result = event["translate_result"]
                mono_path = result.mono_pdf_path
                dual_path = result.dual_pdf_path
                preview_text, preview_buffer = _append_preview(
                    preview_buffer, "翻译完成，可下载结果。"
                )
                yield (
                    preview_text,
                    gr.update(value="### 翻译结果", visible=True),
                    gr.update(
                        value=str(mono_path) if mono_path else None,
                        visible=bool(mono_path),
                    ),
                    gr.update(
                        value=str(dual_path) if dual_path else None,
                        visible=bool(dual_path),
                    ),
                )
                return
            elif event_type == "error":
                error_msg = event.get("error", "未知错误")
                details = event.get("details")
                full_msg = f"翻译失败：{error_msg}"
                if details:
                    full_msg = f"{full_msg}\n{details}"
                preview_text, preview_buffer = _append_preview(preview_buffer, full_msg)
                yield (
                    preview_text,
                    gr.update(visible=False),
                    gr.update(value=None, visible=False),
                    gr.update(value=None, visible=False),
                )
                return

        preview_text, preview_buffer = _append_preview(
            preview_buffer, "翻译未提供结果，请检查日志。"
        )
        yield (
            preview_text,
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )
    except asyncio.CancelledError:
        preview_text, preview_buffer = _append_preview(preview_buffer, "翻译已取消。")
        yield (
            preview_text,
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )
        return
    except gr.Error:
        raise
    except TranslationError as exc:
        preview_text, preview_buffer = _append_preview(
            preview_buffer, f"翻译失败：{exc}"
        )
        yield (
            preview_text,
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )
        return
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Translation failed: %s", exc)
        preview_text, preview_buffer = _append_preview(
            preview_buffer, f"翻译时出现错误：{exc}"
        )
        yield (
            preview_text,
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
        )
        return
    finally:
        state["current_task"] = None


async def cancel_translation(state: dict) -> str:
    """Cancel the ongoing translation task if one exists."""
    if state is None:
        return "当前没有正在运行的翻译任务。"

    task = state.get("current_task")
    if task is None or task.done():
        return "当前没有正在运行的翻译任务。"

    task.cancel()
    await asyncio.sleep(0)
    return "正在取消翻译，请稍候..."


def on_file_uploaded(file_path: str | None) -> str:
    """Show a friendly acknowledgement when a file is selected."""
    if not file_path:
        return "请上传 PDF 文件开始翻译。"
    path = Path(file_path)
    if not path.exists():
        return "上传的文件不可用，请重新选择。"
    return f"已加载文件：{path.name}（{_format_size(path.stat().st_size)}）"


def on_page_range_change(choice: str) -> gr.Update:
    """Toggle the visibility of the custom page input."""
    return gr.update(visible=choice == "Others")


def build_simple_app(max_file_size: int = MAX_FILE_SIZE) -> gr.Blocks:
    """Construct the simplified Gradio interface."""
    css = """
    .input-file > div > div {
        border: 2px dashed #4A90E2;
        padding: 24px;
        background-color: #f8fbff;
    }
    .simple-btn {
        width: 100%;
    }
    """

    with gr.Blocks(title="PDF全文翻译工具", css=css) as demo:
        gr.Markdown("# PDF全文翻译工具")

        app_state = gr.State({"current_task": None, "session_id": None})

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 上传 PDF 文件")
                file_input = gr.File(
                    label="PDF 文件",
                    file_count="single",
                    file_types=[".pdf"],
                    type="filepath",
                    elem_classes=["input-file"],
                )

                gr.Markdown("### 翻译选项")
                lang_from = gr.Dropdown(
                    label="源语言",
                    choices=[label for label, _ in LANGUAGE_OPTIONS],
                    value=DEFAULT_SOURCE_LANGUAGE,
                )
                lang_to = gr.Dropdown(
                    label="目标语言",
                    choices=[label for label, _ in LANGUAGE_OPTIONS],
                    value=DEFAULT_TARGET_LANGUAGE,
                )
                page_range = gr.Radio(
                    label="页码范围",
                    choices=PAGE_OPTIONS,
                    value=PAGE_OPTIONS[0],
                )
                page_input = gr.Textbox(
                    label="自定义页码",
                    placeholder="例如：1-3,5,7",
                    visible=False,
                )

                translate_btn = gr.Button("开始翻译", elem_classes=["simple-btn"])
                cancel_btn = gr.Button("取消", variant="secondary")

                output_title = gr.Markdown(visible=False)
                output_file_mono = gr.File(
                    label="译文 PDF（目标语言）", visible=False, interactive=False
                )
                output_file_dual = gr.File(
                    label="双语对照 PDF", visible=False, interactive=False
                )
            with gr.Column(scale=2):
                preview = gr.Textbox(
                    label="处理进度",
                    value="请上传 PDF 文件并点击“开始翻译”。",
                    lines=12,
                    interactive=False,
                )
                gr.Markdown(
                    "### 使用说明\n"
                    "- 支持上传单个不超过 30MB 的 PDF 文件。\n"
                    "- 可选择常用语言进行翻译。\n"
                    "- 若选择“Others”，请在自定义页码中输入具体页码，例如 `1-3,5`。\n"
                    "- 翻译完成后将在下方提供下载链接。"
                )

        file_input.upload(on_file_uploaded, inputs=file_input, outputs=preview)
        page_range.change(on_page_range_change, inputs=page_range, outputs=page_input)

        translate_btn.click(
            fn=partial(translate_document, file_size_limit=max_file_size),
            inputs=[file_input, lang_from, lang_to, page_range, page_input, app_state],
            outputs=[preview, output_title, output_file_mono, output_file_dual],
        )
        cancel_btn.click(
            fn=cancel_translation,
            inputs=app_state,
            outputs=preview,
        )

    demo.queue()
    return demo


def launch_simple_app(
    *,
    share: bool = False,
    server_port: int = 7860,
    inbrowser: bool = True,
    max_file_size: int = MAX_FILE_SIZE,
    **launch_kwargs: Any,
) -> None:
    """Convenience helper to launch the simplified interface."""
    demo = build_simple_app(max_file_size=max_file_size)
    demo.launch(
        share=share,
        server_port=server_port,
        inbrowser=inbrowser,
        max_file_size=max_file_size,
        **launch_kwargs,
    )


__all__ = ["build_simple_app", "launch_simple_app"]

# Built a simplified Gradio UI for PDF translation aimed at non-technical users.

# Implementation

# Added ready-to-use entry points build_simple_app and launch_simple_app that assemble the streamlined layout and launch parameters (pdf2zh_next/simple_gui.py (line 373), pdf2zh_next/simple_gui.py (line 467)).
# Curated novice-friendly language/page presets with sensible defaults and size checks to keep uploads within 30 MB (pdf2zh_next/simple_gui.py (line 31), pdf2zh_next/simple_gui.py (line 34), pdf2zh_next/simple_gui.py (line 60)).
# Implemented translate_document as a streaming coroutine that clones CLI settings, enforces page selection, updates the preview buffer, and surfaces output downloads when finished (pdf2zh_next/simple_gui.py (line 156), pdf2zh_next/simple_gui.py (line 128), pdf2zh_next/simple_gui.py (line 138)).
# Wired helper actions for cancellation, upload acknowledgement, and page-range toggling to keep the interface responsive (pdf2zh_next/simple_gui.py (line 344), pdf2zh_next/simple_gui.py (line 358), pdf2zh_next/simple_gui.py (line 368)).
# Validation

# python -m compileall pdf2zh_next/simple_gui.py
# Next Steps

# Run from pdf2zh_next.simple_gui import launch_simple_app; launch_simple_app() to exercise the new interface and confirm the UX matches expectations.
