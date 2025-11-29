from __future__ import annotations

import argparse
import asyncio
import io
import logging
import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pypdfium2 as pdfium
import tkinter as tk
from onnxocr.onnx_paddleocr import ONNXPaddleOcr
from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from tkinter import filedialog
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Checkbox, Footer, Header, Input, Label, Log

DEFAULT_FONT_PATH = Path(__file__).resolve().parent.parent / "fonts" / "ipaexg.ttf"


@dataclass
class OCRItem:
    text: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (px)
    score: Optional[float] = None


def render_page_to_pil(page: pdfium.PdfPage, dpi: int = 300):
    """Render a pdfium page to PIL.Image."""
    scale = dpi / 72.0
    return page.render(scale=scale).to_pil()


_OCR_INSTANCE = None


def get_ocr(ocr_factory: Optional[Callable[[], ONNXPaddleOcr]] = None):
    """Lazy-init OCR instance; factory can be swapped for tests."""
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        factory = ocr_factory or (lambda: ONNXPaddleOcr(use_gpu=False, lang="japan"))
        _OCR_INSTANCE = factory()
    return _OCR_INSTANCE


def run_onnx_ocr(pil_img):
    """Run ONNX PaddleOCR on a PIL image."""
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ocr = get_ocr()
    return ocr.ocr(bgr)


def normalize_onnxocr_results(ocr_results) -> List[OCRItem]:
    """Normalize PaddleOCR output to OCRItem list."""
    normalized: List[OCRItem] = []
    for page_items in ocr_results or []:
        if not isinstance(page_items, (list, tuple)):
            continue
        for item in page_items:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            quad = item[0]
            text_tuple = item[1]
            if isinstance(quad, (list, tuple)) and len(quad) == 4:
                try:
                    xs = [p[0] for p in quad]
                    ys = [p[1] for p in quad]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                except Exception:
                    continue
            else:
                continue
            if isinstance(text_tuple, (list, tuple)) and len(text_tuple) >= 1:
                text = text_tuple[0]
                score = text_tuple[1] if len(text_tuple) > 1 else None
            else:
                text = str(text_tuple)
                score = None
            normalized.append(OCRItem(text=text, bbox=(x1, y1, x2, y2), score=score))
    return normalized


def normalize_text_for_pdf(text: str) -> str:
    """Normalize text before embedding to PDF."""
    return unicodedata.normalize("NFKC", text)


def font_size_from_bbox_pt(
    x1_pt: float,
    y1_pt: float,
    x2_pt: float,
    y2_pt: float,
    min_size: float = 6,
    max_size: float = 48,
    scale: float = 0.90,
) -> float:
    height = max(1.0, (y2_pt - y1_pt))
    size = height * scale
    return max(min_size, min(max_size, size))


def make_overlay_pdf_bytes(
    page_w_pt: float,
    page_h_pt: float,
    img_w_px: int,
    img_h_px: int,
    ocr_items: Sequence[OCRItem],
    font_path: Path,
    visible: bool = False,
) -> bytes:
    """Create an overlay PDF (1 page) with OCR text and return bytes."""
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    try:
        pdfmetrics.registerFont(TTFont("JP", str(font_path)))
    except Exception as e:
        raise RuntimeError(f"Failed to register font: {e}") from e
    sx = page_w_pt / float(img_w_px)
    sy = page_h_pt / float(img_h_px)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w_pt, page_h_pt))
    c.setAuthor("overlay-gen")
    c.setTitle("text-layer")
    c.saveState()
    try:
        c.setFillAlpha(1.0 if visible else 0.0)
    except Exception:
        pass
    for item in ocr_items:
        text = normalize_text_for_pdf(item.text)
        x1, y1, x2, y2 = item.bbox
        x1_pt = x1 * sx
        y1_pt = y1 * sy
        x2_pt = x2 * sx
        y2_pt = y2 * sy
        fontsize = font_size_from_bbox_pt(
            x1_pt, y1_pt, x2_pt, y2_pt, min_size=6, max_size=48, scale=0.90
        )
        c.setFont("JP", fontsize)
        _, desc = pdfmetrics.getAscentDescent("JP", fontsize)
        baseline_y = (page_h_pt - y2_pt) - desc
        x_pt = x1_pt + 0.5
        try:
            c.drawString(x_pt, baseline_y, text)
        except Exception:
            safe_text = "".join(ch if ord(ch) < 0x110000 else " " for ch in text)
            c.drawString(x_pt, baseline_y, safe_text)
    c.restoreState()
    c.showPage()
    c.save()
    return buf.getvalue()


def _normalize_rotation(rotation: int) -> int:
    return rotation % 360


def _apply_rotation(page, rotation: int) -> None:
    angle = _normalize_rotation(rotation)
    if angle == 0:
        return
    if angle == 90:
        page.rotate_clockwise(90)
    elif angle == 180:
        page.rotate_clockwise(180)
    elif angle == 270:
        page.rotate_counter_clockwise(90)
    else:
        logging.warning("Unsupported rotation: %s degrees", angle)


def build_overlay_page(
    page_w_pt: float,
    page_h_pt: float,
    pil_img,
    ocr_items: Sequence[OCRItem],
    font_path: Path,
    visible: bool,
):
    """Build overlay page from OCR items."""
    img_w_px, img_h_px = pil_img.size
    overlay_bytes = make_overlay_pdf_bytes(
        page_w_pt=page_w_pt,
        page_h_pt=page_h_pt,
        img_w_px=img_w_px,
        img_h_px=img_h_px,
        ocr_items=ocr_items,
        font_path=font_path,
        visible=visible,
    )
    overlay_reader = PdfReader(io.BytesIO(overlay_bytes))
    return overlay_reader.pages[0]


def create_searchable_pdf_reportlab(
    input_pdf: Path,
    output_pdf: Path,
    dpi: int = 300,
    font_path: Path = DEFAULT_FONT_PATH,
    visible: bool = False,
    ocr_runner: Callable = run_onnx_ocr,
) -> None:
    """Overlay OCR text on existing PDF to make it searchable."""
    reader = PdfReader(str(input_pdf))
    writer = PdfWriter()
    src_doc = pdfium.PdfDocument(str(input_pdf))
    try:
        for page_index, page in enumerate(reader.pages):
            page_w_pt = float(page.mediabox.width)
            page_h_pt = float(page.mediabox.height)
            rotation = getattr(page, "rotation", 0) if hasattr(page, "rotation") else 0
            src_page = src_doc[page_index]
            try:
                pil_img = render_page_to_pil(src_page, dpi=dpi)
            finally:
                try:
                    src_page.close()
                except Exception:
                    pass
            ocr_results_raw = ocr_runner(pil_img)
            ocr_items = normalize_onnxocr_results(ocr_results_raw)
            overlay_page = build_overlay_page(
                page_w_pt=page_w_pt,
                page_h_pt=page_h_pt,
                pil_img=pil_img,
                ocr_items=ocr_items,
                font_path=font_path,
                visible=visible,
            )
            _apply_rotation(overlay_page, rotation or 0)
            page.merge_page(overlay_page)
            writer.add_page(page)
            logging.info(
                "Page %s/%s processed: %s",
                page_index + 1,
                len(reader.pages),
                input_pdf.name,
            )
    finally:
        try:
            src_doc.close()
        except Exception:
            pass
    with output_pdf.open("wb") as f:
        writer.write(f)
    logging.info("Searchable PDF created: %s", output_pdf)


@dataclass
class BatchResult:
    completed: List[Path]
    failed: List[Tuple[Path, str]]


def process_multiple_pdfs(
    input_paths: Sequence[Path],
    output_dir: Path,
    dpi: int = 300,
    font_path: Path = DEFAULT_FONT_PATH,
    visible: bool = False,
) -> BatchResult:
    """Batch OCR embedding for multiple PDFs."""
    completed: List[Path] = []
    failed: List[Tuple[Path, str]] = []

    if not input_paths:
        logging.error("No input files.")
        return BatchResult(completed=[], failed=[(Path("-"), "no input")])
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Invalid output dir: {output_dir}")

    for input_pdf in input_paths:
        try:
            if not input_pdf.is_file():
                raise FileNotFoundError(f"File not found: {input_pdf}")
            base = input_pdf.stem
            output_pdf = output_dir / f"{base}_ocr.pdf"
            create_searchable_pdf_reportlab(
                input_pdf=input_pdf,
                output_pdf=output_pdf,
                dpi=dpi,
                font_path=font_path,
                visible=visible,
            )
            completed.append(output_pdf)
        except Exception as e:
            logging.error("Error processing %s: %s", input_pdf, e)
            failed.append((input_pdf, str(e)))

    if completed:
        logging.info("Success: %s files", len(completed))
        for p in completed:
            logging.info("  -> %s", p)
    if failed:
        logging.error("Failed: %s files", len(failed))
        for ip, msg in failed:
            logging.error("  -> %s (%s)", ip, msg)

    return BatchResult(completed=completed, failed=failed)


def find_pdfs_in_dir(dir_path: Path) -> List[Path]:
    """List PDF files directly under a directory."""
    try:
        entries = list(dir_path.iterdir())
    except Exception as e:
        raise FileNotFoundError(f"Cannot access directory: {dir_path} ({e})")
    pdfs = [p.resolve() for p in entries if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(pdfs)


def resolve_font_path(font: Optional[str]) -> Path:
    """Resolve and validate font path."""
    font_path = Path(font) if font else DEFAULT_FONT_PATH
    font_path = font_path.resolve()
    if not font_path.is_file():
        raise FileNotFoundError(f"Font file not found: {font_path}")
    return font_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed OCR text layer into PDFs (TUI by default, CLI with --cli).",
    )
    parser.add_argument("-d", "--dir", type=str, help="Process all PDFs in a directory")
    parser.add_argument("-f", "--file", nargs="+", help="PDF files to process")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rendering DPI (default 300)",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show overlay text (debug)",
    )
    parser.add_argument(
        "--font",
        type=str,
        default=str(DEFAULT_FONT_PATH),
        help="TrueType font path for overlay text",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (default INFO)",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use CLI mode (otherwise Textual TUI is launched)",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    class ColorFormatter(logging.Formatter):
        ICONS = {
            logging.DEBUG: "🛠 ",
            logging.INFO: "ℹ️ ",
            logging.WARNING: "⚠️ ",
            logging.ERROR: "✖️ ",
            logging.CRITICAL: "💥 ",
        }
        COLORS = {
            logging.DEBUG: "\033[36m",
            logging.INFO: "\033[32m",
            logging.WARNING: "\033[33m",
            logging.ERROR: "\033[31m",
            logging.CRITICAL: "\033[35m",
        }
        RESET = "\033[0m"

        def __init__(self, use_color: bool):
            super().__init__("%(message)s")
            self.use_color = use_color

        def format(self, record: logging.LogRecord) -> str:
            msg = super().format(record)
            icon = self.ICONS.get(record.levelno, "")
            if self.use_color and self._color_enabled(record.levelno):
                color = self.COLORS.get(record.levelno, "")
                return f"{color}{icon}{msg}{self.RESET}"
            return f"{icon}{msg}"

        def _color_enabled(self, levelno: int) -> bool:
            return self.use_color and levelno in self.COLORS

    use_color = sys.stderr.isatty() and os.getenv("NO_COLOR") is None
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter(use_color=use_color))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def run_cli(args: argparse.Namespace) -> int:
    input_paths: List[Path] = []
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            logging.error("Directory not found: %s", dir_path)
            return 1
        input_paths = find_pdfs_in_dir(dir_path)
        if not input_paths:
            logging.error("No PDFs in directory: %s", dir_path)
            return 1
    elif args.file:
        for f in args.file:
            p = Path(f)
            if not p.is_file():
                logging.warning("File not found, skipped: %s", p)
                continue
            if p.suffix.lower() != ".pdf":
                logging.warning("Not a PDF, skipped: %s", p)
                continue
            input_paths.append(p.resolve())
        if not input_paths:
            logging.error("No valid PDF inputs.")
            return 1
    else:
        logging.error("Specify --dir or --file for CLI mode.")
        return 2

    output_dir = (
        Path(args.output).resolve()
        if args.output
        else (Path(args.dir).resolve() if args.dir else Path(input_paths[0]).parent)
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error("Cannot create output dir: %s (%s)", output_dir, e)
        return 1

    try:
        font_path = resolve_font_path(args.font)
    except FileNotFoundError as e:
        logging.error("%s", e)
        return 1

    result = process_multiple_pdfs(
        input_paths=input_paths,
        output_dir=output_dir,
        dpi=args.dpi,
        font_path=font_path,
        visible=args.visible,
    )
    return 0 if not result.failed else 1


class PDFEmbedTUI(App):
    """Textual-based TUI. File/dir selection uses Tk dialogs so mouse-only users are OK."""

    CSS = """
    Screen {
        align: center top;
    }
    #title {
        text-style: bold;
        padding: 1 0;
    }
    #controls {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1 2;
        padding: 1 2;
        border: solid $accent 50%;
    }
    #log {
        height: 12;
        border: tall $background 80%;
        padding: 1 1;
    }
    Button {
        width: 28;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("f", "select_files", "Select files"),
        ("o", "select_output", "Select output"),
        ("s", "start_ocr", "Start OCR"),
    ]

    def __init__(self, font_path: Path, **kwargs):
        super().__init__(**kwargs)
        self.selected_files: List[Path] = []
        self.output_dir: Optional[Path] = None
        self.font_path = font_path
        self.visible = False
        self.dpi = 300

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container():
            yield Label("PDF Embed OCR", id="title")
            with Container(id="controls"):
                yield Button("📂 Select PDF(s) [F]", id="select_files")
                yield Button("🗂 Select output folder [O]", id="select_output")
                yield Horizontal(
                    Label("DPI:", classes="compact"),
                    Input(str(self.dpi), id="dpi_input", placeholder="300", restrict=r"[0-9]+"),
                    classes="row",
                )
                yield Checkbox("Show overlay text (debug)", id="visible_checkbox")
                yield Button("▶️ Start OCR [S]", id="start")
            yield Label("Selected files:", id="files_label")
            yield Log(id="log")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Log).write_line("Select PDFs to begin.")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "select_files":
            await self._select_files()
        elif button_id == "select_output":
            await self._select_output_dir()
        elif button_id == "start":
            await self._start_ocr()

    async def action_select_files(self) -> None:
        await self._select_files()

    async def action_select_output(self) -> None:
        await self._select_output_dir()

    async def action_start_ocr(self) -> None:
        await self._start_ocr()

    async def _select_files(self) -> None:
        log = self.query_one(Log)
        log.write_line("… ファイルダイアログを開いています")
        try:
            paths = await asyncio.to_thread(self._open_file_dialog)
        except Exception as e:
            log.write_line(f"✖️ ファイルダイアログの起動に失敗: {e}")
            return
        if not paths:
            log.write_line("⚠️ 選択がキャンセルされました。")
            return
        self.selected_files = [Path(p).resolve() for p in paths]
        log.clear()
        for p in self.selected_files:
            log.write_line(f"✔ Selected: {p}")

    async def _select_output_dir(self) -> None:
        log = self.query_one(Log)
        log.write_line("… 出力フォルダ選択ダイアログを開いています")
        try:
            selected = await asyncio.to_thread(self._open_dir_dialog)
        except Exception as e:
            log.write_line(f"✖️ フォルダ選択の起動に失敗: {e}")
            return
        if selected:
            self.output_dir = Path(selected).resolve()
            log.write_line(f"Output: {self.output_dir}")
        else:
            log.write_line("⚠️ 選択がキャンセルされました。")

    def _open_file_dialog(self):
        root = tk.Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title="Select PDF files",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        root.destroy()
        return paths

    def _open_dir_dialog(self):
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="Select output folder")
        root.destroy()
        return path

    async def _start_ocr(self) -> None:
        log = self.query_one(Log)
        dpi_input = self.query_one("#dpi_input", Input).value or "300"
        try:
            self.dpi = max(72, int(dpi_input))
        except ValueError:
            log.write_line("⚠️ DPI must be a number.")
            return
        self.visible = self.query_one("#visible_checkbox", Checkbox).value

        if not self.selected_files:
            log.write_line("⚠️ No PDFs selected.")
            return

        output_dir = self.output_dir or self.selected_files[0].parent
        output_dir.mkdir(parents=True, exist_ok=True)
        font_path = resolve_font_path(str(self.font_path))

        log.write_line(f"▶️ Start (DPI={self.dpi}, visible={self.visible})")

        def run_batch():
            return process_multiple_pdfs(
                input_paths=self.selected_files,
                output_dir=output_dir,
                dpi=self.dpi,
                font_path=font_path,
                visible=self.visible,
            )

        result = await asyncio.to_thread(run_batch)

        if result.failed:
            log.write_line(f"✖️ Failed: {len(result.failed)} file(s)")
            for ip, msg in result.failed:
                log.write_line(f"  - {ip} -> {msg}")
        if result.completed:
            log.write_line(f"✅ Success: {len(result.completed)} file(s)")
            for p in result.completed:
                log.write_line(f"  - {p}")

    def action_quit(self) -> None:
        self.exit()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if args.cli:
        return run_cli(args)

    try:
        font_path = resolve_font_path(args.font)
    except FileNotFoundError as e:
        logging.error("%s", e)
        return 1

    app = PDFEmbedTUI(font_path=font_path)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
