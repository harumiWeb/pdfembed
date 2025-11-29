from __future__ import annotations

import argparse
import io
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pypdfium2 as pdfium
from onnxocr.onnx_paddleocr import ONNXPaddleOcr
from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

DEFAULT_FONT_PATH = Path(__file__).resolve().parent.parent / "fonts" / "ipaexg.ttf"


@dataclass
class OCRItem:
    text: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (px)
    score: Optional[float] = None


def render_page_to_pil(page: pdfium.PdfPage, dpi: int = 300):
    """pypdfium2 のページオブジェクトを PIL.Image にレンダリングする。"""
    scale = dpi / 72.0
    return page.render(scale=scale).to_pil()


_OCR_INSTANCE = None


def get_ocr(ocr_factory: Optional[Callable[[], ONNXPaddleOcr]] = None):
    """OCRインスタンスを遅延初期化で取得。テスト用にファクトリ差し替え可。"""
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        factory = ocr_factory or (lambda: ONNXPaddleOcr(use_gpu=False, lang="japan"))
        _OCR_INSTANCE = factory()
    return _OCR_INSTANCE


def run_onnx_ocr(pil_img):
    """PIL.Image を ONNX PaddleOCR で推論し結果を返す。"""
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ocr = get_ocr()
    return ocr.ocr(bgr)


def normalize_onnxocr_results(ocr_results) -> List[OCRItem]:
    """PaddleOCR の結果を OCRItem リストに正規化する。"""
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
    """PDF 埋め込み用にテキストを正規化する。"""
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
    """
    ReportLab で1ページ分のオーバーレイPDFを生成し、bytes を返す。
    visible=True で重ねるテキストを可視化（デバッグ用）。
    """
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    try:
        pdfmetrics.registerFont(TTFont("JP", str(font_path)))
    except Exception as e:
        raise RuntimeError(f"フォント登録に失敗しました: {e}") from e
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
        logging.warning("未対応の回転角: %s 度", angle)


def build_overlay_page(
    page_w_pt: float,
    page_h_pt: float,
    pil_img,
    ocr_items: Sequence[OCRItem],
    font_path: Path,
    visible: bool,
):
    """1ページ分のオーバーレイ PageObject を生成する。"""
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
    """既存PDFにOCRテキストレイヤーを重ねて検索可能PDFを作成する。"""
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
                "ページ %s/%s 処理完了: %s",
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
    logging.info("検索可能 PDF を作成しました: %s", output_pdf)


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
    """複数PDFを一括でOCR埋め込みし、結果を返す。"""
    completed: List[Path] = []
    failed: List[Tuple[Path, str]] = []

    if not input_paths:
        logging.error("入力ファイルが選択されていません。")
        return BatchResult(completed=[], failed=[(Path("-"), "no input")])
    if not output_dir.is_dir():
        raise NotADirectoryError(f"出力ディレクトリが無効です: {output_dir}")

    for input_pdf in input_paths:
        try:
            if not input_pdf.is_file():
                raise FileNotFoundError(f"ファイルが存在しません: {input_pdf}")
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
            logging.error("エラー: %s の処理に失敗しました: %s", input_pdf, e)
            failed.append((input_pdf, str(e)))

    if completed:
        logging.info("成功: %s 件", len(completed))
        for p in completed:
            logging.info("出力: %s", p)
    if failed:
        logging.error("失敗: %s 件", len(failed))
        for ip, msg in failed:
            logging.error("失敗ファイル: %s -> %s", ip, msg)

    return BatchResult(completed=completed, failed=failed)


def find_pdfs_in_dir(dir_path: Path) -> List[Path]:
    """ディレクトリ直下の PDF ファイルを列挙して返す（サブディレクトリは探索しない）。"""
    try:
        entries = list(dir_path.iterdir())
    except Exception as e:
        raise FileNotFoundError(f"ディレクトリにアクセスできません: {dir_path} ({e})")
    pdfs = [p.resolve() for p in entries if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(pdfs)


def resolve_font_path(font: Optional[str]) -> Path:
    """フォントパスを解決し存在確認する。"""
    font_path = Path(font) if font else DEFAULT_FONT_PATH
    font_path = font_path.resolve()
    if not font_path.is_file():
        raise FileNotFoundError(f"フォントファイルが見つかりません: {font_path}")
    return font_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PDFにOCRテキストレイヤーを重ねて検索可能化するツール（CLI版）",
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="このディレクトリ内のPDFを一括処理（サブディレクトリは探索しない）",
    )
    parser.add_argument(
        "-f",
        "--file",
        nargs="+",
        help="処理するPDFファイル（複数指定可）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="出力先ディレクトリ。未指定時は入力場所を使用",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="レンダリング解像度（DPI）。既定300",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="重ねるテキストを可視化（デバッグ用、既定は不可視）",
    )
    parser.add_argument(
        "--font",
        type=str,
        default=str(DEFAULT_FONT_PATH),
        help="テキスト重ね合わせに使用するTrueTypeフォントファイルのパス",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログ出力レベル（既定: INFO）",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    input_paths: List[Path] = []
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            logging.error("ディレクトリが存在しません: %s", dir_path)
            return 1
        input_paths = find_pdfs_in_dir(dir_path)
        if not input_paths:
            logging.error("指定されたディレクトリにPDFがありません: %s", dir_path)
            return 1
    elif args.file:
        for f in args.file:
            p = Path(f)
            if not p.is_file():
                logging.warning("ファイルが存在しません。スキップします: %s", p)
                continue
            if p.suffix.lower() != ".pdf":
                logging.warning("PDFではないためスキップします: %s", p)
                continue
            input_paths.append(p.resolve())
        if not input_paths:
            logging.error("有効な入力PDFが指定されていません。")
            return 1
    else:
        logging.error("CLIモードでは --dir または --file のいずれかを指定してください。")
        return 2

    output_dir = (
        Path(args.output).resolve()
        if args.output
        else (Path(args.dir).resolve() if args.dir else Path(input_paths[0]).parent)
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error("出力ディレクトリを作成できません: %s (%s)", output_dir, e)
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


if __name__ == "__main__":
    raise SystemExit(main())
