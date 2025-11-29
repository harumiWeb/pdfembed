import io
import os
import argparse
import pypdfium2 as pdfium
import numpy as np
import cv2
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import unicodedata
from onnxocr.onnx_paddleocr import ONNXPaddleOcr

FONTS_PATH = "src/fonts/ipaexg.ttf"


def render_page_to_pil(page, dpi=300):
    """
    pypdfium2 のページオブジェクトを PIL.Image にレンダリング
    """
    scale = dpi / 72.0
    pil_img = page.render(scale=scale).to_pil()
    return pil_img


# OCRインスタンスをキャッシュ
_OCR_INSTANCE = None


def get_ocr():
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        _OCR_INSTANCE = ONNXPaddleOcr(use_gpu=False, lang="japan")
    return _OCR_INSTANCE


def run_onnx_ocr(pil_img):
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ocr = get_ocr()
    result = ocr.ocr(bgr)
    return result


def normalize_onnxocr_results(ocr_results):
    normalized = []
    for page_items in ocr_results:
        if not isinstance(page_items, (list, tuple)):
            continue
        for item in page_items:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            quad = item[0]
            text_tuple = item[1]
            # 4点のクアッドから単純に min/max で矩形化
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
            normalized.append({"text": text, "bbox": (x1, y1, x2, y2), "score": score})
    return normalized


def normalize_text_for_pdf(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def font_size_from_bbox_pt(
    x1_pt, y1_pt, x2_pt, y2_pt, min_size=6, max_size=48, scale=0.90
):
    height = max(1.0, (y2_pt - y1_pt))
    size = height * scale
    return max(min_size, min(max_size, size))


def make_overlay_pdf_bytes(
    page_w_pt, page_h_pt, img_w_px, img_h_px, ocr_items, font_path, visible=False
):
    """
    ReportLab で 1ページ分のオーバーレイPDFを生成して bytes を返す。
    page_w_pt, page_h_pt: 元PDFページのサイズ（ポイント）
    img_w_px, img_h_px: OCR元の画像サイズ（ピクセル）
    ocr_items: [{"text": str, "bbox": (x1,y1,x2,y2)}]
    visible: Trueで可視、Falseで不可視（透明）
    """
    if not os.path.isfile(font_path):
        raise FileNotFoundError(font_path)
    try:
        pdfmetrics.registerFont(TTFont("JP", font_path))
    except Exception as e:
        raise RuntimeError(f"フォント登録に失敗: {e}")
    sx = page_w_pt / float(img_w_px)
    sy = page_h_pt / float(img_h_px)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w_pt, page_h_pt))
    c.setAuthor("overlay-gen")
    c.setTitle("text-layer")
    c.saveState()
    try:
        # ReportLabの透明化API
        c.setFillAlpha(1.0 if visible else 0.0)
    except Exception:
        pass
    for item in ocr_items:
        text = normalize_text_for_pdf(item["text"])
        x1, y1, x2, y2 = item["bbox"]
        # ピクセル → ポイント変換
        x1_pt = x1 * sx
        y1_pt = y1 * sy
        x2_pt = x2 * sx
        y2_pt = y2 * sy
        # フォントサイズ（ポイント）
        fontsize = font_size_from_bbox_pt(
            x1_pt, y1_pt, x2_pt, y2_pt, min_size=6, max_size=48, scale=0.90
        )
        c.setFont("JP", fontsize)
        # フォントメトリクス（ascent, descent）を取得
        asc, desc = pdfmetrics.getAscentDescent("JP", fontsize)
        baseline_y = (page_h_pt - y2_pt) - desc
        x_pt = x1_pt + 0.5  # 左に寄せ過ぎ防止の微小オフセット
        try:
            c.drawString(x_pt, baseline_y, text)
        except Exception:
            # フォント未収録文字が含まれるとエラーになるケースへのフォールバック
            safe_text = "".join(ch if ord(ch) < 0x110000 else " " for ch in text)
            c.drawString(x_pt, baseline_y, safe_text)
    c.restoreState()
    c.showPage()
    c.save()
    return buf.getvalue()


def create_searchable_pdf_reportlab(
    input_pdf, output_pdf, dpi=300, font_path=FONTS_PATH, visible=False
):
    """
    ReportLab + pypdf で既存PDFにOCRテキストレイヤーを重ねて検索可能化。
    visible=True にするとデバッグ用にテキストが可視表示される。
    """
    # 元PDFのページサイズ取得（pypdf）
    reader = PdfReader(input_pdf)
    writer = PdfWriter()
    # input_pdf を開いてページごとに画像化（OCR用）
    src_doc = pdfium.PdfDocument(input_pdf)
    try:
        for page_index in range(len(reader.pages)):
            page = reader.pages[page_index]
            page_w_pt = float(page.mediabox.width)
            page_h_pt = float(page.mediabox.height)
            rotation = getattr(page, "rotation", 0) if hasattr(page, "rotation") else 0
            # pypdfium2 のページ取得
            src_page = src_doc[page_index]
            try:
                pil_img = render_page_to_pil(src_page, dpi=dpi)
            finally:
                # pypdfium2 はページを明示的に close 可能
                try:
                    src_page.close()
                except Exception:
                    pass
            img_w_px, img_h_px = pil_img.size
            # OCR
            ocr_results_raw = run_onnx_ocr(pil_img)
            ocr_items = normalize_onnxocr_results(ocr_results_raw)
            # オーバーレイPDF生成
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
            overlay_page = overlay_reader.pages[0]
            if rotation:
                try:
                    overlay_page.rotate(rotation)
                except Exception:
                    if rotation % 360 != 0:
                        angle = rotation % 360
                        if angle == 90:
                            overlay_page.rotate_clockwise(90)
                        elif angle == 180:
                            overlay_page.rotate_clockwise(180)
                        elif angle == 270:
                            overlay_page.rotate_counter_clockwise(90)
            page.merge_page(overlay_page)
            writer.add_page(page)
            print(
                f"ページ {page_index+1}/{len(reader.pages)} 処理完了: {os.path.basename(input_pdf)}"
            )
    finally:
        try:
            src_doc.close()
        except Exception:
            pass
    with open(output_pdf, "wb") as f:
        writer.write(f)
    print("✔ 検索可能 PDF を作成しました:", output_pdf)


def process_multiple_pdfs(
    input_paths,
    output_dir,
    dpi=300,
    font_path=FONTS_PATH,
    visible=False,
):
    """
    選択した複数PDFを指定ディレクトリに一括でOCR埋め込みする。
    出力ファイル名は <元名>_ocr.pdf とする。
    """
    if not input_paths:
        print("入力ファイルが選択されていません。")
        return
    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"出力ディレクトリが無効です: {output_dir}")
    completed = []
    failed = []
    for input_pdf in input_paths:
        try:
            if not os.path.isfile(input_pdf):
                raise FileNotFoundError(f"ファイルが存在しません: {input_pdf}")
            base = os.path.splitext(os.path.basename(input_pdf))[0]
            output_pdf = os.path.join(output_dir, f"{base}_ocr.pdf")
            create_searchable_pdf_reportlab(
                input_pdf=input_pdf,
                output_pdf=output_pdf,
                dpi=dpi,
                font_path=font_path,
                visible=visible,
            )
            completed.append(output_pdf)
        except Exception as e:
            print(f"エラー: {input_pdf} の処理に失敗しました: {e}")
            failed.append((input_pdf, str(e)))
    # 結果を表示
    summary_lines = []
    summary_lines.append(f"成功: {len(completed)} 件")
    summary_lines.append(f"失敗: {len(failed)} 件")
    if completed:
        summary_lines.append("\n出力ファイル:")
        summary_lines.extend([f"- {p}" for p in completed])
    if failed:
        summary_lines.append("\n失敗ファイル:")
        summary_lines.extend([f"- {ip} -> {msg}" for ip, msg in failed])
    summary = "\n".join(summary_lines)
    print(summary)


def find_pdfs_in_dir(dir_path):
    """
    指定ディレクトリ内の PDF ファイル（拡張子 .pdf / .PDF）を列挙して絶対パスで返す。
    サブディレクトリは探索しない。
    """
    try:
        entries = os.listdir(dir_path)
    except Exception as e:
        raise FileNotFoundError(f"ディレクトリにアクセスできません: {dir_path} ({e})")
    pdfs = []
    for name in entries:
        if name.lower().endswith(".pdf"):
            p = os.path.join(dir_path, name)
            if os.path.isfile(p):
                pdfs.append(os.path.abspath(p))
    return sorted(pdfs)


def main():
    """
    CLI モード:
      -d / --dir でディレクトリ内のPDFを一括処理
      -f / --file で単体または複数ファイルを処理
      -o / --output で出力ディレクトリを指定（未指定時は入力の場所を既定に）
      --dpi でレンダリング解像度（既定 300）
      --visible で重ねるテキストを可視化（デバッグ用）
    """
    parser = argparse.ArgumentParser(
        description="PDFにOCRテキストレイヤーを重ねて検索可能化するツール（GUI/CLI両対応）"
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
        help="出力先ディレクトリ（未指定時は入力場所を既定に使用）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="レンダリング解像度（DPI）。既定 300",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="重ねるテキストを可視化（デバッグ用、既定は不可視）",
    )
    parser.add_argument(
        "--font",
        type=str,
        default=FONTS_PATH,
        help="テキスト重ね合わせに使用するTrueTypeフォントファイルのパス",
    )
    args = parser.parse_args()

    # CLI モードの入力検証
    input_paths = []
    if args.dir:
        if not os.path.isdir(args.dir):
            print(f"エラー: ディレクトリが存在しません: {args.dir}")
            return
        input_paths = find_pdfs_in_dir(args.dir)
        if not input_paths:
            print(f"指定されたディレクトリにPDFがありません: {args.dir}")
            return
    elif args.file:
        for f in args.file:
            if not os.path.isfile(f):
                print(f"警告: ファイルが存在しません（スキップ）: {f}")
                continue
            if not f.lower().endswith(".pdf"):
                print(f"警告: PDFではないためスキップ: {f}")
                continue
            input_paths.append(os.path.abspath(f))
        if not input_paths:
            print("有効な入力PDFが指定されていません。")
            return
    else:
        parser.print_help()
        print(
            "\nError: CLIモードでは --dir または --file のいずれかを指定してください。"
        )
        return
    # 出力ディレクトリの決定
    if args.output:
        output_dir = args.output
    else:
        # 既定: ディレクトリ入力ならそのディレクトリ、ファイル入力なら最初のファイルのディレクトリ
        if args.dir:
            output_dir = args.dir
        else:
            output_dir = os.path.dirname(input_paths[0]) or os.getcwd()
    # 出力ディレクトリの用意
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"エラー: 出力ディレクトリを作成できません: {output_dir} ({e})")
        return
    # 一括処理の実行
    process_multiple_pdfs(
        input_paths=input_paths,
        output_dir=output_dir,
        dpi=args.dpi,
        font_path=args.font if args.font else FONTS_PATH,
        visible=args.visible,
    )


if __name__ == "__main__":
    main()
