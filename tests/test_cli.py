import argparse
from pathlib import Path

import pytest

from pdfembed.cli import (
    BatchResult,
    OCRItem,
    _apply_rotation,
    find_pdfs_in_dir,
    normalize_onnxocr_results,
    run_cli,
)


def test_normalize_onnxocr_results_filters_and_maps():
    raw = [
        [([[0, 0], [0, 10], [10, 10], [10, 0]], ("hello", 0.9))],
        "invalid-page",
        [[("bad",)]],
    ]
    items = normalize_onnxocr_results(raw)
    assert len(items) == 1
    assert isinstance(items[0], OCRItem)
    assert items[0].text == "hello"
    assert items[0].bbox == (0, 0, 10, 10)
    assert items[0].score == 0.9


def test_apply_rotation_calls_expected_methods():
    class DummyPage:
        def __init__(self):
            self.calls = []

        def rotate_clockwise(self, angle):
            self.calls.append(("cw", angle))

        def rotate_counter_clockwise(self, angle):
            self.calls.append(("ccw", angle))

    p = DummyPage()
    _apply_rotation(p, 90)
    _apply_rotation(p, 180)
    _apply_rotation(p, 270)
    _apply_rotation(p, 0)
    assert p.calls == [("cw", 90), ("cw", 180), ("ccw", 90)]


def test_find_pdfs_in_dir_filters_only_pdf(tmp_path: Path):
    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "C.PDF").write_text("x")
    result = find_pdfs_in_dir(tmp_path)
    # 並び順はファイルシステム依存のため集合で比較し、名前の集合が一致することのみ確認
    assert set(p.name for p in result) == {"a.pdf", "C.PDF"}


def test_run_cli_returns_error_without_inputs():
    args = argparse.Namespace(
        dir=None,
        file=None,
        output=None,
        dpi=300,
        visible=False,
        font=str(Path(__file__).resolve()),  # dummy
        log_level="ERROR",
    )
    assert run_cli(args) == 2


def test_run_cli_success_with_stubbed_batch(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_text("x")

    called = {}

    def fake_process(input_paths, output_dir, dpi, font_path, visible):
        called["args"] = (list(input_paths), output_dir, dpi, font_path, visible)
        out = output_dir / "doc_ocr.pdf"
        out.write_text("ok")
        return BatchResult(completed=[out], failed=[])

    monkeypatch.setattr("pdfembed.cli.process_multiple_pdfs", fake_process)

    args = argparse.Namespace(
        dir=None,
        file=[str(pdf_path)],
        output=None,
        dpi=200,
        visible=True,
        font=str(pdf_path),  # dummy path, not used by fake
        log_level="ERROR",
    )
    status = run_cli(args)
    assert status == 0
    assert called["args"][2] == 200
    assert called["args"][4] is True