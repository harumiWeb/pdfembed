## pdfembed

ローカルでPDFをOCRし、テキストレイヤーを重ねて検索可能なPDFを作成するツールです。既定ではTextualベースのTUIが起動します。従来のCLIを使う場合は `--cli` を付けてください。

ライセンス: BSD-3-Clause（`LICENSE` を参照）

### 使い方（クイック）

- TUI（既定）:  
  `python -m pdfembed.cli` または `pdfembed`

- CLI:  
  `python -m pdfembed.cli --cli --file sample.pdf --dpi 300`

### TUIの操作

- `f`: PDFを選択（ファイルダイアログが開き、複数選択可能）  
- `o`: 出力先フォルダを選択（フォルダダイアログ。未指定時は最初のPDFと同じ場所）  
- `v`: テキスト可視化のオン/オフ（デバッグ用）  
- `s`: OCR開始  
- `q`: 終了  
- TUIではDPIは固定です。変更が必要な場合はCLIで `--dpi` を指定してください。

OCR実行中は「Processing... please wait」が表示され、完了まで他のキーは無視されます。

### 主なCLIオプション

- `--file <pdf1> [pdf2 ...]` または `--dir <folder>`: 入力PDF  
- `--output <dir>`: 出力先ディレクトリ（既定: 入力と同じ場所）  
- `--dpi <int>`: レンダリングDPI（既定300）  
- `--visible`: テキストを可視化（デバッグ用）  
- `--font <path>`: 重ね合わせに使うTTFフォント  
- `--log-level <LEVEL>`: ログレベル（既定INFO）

### 依存

- Textual（TUI）
- tkinter（ファイルダイアログ、標準ライブラリ）
- onnxocr / pypdfium2 / pypdf / reportlab / opencv-python / numpy
