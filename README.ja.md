## pdfembed

![preview]("https://github.com/user-attachments/assets/0610ecf0-fec0-4657-abe5-c4387a91bbae")

---

ローカルで PDF を OCR し、テキストレイヤーを重ねて検索可能な PDF を作成するツールです。既定では Textual ベースの TUI が起動します。従来の CLI を使う場合は `--cli` を付けてください。

ライセンス: BSD-3-Clause（`LICENSE` を参照）

### 使い方（クイック）

- TUI（既定）:  
  `python -m pdfembed.cli` または `pdfembed`

- CLI:  
  `python -m pdfembed.cli --cli --file sample.pdf --dpi 300`

### TUI の操作

- `f`: PDF を選択（ファイルダイアログが開き、複数選択可能）
- `o`: 出力先フォルダを選択（フォルダダイアログ。未指定時は最初の PDF と同じ場所）
- `v`: テキスト可視化のオン/オフ（デバッグ用）
- `s`: OCR 開始
- `q`: 終了
- TUI では DPI は固定です。変更が必要な場合は CLI で `--dpi` を指定してください。

OCR 実行中は「Processing... please wait」が表示され、完了まで他のキーは無視されます。

### 主な CLI オプション

- `--file <pdf1> [pdf2 ...]` または `--dir <folder>`: 入力 PDF
- `--output <dir>`: 出力先ディレクトリ（既定: 入力と同じ場所）
- `--dpi <int>`: レンダリング DPI（既定 300）
- `--visible`: テキストを可視化（デバッグ用）
- `--font <path>`: 重ね合わせに使う TTF フォント
- `--log-level <LEVEL>`: ログレベル（既定 INFO）

### 依存

- Textual（TUI）
- tkinter（ファイルダイアログ、標準ライブラリ）
- onnxocr / pypdfium2 / pypdf / reportlab / opencv-python / numpy
