## pdfembed

ローカルでPDFをOCRし、テキストレイヤーを重ねて検索可能なPDFを作成するツールです。既定ではTextualベースのTUIが起動し、マウス操作だけでファイル/保存先を選べます。CLIを使う場合は`--cli`を付けてください。

### 使い方

- TUI（既定）:  
  `python -m pdfembed.cli` または `pdfembed`

- CLI:  
  `python -m pdfembed.cli --cli --file sample.pdf --dpi 300`

### TUIの操作

- 「Select PDF(s)」ボタンでPDFを選択（複数可）。
- 「Select output folder」で保存先を選択（未指定時は最初のPDFと同じ場所）。
- DPIやテキスト可視化のオプションを設定し、「Start OCR」で処理を開始。

### 依存

- Textual（TUI）
- tkinter（ファイルダイアログ用、標準ライブラリ）
- onnxocr / pypdfium2 / pypdf / reportlab など
