## pdfembed

CLI/TUI tool to run OCR locally and overlay a searchable text layer on PDFs. By default, a Textual-based TUI launches; use `--cli` for the classic CLI.

### Quickstart

- TUI (default):  
  `python -m pdfembed.cli` or `pdfembed`

- CLI:  
  `python -m pdfembed.cli --cli --file sample.pdf --dpi 300`

### TUI Controls

- `f`: select PDF file(s) (opens a file dialog; multiple selection allowed)  
- `o`: select output folder (opens a folder dialog; defaults to the first PDF's directory)  
- `v`: toggle overlay visibility (debug)  
- `s`: start OCR  
- `q`: quit  
- DPI is fixed to the default in TUI; change via CLI `--dpi` if needed.

While OCR is running, a "Processing... please wait" indicator is shown and other keys are ignored until completion.

### CLI Options (key ones)

- `--file <pdf1> [pdf2 ...]` or `--dir <folder>`: input PDFs  
- `--output <dir>`: output directory (default: input location)  
- `--dpi <int>`: render DPI (default 300)  
- `--visible`: make overlay text visible (debug)  
- `--font <path>`: TTF font for overlay text  
- `--log-level <LEVEL>`: logging level (INFO by default)

### Dependencies

- Textual (TUI)
- tkinter (file dialogs, stdlib)
- onnxocr / pypdfium2 / pypdf / reportlab / opencv-python / numpy
