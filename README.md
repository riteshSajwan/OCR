# formocr

Turns scanned, handwritten-filled QA forms into structured JSON — one object per
table row, keyed by the form's printed column headers, with a confidence score per
value and a review queue for anything uncertain.

See [PLAN.md](PLAN.md) for the architecture and the reasoning behind it.

## Status

**Stage 1 of 3 complete** — Tesseract baseline, full pipeline running end to end on
all 13 sample PDFs. PaddleOCR PP-OCRv5 and TrOCR are the next two engines.

## Setup

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
winget install -e --id UB-Mannheim.TesseractOCR
```

`pytesseract` is only a wrapper — the Tesseract binary is a separate install.
`config.py` finds it on PATH or in the standard Windows install locations.

## Usage

```powershell
$env:PYTHONPATH="src"
.venv\Scripts\python.exe -m formocr run "Minakshi Polymers"          # folder or single PDF
.venv\Scripts\python.exe -m formocr run "Minakshi Polymers" --debug  # + grid overlays
```

Output lands in `out/<name>.json`; `--debug` writes grid overlays to `out/debug/`.
**Look at the overlays before trusting any number** — a misaligned lattice produces
well-formed, confident, completely wrong JSON.

## How it works

```
PDF ─┬─ text layer?  ──► pymupdf extract, no OCR
     └─ scan ──► orient (OSD) ──► deskew ──► detect grid ──► subtract ruling lines
                 ──► classify each cell ──► OCR only what's left ──► template ──► JSON
```

Three decisions carry most of the accuracy:

**Cell-level, not page-level.** The printed ruling lines are detected with morphology,
giving exact row/column boundaries. Each cell is cropped and recognized on its own, so
a cell holding `04` is a trivial problem instead of two digits buried in a 2481×3506
page. The JSON structure falls out for free because every cell knows its row and column.

**Subtract the lattice before reading anything.** Per-cell Otsu thresholding cannot
distinguish blank paper from faint ink — it always finds a split, so it manufactures
ink in empty cells. Thresholding the page once and subtracting the grid we already
detected fixes that, and it also stops the ruling lines from wrecking Tesseract's
page segmentation.

**Classify cells geometrically before OCR.** On MIS PLATING, 763 of 1156 cells
(**66%**) are resolved as empty or as a dash-meaning-zero using pixel statistics alone.
No model ever gets the chance to hallucinate a digit into an empty box, and the run is
several times faster.

## Results — Tesseract baseline

| File | Grid | Rot | Cells resolved w/o OCR | Mean conf | Time |
|---|---|---|---|---|---|
| MIS PLATING | 70×17 | 0° | 763/1156 (66%) | 0.35 | 57s |
| SCRAP NOTE | 19×7 | 180° | 54/133 | 0.42 | 11s |
| PRESS SHOP REJECTION | 37×7 | 180° | — | 0.52 | 13s |
| ST. Pass weld data | 59×29 | 270° | 1273/1711 | 0.46 | 64s |
| SUPPLIER REJECTION | 21×8 | 0° | 115/168 | 0.55 | 9s |

All 13 PDFs in ~270s. The two `OP table` files have a real text layer and correctly
bypass OCR entirely.

### What works

- **Grid detection.** Column counts match the forms exactly (SCRAP NOTE 7, MIS PLATING 17),
  with a dead-consistent row pitch (178px and 40px respectively).
- **Orientation.** Tesseract OSD correctly caught 180° on SCRAP NOTE and PRESS SHOP,
  270° on ST. Pass weld data.
- **Printed text.** `PEEL OFF`, `PLATING SHOP`, `DULL PLATING`, `CLAENING FAULT` — all
  read correctly, including the form's own typo.
- **Empty/dash classification and blank-row suppression.** SCRAP NOTE returns exactly
  its 5 filled rows out of 14 printed lines.

### What does not — and why that was expected

**Tesseract cannot read the handwriting.** Measured on SCRAP NOTE:

| Truth | Tesseract |
|---|---|
| `Carrier AAD` | `Cawier Wan` |
| `WLD 602500` | `WIL Corse` |
| `07` | `2` |

Mean confidence on handwritten cells is 0.34–0.55. This is the documented floor, not a
bug to fix — Tesseract has no handwriting model. It earns its place here by reading the
printed structure, supplying OSD, and proving the pipeline end to end.

## Next

The engine interface (`engines/base.py`) is deliberately narrow so the next two drop in
without touching the pipeline:

1. **PaddleOCR PP-OCRv5** on printed cells — should lift the header/label columns to near 1.0.
   Check Python 3.13 wheel availability first; fall back to a 3.11 venv if missing.
2. **TrOCR** (`microsoft/trocr-base-handwritten`) on handwritten cells — the actual fix
   for the table above. ~1.3 GB, comfortable on the 4 GB GTX 1650 Ti.
3. **Review UI + fine-tune.** Corrections become labelled pairs; fine-tuning TrOCR on this
   specific handwriting is where accuracy stops being mediocre.

Known gaps: ditto-mark detection currently matches on OCR text and misses (`ly`, `jr`),
so it should become geometric — two short near-vertical strokes. `DAILY MIS REPORT
PCOATING` picks the wrong 90°/270° orientation and needs a template.

## Layout

```
src/formocr/
  config.py      paths, Tesseract discovery
  ingest.py      PDF -> native-resolution image, or text-layer passthrough
  orient.py      OSD rotation + Hough deskew
  grid.py        ruling-line detection, cell boxes, line subtraction, debug overlay
  cells.py       empty/dash/content classification, OCR pre-processing
  template.py    per-form YAML layouts
  pipeline.py    orchestration, confidence, arithmetic checks
  cli.py         formocr run
  engines/
    base.py              Engine protocol - implement this to add a model
    tesseract_engine.py  baseline
templates/       mis_plating.yaml, scrap_note.yaml
out/             JSON results, debug/ overlays
```
