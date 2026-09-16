# formocr

Turns scanned, handwritten-filled QA forms into structured JSON — one object per
table row, keyed by the form's printed column headers, with a confidence score per
value and a review queue for anything uncertain.

See [PLAN.md](PLAN.md) for the architecture and the reasoning behind it.

## Status

Two engines working end to end: **Tesseract** (fast baseline, reads the printed
structure) and **TrOCR** (handwriting). PaddleOCR PP-OCRv5 is next.

## Setup

### Core (required)

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
winget install -e --id UB-Mannheim.TesseractOCR
```

`pytesseract` is only a wrapper — the Tesseract binary is a separate install.
`config.py` finds it on PATH or in the standard Windows install locations.

### TrOCR engine (optional)

Only needed for `--engine trocr`. Install torch first and **name the build
explicitly**, or pip resolves to the ~2.5GB CUDA wheel by default:

```powershell
# CPU-only, ~200MB
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
# ...or CUDA, ~2.5GB
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124

.venv\Scripts\python.exe -m pip install -r requirements-trocr.txt
```

`transformers` is pinned below 5.x — `microsoft/trocr-base-handwritten` predates the
fast-tokenizer format, and v5 fails to load it with `Couldn't instantiate the backend
tokenizer`. Installing `sentencepiece`/`tiktoken` as that error suggests does not help.

The first TrOCR run downloads ~1.3GB. It is cached afterwards, so engine startup
drops from ~254s to ~6s.

## Running

Set this once per terminal:

```powershell
cd E:\OCR
$env:PYTHONPATH="src"
```

### Tesseract — fast baseline

```powershell
.venv\Scripts\python.exe -m formocr run "Minakshi Polymers\SCRAP NOTE.pdf"
```

Default engine, so no flag needed.

### TrOCR — handwriting

```powershell
.venv\Scripts\python.exe -m formocr run "Minakshi Polymers\SCRAP NOTE.pdf" --engine trocr
```

### Comparing the two

Output filenames carry the engine name, so both runs can share one directory and
sit side by side rather than overwriting each other:

```powershell
.venv\Scripts\python.exe -m formocr run "Minakshi Polymers\SCRAP NOTE.pdf"
.venv\Scripts\python.exe -m formocr run "Minakshi Polymers\SCRAP NOTE.pdf" --engine trocr

python.exe -m formocr run "Minakshi Polymers" --engine trocr --debug

```

```
out\SCRAP NOTE_tesseract.json
out\SCRAP NOTE_trocr.json
```

### Other arguments

| | |
|---|---|
| `<path>` | a single PDF, or a directory (searched recursively) |
| `--engine tesseract\|trocr` | recognition model; default `tesseract` |
| `--out DIR` | output directory; default `out/` |
| `--debug` | also write grid overlays to `<out>/debug/` |

Measured on `SCRAP NOTE.pdf` (19×7 grid, 5 filled rows):

| | Tesseract | TrOCR |
|---|---|---|
| Time | 11.4s | 90.8s |
| Mean confidence | 0.477 | 0.579 |
| Flagged for review | 22 | 23 |

Structure, rows and rotation come out identical — the engine only affects
recognition, not geometry. TrOCR is ~8× slower here because it is running on CPU.

Output lands in `<out>/<name>_<engine>.json`; `--debug` writes grid overlays to
`<out>/debug/`.
**Look at the overlays before trusting any number** — a misaligned lattice produces
well-formed, confident, completely wrong JSON.

## Notebook — stage-by-stage walkthrough

[`notebooks/formocr_pipeline.ipynb`](notebooks/formocr_pipeline.ipynb) runs the same
pipeline one stage at a time, showing the intermediate image at each step: the raw
scan, the deskewed page, both line masks, the detected lattice, the ink mask, the
de-lined greyscale, sample cell crops, and the review-queue crops beside what each
engine read. It ends with a Tesseract-vs-TrOCR comparison on the same cells.

It **imports `src/formocr`** rather than reimplementing anything, so it cannot drift
from the real pipeline. Use it to understand or debug a stage; use the CLI to process
files.

```powershell
.venv\Scripts\python.exe -m pip install jupyter matplotlib
.venv\Scripts\python.exe -m jupyter lab notebooks/formocr_pipeline.ipynb
```

It is committed **with outputs**, so it reads without being run. Clear them with
`jupyter nbconvert --clear-output --inplace notebooks/formocr_pipeline.ipynb`.

**On Colab** it detects the environment, installs the Tesseract binary via apt, takes
the package as a one-time zip upload (`src/` + `templates/` is all it needs), and then
**prompts you to upload the PDF** — no project checkout required. Mount Drive instead
if you would rather not re-upload each session; `/content/drive/MyDrive/OCR` is one of
the searched paths.

Colab is worth it for **fine-tuning** (a free T4's 16GB beats 4GB of local VRAM); for
everything else local is faster and skips the upload.

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

### TrOCR head-to-head

Ten cells on SCRAP NOTE with hand-typed truth. `rejection_qty` is a numeric column,
so TrOCR runs with constrained decoding (see below):

| Truth | Tesseract | conf | TrOCR | conf |
|---|---|---|---|---|
| `07` | `6 1` | 0.42 | `0 04` | 0.74 |
| `06` | *(empty)* | 0.00 | `0 06` | 0.74 |
| `04` | *(empty)* | 0.00 | `0 94` | 0.61 |
| `05` | *(empty)* | 0.00 | `0 052` | 0.57 |
| `04` | `6` | 0.00 | `0 04` | 0.74 |
| `WLD 602503` | `WLD 602503` | 0.70 | `who 602303` | 0.64 |
| `WLD 602500` | `WIL Corse` | 0.14 | `WLDS GO 2500` | 0.47 |
| `WLD 602509` | `W/L 0) (e804` | 0.28 | `WLD 602509 .` | 0.78 |
| `WLD 602514` | `Wey) ors]` | 0.32 | `WLN 602514` | 0.89 |
| `WLD 602539` | `AIL 0 6025739` | 0.46 | `WH D.059` | 0.45 |

On the numeric column Tesseract gets **0/5**; TrOCR gets **2/5** after coercion. On item
codes both land around 1–2/5, but TrOCR's misses are far closer — `WLN 602514` is one
character out, against Tesseract's `Wey) ors]` — and its confidence tracks correctness
(0.89 on the near-miss, 0.45 on its worst).

TrOCR is clearly better on handwriting and better calibrated. It is still not accurate
enough to trust unreviewed, which is what fine-tuning is for.

**Constrained decoding.** Tesseract gets a charset whitelist for free via
`tessedit_char_whitelist`. TrOCR is a seq2seq language model with no such switch, so
`_AllowedTokensProcessor` masks the decoder at the logits instead — **1698 of 50265
tokens** survive on a numeric column (digits, space, control tokens). The model
therefore cannot spell a letter into a numeric cell at all, rather than emitting one
and having `_coerce` strip it and halve the confidence.

**Caveat on every number here:** n=10, typed by eye. Confidence is what an engine
reports about itself, not accuracy. See *Next*.

## Next

The engine interface (`engines/base.py`) is deliberately narrow, which is how TrOCR
dropped in without touching the pipeline. Still to do:

1. **A ground-truth set.** Every number quoted above is an engine's *self-reported
   confidence*, not accuracy — and an engine can be confidently wrong. Comparing
   engines on confidence is close to meaningless, since they calibrate differently.
   A few hundred hand-typed cells plus a scoring script is the cheapest way to make
   any of these comparisons real, and it must come before choosing an engine.
2. **PaddleOCR PP-OCRv5** on printed cells — should lift the header/label columns to near 1.0.
   Check Python 3.13 wheel availability first; fall back to a 3.11 venv if missing.
3. **Review UI + fine-tune.** Corrections become labelled pairs; fine-tuning TrOCR on this
   specific handwriting is where accuracy stops being mediocre. Stock
   `trocr-base-handwritten` is trained on IAM — English cursive prose, not Indian
   factory shorthand and part codes.

Known gaps: ditto-mark detection currently matches on OCR text and misses (`ly`, `jr`),
so it should become geometric — two short near-vertical strokes. `DAILY MIS REPORT
PCOATING` picks the wrong 90°/270° orientation and needs a template. Constrained
numeric decoding in TrOCR prefixes a spurious `0` (the constraint forcing a digit at
position one); harmless for integer columns since `int()` drops it, but it would
matter if the raw string were ever needed.

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
