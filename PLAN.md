# Handwritten Form OCR → Structured JSON

**Goal:** turn the scanned Meenakshi Polymers QA forms into per-file JSON where each
table row is an object keyed by the printed column headers, with a confidence score
per value and a review queue for anything uncertain.

---

## 1. What we are actually dealing with

Probed from the source files, not assumed:

| Input | Nature | Handling |
|---|---|---|
| `Minakshi Polymers/*.pdf` (11) | Scanned **JPEG only**, no font objects. Printed ruled forms, **handwritten** entries. | Full pipeline below |
| `Minakshi Polymers/OP table/*.pdf` (2) | Digital PDFs with a real text layer (Excel exports) | `pymupdf` text extraction. **No OCR.** |

Two representative pages:

- **SCRAP NOTE** — ~14-row table, handwritten item codes, cursive remarks, ditto marks.
  Scanned **rotated 180°**.
- **MIS PLATING** — ~40 rows x 16 columns of handwritten digits (`04`, `147`), with dashes
  standing in for zero. Printed row labels and column headers.

Native raster is 2481x3506 ≈ **300 DPI A4**. That is enough resolution; do not upscale.

### The three facts that drive the whole design

1. **The printed grid lines are the structure.** Detect them with classical CV and you get
   row/column geometry for free — far more reliable than asking a model to infer a table.
2. **These are fixed forms, reused monthly.** ~8 distinct layouts across 11 files. A small
   per-form template beats general-purpose table understanding by a wide margin.
3. **The forms contain their own arithmetic.** `TOTAL` columns and `TOTAL ... DEFECT` rows are
   sums of their neighbours. That gives us a free correctness check on OCR output.

---

## 2. Model choice

Hardware constraint: **GTX 1650 Ti, 4 GB VRAM** (Turing, CUDA 7.5). That rules out
full-precision 7B+ vision-language models.

| Role | Model | License | Footprint | Why |
|---|---|---|---|---|
| Printed text (headers, labels, doc numbers) | **PaddleOCR PP-OCRv5** | Apache-2.0 | CPU-fine | Best open printed accuracy; ships detection + recognition + document orientation classifier |
| Handwritten cell values | **TrOCR** `microsoft/trocr-base-handwritten` | MIT | ~1.3 GB | Line-level encoder-decoder — exactly matches a one-cell crop. Fine-tunable on our own data |
| Grid / geometry | **OpenCV** morphology + Hough | Apache-2.0 | negligible | Deterministic, debuggable, no training |

**Evaluated and deliberately not chosen for v1:**

- **dots.ocr** (1.7B, MIT) — emits structured JSON directly and fits in 4 GB at 4-bit.
  Best candidate for a second engine to A/B against once the baseline works.
- **Qwen2.5-VL-3B** (Apache-2.0) — strongest open doc-understanding at this size, but tight
  on 4 GB once vision tokens for a 300 DPI page are counted, and slow.
- **Surya** — very good layout/table detection, but the license is not plain Apache; check
  terms before any commercial use.
- **Tesseract / EasyOCR** — Tesseract is effectively unusable on handwriting; EasyOCR is
  strictly worse than PaddleOCR here. Skip both.

### Why not just throw the page at a VLM

A VLM will happily return a clean, well-formed, **confidently wrong** table. On a 40x16 grid
of two-digit numbers it silently drifts — shifting values between columns, inventing plausible
totals. For QA reject data that is worse than no answer, because nothing downstream flags it.
The cell-crop approach makes each recognition trivially scoped and gives us a real
per-cell confidence to gate on.

**Set expectations honestly:** ~75–90% per-cell accuracy on handwritten digits before
fine-tuning. The validation layer (§3.7) and review loop (§3.9) are what make the output
trustworthy — they are core features, not polish.

---

## 3. Pipeline

```
PDF ─┬─ has text layer? ──► pymupdf extract ──────────────────────┐
     │                                                            │
     └─ scanned ──► 1 extract JPEG at native res                  │
                    2 orientation (0/90/180/270)                  │
                    3 deskew                                      │
                    4 grid detection ──► row/col boundaries       │
                    5 cell crops                                  │
                    6 empty / dash classifier ──► null            │
                    7 OCR: printed → Paddle, handwritten → TrOCR  │
                    8 template mapping ──► keys + types           │
                    9 arithmetic validation ──► confidence        │
                                                                  ▼
                                                        JSON + review queue
```

### 3.1 Page extraction
Pull the embedded JPEG directly via `doc.extract_image(xref)` rather than re-rendering with
`get_pixmap()`. The page *is* a single JPEG; re-rendering resamples it and loses detail.

### 3.2 Orientation
SCRAP NOTE proves this is mandatory. Use PaddleOCR's document orientation classifier.
Fallback that needs no extra binary: OCR at 0° and 180°, keep whichever has higher mean
recognition confidence against expected header keywords (`MEENAKSHI`, `DOC.NO`).

### 3.3 Deskew
Estimate skew from the dominant horizontal ruling line angle (`cv2.HoughLinesP`), then
`cv2.warpAffine`. Ruling lines are a much stronger signal here than a projection profile.

### 3.4 Grid detection — the core

1. Grayscale → Sauvola / adaptive threshold, invert so ink is white.
2. Morphological open with a long horizontal kernel `(w//40, 1)` → horizontal line mask.
3. Same with vertical kernel `(1, h//40)` → vertical line mask.
4. `bitwise_and` the two masks → lattice intersection points.
5. Cluster 1-D projection peaks → `xs` column boundaries, `ys` row boundaries.
6. Cells = rectangles between consecutive boundaries, inset ~3 px to drop the ruling stroke.

Merged cells show up as a missing border segment. v1 assumes a regular lattice and **flags**
anomalies rather than guessing; §3.8 templates declare known merges.

### 3.5 Empty and dash detection — do this before any OCR
On MIS PLATING roughly half the cells are blank or a single dash. Classify geometrically:
ink-pixel ratio below threshold → `null`; a low-height wide horizontal blob → `0`.
No model involved. This is the single biggest accuracy and speed win in the pipeline.

### 3.6 Cell OCR
Route by template, not by a classifier: label columns → PaddleOCR, value columns → TrOCR.
Constrain the charset per column (numeric columns accept digits only); anything failing the
constraint is emitted with low confidence rather than silently coerced.

### 3.7 Arithmetic validation
Per form, declare the identities: `row.total == sum(row.value_columns)`,
`TOTAL PLATING DEFECT == sum(rows 1..10)`. Then:

- All identities hold → raise confidence on the whole block.
- One cell is low-confidence and the identity has exactly one unknown → **solve for it**.
- Identity fails with all cells confident → flag the row for review.

### 3.8 Form templates
One YAML per layout, matched by a printed keyword:

```yaml
form_id: mis_plating
match: "DAILY MIS REPORT"
header_rows: [0]
label_columns: [sr_no, defect, reason]
value_type: integer
empty_markers: ["-", "~"]
checks:
  - "row.total == sum(row.values)"
```

### 3.9 Output and review loop
JSON per file, plus `_meta.confidence` per cell and a `review_queue` of cells below
threshold. A small Streamlit page shows the cell crop beside the OCR value for fast
correction — and every correction becomes a labelled training pair. After a few hundred,
fine-tune TrOCR on this handwriting specifically. Accuracy climbs sharply because the
character set is small, repetitive, and written by only a handful of people.

---

## 4. Layout

```
E:\OCR\
  pyproject.toml
  templates/              mis_plating.yaml, scrap_note.yaml, ...
  src/formocr/
    ingest.py             PDF -> page images / text-layer passthrough
    orient.py             rotation + deskew
    grid.py               line detection -> cell boxes
    cells.py              crop, empty/dash classification
    engines/
      paddle.py           printed text
      trocr.py            handwriting
    template.py           YAML load, cell -> key mapping
    validate.py           arithmetic checks, confidence adjustment
    emit.py               JSON writer + review queue
    cli.py                formocr run <path> --out out/
  tests/
  out/
```

---

## 5. Milestones

| # | Deliverable | Notes |
|---|---|---|
| M0 | Text-layer PDFs → JSON | The 2 `OP table` files. Quick win, no OCR |
| M1 | Ingest + orientation + deskew | Verified against the 180° SCRAP NOTE |
| M2 | Grid detection with debug overlay | Visual check is essential — render detected cells |
| M3 | Empty/dash classifier | Measure % of cells resolved with zero OCR |
| M4 | PaddleOCR on printed cells | Headers and row labels correct |
| M5 | TrOCR on handwritten cells | First end-to-end JSON |
| M6 | Templates + arithmetic validation | Accuracy measured, not guessed |
| M7 | Review UI + TrOCR fine-tune | Closes the data flywheel |

M2 is the make-or-break step. If grid detection is solid on both sample forms, the rest is
assembly. Build the debug overlay first.

---

## 6. Risks

- **Python 3.13 + PaddlePaddle.** Paddle's wheel support has historically trailed new CPython
  releases. **Verify before committing**; if 3.13 wheels are missing, create the venv on
  **Python 3.11**. This is the first thing to check at M4.
- **PyMuPDF is AGPL.** Fine for internal use, a problem if this ships in a commercial product.
  Alternative: `pdf2image` + poppler, or `pikepdf` for raw JPEG extraction.
- **4 GB VRAM.** TrOCR-base fits comfortably. Batch cell crops; do not hold Paddle and TrOCR
  on the GPU simultaneously.
- **Cursive remarks** (SCRAP NOTE's "Penetration cut part") are much harder than digits.
  Accept lower confidence on free-text columns and route them to review by default.
- **Ditto marks.** A handwritten ditto means "same as above". Resolve during post-processing
  by inheriting the value from the row above — needs an explicit rule, or the value is lost.

---

## 7. Target output

```json
{
  "file": "MIS PLATING.pdf",
  "form_id": "mis_plating",
  "doc_no": "MPPL/F/QAD/35/PC",
  "date": "23-07-2026",
  "shift": "A",
  "rows": [
    {
      "sr_no": 3,
      "defect": "DULL PLATING",
      "reason": "PLATING SHOP",
      "cr_aael": 25,
      "cr_aad": 20,
      "engine_guard": 31,
      "pedal_brake": 20,
      "pgc": 10,
      "handle": null,
      "total": 106
    }
  ],
  "_meta": {
    "orientation_applied": 0,
    "skew_deg": -0.4,
    "cells_total": 640,
    "cells_empty": 331,
    "mean_confidence": 0.87,
    "checks_passed": 38,
    "checks_failed": 2
  },
  "review_queue": [
    { "row": 7, "column": "handle", "value": "02", "confidence": 0.41,
      "crop": "out/crops/MIS_PLATING_r7_c6.png" }
  ]
}
```
