"""End-to-end: PDF in, structured rows out."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import cells as cellmod
from . import grid as gridmod
from . import ingest, orient, template as tplmod
from .cells import CellKind
from .engines.base import Engine

# Below this, a value goes to the review queue rather than being trusted.
REVIEW_THRESHOLD = 0.60
# Confidence assigned to geometric decisions, which are far more reliable than OCR.
GEOMETRIC_CONFIDENCE = 0.97
# Top slice of the page used to identify which form this is.
HEADER_BAND_FRAC = 0.18


def _coerce(text: str, numeric: bool) -> tuple[Any, bool]:
    """Return (value, clean). `clean` is False when the text fought the column type."""
    text = text.strip()
    if not text:
        return None, False
    if not numeric:
        return text, True
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None, False
    # Non-digit noise in a numeric cell means the read is suspect even if it parsed.
    return int(digits), digits == text


def _is_ditto(text: str, markers: list[str]) -> bool:
    squashed = re.sub(r"\s+", "", text)
    return bool(squashed) and squashed in markers


def process_page(
    img: np.ndarray,
    engine: Engine,
    templates: list[tplmod.Template],
    *,
    debug_dir: Path | None = None,
    stem: str = "page",
) -> dict:
    started = time.time()

    upright, orientation = orient.normalize(img)
    grid = gridmod.detect(upright)

    # Subtract the printed lattice once, up front. Everything downstream reads
    # these instead of the raw page: `ink` for deciding whether a cell holds
    # anything, `delined` for handing clean glyphs to the OCR engine.
    ink = gridmod.content_mask(upright, grid)
    delined = gridmod.remove_lines(upright, grid)

    # Match on the header band only. A full-page read is both slower and worse -
    # the form's own ruling lines wreck Tesseract's page segmentation.
    band = delined[: max(1, int(delined.shape[0] * HEADER_BAND_FRAC))]
    page_text = " ".join(b.text for b in engine.read_page(band))
    tpl = tplmod.match_template(page_text, templates) or tplmod.generic_template(grid.n_cols)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{stem}_grid.png"),
                    gridmod.debug_overlay(upright, grid))

    last_row = grid.n_rows - tpl.trailing_rows_to_drop
    rows: list[dict] = []
    review: list[dict] = []
    stats = {"empty": 0, "dash": 0, "ocr": 0, "low_conf": 0}
    confidences: list[float] = []
    previous: dict[str, Any] = {}

    for r in range(tpl.first_data_row, last_row):
        record: dict[str, Any] = {"_row": r}
        row_conf: dict[str, float] = {}
        # Pre-printed columns (serial numbers, fixed row labels) carry values on
        # every line of the form, including the blank ones. Only handwritten
        # columns can tell us whether a row was actually filled in.
        written: list[Any] = []

        for c in range(grid.n_cols):
            column = tpl.column(c)
            if column is None:
                continue
            cell = grid.cell_at(r, c)
            if cell is None:
                continue

            content = cellmod.classify(cell.crop(ink))

            if content.kind is CellKind.EMPTY:
                stats["empty"] += 1
                record[column.key] = None
                row_conf[column.key] = GEOMETRIC_CONFIDENCE
                continue

            if content.kind is CellKind.DASH:
                stats["dash"] += 1
                record[column.key] = tpl.dash_means
                row_conf[column.key] = GEOMETRIC_CONFIDENCE
                if not column.printed and tpl.dash_means is not None:
                    written.append(tpl.dash_means)
                continue

            stats["ocr"] += 1
            reading = engine.read_cell(cell.crop(delined), numeric=column.numeric)

            if column.key in tpl.ditto_columns and _is_ditto(reading.text, tpl.ditto_markers):
                record[column.key] = previous.get(column.key)
                row_conf[column.key] = 0.75  # inherited, not read
                written.append(record[column.key])
                continue

            value, clean = _coerce(reading.text, column.numeric)
            confidence = reading.confidence * (1.0 if clean else 0.5)
            record[column.key] = value
            row_conf[column.key] = round(confidence, 3)
            confidences.append(confidence)
            if not column.printed and value is not None:
                written.append(value)

            if value is not None and confidence < REVIEW_THRESHOLD:
                stats["low_conf"] += 1
                review.append({
                    "row": r,
                    "column": column.key,
                    "value": value,
                    "raw_text": reading.text,
                    "confidence": round(confidence, 3),
                    "bbox": [cell.x0, cell.y0, cell.x1, cell.y1],
                })

        # A row with nothing but its pre-printed serial number is an unused line.
        if any(v is not None for v in written):
            record["_confidence"] = round(float(np.mean(list(row_conf.values()))), 3)
            rows.append(record)
            previous = {k: v for k, v in record.items() if v is not None}

    checks_passed, checks_failed = _run_checks(rows, tpl)

    return {
        "form_id": tpl.form_id,
        "rows": rows,
        "review_queue": review,
        "_meta": {
            "orientation_applied": orientation.rotation,
            "orientation_method": orientation.method,
            "skew_deg": orientation.skew,
            "grid_rows": grid.n_rows,
            "grid_cols": grid.n_cols,
            "cells_total": len(grid.cells),
            "cells_empty": stats["empty"],
            "cells_dash": stats["dash"],
            "cells_ocr": stats["ocr"],
            "cells_low_confidence": stats["low_conf"],
            "mean_ocr_confidence": round(float(np.mean(confidences)), 3) if confidences else 0.0,
            "matched_header_text": page_text[:200],
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
            "engine": engine.name,
            "seconds": round(time.time() - started, 1),
        },
    }


def _run_checks(rows: list[dict], tpl: tplmod.Template) -> tuple[int, int]:
    """Score the form's own arithmetic. Failures mark rows for review."""
    passed = failed = 0
    for record in rows:
        for check in tpl.checks:
            target = record.get(check.target)
            parts = [record.get(k) for k in check.sum_of]
            if target is None or any(p is None for p in parts):
                continue
            if sum(parts) == target:
                passed += 1
            else:
                failed += 1
                record.setdefault("_check_failures", []).append(
                    f"{check.target}={target} but sum({'+'.join(check.sum_of)})={sum(parts)}"
                )
    return passed, failed


def process_pdf(
    pdf_path: str | Path,
    engine: Engine,
    templates: list[tplmod.Template] | None = None,
    *,
    debug_dir: Path | None = None,
) -> dict:
    pdf_path = Path(pdf_path)
    templates = templates if templates is not None else tplmod.load_all()
    pages = ingest.load_pages(pdf_path)

    result: dict[str, Any] = {"file": pdf_path.name, "pages": []}
    for page in pages:
        if not page.is_scan:
            # Digital PDF: the text is already there, OCR would only add error.
            result["pages"].append({
                "page": page.index,
                "source": "text",
                "text": page.text,
            })
            continue
        payload = process_page(
            page.image, engine, templates,
            debug_dir=debug_dir,
            stem=f"{pdf_path.stem}_p{page.index}".replace(" ", "_"),
        )
        payload["page"] = page.index
        payload["source"] = "scan"
        result["pages"].append(payload)
    return result
