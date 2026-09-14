"""Cell content classification, done geometrically before any OCR runs.

On these forms roughly half the cells are blank or hold a single dash standing in
for zero. Resolving those with pixel statistics instead of a model is the largest
single accuracy and speed win in the pipeline: no engine ever gets the chance to
hallucinate a digit into an empty box.

Classification operates on a page-level ink mask that already has the printed
ruling lines subtracted (see grid.content_mask). Measuring a raw cell crop instead
does not work - per-cell Otsu always finds a split, so it manufactures ink in
empty cells, and the inset crop still carries fragments of the printed lattice.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class CellKind(str, Enum):
    EMPTY = "empty"
    DASH = "dash"  # a stroke meaning zero
    CONTENT = "content"


# Fraction of ink pixels below which a cell is considered blank. Scans carry
# speckle and bleed-through from the reverse side, so this is not zero.
EMPTY_INK_RATIO = 0.004
# Ink that survives as only a few tiny blobs is noise, not a written value.
MIN_COMPONENT_AREA = 12
MIN_CONTENT_AREA_FRAC = 0.002

# A dash is wide relative to its height...
DASH_MIN_ASPECT = 2.2
# ...and occupies only a thin horizontal band of the cell.
DASH_MAX_HEIGHT_FRAC = 0.35
DASH_MAX_INK_RATIO = 0.12


@dataclass
class CellContent:
    kind: CellKind
    ink_ratio: float
    components: int


def classify(mask_crop: np.ndarray) -> CellContent:
    """Classify a cell from its line-subtracted ink mask."""
    if mask_crop.size == 0:
        return CellContent(CellKind.EMPTY, 0.0, 0)

    ink_ratio = float((mask_crop > 0).mean())
    if ink_ratio < EMPTY_INK_RATIO:
        return CellContent(CellKind.EMPTY, ink_ratio, 0)

    contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = [c for c in contours if cv2.contourArea(c) >= MIN_COMPONENT_AREA]
    if not keep:
        return CellContent(CellKind.EMPTY, ink_ratio, 0)

    cell_h, cell_w = mask_crop.shape[:2]
    # Ink spread over a few specks that together cover almost nothing is scanner
    # noise or bleed-through, not handwriting.
    covered = sum(cv2.contourArea(c) for c in keep) / float(cell_h * cell_w)
    if covered < MIN_CONTENT_AREA_FRAC:
        return CellContent(CellKind.EMPTY, ink_ratio, len(keep))

    if len(keep) == 1 and ink_ratio < DASH_MAX_INK_RATIO:
        _, _, w, h = cv2.boundingRect(keep[0])
        if h > 0 and (w / h) >= DASH_MIN_ASPECT and (h / cell_h) <= DASH_MAX_HEIGHT_FRAC:
            return CellContent(CellKind.DASH, ink_ratio, 1)

    return CellContent(CellKind.CONTENT, ink_ratio, len(keep))


def prepare_for_ocr(crop: np.ndarray, scale: int = 3, pad: int = 12) -> np.ndarray:
    """Upscale and pad a cell crop to the size OCR engines expect.

    Tesseract in particular wants an x-height around 30px and a quiet margin;
    handing it a raw 40px-tall cell crop measurably degrades recognition.
    """
    if crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
