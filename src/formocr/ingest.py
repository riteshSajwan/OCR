"""PDF ingestion: digital text layer passthrough, or native-resolution scan extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import pymupdf
import numpy as np

# A page with less real text than this is treated as a scan.
TEXT_LAYER_MIN_CHARS = 40


@dataclass
class Page:
    index: int
    source: str  # "text" | "scan"
    image: np.ndarray | None = None  # BGR, only for scans
    text: str | None = None  # only for digital pages

    @property
    def is_scan(self) -> bool:
        return self.source == "scan"


def _embedded_image(doc: pymupdf.Document, page: pymupdf.Page) -> np.ndarray | None:
    """Pull the page's embedded raster at native resolution.

    These scans are a single full-page JPEG. Extracting it directly preserves every
    original pixel; re-rendering via get_pixmap() would resample and soften the
    handwriting strokes we most need to keep.
    """
    images = page.get_images(full=True)
    if not images:
        return None
    # Largest xref by pixel count - guards against logos/stamps being picked.
    best, best_area = None, -1
    for info in images:
        xref = info[0]
        try:
            raw = doc.extract_image(xref)
        except Exception:
            continue
        area = raw.get("width", 0) * raw.get("height", 0)
        if area > best_area:
            best, best_area = raw, area
    if best is None:
        return None
    buf = np.frombuffer(best["image"], dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _rendered_image(page: pymupdf.Page, dpi: int = 300) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if pix.n == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)


def load_pages(pdf_path: str | Path, dpi: int = 300) -> list[Page]:
    """Load every page, routing digital pages away from OCR entirely."""
    pages: list[Page] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if len(text) >= TEXT_LAYER_MIN_CHARS:
                pages.append(Page(index=i, source="text", text=text))
                continue
            img = _embedded_image(doc, page)
            if img is None:
                img = _rendered_image(page, dpi=dpi)
            pages.append(Page(index=i, source="scan", image=img))
    return pages


def has_text_layer(pdf_path: str | Path) -> bool:
    with pymupdf.open(str(pdf_path)) as doc:
        return any(len(p.get_text().strip()) >= TEXT_LAYER_MIN_CHARS for p in doc)
