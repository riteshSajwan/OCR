"""Table structure from the printed ruling lines.

The forms are pre-printed ruled tables, so the grid is physically drawn on the page.
Recovering it with morphology is deterministic and inspectable - no model has to
infer where the columns are, and every cell arrives already knowing its row/column
index, which is what makes the JSON structure fall out for free.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# A ruling line spans much of the table; this fraction of the strongest projection
# separates real lines from dense text rows.
LINE_PEAK_FRAC = 0.30
# Boundaries closer than this are the two edges of one thick line, not two lines.
MERGE_WITHIN_PX = 12
# Discard slivers that cannot hold a value.
MIN_CELL_W = 18
MIN_CELL_H = 12


@dataclass
class Cell:
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def crop(self, img: np.ndarray, inset: int = 3) -> np.ndarray:
        """Crop the cell interior, insetting to drop the printed ruling stroke."""
        y0 = min(self.y0 + inset, self.y1 - 1)
        y1 = max(self.y1 - inset, y0 + 1)
        x0 = min(self.x0 + inset, self.x1 - 1)
        x1 = max(self.x1 - inset, x0 + 1)
        return img[y0:y1, x0:x1]


@dataclass
class Grid:
    xs: list[int]
    ys: list[int]
    cells: list[Cell]
    horizontal: np.ndarray  # line mask, kept for debugging
    vertical: np.ndarray

    @property
    def n_rows(self) -> int:
        return max(0, len(self.ys) - 1)

    @property
    def n_cols(self) -> int:
        return max(0, len(self.xs) - 1)

    def cell_at(self, row: int, col: int) -> Cell | None:
        for c in self.cells:
            if c.row == row and c.col == col:
                return c
        return None


def binarize(img: np.ndarray) -> np.ndarray:
    """Adaptive threshold to an ink-is-white mask, robust to uneven scan lighting."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=15,
    )


def line_masks(bw: np.ndarray, scale: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Isolate long horizontal and vertical strokes with directional opening."""
    h, w = bw.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // scale), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // scale)))
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)
    # Bridge dashes and scan dropouts along each line's own direction.
    horizontal = cv2.dilate(horizontal, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    vertical = cv2.dilate(vertical, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15)))
    return horizontal, vertical


def _boundaries(mask: np.ndarray, axis: int, peak_frac: float = LINE_PEAK_FRAC) -> list[int]:
    """Collapse a line mask to one coordinate per ruling line."""
    projection = (mask > 0).sum(axis=axis).astype(np.float64)
    if projection.max() <= 0:
        return []
    hits = np.where(projection > projection.max() * peak_frac)[0]
    if hits.size == 0:
        return []

    # Group runs of adjacent indices - one group per physical line.
    groups: list[list[int]] = [[int(hits[0])]]
    for idx in hits[1:]:
        if idx - groups[-1][-1] <= MERGE_WITHIN_PX:
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
    return [int(round(float(np.mean(g)))) for g in groups]


def _drop_tight(coords: list[int], min_gap: int) -> list[int]:
    kept: list[int] = []
    for c in coords:
        if not kept or c - kept[-1] >= min_gap:
            kept.append(c)
    return kept


def detect(img: np.ndarray, scale: int = 40) -> Grid:
    """Detect the table lattice and enumerate its cells."""
    bw = binarize(img)
    horizontal, vertical = line_masks(bw, scale=scale)

    ys = _drop_tight(_boundaries(horizontal, axis=1), MIN_CELL_H)
    xs = _drop_tight(_boundaries(vertical, axis=0), MIN_CELL_W)

    cells: list[Cell] = []
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            y0, y1, x0, x1 = ys[r], ys[r + 1], xs[c], xs[c + 1]
            if (x1 - x0) < MIN_CELL_W or (y1 - y0) < MIN_CELL_H:
                continue
            cells.append(Cell(row=r, col=c, x0=x0, y0=y0, x1=x1, y1=y1))

    return Grid(xs=xs, ys=ys, cells=cells, horizontal=horizontal, vertical=vertical)


def debug_overlay(img: np.ndarray, grid: Grid) -> np.ndarray:
    """Render detected boundaries and cells so the lattice can be eyeballed.

    Build this before trusting any downstream number - a silently wrong grid
    produces confident, well-formed, completely misaligned JSON.
    """
    canvas = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    for y in grid.ys:
        cv2.line(canvas, (0, y), (w, y), (0, 0, 255), 2)
    for x in grid.xs:
        cv2.line(canvas, (x, 0), (x, h), (255, 0, 0), 2)
    for cell in grid.cells:
        cv2.rectangle(canvas, (cell.x0 + 2, cell.y0 + 2), (cell.x1 - 2, cell.y1 - 2),
                      (0, 200, 0), 1)
    return canvas


def line_mask(grid: "Grid") -> np.ndarray:
    """The printed lattice itself, thickened enough to cover its anti-aliased edges."""
    lines = cv2.bitwise_or(grid.horizontal, grid.vertical)
    return cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)


def content_mask(img: np.ndarray, grid: "Grid") -> np.ndarray:
    """Ink mask with the printed ruling lines removed.

    Per-cell Otsu cannot tell blank paper from faint ink - it always finds a split,
    so it manufactures ink in empty cells. Thresholding the page once and then
    subtracting the lattice we already detected is both more honest and faster.
    """
    ink = binarize(img)
    return cv2.bitwise_and(ink, cv2.bitwise_not(line_mask(grid)))


def remove_lines(img: np.ndarray, grid: "Grid") -> np.ndarray:
    """Paint the ruling lines white so OCR sees text on paper, not text in a cage."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    out = gray.copy()
    out[line_mask(grid) > 0] = 255
    # Close the gaps the removal punched through glyph strokes that crossed a line.
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
