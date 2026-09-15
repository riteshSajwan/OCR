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
# Vertical lines get a lower bar than horizontal ones: a photographed page can
# light one side of the table more than the other, weakening some column rules
# relative to the page's single strongest line without weakening them in any
# absolute sense. Swept against every sample: 0.20 recovers the columns that a
# global 0.30 threshold drops on unevenly-lit photos, with zero change to any
# flatbed scan's detected column count; going lower starts manufacturing a
# false column on the densest sample (ST. Pass weld data).
VERTICAL_LINE_PEAK_FRAC = 0.20
# Boundaries closer than this are the two edges of one thick line, not two lines.
MERGE_WITHIN_PX = 12
# Discard slivers that cannot hold a value.
MIN_CELL_W = 18
MIN_CELL_H = 12

# The table's line mask must cover at least this fraction of the page to be
# trusted as the table outline rather than stray marks.
MIN_TABLE_AREA_FRAC = 0.20
# Opposite sides within this fraction of each other's length are treated as a
# simple rotated rectangle (a flatbed scan) rather than true camera perspective.
PERSPECTIVE_SIDE_TOLERANCE = 0.035


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


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    total = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(total)],  # top-left: smallest x+y
        pts[np.argmin(diff)],   # top-right: smallest y-x
        pts[np.argmax(total)],  # bottom-right: largest x+y
        pts[np.argmax(diff)],   # bottom-left: largest y-x
    ], dtype=np.float32)


def find_table_corners(img: np.ndarray) -> np.ndarray | None:
    """Locate the table's four outer corners from its own ruling lines.

    Using the printed lattice rather than the page/background edge works
    regardless of scan vs. photo, cropping, or background clutter - it is the
    same line mask `detect()` already relies on. Returns None when no confident
    quadrilateral is found, so callers can leave the page untouched.
    """
    bw = binarize(img)
    horizontal, vertical = line_masks(bw)
    lines = cv2.bitwise_or(horizontal, vertical)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=2)

    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_TABLE_AREA_FRAC * img.shape[0] * img.shape[1]:
        return None

    hull = cv2.convexHull(largest)
    peri = cv2.arcLength(hull, True)
    for eps_frac in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps_frac * peri, True)
        if len(approx) == 4:
            return _order_corners(approx.reshape(4, 2).astype(np.float32))
    return None


def rectify(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Warp the table to a top-down rectangle if it is genuinely keystoned.

    A flatbed scan's table is already a rectangle, just possibly rotated -
    `orient.normalize` handles that. A photographed page can additionally be
    keystoned (the far edge shorter than the near edge), which a single
    rotation cannot fix and which defeats grid detection: `_boundaries` finds
    ruling lines by their column/row projection peak, and a slanted line
    spreads its pixels across a range of columns instead of one, so the peak
    never clears LINE_PEAK_FRAC. Runs before orientation/skew correction since
    corner-finding is rotation-invariant but a slanted grid is not.
    """
    corners = find_table_corners(img)
    if corners is None:
        return img, False
    tl, tr, br, bl = corners

    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    if min(top, bottom) < 10 or min(left, right) < 10:
        return img, False

    # A simple rotation keeps opposite sides equal length (still a rectangle).
    # Only warp when they genuinely differ - real perspective, not just skew.
    horiz_ratio = abs(top - bottom) / max(top, bottom)
    vert_ratio = abs(left - right) / max(left, right)
    if horiz_ratio < PERSPECTIVE_SIDE_TOLERANCE and vert_ratio < PERSPECTIVE_SIDE_TOLERANCE:
        return img, False

    width = int(round(max(top, bottom)))
    height = int(round(max(left, right)))
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(
        img, matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, True


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


def detect(img: np.ndarray, scale: int = 40, *, relaxed_columns: bool = False) -> Grid:
    """Detect the table lattice and enumerate its cells.

    `relaxed_columns` lowers the vertical-line threshold to VERTICAL_LINE_PEAK_FRAC.
    Only pass this for photographed pages: uneven lighting there can weaken some
    column rules relative to the page's single strongest line without weakening
    them in any absolute sense. On flatbed scans, lighting is uniform enough that
    the standard threshold already finds every column - relaxing it there instead
    risks dense numeric columns coincidentally aligning into a fake line (seen on
    the two busiest sample grids during testing).
    """
    bw = binarize(img)
    horizontal, vertical = line_masks(bw, scale=scale)

    v_peak_frac = VERTICAL_LINE_PEAK_FRAC if relaxed_columns else LINE_PEAK_FRAC
    ys = _drop_tight(_boundaries(horizontal, axis=1), MIN_CELL_H)
    xs = _drop_tight(_boundaries(vertical, axis=0, peak_frac=v_peak_frac), MIN_CELL_W)

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
