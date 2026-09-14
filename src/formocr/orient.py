"""Page orientation (0/90/180/270) and fine deskew.

The SCRAP NOTE scan arrives rotated 180 degrees, so this step is mandatory, not
cosmetic - every downstream stage assumes an upright page.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Below this many degrees the warp costs more in interpolation blur than it buys.
MIN_SKEW_DEG = 0.15
MAX_SKEW_DEG = 5.0


@dataclass
class Orientation:
    rotation: int = 0  # degrees applied, one of 0/90/180/270
    skew: float = 0.0  # degrees applied after rotation
    method: str = "none"  # "osd" | "confidence" | "none"


def _rotate_quarter(img: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _mean_confidence(img: np.ndarray) -> float:
    """Mean word confidence over a page - our orientation tie-breaker."""
    try:
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
    except Exception:
        return 0.0
    confs = [
        float(c)
        for c, t in zip(data["conf"], data["text"])
        if str(t).strip() and float(c) >= 0
    ]
    return float(np.mean(confs)) if confs else 0.0


def detect_rotation(img: np.ndarray) -> tuple[int, str]:
    """Return the rotation needed to make the page upright.

    Tesseract's OSD is the right tool here, but it needs a reasonable amount of
    printed text and throws when it cannot decide. The fallback OCRs the page at
    0 and 180 degrees and keeps whichever reads with higher confidence - which
    works because these forms always carry a printed header.
    """
    try:
        osd = pytesseract.image_to_osd(img, output_type=Output.DICT)
        rotation = int(osd.get("rotate", 0)) % 360
        if rotation in (0, 90, 180, 270):
            return rotation, "osd"
    except Exception:
        pass

    scores = {deg: _mean_confidence(_rotate_quarter(img, deg)) for deg in (0, 180)}
    best = max(scores, key=scores.get)
    return (best, "confidence") if scores[best] > 0 else (0, "none")


def estimate_skew(img: np.ndarray) -> float:
    """Estimate residual skew from the dominant horizontal ruling line angle.

    These forms are ruled tables, so the printed lines are a far stronger signal
    than a projection profile over text rows.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_len = int(gray.shape[1] * 0.25)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=150,
        minLineLength=min_len, maxLineGap=20,
    )
    if lines is None:
        return 0.0

    # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4).
    segments = lines.reshape(-1, 4)

    angles = []
    for x1, y1, x2, y2 in segments:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= MAX_SKEW_DEG:  # near-horizontal only
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < MIN_SKEW_DEG:
        return img
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def normalize(img: np.ndarray) -> tuple[np.ndarray, Orientation]:
    """Bring a scanned page upright and level."""
    rotation, method = detect_rotation(img)
    out = _rotate_quarter(img, rotation)
    angle = estimate_skew(out)
    out = deskew(out, angle)
    applied = angle if abs(angle) >= MIN_SKEW_DEG else 0.0
    return out, Orientation(rotation=rotation, skew=round(applied, 3), method=method)
