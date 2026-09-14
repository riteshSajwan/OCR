"""Engine interface.

Deliberately narrow so PaddleOCR PP-OCRv5 and TrOCR can drop in beside Tesseract
without touching the pipeline. Cell-level recognition is the primitive; page-level
reading exists only for header discovery and orientation fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Reading:
    text: str
    confidence: float  # 0.0 - 1.0
    engine: str

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class Block:
    """A page-level text block with its bounding box."""
    text: str
    confidence: float
    x0: int
    y0: int
    x1: int
    y1: int


@runtime_checkable
class Engine(Protocol):
    name: str

    def read_cell(self, crop: np.ndarray, *, numeric: bool = False) -> Reading:
        """Recognize the contents of a single table cell."""
        ...

    def read_page(self, img: np.ndarray) -> list[Block]:
        """Recognize text blocks across a whole page."""
        ...
