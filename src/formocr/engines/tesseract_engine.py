"""Tesseract engine - the baseline.

Honest about its role: Tesseract is strong on the printed parts of these forms
(headers, row labels, doc numbers) and weak on the handwritten cells. It exists
here to establish a measurable floor and to prove the pipeline end to end before
PaddleOCR and TrOCR arrive. Do not expect usable handwriting accuracy from it.
"""
from __future__ import annotations

import numpy as np
import pytesseract
from pytesseract import Output

from ..cells import prepare_for_ocr
from ..config import configure_tesseract
from .base import Block, Reading

# PSM 6 = assume a single uniform block of text. PSM 7 (a single text line) is the
# more obvious fit for a table cell, but cells here are not reliably one line - a
# remarks cell wraps ("Penetration / cut part"), and PSM 7 forces such a cell onto
# one baseline and garbles it.
CELL_PSM = 6
DIGITS = "0123456789"

# Cell crops are small; Tesseract wants roughly a 30px x-height plus a quiet
# margin, so the crop is upscaled before recognition.
OCR_SCALE = 3


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = CELL_PSM) -> None:
        self.binary = configure_tesseract()
        self.lang = lang
        self.psm = psm

    def _config(self, *, numeric: bool, psm: int | None = None) -> str:
        parts = [f"--psm {psm or self.psm}", "--oem 3"]
        if numeric:
            # Constraining the charset is the cheapest accuracy win available on
            # numeric columns; it cannot turn a 4 into an A.
            parts.append(f"-c tessedit_char_whitelist={DIGITS}")
        return " ".join(parts)

    def read_cell(self, crop: np.ndarray, *, numeric: bool = False) -> Reading:
        if crop.size == 0:
            return Reading("", 0.0, self.name)
        prepared = prepare_for_ocr(crop, scale=OCR_SCALE)
        try:
            data = pytesseract.image_to_data(
                prepared, lang=self.lang,
                config=self._config(numeric=numeric),
                output_type=Output.DICT,
            )
        except Exception:
            return Reading("", 0.0, self.name)

        words, confs = [], []
        for text, conf in zip(data["text"], data["conf"]):
            text = str(text).strip()
            conf = float(conf)
            if text and conf >= 0:
                words.append(text)
                confs.append(conf)
        if not words:
            return Reading("", 0.0, self.name)
        return Reading(" ".join(words), float(np.mean(confs)) / 100.0, self.name)

    def read_page(self, img: np.ndarray) -> list[Block]:
        try:
            data = pytesseract.image_to_data(
                img, lang=self.lang,
                config=self._config(numeric=False, psm=6),
                output_type=Output.DICT,
            )
        except Exception:
            return []

        blocks: list[Block] = []
        for i, text in enumerate(data["text"]):
            text = str(text).strip()
            conf = float(data["conf"][i])
            if not text or conf < 0:
                continue
            x, y = data["left"][i], data["top"][i]
            blocks.append(Block(
                text=text, confidence=conf / 100.0,
                x0=x, y0=y, x1=x + data["width"][i], y1=y + data["height"][i],
            ))
        return blocks
