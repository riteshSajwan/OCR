"""Project paths and Tesseract discovery."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytesseract

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = ROOT / "templates"
OUT_DIR = ROOT / "out"

_TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
]


def find_tesseract() -> str:
    """Locate the tesseract binary; pytesseract is only a wrapper around it."""
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    for candidate in _TESSERACT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError(
        "tesseract binary not found. Install it with:\n"
        "  winget install -e --id UB-Mannheim.TesseractOCR"
    )


def configure_tesseract() -> str:
    path = find_tesseract()
    pytesseract.pytesseract.tesseract_cmd = path
    return path
