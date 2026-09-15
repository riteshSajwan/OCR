"""TrOCR engine - handwriting recognition.

TrOCR is recognition-only: handed an already-cropped line image, it transcribes
it, but it has no text-detection stage of its own. Page-level header discovery
(`read_page`, used for template matching) is delegated to a TesseractEngine
instance rather than reimplemented - that job is finding text on a whole page,
which is a detection problem TrOCR isn't built for, not a handwriting problem.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ..cells import prepare_for_ocr
from .base import Block, Reading
from .tesseract_engine import TesseractEngine

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"


class TrOCREngine:
    name = "trocr"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Reused purely for read_page - see module docstring.
        self._page_engine = TesseractEngine()

    def read_cell(self, crop: np.ndarray, *, numeric: bool = False) -> Reading:
        if crop.size == 0:
            return Reading("", 0.0, self.name)
        prepared = prepare_for_ocr(crop)
        image = Image.fromarray(prepared).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        return Reading(text, self._sequence_confidence(out), self.name)

    def _sequence_confidence(self, out) -> float:
        """Mean per-token probability of the generated sequence, as a 0-1 score.

        TrOCR has no separate confidence output; the token probabilities the
        decoder already computed while generating are the only signal available,
        so this reuses them rather than adding a second inference pass.
        """
        if not out.scores:
            return 0.0
        generated = out.sequences[0][1:]  # drop the decoder start token
        probs = [
            torch.softmax(step_logits[0], dim=-1)[token_id].item()
            for step_logits, token_id in zip(out.scores, generated)
        ]
        return float(np.mean(probs)) if probs else 0.0

    def read_page(self, img: np.ndarray) -> list[Block]:
        return self._page_engine.read_page(img)
