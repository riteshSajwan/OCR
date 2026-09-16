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
from transformers import LogitsProcessor, LogitsProcessorList, TrOCRProcessor, VisionEncoderDecoderModel

from ..cells import prepare_for_ocr
from .base import Block, Reading
from .tesseract_engine import TesseractEngine

DEFAULT_MODEL = "microsoft/trocr-base-handwritten"

# TrOCR's processor resizes every input to 384x384 itself, so upscaling the crop
# beforehand would only resample it twice. Padding still helps; see prepare_for_ocr.
OCR_SCALE = 1.0

# Byte-level BPE marks a leading space with this character.
_SPACE_MARKER = "Ġ"


class _AllowedTokensProcessor(LogitsProcessor):
    """Mask the decoder down to a permitted subset of the vocabulary.

    Tesseract gets charset constraints for free via tessedit_char_whitelist.
    TrOCR is a seq2seq language model with no such switch, so the equivalent has
    to happen at the logits: on a numeric column it will otherwise happily emit
    letters, which _coerce then strips - turning a recoverable read into a
    halved-confidence guess.
    """

    def __init__(self, allowed: torch.Tensor) -> None:
        self.allowed = allowed

    def __call__(self, input_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        mask = self.allowed.to(scores.device)
        return scores.masked_fill(~mask, torch.finfo(scores.dtype).min)


class TrOCREngine:
    name = "trocr"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        # Reused purely for read_page - see module docstring.
        self._page_engine = TesseractEngine()
        self._numeric_processors = LogitsProcessorList(
            [_AllowedTokensProcessor(self._numeric_token_mask())]
        )

    def _numeric_token_mask(self) -> torch.Tensor:
        """Boolean mask over the vocabulary: digits, whitespace, control tokens.

        Built once at load time. Anything that decodes to a non-digit character is
        blocked, so the decoder cannot spell a letter into a numeric cell at all,
        rather than being cleaned up after the fact.
        """
        tokenizer = self.processor.tokenizer
        vocab_size = int(self.model.config.decoder.vocab_size)
        allowed = torch.zeros(vocab_size, dtype=torch.bool)

        tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
        for token_id, token in enumerate(tokens):
            if token is None:
                continue
            body = token.replace(_SPACE_MARKER, "")
            # An all-space token is fine; so is any run of digits.
            if body == "" or body.isdigit():
                allowed[token_id] = True

        # Control tokens must survive or generation can never terminate.
        for special in (
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            self.model.config.decoder_start_token_id,
        ):
            if special is not None and 0 <= special < vocab_size:
                allowed[special] = True
        return allowed

    def read_cell(self, crop: np.ndarray, *, numeric: bool = False) -> Reading:
        if crop.size == 0:
            return Reading("", 0.0, self.name)
        prepared = prepare_for_ocr(crop, scale=OCR_SCALE)
        image = Image.fromarray(prepared).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                pixel_values,
                logits_processor=self._numeric_processors if numeric else None,
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

        Note this reads the *processed* scores, so on numeric columns the
        probabilities are renormalised over the allowed tokens only - a
        constrained read scores its confidence among digits, not against letters
        it was never permitted to emit.
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
