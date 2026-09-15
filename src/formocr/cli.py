"""Command line entry point.

    python -m formocr run "Minakshi Polymers"            # a folder or a single PDF
    python -m formocr run "Minakshi Polymers" --debug    # also write grid overlays
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import OUT_DIR
from .engines.tesseract_engine import TesseractEngine
from .pipeline import process_pdf
from .template import load_all

# TrOCR pulls in torch/transformers and a ~1.3GB model download, so it is
# constructed lazily (see _make_engine) rather than imported here - every other
# engine would otherwise pay that import cost on every invocation.
ENGINES = {
    "tesseract": TesseractEngine,
    "trocr": None,
    # "paddle": PaddleEngine,   # next
}


def _make_engine(name: str):
    if name == "trocr":
        from .engines.trocr_engine import TrOCREngine
        return TrOCREngine()
    return ENGINES[name]()


def _targets(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.pdf"))


def run(args: argparse.Namespace) -> int:
    source = Path(args.path)
    if not source.exists():
        print(f"error: {source} does not exist", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug" if args.debug else None

    engine = _make_engine(args.engine)
    templates = load_all()
    print(f"engine: {engine.name}   templates: {len(templates)}")

    pdfs = _targets(source)
    if not pdfs:
        print(f"error: no PDFs found under {source}", file=sys.stderr)
        return 1

    started = time.time()
    for pdf in pdfs:
        print(f"\n-- {pdf.name}")
        try:
            result = process_pdf(pdf, engine, templates, debug_dir=debug_dir)
        except Exception as exc:  # keep going; one bad scan shouldn't stop a batch
            print(f"   FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        target = out_dir / f"{pdf.stem}.json"
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        for page in result["pages"]:
            if page["source"] == "text":
                print(f"   page {page['page']}: text layer, {len(page['text'])} chars")
                continue
            meta = page["_meta"]
            persp = "persp+ " if meta["perspective_corrected"] else ""
            print(
                f"   page {page['page']}: {persp}{page['form_id']}  "
                f"{len(page['rows'])} rows  "
                f"grid {meta['grid_rows']}x{meta['grid_cols']}  "
                f"rot {meta['orientation_applied']}  "
                f"empty {meta['cells_empty']}/{meta['cells_total']}  "
                f"conf {meta['mean_ocr_confidence']}  "
                f"review {len(page['review_queue'])}  "
                f"{meta['seconds']}s"
            )
        print(f"   -> {target}")

    print(f"\ndone in {time.time() - started:.1f}s -> {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="formocr")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="extract structured JSON from PDFs")
    run_parser.add_argument("path", help="a PDF file or a directory of PDFs")
    run_parser.add_argument("--out", default=str(OUT_DIR), help="output directory")
    run_parser.add_argument("--engine", default="tesseract", choices=sorted(ENGINES))
    run_parser.add_argument("--debug", action="store_true", help="write grid overlays")
    run_parser.set_defaults(func=run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
