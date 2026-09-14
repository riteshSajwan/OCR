"""Per-form layout templates.

These forms are pre-printed and reused every month, so a handful of small YAML
declarations beats general-purpose table understanding: the column keys, types and
arithmetic identities are known facts, not things to be inferred per scan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import TEMPLATES_DIR


@dataclass
class Column:
    key: str
    type: str = "text"
    printed: bool = False

    @property
    def numeric(self) -> bool:
        return self.type == "integer"


@dataclass
class Check:
    target: str
    sum_of: list[str]


@dataclass
class Template:
    form_id: str
    match: list[str]
    columns: list[Column]
    header_row: int = 0
    first_data_row: int = 1
    trailing_rows_to_drop: int = 0
    dash_means: Any = None
    ditto_columns: list[str] = field(default_factory=list)
    ditto_markers: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    description: str = ""

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def column(self, index: int) -> Column | None:
        return self.columns[index] if 0 <= index < len(self.columns) else None


def _parse(raw: dict) -> Template:
    match = raw.get("match", [])
    if isinstance(match, str):
        match = [match]
    return Template(
        form_id=raw["form_id"],
        match=match,
        columns=[Column(**c) for c in raw.get("columns", [])],
        header_row=raw.get("header_row", 0),
        first_data_row=raw.get("first_data_row", 1),
        trailing_rows_to_drop=raw.get("trailing_rows_to_drop", 0),
        dash_means=raw.get("dash_means"),
        ditto_columns=raw.get("ditto_columns", []) or [],
        ditto_markers=raw.get("ditto_markers", []) or [],
        checks=[Check(**c) for c in raw.get("checks", []) or []],
        description=raw.get("description", ""),
    )


def load_all(directory: Path | None = None) -> list[Template]:
    directory = directory or TEMPLATES_DIR
    if not directory.is_dir():
        return []
    templates = []
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw:
            templates.append(_parse(raw))
    return templates


def match_template(page_text: str, templates: list[Template]) -> Template | None:
    """Pick the template whose printed keywords appear in the page's OCR text."""
    haystack = " ".join(page_text.split()).lower()
    best, best_hits = None, 0
    for tpl in templates:
        hits = sum(1 for m in tpl.match if " ".join(m.split()).lower() in haystack)
        if hits > best_hits:
            best, best_hits = tpl, hits
    return best


def generic_template(n_columns: int) -> Template:
    """Fallback when no template matches: positional keys, nothing assumed."""
    return Template(
        form_id="generic",
        match=[],
        columns=[Column(key=f"col_{i}") for i in range(n_columns)],
        header_row=0,
        first_data_row=1,
        description="Auto-generated positional layout; no form template matched.",
    )
