"""PDF text extraction via PyMuPDF."""

from pathlib import Path
from typing import TypedDict

import fitz


class PageContent(TypedDict):
    """Page result: 1-based page number and extracted text."""
    page_number: int
    text: str


def _extract_page_text(page: fitz.Page) -> str:
    """Extract text from a page, preserving blocks and table structure."""
    blocks = page.get_text("blocks")
    pieces: list[str] = []
    for _x0, _y0, _x1, _y1, text, *_rest in blocks:
        if not text:
            continue
        block = text.rstrip()
        if not block:
            continue
        # Heuristic: if the block looks like a table row (lots of digits or
        # delimiters), keep its internal newlines as-is.
        num_digits = sum(ch.isdigit() for ch in block)
        is_table_like = num_digits >= 10 or ("₹" in block or "," in block and any(
            token.strip().isdigit() for token in block.replace(",", " ").split()
        ))
        if is_table_like:
            pieces.append(block)
        else:
            normalized = "\n".join(
                line.rstrip() for line in block.splitlines() if line.strip()
            )
            if normalized:
                pieces.append(normalized)
    return "\n\n".join(pieces).strip()


def load_pdf_pages(
    path: Path | str,
    *,
    debug: bool = False,
    debug_dir: Path | None = None,
) -> list[PageContent]:
    """Extract text from PDF. One PageContent per non-empty page. Optional debug dump."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {path}")

    result: list[PageContent] = []
    doc = fitz.open(path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = _extract_page_text(page)
            if not text:
                continue
            result.append(
                PageContent(
                    page_number=page_num + 1,
                    text=text,
                )
            )
            if debug and debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                out_path = debug_dir / f"{path.stem}_page_{page_num + 1}.txt"
                out_path.write_text(text, encoding="utf-8")
    finally:
        doc.close()

    return result
