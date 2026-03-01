"""PDF loading via PyMuPDF."""

from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

import fitz  # PyMuPDF


@dataclass
class Document:
    """Single document with source metadata."""

    content: str
    source_path: str
    page_start: int
    page_end: int


def load_documents(
    path: Path | str,
    *,
    supported_extensions: set[str] | None = None,
) -> Iterator[Document]:
    """
    Load documents from a file or directory.
    Yields Document objects (PDF pages aggregated by file).
    """
    supported = supported_extensions or {".pdf"}
    path = Path(path)

    if path.is_file():
        if path.suffix.lower() not in supported:
            return
        yield from _load_pdf(path)
    elif path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in supported:
                yield from _load_pdf(p)
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def _load_pdf(file_path: Path) -> Iterator[Document]:
    """Extract text from a PDF file; one Document per file (full text)."""
    doc = fitz.open(file_path)
    try:
        chunks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                chunks.append((page_num + 1, text))
        if chunks:
            full_text = "\n\n".join(t for _, t in chunks)
            page_start = chunks[0][0]
            page_end = chunks[-1][0]
            yield Document(
                content=full_text,
                source_path=str(file_path.resolve()),
                page_start=page_start,
                page_end=page_end,
            )
    finally:
        doc.close()
