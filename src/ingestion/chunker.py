"""Text chunking for RAG."""

import re
from dataclasses import dataclass
from typing import Iterator

from src.ingestion.loader import Document


@dataclass
class Chunk:
    """Text chunk with metadata for retrieval."""

    text: str
    source_path: str
    page_start: int
    page_end: int
    chunk_index: int


def chunk_text(
    document: Document,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Iterator[Chunk]:
    """
    Split document content into overlapping chunks.
    Tries to break on sentence boundaries when possible.
    """
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    text = document.content
    if not text.strip():
        return

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    sentences = _split_sentences(text)
    if not sentences:
        # Fallback: fixed-size character chunks
        for i in range(0, len(text), chunk_size - chunk_overlap):
            segment = text[i : i + chunk_size]
            if segment.strip():
                yield Chunk(
                    text=segment.strip(),
                    source_path=document.source_path,
                    page_start=document.page_start,
                    page_end=document.page_end,
                    chunk_index=i // (chunk_size - chunk_overlap),
                )
        return

    current: list[str] = []
    current_len = 0
    chunk_idx = 0

    for sent in sentences:
        sent_len = len(sent) + 1
        if current_len + sent_len > chunk_size and current:
            chunk_text_str = " ".join(current).strip()
            if chunk_text_str:
                yield Chunk(
                    text=chunk_text_str,
                    source_path=document.source_path,
                    page_start=document.page_start,
                    page_end=document.page_end,
                    chunk_index=chunk_idx,
                )
                chunk_idx += 1
            # Keep overlap
            overlap_len = 0
            overlap_sents: list[str] = []
            for s in reversed(current):
                if overlap_len + len(s) + 1 <= chunk_overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break
            current = overlap_sents
            current_len = overlap_len
        current.append(sent)
        current_len += sent_len

    if current:
        chunk_text_str = " ".join(current).strip()
        if chunk_text_str:
            yield Chunk(
                text=chunk_text_str,
                source_path=document.source_path,
                page_start=document.page_start,
                page_end=document.page_end,
                chunk_index=chunk_idx,
            )


def _split_sentences(text: str) -> list[str]:
    """Simple sentence split on . ! ? followed by space or end."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]
