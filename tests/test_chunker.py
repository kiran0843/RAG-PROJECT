"""Tests for the chunker module."""

from src.ingestion.loader import Document
from src.ingestion.chunker import chunk_text


def test_chunk_text_produces_chunks():
    doc = Document(
        content="First sentence. Second sentence. Third sentence. " * 50,
        source_path="/fake/doc.pdf",
        page_start=1,
        page_end=1,
    )
    chunks = list(chunk_text(doc, chunk_size=100, chunk_overlap=20))
    assert len(chunks) >= 1
    assert all(c.text for c in chunks)
    assert all(c.source_path == doc.source_path for c in chunks)
