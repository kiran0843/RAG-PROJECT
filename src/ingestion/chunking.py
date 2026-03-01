"""Word-based chunking with overlap and page tracking."""

from typing import TypedDict

from src.ingestion.pdf_loader import PageContent


class TextChunk(TypedDict):
    """Single chunk with stable id, text, and source metadata."""

    chunk_id: str
    text: str
    page_number: int
    section_hint: str | None


DEFAULT_CHUNK_SIZE_WORDS = 500
DEFAULT_OVERLAP_WORDS = 100


def chunk_pages(
    pages: list[PageContent],
    *,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
    document_id: str = "doc",
) -> list[TextChunk]:
    """Split pages into overlapping word-based chunks. Table-like blocks are atomic."""
    if overlap_words >= chunk_size_words:
        overlap_words = max(0, chunk_size_words // 5)

    if not pages:
        return []

    chunks: list[TextChunk] = []
    chunk_index = 0

    def flush_chunk(words_with_page: list[tuple[str, int]], hint: str | None) -> None:
        nonlocal chunk_index
        if not words_with_page:
            return
        text = " ".join(w for w, _ in words_with_page).strip()
        if not text:
            return
        page_number = words_with_page[0][1]
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                text=text,
                page_number=page_number,
                section_hint=hint,
            )
        )
        chunk_index += 1

    current: list[tuple[str, int]] = []
    current_hint: str | None = None

    for p in pages:
        page_num = p["page_number"]
        raw_text = (p.get("text") or "").strip()
        if not raw_text:
            continue

        blocks = [b.strip() for b in raw_text.split("\n\n") if b.strip()]
        for block in blocks:
            num_digits = sum(ch.isdigit() for ch in block)
            is_table_like = num_digits >= 10 or ("₹" in block or "," in block and any(
                token.strip().isdigit() for token in block.replace(",", " ").split()
            ))

            block_words = block.split()
            if not block_words:
                continue

            first_line = block.splitlines()[0].strip()
            if current_hint is None and first_line:
                current_hint = first_line[:120]

            if is_table_like:
                if current:
                    flush_chunk(current, current_hint)
                    if overlap_words > 0:
                        current = current[-overlap_words:]
                    else:
                        current = []
                    current_hint = first_line[:120] if first_line else None

                atomic = [(w, page_num) for w in block_words]
                flush_chunk(atomic, first_line[:120] if first_line else None)
                current = []
                current_hint = None
                continue

            for w in block_words:
                current.append((w, page_num))
                if len(current) >= chunk_size_words + overlap_words:
                    flush_chunk(current[:chunk_size_words], current_hint)
                    if overlap_words > 0:
                        current = current[chunk_size_words - overlap_words :]
                    else:
                        current = []
                    current_hint = None

    if current:
        flush_chunk(current, current_hint)

    return chunks
