"""Build FAISS index from PDFs in data/raw_documents."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from config.settings import (
    RAW_DOCS_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    CHUNKS_METADATA_PATH,
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS,
)
from src.ingestion import load_pdf_pages, clean_financial_text, chunk_pages
from src.ingestion.financial_tables import (
    build_financial_sentence_chunks,
    get_pages_with_financial_tables_excluded,
)
from src.embeddings import get_encoder
from src.vector_store import FAISSStore


def main() -> int:
    pdf_files = sorted(RAW_DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDFs found. Add files to %s", RAW_DOCS_DIR)
        return 1

    all_texts: list[str] = []
    all_metadata: list[dict] = []
    total_pages = 0

    for pdf_path in pdf_files:
        pages = load_pdf_pages(pdf_path)
        if not pages:
            continue
        total_pages += len(pages)
        for p in pages:
            p["text"] = clean_financial_text(p.get("text", ""))
        # Structured financial extraction: one chunk per metric.
        fin_chunks = build_financial_sentence_chunks(pages, document_id=pdf_path.stem)
        # Normal chunking excludes raw table text; we rely on structured chunks.
        pages_for_chunking = get_pages_with_financial_tables_excluded(pages)
        chunks = chunk_pages(
            pages_for_chunking,
            chunk_size_words=CHUNK_SIZE_WORDS,
            overlap_words=CHUNK_OVERLAP_WORDS,
            document_id=pdf_path.stem,
        )
        if fin_chunks:
            logger.info("Added %d structured chunks from %s", len(fin_chunks), pdf_path.name)
            chunks.extend(fin_chunks)
        for c in chunks:
            meta = {
                "text": c["text"],
                "page_number": c["page_number"],
                "chunk_id": c["chunk_id"],
                "section_hint": c.get("section_hint"),
                "source_path": str(pdf_path.resolve()),
            }
            if c.get("is_structured_metric"):
                meta["metric_name"] = c["metric_name"]
                meta["year"] = c["year"]
                meta["report_type"] = c["report_type"]
                meta["is_structured_metric"] = True
            all_texts.append(c["text"])
            all_metadata.append(meta)

    if not all_texts:
        logger.error("No chunks produced from any PDF.")
        return 1

    num_chunks = len(all_texts)
    logger.info("Loaded %d pages across %d documents. Prepared %d chunks.", total_pages, len(pdf_files), num_chunks)
    logger.info("Encoding %d chunks with %s", num_chunks, EMBEDDING_MODEL)
    encoder = get_encoder(EMBEDDING_MODEL)
    embeddings = encoder.embed_chunks(all_texts, show_progress=True)
    dim = encoder.dimension
    store = FAISSStore(dimension=dim)
    store.add(embeddings, metadata=all_metadata)
    store.save(FAISS_INDEX_PATH, CHUNKS_METADATA_PATH)
    logger.info("Saved index to %s (n_vectors=%d)", FAISS_INDEX_PATH, store.n_vectors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
