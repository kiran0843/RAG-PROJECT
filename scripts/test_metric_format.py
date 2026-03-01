"""Test metric dual-response formatting."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    CHUNKS_METADATA_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from src.embeddings import get_encoder
from src.llm.ollama_client import OllamaClient
from src.rag.pipeline import RAGPipeline
from src.vector_store.faiss_store import FAISSStore


def main() -> int:
    query = "What was the net loss for FY 2024?"
    print(f"Query: {query}")
    print("-" * 60)

    encoder = get_encoder(EMBEDDING_MODEL)
    store = FAISSStore(dimension=EMBEDDING_DIM)
    store.load(FAISS_INDEX_PATH, CHUNKS_METADATA_PATH)
    llm = OllamaClient(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    rag = RAGPipeline(encoder=encoder, store=store, llm=llm, top_k=5)

    result = rag.run(query)
    answer = result.get("answer", "")

    print("Answer:")
    print(answer)
    print("-" * 60)

    expected_standalone = "(18,880)"
    expected_consolidated = "(23,502)"
    ok = expected_standalone in answer and expected_consolidated in answer
    ok = ok and "Standalone" in answer and "Consolidated" in answer
    ok = ok and "ambiguity" not in answer.lower() and "warning" not in answer.lower()

    if ok:
        print("PASS: Both values present, structured format, no warning language.")
    else:
        print("FAIL: Check format requirements.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
