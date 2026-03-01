"""Verify net loss FY 2024 retrieval returns standalone and consolidated values."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import FAISS_INDEX_PATH, CHUNKS_METADATA_PATH, EMBEDDING_MODEL, EMBEDDING_DIM
from src.embeddings import get_encoder
from src.vector_store import FAISSStore
from src.retriever import Retriever


def main() -> int:
    query = "What was the net loss for FY 2024?"
    expected_standalone = "(18,880)"
    expected_consolidated = "(23,502)"

    print(f"Query: {query}")
    print("Expected: Standalone (18,880) million, Consolidated (23,502) million")
    print("-" * 60)

    encoder = get_encoder(EMBEDDING_MODEL)
    store = FAISSStore(dimension=EMBEDDING_DIM)
    store.load(FAISS_INDEX_PATH, CHUNKS_METADATA_PATH)
    retriever = Retriever(encoder, store)

    results = retriever.retrieve(query)
    if not results:
        print("FAIL: No chunks returned.")
        return 1

    texts = [r.get("metadata", {}).get("text", "") for r in results]
    meta_info = [
        (r.get("metadata", {}).get("report_type"), r.get("metadata", {}).get("metric_name"))
        for r in results
    ]

    has_standalone = any(expected_standalone in t for t in texts)
    has_consolidated = any(expected_consolidated in t for t in texts)
    wrong_numbers = []
    for t in texts:
        if "million" in t or ")" in t:
            nums = re.findall(r"\(\d[\d,]*\)|\d[\d,]*\.?\d*", t)
            for n in nums:
                if n not in (expected_standalone, expected_consolidated, "2024", "2023"):
                    if "million" in t or "profit" in t.lower() or "loss" in t.lower():
                        wrong_numbers.append((t[:80], n))

    print(f"Returned {len(results)} chunks:")
    for i, (text, (rt, mn)) in enumerate(zip(texts, meta_info)):
        print(f"  [{i+1}] {rt} | {mn}")
        print(f"      {text[:100]}...")

    ok = has_standalone and has_consolidated and len(results) == 2
    if ok:
        print("-" * 60)
        print("PASS: Context contains exactly Standalone (18,880) and Consolidated (23,502).")
    else:
        print("-" * 60)
        if not has_standalone:
            print("FAIL: Missing Standalone (18,880)")
        if not has_consolidated:
            print("FAIL: Missing Consolidated (23,502)")
        if len(results) != 2:
            print(f"FAIL: Expected 2 chunks, got {len(results)}")
        if wrong_numbers:
            print("FAIL: Unexpected numbers in context:", wrong_numbers)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
