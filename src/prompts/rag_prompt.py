"""RAG prompt templates."""

RAG_SYSTEM_INSTRUCTIONS = """Answer using ONLY the provided context. No external knowledge.

- Answer strictly from the context. If insufficient, say "I cannot find this in the provided documents."
- Cite page numbers (e.g. [Page 5]). Use [Page N] labels from the context.
- Be concise. No filler.
- Do not add information absent from context.
- Off-topic or empty context: respond that you have no relevant context.
- Financial metrics with both standalone and consolidated: return BOTH. Format: "For FY YYYY: - Standalone [metric]: [value]. - Consolidated [metric]: [value]." No ambiguity or warning language."""


CONTEXT_BLOCK_TEMPLATE = """## Context

{context}

## Question

{question}

## Answer"""


def build_rag_user_message(context: str, question: str) -> str:
    """Build user message: context block and question."""
    context = (context or "").strip()
    question = (question or "").strip()
    return CONTEXT_BLOCK_TEMPLATE.format(context=context or "(No context provided.)", question=question)
