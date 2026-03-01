"""Conservative text cleaning for financial reports."""

import re


def clean_financial_text(text: str) -> str:
    """Collapse newlines, normalize spaces. Preserves numbers and punctuation."""
    if not text or not isinstance(text, str):
        return ""

    text = re.sub(r"\n[\s\n]*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
