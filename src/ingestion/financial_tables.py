"""Structured financial table extraction. One chunk per metric."""

from __future__ import annotations

import re
from src.ingestion.pdf_loader import PageContent
from src.ingestion.chunking import TextChunk


_YEAR_HEADER_RE = re.compile(
    r"For the Year ended March 31,\s*(\d{4}).*?For the Year ended March 31,\s*(\d{4})",
    re.IGNORECASE | re.DOTALL,
)
_YEAR_FALLBACK_RE = re.compile(r"March\s+31,?\s*(\d{4})", re.IGNORECASE)

_NUMBER_RE = re.compile(r"[\(\-]?\d[\d,]*\.?\d*\)?")

_METRIC_SPECS = [
    (r"Net Sales\s*/?\s*Income from Business Operations", "Net Sales/Income from Business Operations", "amount"),
    (r"Total Income", "Total Income", "amount"),
    (r"Less:?\s*Total expenses including Depreciation", "Total expenses including Depreciation", "plural_amount"),
    (r"Total expenses including Depreciation", "Total expenses including Depreciation", "plural_amount"),
    (r"Less:?\s*Exceptional Items \+ Taxes", "Less: Exceptional Items + Taxes", "amount"),
    (r"Net Profit/\(Loss\) after Tax", "Net Profit/(Loss) after Tax", "amount"),
    (r"Profit/\(Loss\) after Tax", "Net Profit/(Loss) after Tax", "amount"),
    (r"Other comprehensive income", "Other comprehensive income", "amount"),
    (r"Total comprehensive loss for the year, net of tax", "Total comprehensive loss for the year, net of tax", "amount"),
    (r"Earnings per share\s*\(.*?Basic and Diluted\)\*?", "Earnings per share (Basic and Diluted)", "eps"),
]


def _extract_years(segment: str) -> tuple[str, str] | None:
    """Return (year_current, year_prev) or None."""
    m = _YEAR_HEADER_RE.search(segment)
    if m:
        return m.group(1), m.group(2)
    years = _YEAR_FALLBACK_RE.findall(segment)
    if len(years) >= 2:
        return years[0], years[1]
    return None


def _extract_two_numbers(text: str) -> tuple[str, str] | None:
    """Extract first two numeric values from text (FY2024, FY2023)."""
    nums = _NUMBER_RE.findall(text)
    if len(nums) < 2:
        return None
    return nums[0], nums[1]


def _parse_rows(segment: str) -> list[tuple[str, str, str, str]]:
    """Parse segment into (display_name, metric_type, val_cur, val_prev) rows."""
    years = _extract_years(segment)
    if not years:
        return []

    rows: list[tuple[str, str, str, str]] = []
    lines = [l.strip() for l in segment.splitlines() if l.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]
        for pattern, display_name, metric_type in _METRIC_SPECS:
            m = re.search(pattern, line, re.IGNORECASE)
            if not m:
                continue
            nums = _extract_two_numbers(line)
            if not nums and i + 2 < len(lines):
                combined = f"{lines[i + 1]} {lines[i + 2]}"
                nums = _extract_two_numbers(combined)
            if not nums and i + 1 < len(lines):
                nums = _extract_two_numbers(lines[i + 1])
            if not nums:
                break
            val_cur, val_prev = nums
            rows.append((display_name, metric_type, val_cur, val_prev))
            break
        i += 1
    return rows


def _build_sentence(
    metric_name: str,
    metric_type: str,
    value: str,
    year: str,
    context_label: str,
) -> str:
    """Build one semantic sentence for a metric and value."""
    verb = "were" if metric_type == "plural_amount" else "was"
    base = f"For the year ended March 31, {year}, the {context_label} {metric_name} {verb} {value}"
    if metric_type == "amount" or metric_type == "plural_amount":
        base += " million."
    else:
        base += "."
    return base


def _build_chunks_from_segment(
    segment: str,
    *,
    page_number: int,
    document_id: str,
    context_label: str,
    base_index: int,
) -> list[dict]:
    """Parse segment into one chunk per (metric, year) with metadata."""
    years = _extract_years(segment)
    if not years:
        return []
    year_current, year_prev = years

    rows = _parse_rows(segment)
    if not rows:
        return []

    chunks: list[dict] = []
    idx = 0
    for display_name, metric_type, val_cur, val_prev in rows:
        for year, val in [(year_current, val_cur), (year_prev, val_prev)]:
            sent = _build_sentence(
                display_name, metric_type, val, year, context_label
            )
            chunk_id = f"{document_id}_fin_{page_number}_{base_index}_{idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": sent,
                "page_number": page_number,
                "section_hint": f"{context_label.capitalize()}: {display_name}",
                "metric_name": display_name,
                "year": int(year),
                "report_type": context_label,
                "is_structured_metric": True,
            })
            idx += 1
    return chunks


def _get_financial_segment_ranges(text: str) -> list[tuple[int, int]]:
    """Return (start, end) ranges of financial table segments in page text."""
    lower = text.lower()
    ranges: list[tuple[int, int]] = []
    standalone_marker = "the standalone performance as per standalone financial statements is as under"
    consolidated_marker = "the consolidated performance as per consolidated financial statements is as under"

    if standalone_marker in lower:
        start = lower.index(standalone_marker)
        end = lower.index(consolidated_marker) if consolidated_marker in lower else len(text)
        ranges.append((start, end))

    if consolidated_marker in lower:
        start = lower.index(consolidated_marker)
        ranges.append((start, len(text)))

    return ranges


def get_pages_with_financial_tables_excluded(pages: list[PageContent]) -> list[PageContent]:
    """Return pages with financial table segments removed from text."""
    result: list[PageContent] = []
    for p in pages:
        text = p.get("text") or ""
        ranges = _get_financial_segment_ranges(text)
        if not ranges:
            result.append(dict(p))
            continue
        for start, end in sorted(ranges, key=lambda r: -r[0]):
            text = text[:start] + text[end:]
        result.append({**p, "text": text})
    return result


def _page_has_section_marker(text: str) -> bool:
    """Return True if page has standalone or consolidated section markers in text."""
    lower = (text or "").lower()
    standalone_marker = "the standalone performance as per standalone financial statements is as under"
    consolidated_marker = "the consolidated performance as per consolidated financial statements is as under"
    return standalone_marker in lower or consolidated_marker in lower


def build_financial_sentence_chunks(
    pages: list[PageContent],
    *,
    document_id: str,
) -> list[TextChunk]:
    """Detect financial tables, parse rows, return one TextChunk per metric."""
    extra: list[TextChunk] = []
    standalone_marker = "the standalone performance as per standalone financial statements is as under"
    consolidated_marker = "the consolidated performance as per consolidated financial statements is as under"

    for i, p in enumerate(pages):
        page_num = p["page_number"]
        text = (p.get("text") or "").strip()
        if not text:
            continue

        lower = text.lower()

        if standalone_marker in lower:
            seg = text[lower.index(standalone_marker) :]
            if consolidated_marker in lower:
                cut = lower.index(consolidated_marker)
                seg = text[lower.index(standalone_marker) : cut]
            extra.extend(
                _build_chunks_from_segment(
                    seg,
                    page_number=page_num,
                    document_id=document_id,
                    context_label="standalone",
                    base_index=len(extra),
                )
            )

        if consolidated_marker in lower:
            seg = text[lower.index(consolidated_marker) :]
            if i + 1 < len(pages):
                next_text = (pages[i + 1].get("text") or "").strip()
                if next_text and not _page_has_section_marker(next_text):
                    seg = seg + "\n" + next_text
            extra.extend(
                _build_chunks_from_segment(
                    seg,
                    page_number=page_num,
                    document_id=document_id,
                    context_label="consolidated",
                    base_index=len(extra),
                )
            )

    return extra
