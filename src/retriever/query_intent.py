"""Query intent classification for metric retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QueryIntent:
    """Parsed intent from a financial metric query."""

    target_metric: str
    year: int | None
    report_type: Literal["standalone", "consolidated"] | None


METRIC_NET_PROFIT = "Net Profit/(Loss) after Tax"
METRIC_TOTAL_INCOME = "Total Income"
METRIC_TOTAL_EXPENSES = "Total expenses including Depreciation"
METRIC_TOTAL_COMPREHENSIVE = "Total comprehensive loss for the year, net of tax"
METRIC_EARNINGS_PER_SHARE = "Earnings per share (Basic and Diluted)"
METRIC_NET_SALES = "Net Sales/Income from Business Operations"
METRIC_EXCEPTIONAL_ITEMS = "Less: Exceptional Items + Taxes"
METRIC_OTHER_COMPREHENSIVE = "Other comprehensive income"


_METRIC_PATTERNS: list[tuple[list[str], str]] = [
    (["net loss", "net profit", "profit after tax", "loss after tax"], METRIC_NET_PROFIT),
    (["total income"], METRIC_TOTAL_INCOME),
    (["total expenses", "expenses including depreciation"], METRIC_TOTAL_EXPENSES),
    (["total comprehensive", "comprehensive loss"], METRIC_TOTAL_COMPREHENSIVE),
    (["earnings per share", " eps", "eps "], METRIC_EARNINGS_PER_SHARE),
    (["net sales", "income from business operations"], METRIC_NET_SALES),
    (["exceptional items", "exceptional items + taxes"], METRIC_EXCEPTIONAL_ITEMS),
    (["other comprehensive income"], METRIC_OTHER_COMPREHENSIVE),
]

_YEAR_RE = re.compile(
    r"\b(?:fy\s*|financial\s+year\s+)?(20\d{2})\b",
    re.IGNORECASE,
)

_REPORT_TYPE_RE = re.compile(
    r"\b(standalone|consolidated)\b",
    re.IGNORECASE,
)


def classify_query_intent(query: str) -> QueryIntent | None:
    """Return QueryIntent if query targets a known metric, else None."""
    if not query or not query.strip():
        return None

    q_lower = query.lower().strip()

    target_metric: str | None = None
    for phrases, metric in _METRIC_PATTERNS:
        if any(p in q_lower for p in phrases):
            target_metric = metric
            break

    if not target_metric:
        return None

    year_match = _YEAR_RE.search(query)
    year = int(year_match.group(1)) if year_match else None

    report_match = _REPORT_TYPE_RE.search(query)
    report_type: Literal["standalone", "consolidated"] | None = None
    if report_match:
        report_type = report_match.group(1).lower()
        if report_type not in ("standalone", "consolidated"):
            report_type = None

    return QueryIntent(
        target_metric=target_metric,
        year=year,
        report_type=report_type,
    )
