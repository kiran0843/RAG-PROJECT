"""Query expansion for narrative retrieval. Not applied to metric path."""

_EXPANSION_RULES: list[tuple[str, list[str]]] = [
    ("POSH", ["ICC", "sexual harassment", "complaints received"]),
    ("Board meetings", ["meetings of the Board", "number of meetings held"]),
    ("complaints", ["cases received"]),
]


def expand_for_narrative(query: str) -> str:
    """Append expansion phrases when triggers match. Returns expanded query."""
    if not query or not query.strip():
        return query

    q_lower = query.lower()
    expansions: list[str] = []

    for trigger, phrases in _EXPANSION_RULES:
        if trigger.lower() in q_lower:
            expansions.extend(phrases)

    if not expansions:
        return query

    seen: set[str] = set()
    unique: list[str] = []
    for p in expansions:
        if p.lower() not in seen:
            seen.add(p.lower())
            unique.append(p)

    return f"{query.strip()} {' '.join(unique)}"
