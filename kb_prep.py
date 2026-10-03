"""Trim Ayaz's skills knowledge base down to what Jarvis needs to answer visitors.

The knowledge base doubles as Ayaz's resume-building source, so it carries notes Jarvis never uses: where each
fact came from, which resume versions used which wording, and how to frame summaries per role type. Sending those
on every question costs tokens and can leak into answers. Shared by app.py, tools/grounding.py, and the deploy
check, so all three see exactly the same text.
"""
import os
import re
from pathlib import Path

# Jarvis's private answer rules (exact metric wording, context-only facts, extra off-limits topics). Kept out of
# the public repo: the TWIN_RULES_TEXT secret when hosted, or this file next to the knowledge base locally.
RULES_PATH = Path(os.getenv("TWIN_RULES_PATH", Path(__file__).parent.parent / "Jarvis_Private_Rules.md"))


def private_rules() -> str:
    """Private rules, appended to persona.md wherever Jarvis's prompt is built ('' if none are set)."""
    text = os.getenv("TWIN_RULES_TEXT") or (RULES_PATH.read_text() if RULES_PATH.exists() else "")
    return f"\n\n{text.strip()}\n" if text.strip() else ""


def jarvis_kb(kb: str) -> str:
    # Source-version keys come from the knowledge base's own source table ("| KEY | file | date | framing |"),
    # so this public file never lists them.
    first = re.search(r"(?m)^## 1\.", kb)
    keys = re.findall(r"(?m)^\| ([A-Z][A-Z0-9]{1,5}) \|", kb[: first.start()] if first else "")
    # Provenance header + source-versions table: everything before section 1.
    if first:
        kb = "# Ayaz Shaik — Skills & Experience Knowledge Base\n\n" + kb[first.start():]
    # Section 8 (resume summary framings per role type).
    kb = re.sub(r"(?ms)^## 8\..*?(?=^## )", "", kb)
    # Resume-wording notes inside achievements: 'Wording used (<application>, <date>): "..."'
    kb = re.sub(r'\s*Wording used \([^)]*\): "[^"]*"\.?', "", kb)
    # Source-version tags like (ABC) or (ABC, XYZ) — resume bookkeeping, not facts.
    if keys:
        tags = "(?:" + "|".join(map(re.escape, keys)) + ")"
        kb = re.sub(rf"\s*\((?:{tags})(?:,\s*{tags})*\)", "", kb)
        kb = re.sub(rf"\s*\((?:{tags}) [^)]*\)", "", kb)  # e.g. "(ABC lists these under AI Projects)"
    # Jarvis may share LinkedIn and email only (persona.md), so the phone number isn't sent at all.
    kb = re.sub(r"\s*·\s*\(?\d{3}\)?[ -]?\d{3}-\d{4}", "", kb)
    return re.sub(r"\n{3,}", "\n\n", kb).strip() + "\n"
