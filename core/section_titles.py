"""Safe display handling for parsed paper section titles."""

from __future__ import annotations

import re


def clean_section_title(
    title: str,
    index: int,
) -> str:
    """Preserve the paper's original section title with a safe noisy-text fallback."""
    cleaned = re.sub(r"\s+", " ", str(title or "")).strip()
    letters = re.findall(r"[A-Za-z\u4e00-\u9fff]", cleaned)
    symbols = re.findall(r"[^A-Za-z0-9\u4e00-\u9fff\s.\-:/&]", cleaned)
    looks_noisy = (
        not cleaned
        or len(letters) < 2
        or "\ufffd" in cleaned
        or "�" in cleaned
        or len(symbols) / max(len(cleaned), 1) > 0.16
    )
    return f"章节 {index + 1}" if looks_noisy else cleaned
