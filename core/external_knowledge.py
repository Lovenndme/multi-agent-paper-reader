"""Optional academic metadata lookup for follow-up questions that need outside context."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_RECOMMENDATIONS_URL = (
    "https://api.semanticscholar.org/recommendations/v1/papers/forpaper"
)
EXTERNAL_QUERY_MARKERS = (
    "最新", "近期", "最近", "其他论文", "相关论文", "外部资料", "领域现状",
    "文献综述", "后续研究进展", "和现有工作相比", "与现有工作相比", "相比其他论文",
    "和其他论文相比", "与其他论文相比", "state of the art", "latest", "recent",
    "other paper", "related paper", "compare with other", "comparison with other",
    "literature review",
)
LATEST_QUERY_MARKERS = ("最新", "近期", "最近", "领域现状", "latest", "recent", "state of the art")


@dataclass(frozen=True)
class ExternalAcademicSource:
    id: str
    title: str
    year: int | None
    authors: str
    abstract: str
    url: str


def should_search_external(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in EXTERNAL_QUERY_MARKERS)


def search_external_academic_sources(
    question: str,
    paper_title: str,
    *,
    limit: int = 3,
    timeout: float = 8.0,
) -> list[ExternalAcademicSource]:
    """Find related Semantic Scholar metadata/abstracts when outside literature is needed."""
    if not should_search_external(question):
        return []

    # The paper title is normally English even when the follow-up question is
    # Chinese. Search it first, then use Semantic Scholar's paper-aware
    # recommendation endpoint instead of mixing two languages into one query.
    query = " ".join((paper_title or question).split())[:300]
    params = urlencode(
        {
            "query": query,
            "limit": 5,
            "fields": "paperId,title,year,authors,abstract,url,externalIds",
        }
    )
    payload = _request_json(f"{SEMANTIC_SCHOLAR_SEARCH_URL}?{params}", timeout)
    if not payload:
        return []

    search_results = payload.get("data", [])
    normalized_title = _normalize_title(paper_title)
    matched_paper = next(
        (
            item
            for item in search_results
            if normalized_title
            and _normalize_title(str(item.get("title") or "")) == normalized_title
            and item.get("paperId")
        ),
        None,
    )
    if matched_paper:
        pool = "recent" if any(marker in question.lower() for marker in LATEST_QUERY_MARKERS) else "all-cs"
        recommendation_params = urlencode(
            {
                "limit": max(1, min(limit, 5)),
                "from": pool,
                "fields": "title,year,authors,abstract,url,externalIds",
            }
        )
        recommendation_url = (
            f"{SEMANTIC_SCHOLAR_RECOMMENDATIONS_URL}/{matched_paper['paperId']}"
            f"?{recommendation_params}"
        )
        recommendation_payload = _request_json(recommendation_url, timeout)
        recommended = (
            recommendation_payload.get("recommendedPapers", [])
            if recommendation_payload
            else []
        )
        if recommended:
            return _parse_sources(recommended, limit)

    fallback_results = [
        item
        for item in search_results
        if _normalize_title(str(item.get("title") or "")) != normalized_title
    ]
    return _parse_sources(fallback_results, limit)


def _request_json(url: str, timeout: float) -> dict:
    headers = {"User-Agent": "multi-agent-paper-reader/0.1"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS hosts
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def _parse_sources(items: list[dict], limit: int) -> list[ExternalAcademicSource]:
    sources: list[ExternalAcademicSource] = []
    for item in items[:limit]:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        authors = ", ".join(
            str(author.get("name") or "").strip()
            for author in item.get("authors", [])[:5]
            if author.get("name")
        )
        external_ids = item.get("externalIds") or {}
        doi = external_ids.get("DOI")
        url = str(item.get("url") or (f"https://doi.org/{doi}" if doi else "")).strip()
        sources.append(
            ExternalAcademicSource(
                id=f"S{len(sources) + 1}",
                title=title,
                year=item.get("year") if isinstance(item.get("year"), int) else None,
                authors=authors,
                abstract=str(item.get("abstract") or "").strip()[:1600],
                url=url,
            )
        )
    return sources


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def format_external_sources(sources: list[ExternalAcademicSource]) -> str:
    blocks: list[str] = []
    for source in sources:
        heading = f"[{source.id} | external abstract | {source.year or 'year unknown'}] {source.title}"
        details = [heading]
        if source.authors:
            details.append(f"Authors: {source.authors}")
        if source.abstract:
            details.append(f"Abstract: {source.abstract}")
        if source.url:
            details.append(f"URL: {source.url}")
        blocks.append("\n".join(details))
    return "\n\n".join(blocks)
