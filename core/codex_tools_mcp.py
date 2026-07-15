"""stdio MCP adapter for the paper reader's fixed read-only tool surface."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypeVar

from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations

from core import codex_tools


ResultT = TypeVar("ResultT")
READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
mcp = FastMCP(
    "paper_reader",
    instructions=(
        "Read-only academic-paper tools bound by the host to exactly one current paper. "
        "Never ask for or infer filesystem paths, history IDs, or coordinates."
    ),
    log_level="ERROR",
)


def _call(function: Callable[..., ResultT], *args: Any, **kwargs: Any) -> ResultT:
    try:
        return function(*args, **kwargs)
    except codex_tools.PaperToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - never expose local paths/runtime internals
        raise codex_tools.PaperToolError("论文只读工具暂时无法完成本次请求。") from exc


@mcp.tool(annotations=READ_ONLY, structured_output=True)
def paper_search_evidence(
    query: str,
    kinds: list[str] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search original text/table/figure evidence in the currently bound paper."""
    return _call(codex_tools.paper_search_evidence, query, kinds=kinds, limit=limit)


@mcp.tool(annotations=READ_ONLY, structured_output=True)
def paper_get_section(section: str, max_chars: int = 12_000) -> dict[str, Any]:
    """Read one section by exact title or 1-based section number."""
    return _call(codex_tools.paper_get_section, section, max_chars=max_chars)


@mcp.tool(annotations=READ_ONLY, structured_output=True)
def paper_get_page(page: int, page_count: int = 1, max_chars: int = 12_000) -> dict[str, Any]:
    """Read one or two 1-based pages from the bound paper PDF."""
    return _call(
        codex_tools.paper_get_page,
        page,
        page_count=page_count,
        max_chars=max_chars,
    )


@mcp.tool(annotations=READ_ONLY, structured_output=True)
def paper_get_figure(figure_id: str) -> dict[str, Any]:
    """Read caption and verified visual summary for an Fxxx evidence ID."""
    return _call(codex_tools.paper_get_figure, figure_id)


@mcp.tool(annotations=READ_ONLY, structured_output=True)
def paper_get_table(
    table_id: str,
    max_rows: int = 40,
    max_cols: int = 12,
) -> dict[str, Any]:
    """Read bounded cells and caption for a Txxx evidence ID."""
    return _call(
        codex_tools.paper_get_table,
        table_id,
        max_rows=max_rows,
        max_cols=max_cols,
    )


@mcp.tool(annotations=READ_ONLY)
def paper_get_visual_region(region_id: str) -> list[Any]:
    """Render only a parser-verified Fxxx/Txxx crop, never an arbitrary bbox or path."""
    metadata, png = _call(codex_tools.paper_get_visual_region, region_id)
    return [
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        Image(data=png, format="png"),
    ]


@mcp.tool(annotations=READ_ONLY, structured_output=True)
def paper_recall_memory(query: str, limit: int = 3) -> dict[str, Any]:
    """Recall bounded long-term notes from the current paper namespace only."""
    return _call(codex_tools.paper_recall_memory, query, limit=limit)


@mcp.tool(annotations=READ_ONLY, structured_output=True)
def calculate(expression: str) -> dict[str, Any]:
    """Evaluate bounded arithmetic with an AST whitelist and no Python eval."""
    return _call(codex_tools.calculate, expression)


if __name__ == "__main__":
    # stdout is reserved exclusively for MCP JSON-RPC framing.
    mcp.run(transport="stdio")
