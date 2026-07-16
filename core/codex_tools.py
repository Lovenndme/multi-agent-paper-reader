"""Capability-bound, read-only tools for Codex paper turns."""

from __future__ import annotations

import ast
import json
import math
import os
import re
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fitz

from core.evidence import EvidenceSnippet, build_evidence_index
from core.history import load_paper_analysis, retained_paper_pdf_path
from core.pdf_parser import FigureBlock, ParsedPaper, TableBlock, parse_pdf
from core.pdf_rendering import (
    plan_figure_preview_render,
    plan_table_preview_render,
    render_page_preview_png,
    render_planned_png,
)


ROOT = Path(__file__).resolve().parent.parent
_CONTEXT_ENV = "PAPER_READER_CODEX_CONTEXT_FILE"
_MAX_CONTEXT_BYTES = 16 * 1024 * 1024
_MAX_TOOL_TEXT = 16_000
_MAX_VISUAL_BYTES = 4 * 1024 * 1024
_MAX_OVERVIEW_ASSETS = 100
_CONTEXT_MAX_AGE_SECONDS = 6 * 60 * 60
_SAFE_CONTEXT_NAME = re.compile(r"^[0-9a-f]{48}\.json$")
_SAFE_EVIDENCE_ID = re.compile(r"^[ETF]\d{3,6}$", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_-]+|[\u4e00-\u9fff]{2,}")


class PaperToolError(ValueError):
    """Safe tool failure that never includes host paths or credentials."""


@dataclass
class CodexToolContextHandle:
    path: Path

    def close(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self) -> "CodexToolContextHandle":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def create_codex_tool_context(
    *,
    snippets: Iterable[EvidenceSnippet],
    context: dict[str, Any] | None = None,
    paper: ParsedPaper | None = None,
    pdf_path: str | Path | None = None,
    history_id: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> CodexToolContextHandle:
    """Create one private capability file owned by the current paper turn."""
    safe_pdf = _trusted_pdf_path(pdf_path)
    analysis = _bounded_analysis_context(context or {})
    raw_manifest = (
        build_codex_paper_manifest(paper)
        if paper is not None
        else manifest or _manifest_from_analysis(analysis)
    )
    bounded_manifest = _bounded_paper_manifest(raw_manifest)
    payload: dict[str, Any] = {
        "version": 1,
        "history_id": history_id if history_id and re.fullmatch(r"[A-Za-z0-9_-]{1,120}", history_id) else None,
        "pdf_path": str(safe_pdf) if safe_pdf else None,
        "snippets": [_bounded_snippet(snippet) for snippet in list(snippets)[:800]],
        "analysis": analysis,
        "paper": bounded_manifest["paper"],
        "section_index": bounded_manifest["section_index"],
        "sections": [],
        "tables": bounded_manifest["tables"],
        "figures": bounded_manifest["figures"],
    }
    if paper is not None:
        payload["sections"] = [
            {
                "title": str(section.title or "")[:500],
                "content": str(section.content or "")[:_MAX_TOOL_TEXT],
                "content_truncated": len(str(section.content or "")) > _MAX_TOOL_TEXT,
                "page_start": int(section.page_start),
                "page_end": int(section.page_end),
            }
            for section in paper.sections[:300]
        ]

    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_CONTEXT_BYTES:
        payload["analysis"] = {}
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_CONTEXT_BYTES:
        # Section bodies duplicate text snippets. Keep the evidence index and
        # visual metadata so all tools still fail safely and page lookup can
        # recover exact text from the bound PDF.
        payload["sections"] = []
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_CONTEXT_BYTES:
        raise PaperToolError("论文工具上下文过大，无法安全启用。")

    directory = _context_directory()
    _prune_stale_contexts(directory)
    context_path = directory / f"{secrets.token_hex(24)}.json"
    descriptor = os.open(context_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        context_path.unlink(missing_ok=True)
        raise
    return CodexToolContextHandle(context_path)


def create_codex_tool_context_from_history(history_id: str) -> CodexToolContextHandle:
    """Bind a tool context to one saved paper selected by the trusted application."""
    stored = load_paper_analysis(history_id)
    if stored is None:
        raise PaperToolError("已保存论文不存在。")
    pdf_path = retained_paper_pdf_path(history_id)
    manifest = stored.get("paper_manifest")
    paper: ParsedPaper | None = None
    snippets = stored["snippets"]
    if not isinstance(manifest, dict) and pdf_path is not None:
        paper = parse_pdf(pdf_path)
        snippets = build_evidence_index(paper)
        manifest = build_codex_paper_manifest(paper)
        manifest["paper"]["legacy_reparsed"] = True
    return create_codex_tool_context(
        snippets=snippets,
        context=stored["result"],
        paper=paper,
        pdf_path=pdf_path,
        history_id=history_id,
        manifest=manifest if isinstance(manifest, dict) else None,
    )


def validate_codex_tool_context_path(value: str | Path) -> Path:
    """Validate a host-created capability path before exposing it to an MCP process."""
    try:
        path = Path(value).expanduser().resolve(strict=True)
        path.relative_to(_context_directory().resolve(strict=True))
        file_stat = path.stat()
    except (OSError, RuntimeError, ValueError) as exc:
        raise PaperToolError("论文工具上下文无效或已过期。") from exc
    if not _SAFE_CONTEXT_NAME.fullmatch(path.name):
        raise PaperToolError("论文工具上下文标识无效。")
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > _MAX_CONTEXT_BYTES:
        raise PaperToolError("论文工具上下文文件无效。")
    if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
        raise PaperToolError("论文工具上下文所有者无效。")
    # Windows reports synthetic POSIX mode bits (typically 0o666) that do not
    # describe the file's ACL.  The file inherits the current user's protected
    # data-directory ACL there; enforce Unix owner-only bits only on POSIX.
    if os.name != "nt" and file_stat.st_mode & 0o077:
        raise PaperToolError("论文工具上下文权限不安全。")
    return path


def load_bound_context() -> dict[str, Any]:
    """Load only the context path injected by the trusted Codex thread config."""
    raw_path = os.environ.get(_CONTEXT_ENV)
    if not raw_path:
        raise PaperToolError("本轮没有绑定论文工具上下文。")
    path = validate_codex_tool_context_path(raw_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PaperToolError("论文工具上下文无法读取。") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise PaperToolError("论文工具上下文版本无效。")
    return payload


def build_codex_paper_manifest(paper: ParsedPaper) -> dict[str, Any]:
    """Serialize stable, path-free paper navigation and visual records for history."""
    page_count = max(
        [section.page_end + 1 for section in paper.sections]
        + [table.page + 1 for table in paper.tables]
        + [figure.page + 1 for figure in paper.figures]
        + [0]
    )
    return {
        "version": 1,
        "paper": {
            "title": str(paper.title or "Untitled Paper")[:1_000],
            "page_count": page_count,
            "parser_backend": str(paper.metadata.get("parser_backend") or "")[:100],
            "metadata": _bounded_paper_metadata(paper.metadata),
            "legacy_reparsed": False,
        },
        "section_index": [
            {
                "index": index,
                "title": str(section.title or "Untitled section")[:500],
                "page_start": max(0, int(section.page_start)),
                "page_end": max(0, int(section.page_end)),
                "chars": len(str(section.content or "")),
            }
            for index, section in enumerate(paper.sections[:300], start=1)
        ],
        "tables": [
            _bounded_table(table, index)
            for index, table in enumerate(paper.tables[:500], start=1)
        ],
        "figures": [
            _bounded_figure(figure, index)
            for index, figure in enumerate(paper.figures[:500], start=1)
        ],
    }


def paper_get_overview(
    *,
    asset_offset: int = 0,
    asset_limit: int = 50,
) -> dict[str, Any]:
    """Return bounded paper metadata, section outline, and discoverable F/T assets."""
    context = load_bound_context()
    paper = context.get("paper") if isinstance(context.get("paper"), dict) else {}
    sections = (
        context.get("section_index")
        if isinstance(context.get("section_index"), list)
        else []
    )
    figures = context.get("figures") if isinstance(context.get("figures"), list) else []
    tables = context.get("tables") if isinstance(context.get("tables"), list) else []
    assets: list[dict[str, Any]] = []
    for kind, records in (("figure", figures), ("table", tables)):
        for record in records:
            if not isinstance(record, dict):
                continue
            assets.append(
                {
                    "id": str(record.get("id") or "")[:16],
                    "kind": kind,
                    "page": max(0, int(record.get("page", 0))) + 1,
                    "caption": str(record.get("caption") or "")[:800],
                    "bbox_verified": _bbox(record.get("bbox")) is not None,
                }
            )
    assets.sort(key=lambda item: (item["page"], item["kind"], item["id"]))
    offset = max(0, min(int(asset_offset), len(assets)))
    limit = max(1, min(int(asset_limit), _MAX_OVERVIEW_ASSETS))
    selected = assets[offset : offset + limit]
    next_offset = offset + len(selected) if offset + len(selected) < len(assets) else None
    page_count = _nonnegative_int(paper.get("page_count"))
    if context.get("pdf_path"):
        document = fitz.open(str(_context_pdf(context)))
        try:
            page_count = document.page_count
        finally:
            document.close()
    return {
        "title": str(paper.get("title") or "Untitled Paper")[:1_000],
        "page_count": page_count,
        "parser_backend": str(paper.get("parser_backend") or "")[:100],
        "metadata": _bounded_paper_metadata(paper.get("metadata")),
        "sections": [
            {
                "index": max(1, int(section.get("index", index))),
                "title": str(section.get("title") or "Untitled section")[:500],
                "page_start": max(0, int(section.get("page_start", 0))) + 1,
                "page_end": max(0, int(section.get("page_end", 0))) + 1,
                "chars": max(0, int(section.get("chars", 0))),
            }
            for index, section in enumerate(sections[:300], start=1)
            if isinstance(section, dict)
        ],
        "counts": {
            "sections": len(sections),
            "figures": len(figures),
            "tables": len(tables),
        },
        "assets": selected,
        "asset_offset": offset,
        "next_asset_offset": next_offset,
        "assets_truncated": next_offset is not None,
        "legacy_reparsed": bool(paper.get("legacy_reparsed")),
    }


def paper_search_evidence(
    query: str,
    *,
    kinds: list[str] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search bounded original-paper evidence in the current capability context."""
    clean_query = " ".join(str(query).split())[:500]
    if not clean_query:
        raise PaperToolError("检索词不能为空。")
    allowed_kinds = {"text", "table", "figure"}
    selected_kinds = {str(kind).lower() for kind in (kinds or allowed_kinds)}
    if not selected_kinds or not selected_kinds <= allowed_kinds:
        raise PaperToolError("证据类型仅支持 text、table、figure。")
    bounded_limit = max(1, min(int(limit), 8))
    terms = _query_terms(clean_query)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, snippet in enumerate(_snippets(load_bound_context())):
        if snippet["kind"] not in selected_kinds:
            continue
        section = snippet["section"].casefold()
        text = snippet["text"].casefold()
        score = 0.0
        if clean_query.casefold() in text:
            score += 12.0
        for term in terms:
            if term in section:
                score += 4.0
            score += min(text.count(term), 5)
        if score > 0:
            scored.append((score, -index, snippet))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    matches = [_snippet_result(item[2], max_chars=2400) for item in scored[:bounded_limit]]
    return {"query": clean_query, "matches": matches, "count": len(matches)}


def paper_get_section(section: str | int, *, max_chars: int = 12_000) -> dict[str, Any]:
    """Return one exact/fuzzy section assembled from bounded original snippets."""
    context = load_bound_context()
    bounded_chars = max(500, min(int(max_chars), _MAX_TOOL_TEXT))
    sections = context.get("sections") if isinstance(context.get("sections"), list) else []
    if sections:
        chosen = _select_named_record(sections, section, title_key="title")
        content = str(chosen.get("content") or "")[:bounded_chars]
        return {
            "title": str(chosen.get("title") or "Untitled section"),
            "page_start": int(chosen.get("page_start", 0)) + 1,
            "page_end": int(chosen.get("page_end", 0)) + 1,
            "text": content,
            "truncated": bool(chosen.get("content_truncated"))
            or len(str(chosen.get("content") or "")) > len(content),
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for snippet in _snippets(context):
        if snippet["kind"] == "text":
            grouped.setdefault(snippet["section"], []).append(snippet)
    records = [{"title": title, "items": items} for title, items in grouped.items()]
    chosen = _select_named_record(records, section, title_key="title")
    text = "\n\n".join(item["text"] for item in chosen["items"])
    output = text[:bounded_chars]
    return {
        "title": chosen["title"],
        "page_start": min(item["page_start"] for item in chosen["items"]) + 1,
        "page_end": max(item["page_end"] for item in chosen["items"]) + 1,
        "text": output,
        "truncated": len(text) > len(output),
    }


def paper_get_page(page: int, *, page_count: int = 1, max_chars: int = 12_000) -> dict[str, Any]:
    """Extract at most two 1-based PDF pages from the bound retained paper."""
    context = load_bound_context()
    pdf_path = _context_pdf(context)
    start = int(page)
    count = max(1, min(int(page_count), 2))
    bounded_chars = max(500, min(int(max_chars), 12_000))
    if start < 1:
        raise PaperToolError("页码必须从 1 开始。")
    document = fitz.open(str(pdf_path))
    try:
        if start > document.page_count or start + count - 1 > document.page_count:
            raise PaperToolError("请求页码超出论文范围。")
        blocks = [document[index - 1].get_text("text") for index in range(start, start + count)]
    finally:
        document.close()
    text = "\n\n".join(blocks)
    output = text[:bounded_chars]
    return {
        "page_start": start,
        "page_end": start + count - 1,
        "text": output,
        "truncated": len(text) > len(output),
    }


def paper_get_page_image(page: int, *, dpi: int = 120) -> tuple[dict[str, Any], bytes]:
    """Render one explicitly requested page; never acts as a visual-region fallback."""
    context = load_bound_context()
    pdf_path = _context_pdf(context)
    requested_page = int(page)
    bounded_dpi = max(96, min(int(dpi), 144))
    if requested_page < 1:
        raise PaperToolError("页码必须从 1 开始。")
    document = fitz.open(str(pdf_path))
    try:
        if requested_page > document.page_count:
            raise PaperToolError("请求页码超出论文范围。")
        page_object = document[requested_page - 1]
        width_points = float(page_object.rect.width)
        height_points = float(page_object.rect.height)
        try:
            png = render_page_preview_png(document, requested_page - 1, dpi=bounded_dpi)
        except ValueError as exc:
            raise PaperToolError("该 PDF 页面超过安全渲染范围。") from exc
        page_count = document.page_count
    finally:
        document.close()
    if len(png) > _MAX_VISUAL_BYTES:
        raise PaperToolError("整页图像超过安全大小限制。")
    return {
        "page": requested_page,
        "page_count": page_count,
        "width_points": round(width_points, 2),
        "height_points": round(height_points, 2),
        "dpi": bounded_dpi,
        "media_type": "image/png",
        "bytes": len(png),
    }, png


def paper_get_figure(figure_id: str) -> dict[str, Any]:
    """Return bounded caption/summary metadata for one Fxxx visual."""
    evidence_id = _validated_evidence_id(figure_id, prefix="F")
    context = load_bound_context()
    record = _record_by_id(context.get("figures"), evidence_id)
    if record is not None:
        return {
            "id": evidence_id,
            "page": int(record.get("page", 0)) + 1,
            "caption": str(record.get("caption") or "")[:4000],
            "visual_summary": str(record.get("visual_summary") or "")[:6000],
            "bbox_verified": bool(record.get("bbox")),
        }
    return _evidence_by_id(context, evidence_id, kind="figure")


def paper_get_table(
    table_id: str,
    *,
    max_rows: int = 40,
    max_cols: int = 12,
) -> dict[str, Any]:
    """Return one bounded Txxx table without accepting arbitrary paths or coordinates."""
    evidence_id = _validated_evidence_id(table_id, prefix="T")
    context = load_bound_context()
    record = _record_by_id(context.get("tables"), evidence_id)
    if record is not None:
        rows = record.get("rows") if isinstance(record.get("rows"), list) else []
        row_limit = max(1, min(int(max_rows), 40))
        col_limit = max(1, min(int(max_cols), 12))
        bounded_rows = [
            [str(cell)[:1000] for cell in row[:col_limit]]
            for row in rows[:row_limit]
            if isinstance(row, list)
        ]
        return {
            "id": evidence_id,
            "page": int(record.get("page", 0)) + 1,
            "caption": str(record.get("caption") or "")[:4000],
            "rows": bounded_rows,
            "truncated": len(rows) > len(bounded_rows) or any(
                isinstance(row, list) and len(row) > col_limit for row in rows[:row_limit]
            ) or bool(record.get("rows_truncated")),
            "bbox_verified": bool(record.get("bbox")),
        }
    return _evidence_by_id(context, evidence_id, kind="table")


def paper_get_visual_region(region_id: str) -> tuple[dict[str, Any], bytes]:
    """Render only a parser-verified Fxxx/Txxx region; never accepts bbox or file paths."""
    evidence_id = _validated_evidence_id(region_id)
    context = load_bound_context()
    pdf_path = _context_pdf(context)
    prefix = evidence_id[0]
    records_key = "figures" if prefix == "F" else "tables"
    record = _record_by_id(context.get(records_key), evidence_id)
    if record is None:
        raise PaperToolError("视觉区域不存在。")
    if not record.get("bbox"):
        raise PaperToolError("该视觉区域没有经过验证的边界框，已拒绝整页回退。")
    if prefix == "F":
        item = FigureBlock(
            page=int(record.get("page", 0)),
            caption=str(record.get("caption") or ""),
            image_index=int(record.get("image_index", 0)),
            bbox=_bbox(record.get("bbox")),
            caption_bbox=_bbox(record.get("caption_bbox")),
            visual_summary=str(record.get("visual_summary") or ""),
        )
    else:
        item = TableBlock(
            page=int(record.get("page", 0)),
            rows=[[str(cell) for cell in row] for row in record.get("rows", []) if isinstance(row, list)],
            caption=str(record.get("caption") or ""),
            bbox=_bbox(record.get("bbox")),
            caption_bbox=_bbox(record.get("caption_bbox")),
        )
    if getattr(item, "bbox", None) is None:
        raise PaperToolError("该视觉区域没有经过验证的边界框，已拒绝整页回退。")
    document = fitz.open(str(pdf_path))
    try:
        try:
            if item.page < 0 or item.page >= document.page_count:
                raise ValueError("Visual region page is outside the retained PDF.")
            page = document[item.page]
            plan = (
                plan_figure_preview_render(page, item, dpi=144)
                if prefix == "F"
                else plan_table_preview_render(page, item, dpi=144)
            )
            png = render_planned_png(page, plan)
        except ValueError as exc:
            raise PaperToolError("该视觉区域超过安全渲染范围。") from exc
    finally:
        document.close()
    if len(png) > _MAX_VISUAL_BYTES:
        raise PaperToolError("视觉区域预览超过安全大小限制。")
    metadata = {
        "id": evidence_id,
        "kind": "figure" if prefix == "F" else "table",
        "page": int(item.page) + 1,
        "caption": str(getattr(item, "caption", "") or "")[:4000],
        "dpi": plan.dpi,
        "width_px": plan.width_px,
        "height_px": plan.height_px,
        "quality": plan.policy,
        "media_type": "image/png",
        "bytes": len(png),
    }
    return metadata, png


def paper_recall_memory(query: str, *, limit: int = 3) -> dict[str, Any]:
    """Recall bounded LangMem records for the already-bound paper only."""
    context = load_bound_context()
    history_id = context.get("history_id")
    if not history_id:
        return {"items": [], "count": 0, "message": "本轮论文尚未建立持久化记忆。"}
    clean_query = " ".join(str(query).split())[:500]
    if not clean_query:
        raise PaperToolError("记忆检索词不能为空。")
    from core.langmem_store import list_langmem_memories

    items = list_langmem_memories(
        str(history_id),
        query=clean_query,
        limit=max(1, min(int(limit), 5)),
    )
    safe_items = [
        {
            "topic": str(item.get("topic") or "")[:300],
            "type": str(item.get("type") or "")[:80],
            "content": str(item.get("content") or "")[:2400],
            "description": str(item.get("description") or "")[:800],
            "score": item.get("score"),
        }
        for item in items[:5]
    ]
    return {"items": safe_items, "count": len(safe_items)}


def calculate(expression: str) -> dict[str, Any]:
    """Evaluate a small arithmetic expression through an AST whitelist, never eval()."""
    clean = str(expression).strip()
    if not clean or len(clean) > 256:
        raise PaperToolError("计算表达式必须为 1 到 256 个字符。")
    try:
        tree = ast.parse(clean, mode="eval")
    except SyntaxError as exc:
        raise PaperToolError("计算表达式语法无效。") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise PaperToolError("计算表达式过于复杂。")
    value = _evaluate_math_node(tree.body)
    if not math.isfinite(value) or abs(value) > 1e100:
        raise PaperToolError("计算结果超出安全范围。")
    return {"expression": clean, "result": value}


def _evaluate_math_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return _bounded_number(float(node.value))
    if isinstance(node, ast.Name) and node.id in {"pi", "e"}:
        return math.pi if node.id == "pi" else math.e
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_math_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    ):
        left = _evaluate_math_node(node.left)
        right = _evaluate_math_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise PaperToolError("指数超出安全范围。")
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
            ast.Pow: lambda: left**right,
        }
        try:
            return _bounded_number(float(operations[type(node.op)]()))
        except (OverflowError, ZeroDivisionError, ValueError) as exc:
            raise PaperToolError("计算无法在安全范围内完成。") from exc
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        functions = {
            "abs": (abs, 1),
            "sqrt": (math.sqrt, 1),
            "log": (math.log, 1),
            "log10": (math.log10, 1),
            "round": (round, 1),
            "min": (min, None),
            "max": (max, None),
        }
        spec = functions.get(node.func.id)
        if spec is None or not node.args or len(node.args) > 8:
            raise PaperToolError("计算函数不在允许列表中。")
        function, arity = spec
        if arity is not None and len(node.args) != arity:
            raise PaperToolError("计算函数参数数量无效。")
        values = [_evaluate_math_node(argument) for argument in node.args]
        try:
            return _bounded_number(float(function(*values)))
        except (OverflowError, ValueError) as exc:
            raise PaperToolError("计算函数无法在安全范围内完成。") from exc
    raise PaperToolError("表达式包含不允许的语法。")


def _bounded_number(value: float) -> float:
    if not math.isfinite(value) or abs(value) > 1e100:
        raise PaperToolError("计算中间值超出安全范围。")
    return value


def _context_directory() -> Path:
    data_root = Path(os.environ.get("PAPER_READER_DATA_DIR") or ROOT / ".paper-reader").expanduser().resolve()
    directory = data_root / "codex-tool-contexts"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    return directory


def _prune_stale_contexts(directory: Path) -> None:
    """Remove abandoned capability files without touching active recent turns."""
    cutoff = time.time() - _CONTEXT_MAX_AGE_SECONDS
    try:
        entries = tuple(directory.iterdir())
    except OSError:
        return
    for path in entries:
        if not _SAFE_CONTEXT_NAME.fullmatch(path.name):
            continue
        try:
            file_stat = path.lstat()
            if stat.S_ISREG(file_stat.st_mode) and file_stat.st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


def _trusted_pdf_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise PaperToolError("论文 PDF 不存在。") from exc
    if not path.is_file() or path.suffix.lower() != ".pdf":
        raise PaperToolError("论文 PDF 无效。")
    return path


def _context_pdf(context: dict[str, Any]) -> Path:
    raw = context.get("pdf_path")
    if not raw:
        raise PaperToolError("本轮没有可读取的论文 PDF。")
    return _trusted_pdf_path(str(raw)) or Path()


def _bounded_analysis_context(context: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "mode",
        "paper",
        "method_output",
        "experiment_output",
        "critic_output",
        "summary_output",
        "assessment",
        "evidence_index",
    }
    return {key: context[key] for key in allowed if key in context}


def _manifest_from_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    paper = analysis.get("paper") if isinstance(analysis.get("paper"), dict) else {}
    metadata = paper.get("metadata") if isinstance(paper.get("metadata"), dict) else {}
    return {
        "version": 1,
        "paper": {
            "title": str(paper.get("title") or "Untitled Paper")[:1_000],
            "page_count": _nonnegative_int(paper.get("pages")),
            "parser_backend": str(metadata.get("parser_backend") or "")[:100],
            "metadata": metadata,
            "legacy_reparsed": False,
        },
        "section_index": paper.get("sections") if isinstance(paper.get("sections"), list) else [],
        "tables": [],
        "figures": [],
    }


def _bounded_paper_manifest(value: Any) -> dict[str, Any]:
    manifest = value if isinstance(value, dict) else {}
    paper = manifest.get("paper") if isinstance(manifest.get("paper"), dict) else {}
    section_records = (
        manifest.get("section_index")
        if isinstance(manifest.get("section_index"), list)
        else []
    )
    table_records = manifest.get("tables") if isinstance(manifest.get("tables"), list) else []
    figure_records = manifest.get("figures") if isinstance(manifest.get("figures"), list) else []

    sections = [
        {
            "index": index,
            "title": str(record.get("title") or "Untitled section")[:500],
            "page_start": _nonnegative_int(record.get("page_start")),
            "page_end": max(
                _nonnegative_int(record.get("page_start")),
                _nonnegative_int(record.get("page_end")),
            ),
            "chars": _nonnegative_int(record.get("chars")),
        }
        for index, record in enumerate(section_records[:300], start=1)
        if isinstance(record, dict)
    ]
    tables = [
        _bounded_manifest_table(record, index)
        for index, record in enumerate(table_records[:500], start=1)
        if isinstance(record, dict)
    ]
    figures = [
        _bounded_manifest_figure(record, index)
        for index, record in enumerate(figure_records[:500], start=1)
        if isinstance(record, dict)
    ]
    derived_pages = max(
        [record["page_end"] + 1 for record in sections]
        + [record["page"] + 1 for record in tables]
        + [record["page"] + 1 for record in figures]
        + [0]
    )
    return {
        "version": 1,
        "paper": {
            "title": str(paper.get("title") or "Untitled Paper")[:1_000],
            "page_count": max(_nonnegative_int(paper.get("page_count")), derived_pages),
            "parser_backend": str(paper.get("parser_backend") or "")[:100],
            "metadata": _bounded_paper_metadata(paper.get("metadata")),
            "legacy_reparsed": bool(paper.get("legacy_reparsed")),
        },
        "section_index": sections,
        "tables": tables,
        "figures": figures,
    }


def _bounded_manifest_table(record: dict[str, Any], index: int) -> dict[str, Any]:
    rows = record.get("rows") if isinstance(record.get("rows"), list) else []
    bounded_rows = [
        [str(cell)[:1_000] for cell in row[:12]]
        for row in rows[:40]
        if isinstance(row, list)
    ]
    return {
        "id": _manifest_evidence_id(record.get("id"), "T", index),
        "page": _nonnegative_int(record.get("page")),
        "rows": bounded_rows,
        "rows_truncated": bool(record.get("rows_truncated"))
        or len(rows) > len(bounded_rows)
        or any(isinstance(row, list) and len(row) > 12 for row in rows[:40]),
        "caption": str(record.get("caption") or "")[:4_000],
        "bbox": _bbox(record.get("bbox")),
        "caption_bbox": _bbox(record.get("caption_bbox")),
    }


def _bounded_manifest_figure(record: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": _manifest_evidence_id(record.get("id"), "F", index),
        "page": _nonnegative_int(record.get("page")),
        "caption": str(record.get("caption") or "")[:4_000],
        "image_index": _nonnegative_int(record.get("image_index")),
        "bbox": _bbox(record.get("bbox")),
        "caption_bbox": _bbox(record.get("caption_bbox")),
        "visual_summary": str(record.get("visual_summary") or "")[:6_000],
    }


def _bounded_paper_metadata(value: Any) -> dict[str, str]:
    metadata = value if isinstance(value, dict) else {}
    allowed_keys = ("author", "subject", "keywords", "creationDate", "modDate")
    return {
        key: str(metadata[key])[:1_000]
        for key in allowed_keys
        if metadata.get(key) not in (None, "")
    }


def _manifest_evidence_id(value: Any, prefix: str, index: int) -> str:
    candidate = str(value or "").upper()
    if _SAFE_EVIDENCE_ID.fullmatch(candidate) and candidate.startswith(prefix):
        return candidate
    return f"{prefix}{index:03d}"


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _bounded_snippet(snippet: EvidenceSnippet) -> dict[str, Any]:
    text = str(snippet.text or "")
    return {
        "id": str(snippet.id or "")[:16],
        "section": str(snippet.section or "Untitled")[:500],
        "page_start": max(0, int(snippet.page_start)),
        "page_end": max(0, int(snippet.page_end)),
        "text": text[:12_000],
        "kind": str(snippet.kind or "text").lower(),
        "text_truncated": len(text) > 12_000,
    }


def _bounded_table(table: TableBlock, index: int) -> dict[str, Any]:
    rows = table.rows if isinstance(table.rows, list) else []
    bounded_rows = [
        [str(cell)[:1_000] for cell in row[:12]]
        for row in rows[:40]
        if isinstance(row, list)
    ]
    return {
        "id": f"T{index:03d}",
        "page": int(table.page),
        "rows": bounded_rows,
        "rows_truncated": len(rows) > 40
        or any(isinstance(row, list) and len(row) > 12 for row in rows[:40]),
        "caption": str(table.caption or "")[:4_000],
        "bbox": table.bbox,
        "caption_bbox": table.caption_bbox,
    }


def _bounded_figure(figure: FigureBlock, index: int) -> dict[str, Any]:
    return {
        "id": f"F{index:03d}",
        "page": int(figure.page),
        "caption": str(figure.caption or "")[:4_000],
        "image_index": int(figure.image_index or 0),
        "bbox": figure.bbox,
        "caption_bbox": figure.caption_bbox,
        "visual_summary": str(figure.visual_summary or "")[:6_000],
    }


def _snippets(context: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in context.get("snippets", []):
        if not isinstance(item, dict):
            continue
        output.append(
            {
                "id": str(item.get("id") or "")[:16],
                "section": str(item.get("section") or "Untitled")[:500],
                "page_start": max(0, int(item.get("page_start", 0))),
                "page_end": max(0, int(item.get("page_end", 0))),
                "text": str(item.get("text") or "")[:_MAX_TOOL_TEXT],
                "kind": str(item.get("kind") or "text").lower(),
                "text_truncated": bool(item.get("text_truncated")),
            }
        )
    return output


def _query_terms(query: str) -> set[str]:
    terms: set[str] = set()
    for match in _WORD.findall(query.casefold()):
        terms.add(match)
        if re.fullmatch(r"[\u4e00-\u9fff]+", match):
            terms.update(match[index : index + 2] for index in range(len(match) - 1))
    return {term for term in terms if len(term) >= 2}


def _snippet_result(snippet: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    text = snippet["text"][:max_chars]
    start = snippet["page_start"] + 1
    end = snippet["page_end"] + 1
    return {
        "id": snippet["id"],
        "kind": snippet["kind"],
        "section": snippet["section"],
        "page": f"p.{start}" if start == end else f"pp.{start}-{end}",
        "text": text,
        "truncated": bool(snippet.get("text_truncated")) or len(snippet["text"]) > len(text),
    }


def _select_named_record(records: list[dict[str, Any]], selector: str | int, *, title_key: str) -> dict[str, Any]:
    if not records:
        raise PaperToolError("论文中没有可用章节。")
    if isinstance(selector, int) or str(selector).strip().isdigit():
        index = int(selector) - 1
        if 0 <= index < len(records):
            return records[index]
        raise PaperToolError("章节编号超出范围。")
    query = " ".join(str(selector).casefold().split())[:300]
    exact = [record for record in records if str(record.get(title_key) or "").casefold() == query]
    if exact:
        return exact[0]
    partial = [record for record in records if query in str(record.get(title_key) or "").casefold()]
    if len(partial) == 1:
        return partial[0]
    raise PaperToolError("章节名称不存在或不唯一，请使用完整标题或 1-based 编号。")


def _validated_evidence_id(value: str, *, prefix: str | None = None) -> str:
    evidence_id = str(value).upper().strip()
    if not _SAFE_EVIDENCE_ID.fullmatch(evidence_id):
        raise PaperToolError("证据区域 ID 无效。")
    if prefix and not evidence_id.startswith(prefix):
        raise PaperToolError(f"该工具仅接受 {prefix} 开头的证据 ID。")
    if evidence_id[0] not in {"F", "T"} and prefix is None:
        raise PaperToolError("视觉工具仅接受 Fxxx 或 Txxx。")
    return evidence_id


def _record_by_id(records: Any, evidence_id: str) -> dict[str, Any] | None:
    if not isinstance(records, list):
        return None
    return next(
        (record for record in records if isinstance(record, dict) and str(record.get("id")).upper() == evidence_id),
        None,
    )


def _evidence_by_id(context: dict[str, Any], evidence_id: str, *, kind: str) -> dict[str, Any]:
    snippet = next((item for item in _snippets(context) if item["id"].upper() == evidence_id), None)
    if snippet is None or snippet["kind"] != kind:
        raise PaperToolError("证据 ID 不存在。")
    return _snippet_result(snippet, max_chars=8000)


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return values if all(math.isfinite(item) for item in values) else None
