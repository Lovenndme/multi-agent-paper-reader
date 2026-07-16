"""FastAPI web app for the multi-agent paper reader."""

from __future__ import annotations

import ipaddress
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Iterable
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from agents.critic_agent import run_critic_agent, stream_critic_agent
from agents.comparison_agent import stream_comparison_agent
from agents.experiment_agent import run_experiment_agent, stream_experiment_agent
from agents.method_agent import run_method_agent, stream_method_agent
from agents.summary_agent import run_summary_agent, stream_summary_agent
from core.analysis_progress import AnalysisProgressTracker
from core.assessment import build_analysis_assessment
from core.chat import (
    PaperChatRequest,
    build_chat_prompt,
    demo_chat_reply,
    estimate_chat_tokens,
    hide_evidence_citations,
    resolve_chat_model_route,
    store_analysis_session,
    stream_chat_reply,
)
from core.chat_memory import (
    ConversationCreateRequest,
    ConversationUpdateRequest,
    add_conversation_message,
    create_conversation,
    delete_conversation,
    drain_memory_refreshes,
    get_conversation_summary,
    list_conversations,
    load_conversation,
    rename_conversation,
    schedule_conversation_title,
    schedule_memory_refresh,
)
from core.comparison import (
    ComparisonCreateRequest,
    build_comparison_assessment,
    build_comparison_evidence_catalog,
    demo_comparison_output,
    load_comparison_sources,
    sanitize_comparison_output,
)
from core.comparison_chat import (
    ComparisonChatRequest,
    build_comparison_chat_prompt,
    demo_comparison_chat_reply,
    stream_comparison_chat_reply,
)
from core.comparison_history import (
    ComparisonConversationCreateRequest,
    ComparisonConversationUpdateRequest,
    add_comparison_message,
    comparison_exists,
    create_comparison_conversation,
    delete_comparison,
    delete_comparison_conversation,
    get_comparison_conversation_summary,
    list_comparison_conversations,
    list_comparisons,
    load_comparison,
    load_comparison_conversation,
    rename_comparison_conversation,
    save_comparison,
    schedule_comparison_conversation_title,
)
from core.codex_sdk import CodexSDKError, close_codex_sdk_service, get_codex_sdk_service
from core.codex_tools import build_codex_paper_manifest, create_codex_tool_context
from core.evidence import build_evidence_index, evidence_context_for_agent, evidence_payload
from core.history import (
    delete_paper_history,
    list_paper_history,
    load_paper_analysis,
    paper_history_exists,
    save_paper_analysis,
)
from core.model_health import model_catalog_health
from core.pdf_parser import ParsedPaper, parse_pdf
from core.public_analysis import (
    public_agent_output,
    public_analysis_payload,
    sanitize_visible_text,
)
from core.model_providers import (
    provider_label,
    provider_spec,
    selected_text_model,
    selected_text_model_label,
    selected_text_mode,
    selected_vision_model,
    text_provider_id,
    vision_enabled,
    vision_provider_id,
)
from core.schemas import CriticOutput, ExperimentOutput, MethodOutput, SummaryOutput
from core.section_titles import clean_section_title
from core.settings import (
    PROJECT_VERSION,
    ApiKeySettingsRequest,
    ApiKeyValidationError,
    ModelRoutingSettingsRequest,
    ModelRoutingValidationError,
    ProviderApiKeySettingsRequest,
    application_settings_payload,
    configure_glm_api_key,
    configure_model_routing,
    configure_provider_api_key,
)
from core.vision import enrich_paper_figures_with_vision
from utils.llm import is_llm_configured, is_vision_configured


ROOT = Path(__file__).parent
FRONTEND_DIST = ROOT / "frontend-prototype" / "dist"
FRONTEND_BUILD_METADATA = "build-meta.json"
FRONTEND_REBUILD_COMMANDS = (
    "npm --prefix frontend-prototype ci；"
    "npm --prefix frontend-prototype run build"
)

load_dotenv(
    Path(os.environ.get("PAPER_READER_ENV_PATH", ROOT / ".env")),
    override=False,
)


@asynccontextmanager
async def _lifespan(_: FastAPI):
    yield
    drain_memory_refreshes(timeout=60.0)
    close_codex_sdk_service()

app = FastAPI(
    title="Multi-Agent Paper Reader",
    description="Upload a paper PDF and generate structured multi-agent reading notes.",
    version=PROJECT_VERSION,
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _frontend_build_status() -> dict[str, Any]:
    """Read frontend build metadata without preventing API-only startup."""
    index = FRONTEND_DIST / "index.html"
    metadata_path = FRONTEND_DIST / FRONTEND_BUILD_METADATA
    frontend_version: str | None = None
    metadata_error: str | None = None

    if not index.is_file():
        metadata_error = "前端构建不存在或不完整。"
    elif not metadata_path.is_file():
        metadata_error = "前端构建缺少版本元数据。"
    else:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidate = metadata.get("project_version")
            if metadata.get("schema_version") != 1:
                metadata_error = "前端构建版本元数据格式不受支持。"
            elif isinstance(candidate, str) and candidate.strip():
                frontend_version = candidate.strip()
            else:
                metadata_error = "前端构建版本元数据无效。"
        except (OSError, UnicodeError, json.JSONDecodeError):
            metadata_error = "前端构建版本元数据无法读取。"

    return {
        "dist_exists": FRONTEND_DIST.exists(),
        "index_exists": index.is_file(),
        "frontend_version": frontend_version,
        "frontend_version_match": bool(
            frontend_version and frontend_version == PROJECT_VERSION
        ),
        "metadata_error": metadata_error,
    }


def _frontend_build_error(status: dict[str, Any]) -> str:
    """Return an actionable Chinese error for an unusable frontend build."""
    if status["metadata_error"]:
        reason = status["metadata_error"]
    else:
        reason = (
            "前后端版本不一致："
            f"后端为 {PROJECT_VERSION}，前端为 {status['frontend_version']}。"
        )
    return (
        f"{reason} 为避免继续加载旧版界面，前端服务已暂停。"
        "请在项目根目录依次运行："
        f"{FRONTEND_REBUILD_COMMANDS}；然后重启服务。"
    )


def _section_payload(
    paper: ParsedPaper,
) -> list[dict[str, Any]]:
    return [
        {
            "title": section.title,
            "display_title": clean_section_title(section.title, index),
            "page_start": section.page_start,
            "page_end": section.page_end,
            "chars": len(section.content),
        }
        for index, section in enumerate(paper.sections)
    ]


def _paper_payload(
    paper: ParsedPaper,
    filename: str,
    file_size: int,
) -> dict[str, Any]:
    page_count = max((section.page_end for section in paper.sections), default=-1) + 1
    return {
        "title": paper.title,
        "filename": filename,
        "size_bytes": file_size,
        "pages": page_count,
        "sections_count": len(paper.sections),
        "sections": _section_payload(paper),
        "metadata": paper.metadata,
    }


def _demo_evidence(paper: ParsedPaper) -> list[dict[str, str]]:
    first_section = paper.sections[0].title if paper.sections else "Full Paper"
    return [
        {
            "id": "E001",
            "section": first_section,
            "page": "p.1",
            "quote": "Demo 模式未调用模型，仅用于验证上传、解析和渲染链路。",
            "note": "该证据说明当前输出是确定性的演示结果，不是论文内容判断。",
        }
    ]


def _demo_outputs(paper: ParsedPaper) -> dict[str, Any]:
    title = paper.title or "Uploaded Paper"
    demo_evidence = _demo_evidence(paper)
    method = MethodOutput(
        research_problem=(
            f"识别 {title} 所解决的核心研究问题及其主要技术路线。"
        ),
        proposed_method=(
            "Demo 模式已成功解析 PDF。配置当前文本模型的 API Key 后，可让真实 MethodAgent "
            "分析论文中与方法相关的章节。"
        ),
        key_components=[
            "PDF 解析器与章节路由",
            "MethodAgent 方法分析提示词",
            "ExperimentAgent 实验分析提示词",
            "CriticAgent 批判性评审提示词",
            "SummaryAgent 综合整理步骤",
        ],
        innovations=[
            "结构化的多 Agent 论文研读流程",
            "面向不同 Agent 的章节路由，减少无关上下文",
        ],
        differences_from_prior=(
            "该 Demo 结果用于证明前后端链路已经连通；在未运行真实 LLM 时，"
            "不会对论文的具体创新性作出判断。"
        ),
        implementation_details="在 Settings 中配置模型厂商与 API Key，即可运行真实结构化分析。",
        evidence=demo_evidence,
    )
    experiment = ExperimentOutput(
        datasets=["Demo 模式：真实数据集信息需要由 LLM 提取"],
        metrics=["Demo 模式：真实评估指标需要由 LLM 提取"],
        main_results=(
            "后端已收到并成功解析 PDF。配置模型凭证后即可提取真实实验结果。"
        ),
        comparison_with_baselines=(
            "Demo 模式不会虚构基线对比结果。"
        ),
        ablation_study=None,
        notable_findings=[
            f"解析器共识别出 {len(paper.sections)} 个章节。",
            f"提取出的正文约包含 {len(paper.full_text):,} 个字符。",
        ],
        evidence=demo_evidence,
    )
    critic = CriticOutput(
        novelty_score=3,
        novelty_justification=(
            "Demo 模式无法公正评估论文创新性，该占位结果仅用于验证完整响应结构。"
        ),
        strengths=[
            "上传、解析与响应序列化链路已经连通。",
            "前端能够渲染四个 Agent 的全部输出结构。",
        ],
        limitations=[
            "Demo 模式没有执行真实 LLM 评审。",
            "针对论文的具体批判性分析需要当前厂商的 API Key 和兼容模型。",
        ],
        potential_improvements=[
            "配置真实模型后重新运行分析。",
            "在生产环境中为超长论文增加分块处理。",
        ],
        broader_impact=None,
        evidence=demo_evidence,
    )
    summary = SummaryOutput(
        one_sentence_summary=(
            f"{title} 已成功上传并完成解析；配置真实 LLM Key 后可生成针对该论文的研读笔记。"
        ),
        core_contributions=[
            "前端上传的 PDF 已能到达 Python 后端。",
            "后端复用了项目现有的 PDF 解析器。",
            "API 返回与四 Agent 流程一致的结构化数据契约。",
            "配置模型凭证后即可运行真实 LangGraph 分析流程。",
        ],
        method_highlights=method.proposed_method,
        experiment_highlights=experiment.main_results,
        limitations_and_future_work=(
            "当前是确定性的 Demo 响应。在 Settings 中配置模型厂商与 API Key 后，"
            "即可运行真实多 Agent 分析。"
        ),
        reading_notes=(
            "Demo 模式可在不消耗模型 Token 的情况下验证部署、上传、解析和界面渲染。"
        ),
        evidence=demo_evidence,
    )
    snippets = build_evidence_index(paper)
    assessment = build_analysis_assessment(
        paper,
        snippets,
        method,
        experiment,
        critic,
        summary,
        demo=True,
    )
    return {
        "method_output": method.model_dump(),
        "experiment_output": experiment.model_dump(),
        "critic_output": critic.model_dump(),
        "summary_output": summary.model_dump(),
        "assessment": assessment.model_dump(),
    }


def _live_outputs(paper: ParsedPaper, pdf_path: Path | None = None) -> dict[str, Any]:
    if pdf_path is not None:
        enrich_paper_figures_with_vision(pdf_path, paper)
    snippets = build_evidence_index(paper)
    tool_context = (
        create_codex_tool_context(snippets=snippets, paper=paper, pdf_path=pdf_path)
        if text_provider_id() == "codex"
        else None
    )
    tool_path = tool_context.path if tool_context else None
    try:
        method = run_method_agent(
            evidence_context_for_agent(snippets, "method") or paper.get_sections_for_agent("method"),
            tool_context_path=tool_path,
        )
        experiment = run_experiment_agent(
            evidence_context_for_agent(snippets, "experiment") or paper.get_sections_for_agent("experiment"),
            tool_context_path=tool_path,
        )
        critic = run_critic_agent(
            evidence_context_for_agent(snippets, "critic") or paper.get_sections_for_agent("critic"),
            tool_context_path=tool_path,
        )
        summary = run_summary_agent(
            paper_title=paper.title,
            method_output=method,
            experiment_output=experiment,
            critic_output=critic,
            tool_context_path=tool_path,
        )
    finally:
        if tool_context:
            tool_context.close()
    assessment = build_analysis_assessment(
        paper,
        snippets,
        method,
        experiment,
        critic,
        summary,
    )
    return {
        "evidence_index": evidence_payload(snippets),
        "method_output": method.model_dump(),
        "experiment_output": experiment.model_dump(),
        "critic_output": critic.model_dump(),
        "summary_output": summary.model_dump(),
        "assessment": assessment.model_dump(),
    }


def _stream_event(event_type: str, **payload: Any) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


def _model_runtime_payload() -> dict[str, Any]:
    """Describe the active route so saved analyses remain reproducible."""
    text_provider = text_provider_id()
    visual_provider = vision_provider_id()
    payload = {
        "text_provider": text_provider,
        "text_provider_label": provider_label(text_provider),
        "text_model": selected_text_model(),
        "text_model_label": selected_text_model_label(),
        "text_mode": selected_text_mode(),
        "vision_enabled": vision_enabled(),
        "vision_provider": visual_provider,
        "vision_provider_label": provider_label(visual_provider),
        "vision_model": selected_vision_model(),
    }
    if text_provider == "codex":
        payload["codex_security_profile"] = (
            get_codex_sdk_service().status().get("security_profile") or {}
        )
    return payload


def _missing_model_key_message() -> str:
    provider_id = text_provider_id()
    if provider_spec(provider_id).credential_type == "codex_login":
        return "本机 Codex 尚未登录 ChatGPT，请在 Settings 中连接 Codex 订阅。"
    return f"{provider_label(provider_id)} API Key 未配置，请在 Settings 中添加当前文本模型所需的密钥。"


def _require_local_codex_request(request: Request) -> None:
    host = request.client.host if request.client else ""
    if host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(
            status_code=403,
            detail="Codex 订阅连接仅允许从运行服务的本机发起。",
        )
    request_host = request.headers.get("host", "").strip().lower()
    if host != "testclient" and not _is_loopback_http_host(request_host):
        raise HTTPException(
            status_code=403,
            detail="Codex 订阅连接仅接受 localhost Host，已拒绝潜在的 DNS 重绑定请求。",
        )
    origin = request.headers.get("origin", "").strip()
    if origin:
        parsed_origin = urlparse(origin)
        same_origin = (
            parsed_origin.scheme in {"http", "https"}
            and parsed_origin.netloc.lower() == request_host
        )
        if not same_origin:
            raise HTTPException(
                status_code=403,
                detail="Codex 订阅连接拒绝跨站请求。",
            )


def _is_loopback_http_host(value: str) -> bool:
    hostname = urlparse(f"//{value}").hostname
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


_AGENT_START_SUMMARIES = {
    "method": "已选取方法相关章节与证据，正在识别研究问题、方法组件和创新点。",
    "experiment": "已选取实验章节、表格和指标证据，正在核对数据集、基线与主要结果。",
    "critic": "已汇集论文主张和支撑证据，正在评估创新性、优点、局限与证据覆盖。",
    "summary": "已收到三个专业 Agent 的结构化结论，正在综合冲突与不确定性并生成最终笔记。",
}

_AGENT_COMPLETE_SUMMARIES = {
    "method": "方法分析已完成，研究问题、关键组件和创新点已整理。",
    "experiment": "实验分析已完成，数据集、指标、基线和结果已核对。",
    "critic": "批判性评审已完成，创新性、优点、局限和改进方向已整理。",
    "summary": "最终研读笔记已完成，已保留上游结论中的不确定性与冲突。",
}


def _run_streaming_agent(
    agent_id: str,
    stream_fn,
    paper_text: str,
    event_queue: Queue[str],
    tracker: AnalysisProgressTracker,
    tool_context_path: Path | None = None,
):
    progress_buffers: dict[str, str] = {}

    def on_progress(delta: str, progress_id: str) -> None:
        combined = f"{progress_buffers.get(progress_id, '')}{delta}"
        progress_buffers[progress_id] = combined
        visible_summary = sanitize_visible_text(combined)
        if not visible_summary:
            return
        payload = tracker.progress(
            agent_id,
            visible_summary,
            source="native_reasoning_summary",
            progress_id=progress_id,
        )
        event_queue.put(_stream_event("agent_progress", **payload))

    def on_activity(summary: str, progress_id: str) -> None:
        visible_summary = sanitize_visible_text(summary)
        if not visible_summary:
            return
        payload = tracker.progress(
            agent_id,
            visible_summary,
            source="tool_activity",
            progress_id=progress_id,
        )
        event_queue.put(_stream_event("agent_progress", **payload))

    return stream_fn(
        paper_text,
        on_token=None,
        on_progress=on_progress,
        on_activity=on_activity,
        tool_context_path=tool_context_path,
    )


def _drain_agent_events(event_queue: Queue[str]) -> Iterable[str]:
    while True:
        try:
            yield event_queue.get_nowait()
        except Empty:
            return


def _stream_summary_agent_events(
    paper_title: str,
    method_output: MethodOutput,
    experiment_output: ExperimentOutput,
    critic_output: CriticOutput,
    tracker: AnalysisProgressTracker,
    tool_context_path: Path | None = None,
) -> Iterable[tuple[str, SummaryOutput | None]]:
    event_queue: Queue[str] = Queue()
    progress_buffers: dict[str, str] = {}

    def on_progress(delta: str, progress_id: str) -> None:
        combined = f"{progress_buffers.get(progress_id, '')}{delta}"
        progress_buffers[progress_id] = combined
        visible_summary = sanitize_visible_text(combined)
        if not visible_summary:
            return
        payload = tracker.progress(
            "summary",
            visible_summary,
            source="native_reasoning_summary",
            progress_id=progress_id,
        )
        event_queue.put(_stream_event("agent_progress", **payload))

    def on_activity(summary: str, progress_id: str) -> None:
        visible_summary = sanitize_visible_text(summary)
        if not visible_summary:
            return
        payload = tracker.progress(
            "summary",
            visible_summary,
            source="tool_activity",
            progress_id=progress_id,
        )
        event_queue.put(_stream_event("agent_progress", **payload))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            stream_summary_agent,
            paper_title,
            method_output,
            experiment_output,
            critic_output,
            on_token=None,
            on_progress=on_progress,
            on_activity=on_activity,
            tool_context_path=tool_context_path,
        )
        while not future.done():
            for event in _drain_agent_events(event_queue):
                yield event, None
            time.sleep(0.05)

        for event in _drain_agent_events(event_queue):
            yield event, None
        yield "", future.result()


def _stream_demo_analysis(
    paper: ParsedPaper,
    filename: str,
    file_size: int,
    pdf_data: bytes,
    tracker: AnalysisProgressTracker | None = None,
) -> Iterable[str]:
    tracker = tracker or AnalysisProgressTracker()
    paper_payload = _paper_payload(paper, filename, file_size)
    snippets = build_evidence_index(paper)
    index_payload = evidence_payload(snippets)
    outputs = _demo_outputs(paper)
    yield _stream_event("paper", mode="demo", paper=paper_payload, message="PDF parsed")
    yield _stream_event(
        "agent_progress",
        **tracker.progress(
            "system",
            f"PDF 解析完成，共识别 {paper_payload['sections_count']} 个章节。",
            progress_id="paper-parsed",
        ),
    )
    yield _stream_event(
        "evidence_index",
        evidence_count=len(index_payload),
        message=f"Built {len(snippets)} evidence snippets",
    )
    yield _stream_event(
        "agent_progress",
        **tracker.progress(
            "system",
            f"已建立 {len(index_payload)} 个文本、表格或图像证据片段。",
            progress_id="evidence-index",
        ),
    )

    for agent_id, output_key in (
        ("method", "method_output"),
        ("experiment", "experiment_output"),
        ("critic", "critic_output"),
    ):
        yield _stream_event(
            "agent_started",
            **tracker.start_agent(agent_id, _AGENT_START_SUMMARIES[agent_id]),
        )
        time.sleep(0.12)
        yield _stream_event(
            "agent_progress",
            **tracker.progress(
                agent_id,
                "Demo 模式正在根据已解析的论文内容生成可验证的界面示例。",
                progress_id=f"{agent_id}-demo",
            ),
        )
        yield _stream_event(
            "agent_complete",
            **tracker.complete_agent(agent_id, _AGENT_COMPLETE_SUMMARIES[agent_id]),
            output_key=output_key,
            output=public_agent_output(outputs[output_key]),
        )

    yield _stream_event(
        "agent_started",
        **tracker.start_agent("summary", _AGENT_START_SUMMARIES["summary"]),
    )
    time.sleep(0.12)
    yield _stream_event(
        "agent_complete",
        **tracker.complete_agent("summary", _AGENT_COMPLETE_SUMMARIES["summary"]),
        output_key="summary_output",
        output=outputs["summary_output"],
    )
    analysis_process = tracker.finish()
    result_payload = {
        "mode": "demo",
        "analysis_id": None,
        "model_config": _model_runtime_payload(),
        "paper": paper_payload,
        "evidence_index": index_payload,
        "analysis_process": analysis_process,
        **outputs,
    }
    history_id: str | None = None
    try:
        history_id = save_paper_analysis(
            pdf_data=pdf_data,
            result=result_payload,
            snippets=snippets,
            paper_manifest=build_codex_paper_manifest(paper),
        )
    except Exception as exc:  # noqa: BLE001 - preserve analysis if storage fails
        yield _stream_event("history_error", message=f"Could not save paper history: {exc}")
    yield _stream_event(
        "complete",
        history_id=history_id,
        **public_analysis_payload(result_payload),
    )


def _stream_live_analysis(
    paper: ParsedPaper,
    filename: str,
    file_size: int,
    pdf_data: bytes,
    pdf_path: Path | None = None,
    tracker: AnalysisProgressTracker | None = None,
) -> Iterable[str]:
    tracker = tracker or AnalysisProgressTracker()
    paper_payload = _paper_payload(paper, filename, file_size)
    yield _stream_event("paper", mode="live", paper=paper_payload, message="PDF parsed")
    yield _stream_event(
        "agent_progress",
        **tracker.progress(
            "system",
            f"PDF 解析完成，共识别 {paper_payload['sections_count']} 个章节。",
            progress_id="paper-parsed",
        ),
    )
    if pdf_path is not None:
        yield _stream_event(
            "vision_started",
            message=f"Vision enrichment started for {len(paper.figures)} visual candidates",
        )
        yield _stream_event(
            "agent_progress",
            **tracker.progress(
                "system",
                f"正在检查 {len(paper.figures)} 个图表候选区域并补充视觉证据。",
                progress_id="vision",
            ),
        )
        try:
            vision_result = enrich_paper_figures_with_vision(pdf_path, paper)
            yield _stream_event(
                "vision_complete",
                total_figures=vision_result.total_figures,
                attempted=vision_result.attempted,
                enriched=vision_result.enriched,
                skipped=vision_result.skipped,
                errors=vision_result.errors,
                message=(
                    f"Vision enrichment complete: {vision_result.enriched}/"
                    f"{vision_result.total_figures} figures enriched"
                ),
            )
            yield _stream_event(
                "agent_progress",
                **tracker.progress(
                    "system",
                    f"视觉检查完成，已补充 {vision_result.enriched} 个图表摘要。",
                    progress_id="vision",
                ),
            )
        except Exception as exc:  # noqa: BLE001 - keep text/table analysis alive
            yield _stream_event("vision_error", message=f"Vision enrichment failed: {exc}")
            yield _stream_event(
                "agent_progress",
                **tracker.progress(
                    "system",
                    "视觉摘要不可用，分析将继续使用正文、表格和图注证据。",
                    progress_id="vision",
                ),
            )
    snippets = build_evidence_index(paper)
    index_payload = evidence_payload(snippets)
    tool_context = (
        create_codex_tool_context(
            snippets=snippets,
            paper=paper,
            pdf_path=pdf_path,
        )
        if text_provider_id() == "codex"
        else None
    )
    tool_path = tool_context.path if tool_context else None
    yield _stream_event(
        "evidence_index",
        evidence_count=len(index_payload),
        message=f"Built {len(snippets)} evidence snippets",
    )
    yield _stream_event(
        "agent_progress",
        **tracker.progress(
            "system",
            f"已建立 {len(index_payload)} 个文本、表格或图像证据片段。",
            progress_id="evidence-index",
        ),
    )

    agent_jobs = {
        "method": (
            "method_output",
            stream_method_agent,
            evidence_context_for_agent(snippets, "method") or paper.get_sections_for_agent("method"),
        ),
        "experiment": (
            "experiment_output",
            stream_experiment_agent,
            evidence_context_for_agent(snippets, "experiment") or paper.get_sections_for_agent("experiment"),
        ),
        "critic": (
            "critic_output",
            stream_critic_agent,
            evidence_context_for_agent(snippets, "critic") or paper.get_sections_for_agent("critic"),
        ),
    }

    outputs: dict[str, Any] = {}
    event_queue: Queue[str] = Queue()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for agent_id, (output_key, fn, paper_text) in agent_jobs.items():
            yield _stream_event(
                "agent_started",
                **tracker.start_agent(agent_id, _AGENT_START_SUMMARIES[agent_id]),
            )
            futures[
                executor.submit(
                    _run_streaming_agent,
                    agent_id,
                    fn,
                    paper_text,
                    event_queue,
                    tracker,
                    tool_path,
                )
            ] = (
                agent_id,
                output_key,
            )

        pending = set(futures)
        while pending:
            for event in _drain_agent_events(event_queue):
                yield event

            completed = [future for future in pending if future.done()]
            for future in completed:
                pending.remove(future)
                agent_id, output_key = futures[future]
                try:
                    agent_output = future.result()
                except Exception as exc:  # noqa: BLE001 - stream actionable UI error
                    failed_process = tracker.fail_agent(
                        agent_id,
                        f"{agent_id} 分析失败，无法生成可靠结果。",
                    )
                    yield _stream_event(
                        "error",
                        message=f"{agent_id} failed: {exc}",
                        **failed_process,
                        analysis_process=tracker.finish(status="failed"),
                    )
                    if tool_context:
                        tool_context.close()
                    return

                output_payload = agent_output.model_dump()
                outputs[output_key] = output_payload
                yield _stream_event(
                    "agent_complete",
                    **tracker.complete_agent(agent_id, _AGENT_COMPLETE_SUMMARIES[agent_id]),
                    output_key=output_key,
                    output=public_agent_output(output_payload),
                )

            if pending:
                time.sleep(0.05)

        for event in _drain_agent_events(event_queue):
            yield event

    yield _stream_event(
        "agent_started",
        **tracker.start_agent("summary", _AGENT_START_SUMMARIES["summary"]),
    )
    try:
        method_model = MethodOutput.model_validate(outputs["method_output"])
        experiment_model = ExperimentOutput.model_validate(outputs["experiment_output"])
        critic_model = CriticOutput.model_validate(outputs["critic_output"])
        summary_model: SummaryOutput | None = None
        for event, maybe_summary in _stream_summary_agent_events(
            paper.title,
            method_model,
            experiment_model,
            critic_model,
            tracker,
            tool_path,
        ):
            if event:
                yield event
            if maybe_summary:
                summary_model = maybe_summary
        if summary_model is None:
            raise RuntimeError("SummaryAgent finished without a parsed result.")
        summary_output = summary_model
    except Exception as exc:  # noqa: BLE001 - stream actionable UI error
        failed_process = tracker.fail_agent(
            "summary",
            "总结 Agent 失败，无法生成可靠的最终笔记。",
        )
        yield _stream_event(
            "error",
            message=f"summary failed: {exc}",
            **failed_process,
            analysis_process=tracker.finish(status="failed"),
        )
        if tool_context:
            tool_context.close()
        return

    if tool_context:
        tool_context.close()

    outputs["summary_output"] = summary_output.model_dump()
    outputs["assessment"] = build_analysis_assessment(
        paper,
        snippets,
        method_model,
        experiment_model,
        critic_model,
        summary_output,
    ).model_dump()
    analysis_id = store_analysis_session(
        snippets,
        {
            "mode": "live",
            "model_config": _model_runtime_payload(),
            "paper": paper_payload,
            "evidence_index": index_payload,
            **outputs,
        },
    )
    yield _stream_event(
        "agent_complete",
        **tracker.complete_agent("summary", _AGENT_COMPLETE_SUMMARIES["summary"]),
        output_key="summary_output",
        output=public_agent_output(outputs["summary_output"]),
    )
    analysis_process = tracker.finish()
    result_payload = {
        "mode": "live",
        "analysis_id": analysis_id,
        "model_config": _model_runtime_payload(),
        "paper": paper_payload,
        "evidence_index": index_payload,
        "analysis_process": analysis_process,
        **outputs,
    }
    history_id: str | None = None
    try:
        history_id = save_paper_analysis(
            pdf_data=pdf_data,
            result=result_payload,
            snippets=snippets,
            paper_manifest=build_codex_paper_manifest(paper),
        )
    except Exception as exc:  # noqa: BLE001 - preserve analysis if storage fails
        yield _stream_event("history_error", message=f"Could not save paper history: {exc}")
    yield _stream_event(
        "complete",
        history_id=history_id,
        **public_analysis_payload(result_payload),
    )


def _stream_analyze_response(
    filename: str,
    data: bytes,
    *,
    demo: bool,
) -> Iterable[str]:
    tracker = AnalysisProgressTracker()
    yield _stream_event("analysis_started", **tracker.started_payload())
    with tempfile.TemporaryDirectory(prefix="paper-reader-") as tmpdir:
        pdf_path = Path(tmpdir) / filename
        pdf_path.write_bytes(data)
        try:
            parsed = parse_pdf(pdf_path)
        except Exception as exc:  # noqa: BLE001 - stream parser details for UI
            yield _stream_event(
                "error",
                message=f"Could not parse PDF: {exc}",
                analysis_process=tracker.finish(status="failed"),
            )
            return

        if demo:
            yield from _stream_demo_analysis(parsed, filename, len(data), data, tracker)
        else:
            yield from _stream_live_analysis(parsed, filename, len(data), data, pdf_path, tracker)


def _stream_chat_response(request: PaperChatRequest, *, demo: bool) -> Iterable[str]:
    """Emit newline-delimited follow-up chat events."""
    conversation_id: str | None = request.conversation_id
    try:
        effective_request = request
        prompt = None
        user_message: dict[str, Any] | None = None
        assistant_message: dict[str, Any] | None = None
        answer_chunks: list[str] = []
        model_trace: dict[str, Any] | None = None if demo else {}

        if not demo and (request.conversation_id or request.history_id):
            if request.conversation_id:
                conversation = get_conversation_summary(request.conversation_id)
                if request.history_id and conversation["history_id"] != request.history_id:
                    raise ValueError("Conversation does not belong to the current paper.")
            elif request.history_id:
                conversation = create_conversation(request.history_id)
            else:  # pragma: no cover - guarded by the outer condition
                conversation = None
            conversation_id = conversation["id"] if conversation else None
            effective_request = request.model_copy(update={"conversation_id": conversation_id})
            prompt = build_chat_prompt(effective_request)
            user_message = add_conversation_message(
                conversation_id,
                role="user",
                content=request.question,
                quote=request.selected_text,
            )

        if demo:
            reply = demo_chat_reply(effective_request)
            for index in range(0, len(reply), 28):
                token = reply[index : index + 28]
                answer_chunks.append(token)
                yield _stream_event("token", text=token)
                time.sleep(0.015)
        else:
            for token in stream_chat_reply(
                effective_request,
                messages=prompt.messages if prompt else None,
                trace=model_trace,
            ):
                answer_chunks.append(token)
                yield _stream_event("token", text=token)

        visible_answer = hide_evidence_citations("".join(answer_chunks))
        if conversation_id and visible_answer:
            memory_provider, memory_model, memory_mode = resolve_chat_model_route(effective_request)
            assistant_message = add_conversation_message(
                conversation_id,
                role="assistant",
                content=visible_answer,
                model_trace=model_trace,
            )
            memory_refresh_scheduled = schedule_memory_refresh(
                conversation_id,
                context_token_count=(
                    (prompt.stats.estimated_input_tokens if prompt else 0)
                    + estimate_chat_tokens(visible_answer)
                ),
                text_provider=memory_provider,
                text_model=memory_model,
                text_mode=memory_mode,
            )
            title_generation_scheduled = bool(
                user_message
                and user_message.get("title_generation_eligible")
                and schedule_conversation_title(
                    conversation_id,
                    request.question,
                    expected_title=str(user_message.get("provisional_title") or ""),
                )
            )
            conversation = get_conversation_summary(conversation_id)
        else:
            memory_refresh_scheduled = False
            title_generation_scheduled = False
            conversation = None
        yield _stream_event(
            "complete",
            provider=(model_trace or {}).get("provider") or text_provider_id(),
            model=(model_trace or {}).get("requested_model") or selected_text_model(),
            model_trace=model_trace,
            external_sources=(model_trace or {}).get("external_sources") or [],
            conversation_id=conversation_id,
            conversation=conversation,
            user_message=user_message,
            assistant_message=assistant_message,
            memory_refresh_scheduled=memory_refresh_scheduled,
            title_generation_scheduled=title_generation_scheduled,
            prompt_stats=prompt.stats.__dict__ if prompt else None,
        )
    except Exception as exc:  # noqa: BLE001 - stream actionable chat errors
        yield _stream_event(
            "error",
            message=f"追问失败：{exc}",
            conversation_id=conversation_id,
        )


def _stream_comparison_response(
    request: ComparisonCreateRequest,
    *,
    demo: bool,
) -> Iterable[str]:
    """Emit an evidence-grounded comparison and persist the completed workspace."""
    try:
        sources = load_comparison_sources(request.history_ids)
        yield _stream_event(
            "comparison_started",
            focus=request.focus,
            paper_count=len(sources),
            message="正在读取历史论文与完整证据",
        )
        for source in sources:
            yield _stream_event(
                "paper_loaded",
                label=source.label,
                history_id=source.history_id,
                title=source.title,
                evidence_count=len(source.snippets),
                message=f"{source.label} 已载入",
            )

        if demo:
            output = demo_comparison_output(sources, request)
            serialized = output.model_dump_json()
            for index in range(0, len(serialized), 100):
                yield _stream_event("comparison_token", text=serialized[index : index + 100])
                time.sleep(0.01)
        else:
            event_queue: Queue[str] = Queue()

            def on_token(token: str) -> None:
                event_queue.put(_stream_event("comparison_token", text=token))

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(stream_comparison_agent, sources, request, on_token)
                while not future.done():
                    for event in _drain_agent_events(event_queue):
                        yield event
                    time.sleep(0.05)
                for event in _drain_agent_events(event_queue):
                    yield event
                output = future.result()

        sanitized = sanitize_comparison_output(output, sources, request)
        assessment = build_comparison_assessment(sanitized)
        result = {
            "mode": "demo" if demo else "live",
            "comparison": sanitized.model_dump(),
            "assessment": assessment.model_dump(),
            "evidence_catalog": build_comparison_evidence_catalog(sanitized, sources),
        }
        comparison_id = save_comparison(
            result=result,
            sources=sources,
            request=request,
        )
        yield _stream_event(
            "complete",
            comparison_id=comparison_id,
            **result,
        )
    except Exception as exc:  # noqa: BLE001 - stream actionable comparison errors
        yield _stream_event("error", message=f"多论文对比失败：{exc}")


def _stream_comparison_chat_response(
    request: ComparisonChatRequest,
    *,
    demo: bool,
) -> Iterable[str]:
    """Stream and persist one cross-paper follow-up answer."""
    conversation_id = request.conversation_id
    try:
        if conversation_id:
            conversation = get_comparison_conversation_summary(conversation_id)
            if conversation["comparison_id"] != request.comparison_id:
                raise ValueError("Conversation does not belong to the current comparison.")
        else:
            conversation = create_comparison_conversation(request.comparison_id)
            conversation_id = conversation["id"]
        effective_request = request.model_copy(update={"conversation_id": conversation_id})
        prompt = None if demo else build_comparison_chat_prompt(effective_request)
        model_trace: dict[str, Any] | None = None if demo else {}
        user_message = add_comparison_message(
            conversation_id,
            role="user",
            content=request.question,
            quote=request.selected_text,
        )
        answer_chunks: list[str] = []
        if demo:
            reply = demo_comparison_chat_reply(effective_request)
            for index in range(0, len(reply), 28):
                token = reply[index : index + 28]
                answer_chunks.append(token)
                yield _stream_event("token", text=token)
                time.sleep(0.015)
        else:
            for token in stream_comparison_chat_reply(
                effective_request,
                messages=prompt.messages if prompt else None,
                trace=model_trace,
            ):
                answer_chunks.append(token)
                yield _stream_event("token", text=token)
        assistant_message = add_comparison_message(
            conversation_id,
            role="assistant",
            content="".join(answer_chunks),
            model_trace=model_trace,
        )
        title_generation_scheduled = bool(
            not demo
            and user_message.get("title_generation_eligible")
            and schedule_comparison_conversation_title(
                conversation_id,
                request.question,
                expected_title=str(user_message.get("provisional_title") or ""),
            )
        )
        yield _stream_event(
            "complete",
            provider=(model_trace or {}).get("provider") or text_provider_id(),
            model=(model_trace or {}).get("requested_model") or selected_text_model(),
            model_trace=model_trace,
            external_sources=(model_trace or {}).get("external_sources") or [],
            conversation_id=conversation_id,
            conversation=get_comparison_conversation_summary(conversation_id),
            user_message=user_message,
            assistant_message=assistant_message,
            title_generation_scheduled=title_generation_scheduled,
            prompt_stats=prompt.stats.__dict__ if prompt else None,
        )
    except Exception as exc:  # noqa: BLE001 - stream actionable comparison chat errors
        yield _stream_event(
            "error",
            message=f"跨论文追问失败：{exc}",
            conversation_id=conversation_id,
        )


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Report whether the backend and live LLM configuration are available."""
    frontend_status = _frontend_build_status()
    return {
        "ok": frontend_status["frontend_version_match"],
        "version": PROJECT_VERSION,
        "frontend_dist": frontend_status["dist_exists"],
        "frontend_version": frontend_status["frontend_version"],
        "frontend_version_match": frontend_status["frontend_version_match"],
        "frontend_error": (
            None
            if frontend_status["frontend_version_match"]
            else _frontend_build_error(frontend_status)
        ),
        "llm_configured": is_llm_configured(),
        "provider": text_provider_id(),
        "model": selected_text_model(),
        "model_label": selected_text_model_label(),
        "vision_configured": is_vision_configured(),
        "vision_provider": vision_provider_id(),
        "vision_model": selected_vision_model(),
    }


@app.get("/api/settings")
def application_settings() -> dict[str, Any]:
    """Return public model and version information for the settings dialog."""
    return application_settings_payload()


@app.get("/api/settings/model-health")
def application_model_health(
    force: bool = Query(default=False),
) -> dict[str, Any]:
    """Check configured provider catalogs without exposing credentials."""
    return model_catalog_health(force=force)


@app.get("/api/settings/codex/status")
def codex_subscription_status(
    force: bool = Query(default=False),
) -> dict[str, Any]:
    """Return credential-safe local Codex account and model metadata."""
    service = get_codex_sdk_service()
    status = service.status(force=force)
    models = ()
    if status.get("authenticated"):
        try:
            models = service.models(force=force)
        except CodexSDKError:
            models = ()
        # model/list records its own safe ready/empty/error state.
        status = service.status()
    return {
        "ok": bool(
            status.get("runtime_ready")
            and (
                not status.get("authenticated")
                or status.get("model_catalog_ready")
            )
        ),
        "status": status,
        "models": [
            {
                "id": model.id,
                "label": model.label,
                "description": model.description,
                "recommended": model.recommended,
                "supports_image": model.supports_image,
                "default_effort": model.default_effort,
                "efforts": [
                    {
                        "id": effort,
                        "description": next(
                            (text for item, text in model.efforts if item == effort),
                            "",
                        ),
                        "available": any(item == effort for item, _ in model.efforts),
                        "disabled_reason": (
                            None
                            if any(item == effort for item, _ in model.efforts)
                            else f"{model.label} 当前不支持该推理强度。"
                        ),
                        "execution_kind": "multi_agent" if effort == "ultra" else "single_agent",
                    }
                    for effort in ("low", "medium", "high", "xhigh", "max", "ultra")
                ],
            }
            for model in models
        ],
    }


@app.post("/api/settings/codex/login")
def start_codex_subscription_login(request: Request) -> dict[str, Any]:
    """Start the SDK browser flow only for a browser reaching this local host."""
    _require_local_codex_request(request)
    service = get_codex_sdk_service()
    status = service.status(force=True)
    if status.get("authenticated"):
        return {"ok": True, "already_authenticated": True, "status": status}
    try:
        login = service.start_chatgpt_login()
    except CodexSDKError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "already_authenticated": False, **login}


@app.post("/api/settings/codex/login/device")
def start_codex_subscription_device_login(request: Request) -> dict[str, Any]:
    """Start the official device-code fallback from this local application only."""
    _require_local_codex_request(request)
    service = get_codex_sdk_service()
    status = service.status(force=True)
    if status.get("authenticated"):
        return {"ok": True, "already_authenticated": True, "status": status}
    try:
        login = service.start_chatgpt_device_login()
    except CodexSDKError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "already_authenticated": False, **login}


@app.delete("/api/settings/codex/session")
def logout_codex_subscription(request: Request) -> dict[str, Any]:
    """Clear the official local Codex login after an explicit same-origin request."""
    _require_local_codex_request(request)
    service = get_codex_sdk_service()
    try:
        service.logout()
    except CodexSDKError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "status": service.status(force=True)}


@app.get("/api/settings/codex/login/{login_id}")
def codex_subscription_login_state(login_id: str, request: Request) -> dict[str, Any]:
    """Poll one SDK-managed login without exposing account credentials."""
    _require_local_codex_request(request)
    if not login_id or len(login_id) > 100:
        raise HTTPException(status_code=404, detail="Codex 登录任务不存在。")
    service = get_codex_sdk_service()
    state = service.login_state(login_id)
    payload: dict[str, Any] = {"ok": state.get("status") != "unknown", **state}
    if state.get("status") == "success":
        payload["account"] = service.status(force=True)
    return payload


@app.post("/api/settings/api-key")
def update_application_api_key(request: ApiKeySettingsRequest) -> dict[str, Any]:
    """Validate and persist a GLM key without ever returning the secret."""
    try:
        settings = configure_glm_api_key(request.api_key.get_secret_value())
    except ApiKeyValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, "settings": settings}


@app.post("/api/settings/providers/{provider_id}/api-key")
def update_provider_api_key(
    provider_id: str,
    request: ProviderApiKeySettingsRequest,
) -> dict[str, Any]:
    """Validate and persist one provider key without returning secret material."""
    try:
        settings = configure_provider_api_key(
            provider_id,
            request.api_key.get_secret_value(),
            base_url=request.base_url,
            protocol=request.protocol,
            provider_name=request.provider_name,
            text_model=request.text_model,
            vision_model=request.vision_model,
        )
    except ApiKeyValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, "settings": settings}


@app.put("/api/settings/routing")
def update_model_routing(request: ModelRoutingSettingsRequest) -> dict[str, Any]:
    """Persist active text and vision routes and activate them immediately."""
    try:
        settings = configure_model_routing(request)
    except ModelRoutingValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ok": True, "settings": settings}


@app.get("/api/comparisons")
def comparison_history(
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """List persisted multi-paper comparison workspaces."""
    return {"items": list_comparisons(limit=limit)}


@app.post("/api/comparisons/stream")
def compare_saved_papers(
    request: ComparisonCreateRequest,
    demo: bool = Query(default=False, description="Return deterministic comparison output."),
) -> StreamingResponse:
    """Compare two to four saved papers and stream the structured result."""
    if not demo and not is_llm_configured():
        raise HTTPException(status_code=503, detail=_missing_model_key_message())
    return StreamingResponse(
        _stream_comparison_response(request, demo=demo),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/comparisons/chat/stream")
def chat_with_comparison(
    request: ComparisonChatRequest,
    demo: bool = Query(default=False, description="Return a deterministic comparison reply."),
) -> StreamingResponse:
    """Continue a persisted, evidence-grounded cross-paper conversation."""
    if not demo and not is_llm_configured():
        raise HTTPException(status_code=503, detail=_missing_model_key_message())
    if not comparison_exists(request.comparison_id):
        raise HTTPException(status_code=404, detail="Comparison workspace was not found.")
    return StreamingResponse(
        _stream_comparison_chat_response(request, demo=demo),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/comparisons/chat/conversations/{conversation_id}")
def comparison_conversation_detail(conversation_id: str) -> dict[str, Any]:
    try:
        return load_comparison_conversation(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


@app.patch("/api/comparisons/chat/conversations/{conversation_id}")
def update_comparison_conversation(
    conversation_id: str,
    request: ComparisonConversationUpdateRequest,
) -> dict[str, Any]:
    try:
        return {"conversation": rename_comparison_conversation(conversation_id, request.title)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/comparisons/chat/conversations/{conversation_id}")
def remove_comparison_conversation(conversation_id: str) -> dict[str, bool]:
    if not delete_comparison_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation was not found.")
    return {"ok": True}


@app.get("/api/comparisons/{comparison_id}/conversations")
def comparison_conversations(comparison_id: str) -> dict[str, Any]:
    if not comparison_exists(comparison_id):
        raise HTTPException(status_code=404, detail="Comparison workspace was not found.")
    return {"items": list_comparison_conversations(comparison_id)}


@app.post("/api/comparisons/{comparison_id}/conversations")
def create_comparison_chat_conversation(
    comparison_id: str,
    request: ComparisonConversationCreateRequest,
) -> dict[str, Any]:
    try:
        return {
            "conversation": create_comparison_conversation(
                comparison_id,
                title=request.title,
            )
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


@app.get("/api/comparisons/{comparison_id}")
def comparison_detail(comparison_id: str) -> dict[str, Any]:
    stored = load_comparison(comparison_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Comparison workspace was not found.")
    return {
        "comparison_id": comparison_id,
        **stored["result"],
        "workspace": stored["workspace"],
        "papers": stored["papers"],
    }


@app.delete("/api/comparisons/{comparison_id}")
def remove_comparison(comparison_id: str) -> dict[str, bool]:
    if not delete_comparison(comparison_id):
        raise HTTPException(status_code=404, detail="Comparison workspace was not found.")
    return {"ok": True}


@app.get("/api/history")
def paper_history(
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    """List locally persisted paper analyses, newest first."""
    return {"items": list_paper_history(limit=limit)}


@app.get("/api/history/{history_id}/conversations")
def paper_chat_conversations(history_id: str) -> dict[str, Any]:
    """List persisted follow-up conversations for one saved paper."""
    if not paper_history_exists(history_id):
        raise HTTPException(status_code=404, detail="Saved paper analysis was not found.")
    return {"items": list_conversations(history_id)}


@app.post("/api/history/{history_id}/conversations")
def create_paper_chat_conversation(
    history_id: str,
    request: ConversationCreateRequest,
) -> dict[str, Any]:
    """Start an independent follow-up conversation for one paper."""
    try:
        return {"conversation": create_conversation(history_id, title=request.title)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


@app.get("/api/chat/conversations/{conversation_id}")
def conversation_detail(conversation_id: str) -> dict[str, Any]:
    """Restore complete original messages for one conversation."""
    try:
        return load_conversation(conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc


@app.patch("/api/chat/conversations/{conversation_id}")
def update_conversation(
    conversation_id: str,
    request: ConversationUpdateRequest,
) -> dict[str, Any]:
    """Rename one follow-up conversation."""
    try:
        return {"conversation": rename_conversation(conversation_id, request.title)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/chat/conversations/{conversation_id}")
def remove_conversation(conversation_id: str) -> dict[str, bool]:
    """Delete one conversation without affecting the saved paper."""
    if not delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation was not found.")
    return {"ok": True}


@app.get("/api/history/{history_id}")
def history_analysis(history_id: str) -> dict[str, Any]:
    """Restore one saved analysis and recreate its grounded chat session."""
    stored = load_paper_analysis(history_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Saved paper analysis was not found.")

    result = dict(stored["result"])
    snippets = stored["snippets"]
    analysis_id: str | None = None
    if result.get("mode") == "live" and snippets:
        analysis_id = store_analysis_session(snippets, result)
    result["analysis_id"] = analysis_id
    result["history_id"] = history_id
    result["history_item"] = stored["history"]
    return public_analysis_payload(result)


@app.delete("/api/history/{history_id}")
def remove_history_analysis(history_id: str) -> dict[str, bool]:
    """Delete one saved analysis and its retained PDF."""
    if not delete_paper_history(history_id):
        raise HTTPException(status_code=404, detail="Saved paper analysis was not found.")
    return {"ok": True}


@app.post("/api/papers/preview")
async def preview_paper(file: UploadFile = File(...)) -> dict[str, Any]:
    """Parse PDF metadata and sections without running agents or model calls."""
    filename = Path(file.filename or "paper.pdf").name or "paper.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    with tempfile.TemporaryDirectory(prefix="paper-reader-preview-") as tmpdir:
        pdf_path = Path(tmpdir) / filename
        pdf_path.write_bytes(data)
        try:
            parsed = parse_pdf(pdf_path, layout=False)
        except Exception as exc:  # noqa: BLE001 - return actionable parser details
            raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}") from exc

    return {"paper": _paper_payload(parsed, filename, len(data))}


@app.post("/api/analyze")
async def analyze_paper(
    file: UploadFile = File(...),
    demo: bool = Query(default=False, description="Return deterministic demo output."),
) -> dict[str, Any]:
    """Upload a PDF, parse it, and run the paper-reading pipeline."""
    filename = Path(file.filename or "paper.pdf").name or "paper.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    with tempfile.TemporaryDirectory(prefix="paper-reader-") as tmpdir:
        pdf_path = Path(tmpdir) / filename
        pdf_path.write_bytes(data)
        try:
            parsed = parse_pdf(pdf_path)
        except Exception as exc:  # noqa: BLE001 - preserve useful parser details for UI
            raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}") from exc

        if demo:
            outputs = _demo_outputs(parsed)
            mode = "demo"
        else:
            if not is_llm_configured():
                raise HTTPException(
                    status_code=503,
                    detail=_missing_model_key_message(),
                )
            try:
                outputs = _live_outputs(parsed, pdf_path)
            except Exception as exc:  # noqa: BLE001 - return actionable UI error
                raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
            mode = "live"

    paper_payload = _paper_payload(parsed, filename, len(data))
    snippets = build_evidence_index(parsed)
    index_payload = outputs.get("evidence_index")
    if not isinstance(index_payload, list):
        index_payload = evidence_payload(snippets)
        outputs["evidence_index"] = index_payload
    analysis_id: str | None = None
    if mode == "live":
        analysis_id = store_analysis_session(
            snippets,
            {
                "mode": mode,
                "model_config": _model_runtime_payload(),
                "paper": paper_payload,
                **outputs,
            },
        )

    result_payload = {
        "mode": mode,
        "analysis_id": analysis_id,
        "model_config": _model_runtime_payload(),
        "paper": paper_payload,
        **outputs,
    }
    try:
        result_payload["history_id"] = save_paper_analysis(
            pdf_data=data,
            result=result_payload,
            snippets=snippets,
            paper_manifest=build_codex_paper_manifest(parsed),
        )
    except Exception as exc:  # noqa: BLE001 - return analysis with an actionable warning
        result_payload["history_id"] = None
        result_payload["history_warning"] = f"Could not save paper history: {exc}"
    return public_analysis_payload(result_payload)


@app.post("/api/analyze/stream")
async def analyze_paper_stream(
    file: UploadFile = File(...),
    demo: bool = Query(default=False, description="Return deterministic demo output."),
) -> StreamingResponse:
    """Upload a PDF and stream parsing, agent, and final analysis events."""
    filename = Path(file.filename or "paper.pdf").name or "paper.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")
    if not demo and not is_llm_configured():
        raise HTTPException(
            status_code=503,
            detail=_missing_model_key_message(),
        )

    return StreamingResponse(
        _stream_analyze_response(filename, data, demo=demo),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/stream")
def chat_with_paper(
    request: PaperChatRequest,
    demo: bool = Query(default=False, description="Return a deterministic demo reply."),
) -> StreamingResponse:
    """Continue the paper-reading conversation with analysis-grounded context."""
    if not demo and not is_llm_configured():
        raise HTTPException(
            status_code=503,
            detail=_missing_model_key_message(),
        )
    return StreamingResponse(
        _stream_chat_response(request, demo=demo),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def serve_frontend(path: str) -> FileResponse:
    """Serve the built React app when `frontend-prototype/dist` exists."""
    index = FRONTEND_DIST / "index.html"
    frontend_status = _frontend_build_status()
    if not frontend_status["frontend_version_match"]:
        raise HTTPException(
            status_code=503,
            detail=_frontend_build_error(frontend_status),
            headers={"Cache-Control": "no-store"},
        )
    requested = (FRONTEND_DIST / path).resolve()
    if path and requested.is_file() and FRONTEND_DIST.resolve() in requested.parents:
        if requested == index.resolve():
            return FileResponse(index, headers={"Cache-Control": "no-cache"})
        return FileResponse(requested)
    return FileResponse(index, headers={"Cache-Control": "no-cache"})
