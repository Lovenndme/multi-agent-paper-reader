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

from agents.comparison_agent import stream_comparison_agent
from core.analysis_events import AnalysisEvent, AnalysisOrchestratorError, AnalysisRequest
from core.analysis_orchestrator import (
    build_paper_payload,
    get_paper_analysis_orchestrator,
    missing_model_key_message,
)
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
from core.history import (
    delete_paper_history,
    list_paper_history,
    load_paper_analysis,
    paper_history_exists,
)
from core.model_health import model_catalog_health
from core.pdf_parser import parse_pdf
from core.public_analysis import public_analysis_payload
from core.model_providers import (
    selected_text_model,
    selected_text_model_label,
    selected_vision_model,
    text_provider_id,
    vision_provider_id,
)
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
ANALYSIS_ORCHESTRATOR = get_paper_analysis_orchestrator()


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


def _stream_event(event_type: str, **payload: Any) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=False) + "\n"


def _missing_model_key_message() -> str:
    return missing_model_key_message()


def _analysis_http_exception(error: AnalysisOrchestratorError) -> HTTPException:
    status_code = {
        "request": 400,
        "parse": 422,
        "configuration": 503,
    }.get(error.category, 500)
    return HTTPException(status_code=status_code, detail=error.message)


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


def _drain_agent_events(event_queue: Queue[str]) -> Iterable[str]:
    while True:
        try:
            yield event_queue.get_nowait()
        except Empty:
            return


def _serialize_analysis_events(events: Iterable[AnalysisEvent]) -> Iterable[str]:
    for event in events:
        yield _stream_event(event.type, **event.payload)


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

    return {"paper": build_paper_payload(parsed, filename, len(data))}


@app.post("/api/analyze")
async def analyze_paper(
    file: UploadFile = File(...),
    demo: bool = Query(default=False, description="Return deterministic demo output."),
) -> dict[str, Any]:
    """Run one complete paper-analysis task through the Orchestrator."""
    filename = Path(file.filename or "paper.pdf").name or "paper.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    try:
        result = ANALYSIS_ORCHESTRATOR.run(
            AnalysisRequest(
                filename=filename,
                pdf_data=data,
                demo=demo,
            )
        )
    except AnalysisOrchestratorError as exc:
        raise _analysis_http_exception(exc) from exc
    return result.as_dict()


@app.post("/api/analyze/stream")
async def analyze_paper_stream(
    file: UploadFile = File(...),
    demo: bool = Query(default=False, description="Return deterministic demo output."),
) -> StreamingResponse:
    """Stream transport-neutral Orchestrator events as newline-delimited JSON."""
    filename = Path(file.filename or "paper.pdf").name or "paper.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    analysis_request = AnalysisRequest(
        filename=filename,
        pdf_data=data,
        demo=demo,
    )
    try:
        ANALYSIS_ORCHESTRATOR.validate(analysis_request)
    except AnalysisOrchestratorError as exc:
        raise _analysis_http_exception(exc) from exc

    return StreamingResponse(
        _serialize_analysis_events(
            ANALYSIS_ORCHESTRATOR.stream(analysis_request)
        ),
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
