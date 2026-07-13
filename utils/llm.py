"""LLM call wrapper. Switch models via .env configuration."""

import base64
import json
import os
import re
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any, TypeVar
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from core.model_providers import (
    model_display_label,
    provider_api_key,
    provider_base_url,
    provider_spec,
    selected_text_model,
    selected_vision_model,
    text_provider_id,
    vision_enabled,
    vision_provider_id,
)

_env_path = Path(
    os.environ.get(
        "PAPER_READER_ENV_PATH",
        Path(__file__).parent.parent / ".env",
    )
)
load_dotenv(dotenv_path=_env_path, override=False)

# Temperature env var lets users override without code changes.
# Default 1.0 is required by some providers (e.g. kimi-k2.5).
_DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "1.0"))
SchemaT = TypeVar("SchemaT", bound=BaseModel)


def get_api_key(provider_id: str | None = None) -> str | None:
    """Return the credential for one provider without exposing other keys."""
    return provider_api_key(provider_id or text_provider_id())


def get_base_url(provider_id: str | None = None) -> str:
    """Return the OpenAI-compatible base URL for one provider."""
    return provider_base_url(provider_id or text_provider_id())


def is_llm_configured() -> bool:
    """Return whether an API key is available for live analysis."""
    return bool(get_api_key())


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return a cached ChatOpenAI instance configured from environment variables."""
    provider_id = text_provider_id()
    api_key = get_api_key(provider_id)
    if not api_key:
        raise EnvironmentError(_missing_key_message(provider_id))

    return ChatOpenAI(
        **_client_kwargs(
            provider_id=provider_id,
            model=selected_text_model(),
            api_key=api_key,
            temperature=_DEFAULT_TEMPERATURE,
            timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "240")),
            max_retries=3,
        )
    )


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatOpenAI:
    """Return the same text model with a lower temperature for grounded paper QA."""
    provider_id = text_provider_id()
    api_key = get_api_key(provider_id)
    if not api_key:
        raise EnvironmentError(_missing_key_message(provider_id))

    return ChatOpenAI(
        **_client_kwargs(
            provider_id=provider_id,
            model=selected_text_model(),
            api_key=api_key,
            temperature=float(os.environ.get("CHAT_TEMPERATURE", "0.25")),
            timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "240")),
            max_retries=3,
        )
    )


@lru_cache(maxsize=1)
def get_vision_llm() -> ChatOpenAI:
    """Return a cached vision-capable ChatOpenAI-compatible client."""
    provider_id = vision_provider_id()
    api_key = get_api_key(provider_id)
    if not api_key:
        raise EnvironmentError(_missing_key_message(provider_id))

    return ChatOpenAI(
        **_client_kwargs(
            provider_id=provider_id,
            model=selected_vision_model(),
            api_key=api_key,
            temperature=float(os.environ.get("VISION_TEMPERATURE", "0.2")),
            timeout=float(os.environ.get("VISION_TIMEOUT_SECONDS", "180")),
            max_retries=2,
        )
    )


def is_vision_configured() -> bool:
    """Return whether the runtime has enough config to call a vision model."""
    provider_id = vision_provider_id()
    return (
        vision_enabled()
        and bool(provider_spec(provider_id).vision_models)
        and bool(get_api_key(provider_id))
        and bool(selected_vision_model())
    )


def _client_kwargs(
    *,
    provider_id: str,
    model: str,
    api_key: str,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "base_url": get_base_url(provider_id),
        "include_response_headers": True,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    # Current GPT-5-family endpoints may reject sampling parameters depending
    # on the selected reasoning mode. Omitting temperature works for all modes.
    if not (provider_id == "openai" and model.startswith("gpt-5")):
        kwargs["temperature"] = temperature
    return kwargs


def _missing_key_message(provider_id: str) -> str:
    spec = provider_spec(provider_id)
    return (
        f"{spec.label} API Key 未配置。请在 Settings 中为当前厂商添加密钥，"
        f"或在 .env 中设置 {spec.api_key_env}。"
    )


def reset_llm_clients() -> None:
    """Discard cached clients after runtime credentials change."""
    get_llm.cache_clear()
    get_chat_llm.cache_clear()
    get_vision_llm.cache_clear()


def start_text_model_call_trace(llm: ChatOpenAI) -> dict[str, Any]:
    """Snapshot the actual cached client used for one text-model request."""
    provider_id = text_provider_id()
    requested_model = str(getattr(llm, "model_name", "") or selected_text_model())
    base_url = str(getattr(llm, "openai_api_base", "") or get_base_url(provider_id))
    return {
        "provider": provider_id,
        "provider_label": provider_spec(provider_id).label,
        "requested_model": requested_model,
        "requested_model_label": model_display_label(provider_id, "text", requested_model),
        "endpoint_host": urlparse(base_url).netloc,
        "upstream_model": None,
        "request_id": None,
        "verification": "route_recorded",
    }


def update_text_model_call_trace(trace: dict[str, Any], response: Any) -> None:
    """Merge upstream response metadata into a request trace without secrets."""
    metadata = dict(getattr(response, "response_metadata", {}) or {})
    upstream_model = metadata.get("model_name") or metadata.get("model")
    if upstream_model:
        trace["upstream_model"] = str(upstream_model)

    headers = metadata.get("headers")
    if isinstance(headers, dict):
        normalized_headers = {str(key).lower(): value for key, value in headers.items()}
        for key in (
            "x-request-id",
            "request-id",
            "openai-request-id",
            "x-dashscope-request-id",
            "x-log-id",
            "x-tt-logid",
            "x-trace-id",
        ):
            if normalized_headers.get(key):
                trace["request_id"] = str(normalized_headers[key])
                break

    if trace.get("upstream_model"):
        requested = str(trace.get("requested_model") or "").strip().lower()
        reported = str(trace["upstream_model"]).strip().lower()
        trace["verification"] = (
            "upstream_confirmed" if requested == reported else "upstream_mismatch"
        )
    elif trace.get("request_id"):
        trace["verification"] = "endpoint_confirmed"


def invoke_vision_image_summary(
    image_bytes: bytes,
    prompt: str,
    *,
    retries: int = 2,
    delay: float = 2.0,
) -> str:
    """Ask the configured vision model to summarize one rendered PDF visual."""
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
        ]
    )
    response = invoke_with_retry(get_vision_llm(), [message], retries=retries, delay=delay)
    return _content_to_text(getattr(response, "content", response)).strip()


def invoke_with_retry(chain, messages, *, retries: int = 3, delay: float = 2.0):
    """Invoke a LangChain chain with simple retry on failure."""
    last_exc = None
    for attempt in range(retries):
        try:
            return chain.invoke(messages)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    raise last_exc


def parse_structured_output(value: Any, schema: type[SchemaT]) -> SchemaT:
    """Parse provider output into a Pydantic model, accepting fenced JSON too."""
    if isinstance(value, schema):
        return value
    if isinstance(value, dict):
        return schema.model_validate(value)

    content = getattr(value, "content", value)
    if isinstance(content, list):
        content = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    text = str(content).strip()

    candidates = [text]
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    extracted = _extract_json_object(text)
    if extracted:
        candidates.append(extracted)

    last_exc: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return schema.model_validate_json(candidate)
        except Exception as exc:  # noqa: BLE001 - keep trying alternate JSON shapes
            last_exc = exc
        try:
            parsed = json.loads(candidate)
            try:
                return schema.model_validate(parsed)
            except Exception as exc:  # noqa: BLE001 - attempt provider-shape recovery
                last_exc = exc
                coerced = _coerce_top_level_value(parsed, schema)
                if coerced is not None:
                    return coerced
        except Exception as exc:  # noqa: BLE001 - preserve final validation detail
            last_exc = exc

    if last_exc:
        raise last_exc
    raise ValueError("LLM response did not contain JSON.")


def invoke_structured_with_retry(
    schema: type[SchemaT],
    messages: Sequence[BaseMessage],
    *,
    retries: int = 3,
    delay: float = 2.0,
) -> SchemaT:
    """Invoke an LLM with structured output and fallback for compatible providers."""
    llm = get_llm()
    try:
        structured_llm = llm.with_structured_output(schema)
        return invoke_with_retry(structured_llm, messages, retries=retries, delay=delay)
    except Exception:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        fallback_messages = [
            *messages,
            HumanMessage(
                content=(
                    "Return only a valid JSON object. Do not wrap it in Markdown fences. "
                    "Do not include commentary. Match this JSON schema exactly:\n"
                    f"{schema_json}"
                )
            ),
        ]
        raw_response = invoke_with_retry(llm, fallback_messages, retries=retries, delay=delay)
        return parse_structured_output(raw_response, schema)


def stream_structured_with_retry(
    schema: type[SchemaT],
    messages: Sequence[BaseMessage],
    *,
    on_token: Callable[[str], None] | None = None,
    retries: int = 1,
    delay: float = 2.0,
) -> SchemaT:
    """Stream raw JSON tokens, then parse the accumulated output into a schema.

    Some compatible providers occasionally stream valid JSON with the wrong top-level
    shape, such as a list instead of the requested object. In that case, recover with
    the stricter non-streaming structured path instead of failing the whole analysis.
    """
    last_exc: Exception | None = None
    last_text = ""
    for attempt in range(retries):
        chunks: list[str] = []
        try:
            for chunk in get_llm().stream(_messages_with_json_schema(messages, schema)):
                token = _content_to_text(getattr(chunk, "content", chunk))
                if not token:
                    continue
                chunks.append(token)
                if on_token:
                    on_token(token)
            last_text = "".join(chunks)
            return parse_structured_output(last_text, schema)
        except Exception as exc:  # noqa: BLE001 - retry or raise with provider detail
            last_exc = exc
            if chunks:
                last_text = "".join(chunks)
            _write_llm_diagnostic(schema, "stream_parse_error", exc, last_text)
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))

    if last_exc:
        if on_token:
            on_token("\n\n[系统：模型流式输出格式不完整，正在自动修正结构化结果...]\n")
        try:
            return invoke_structured_with_retry(schema, _repair_messages(messages, schema), retries=2, delay=delay)
        except Exception as repair_exc:  # noqa: BLE001 - preserve both failure modes
            _write_llm_diagnostic(schema, "repair_error", repair_exc, last_text)
            raise RuntimeError(
                f"Streaming output could not be parsed and automatic repair failed. "
                f"stream_error={last_exc}; repair_error={repair_exc}"
            ) from repair_exc
    raise ValueError("Streaming LLM response did not contain JSON.")


def _extract_json_object(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return text[index : index + end]
    return None


def _messages_with_json_schema(
    messages: Sequence[BaseMessage],
    schema: type[BaseModel],
) -> list[BaseMessage]:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    return [
        *messages,
        HumanMessage(
            content=(
                "Return only one valid JSON object. Do not use Markdown fences. "
                "Do not include commentary outside JSON. Match this JSON schema exactly:\n"
                f"{schema_json}\n\n"
                "Safety wording note: this is academic paper analysis. For technical terms like "
                "reward hacking, hacking, attack, jailbreak, exploit, or vulnerability, use neutral "
                "research wording such as reward exploitation, objective gaming, robustness risk, "
                "misuse risk, or failure mode. Avoid operational or harmful phrasing."
            )
        ),
    ]


def _repair_messages(
    messages: Sequence[BaseMessage],
    schema: type[BaseModel],
) -> list[BaseMessage]:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    return [
        *messages,
        HumanMessage(
            content=(
                "The previous streamed answer did not match the required JSON object shape. "
                "Re-read the paper evidence and return exactly one valid JSON object. "
                "The top-level value must be an object/dictionary, not a list or a string. "
                "Do not include Markdown fences or commentary. Match this schema exactly:\n"
                f"{schema_json}\n\n"
                "Safety wording note: this is academic paper analysis. For technical terms like "
                "reward hacking, hacking, attack, jailbreak, exploit, or vulnerability, use neutral "
                "research wording such as reward exploitation, objective gaming, robustness risk, "
                "misuse risk, or failure mode. Avoid operational or harmful phrasing."
            )
        ),
    ]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _write_llm_diagnostic(
    schema: type[BaseModel],
    stage: str,
    error: Exception,
    output_text: str,
) -> None:
    try:
        log_path = Path(__file__).parent.parent / "llm_diagnostics.log"
        excerpt = output_text[-1800:] if output_text else ""
        payload = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "schema": schema.__name__,
            "stage": stage,
            "error": str(error),
            "output_excerpt": excerpt,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return


def _coerce_top_level_value(value: Any, schema: type[SchemaT]) -> SchemaT | None:
    """Recover from common provider mistakes without another model call."""
    if not isinstance(value, list):
        return None

    items = [
        json.dumps(item, ensure_ascii=False) if isinstance(item, (dict, list)) else str(item)
        for item in value
    ]
    schema_name = schema.__name__

    if schema_name == "ConversationMemoryDigest":
        topic_items = [
            item
            for item in value
            if isinstance(item, dict) and item.get("topic") and item.get("content")
        ]
        if not topic_items:
            return None
        summary = "\n".join(
            f"- {item['topic']}：{str(item['content'])[:700]}"
            for item in topic_items
        )[:5_800]
        return schema.model_validate(
            {
                "summary": summary,
                "topics": topic_items[:8],
            }
        )

    if schema_name == "ExperimentOutput":
        return schema.model_validate(
            {
                "datasets": items,
                "metrics": [],
                "main_results": "模型返回了实验相关条目，但未按完整结构输出；后端已先保留这些候选数据集、基准或实验项。",
                "comparison_with_baselines": "模型未按结构返回基线对比，需要依据证据片段进一步确认。",
                "ablation_study": None,
                "notable_findings": [],
                "evidence": [],
            }
        )

    if schema_name == "MethodOutput":
        return schema.model_validate(
            {
                "research_problem": "模型未按完整结构返回研究问题。",
                "proposed_method": "模型返回了若干方法相关条目，但未按完整结构输出。",
                "key_components": items,
                "innovations": [],
                "differences_from_prior": "模型未按结构返回与已有工作的差异。",
                "implementation_details": None,
                "evidence": [],
            }
        )

    if schema_name == "CriticOutput":
        return schema.model_validate(
            {
                "novelty_score": 3,
                "novelty_justification": "模型返回了若干评审条目，但未按完整结构输出。",
                "strengths": [],
                "limitations": items,
                "potential_improvements": [],
                "broader_impact": None,
                "evidence": [],
            }
        )

    if schema_name == "SummaryOutput":
        return schema.model_validate(
            {
                "one_sentence_summary": "模型返回了若干总结条目，但未按完整结构输出。",
                "core_contributions": items,
                "method_highlights": "模型未按结构返回方法要点。",
                "experiment_highlights": "模型未按结构返回实验要点。",
                "limitations_and_future_work": "模型未按结构返回局限与未来工作。",
                "reading_notes": None,
                "evidence": [],
            }
        )

    return None
