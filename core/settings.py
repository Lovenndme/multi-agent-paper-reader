"""Local application settings and secure multi-provider onboarding."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import set_key
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, SecretStr

from core.model_health import invalidate_model_catalog_health_cache
from core.model_providers import (
    ModelModeSpec,
    ModelSpec,
    PROVIDERS,
    model_mode_request_body,
    model_is_known,
    model_modes,
    provider_api_key,
    provider_base_url,
    provider_credential_configured,
    provider_label,
    provider_protocol,
    provider_spec,
    selected_text_model,
    selected_text_model_label,
    selected_text_mode,
    selected_vision_model,
    text_provider_id,
    vision_enabled,
    vision_provider_id,
)
from utils.llm import is_llm_configured, is_vision_configured, reset_llm_clients


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = Path(os.environ.get("PAPER_READER_ENV_PATH", PROJECT_ROOT / ".env"))
PROJECT_VERSION = "V1.6.2"
_SETTINGS_LOCK = threading.Lock()
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class ApiKeySettingsRequest(BaseModel):
    """Legacy GLM request kept for old clients."""

    api_key: SecretStr


class ProviderApiKeySettingsRequest(BaseModel):
    """Receive one provider credential without exposing it in logs."""

    api_key: SecretStr
    base_url: str | None = None
    protocol: str | None = None
    provider_name: str | None = None
    text_model: str | None = None
    vision_model: str | None = None


class ModelRoutingSettingsRequest(BaseModel):
    """Select a text route; vision is paired to the same provider."""

    text_provider: str
    text_model: str
    text_mode: str | None = None
    vision_enabled: bool = True
    vision_provider: str | None = None
    vision_model: str | None = None


class ApiKeyValidationError(RuntimeError):
    """Raised when a provider rejects a key or cannot be reached."""


class ModelRoutingValidationError(ValueError):
    """Raised when a provider/model route cannot be used."""


def application_settings_payload() -> dict[str, Any]:
    """Return public configuration metadata without any secret material."""
    active_text_provider = text_provider_id()
    active_vision_provider = vision_provider_id()
    text_configured = is_llm_configured()
    vision_configured = is_vision_configured()

    providers = []
    for spec in PROVIDERS.values():
        custom_text_model = os.environ.get("CUSTOM_TEXT_MODEL", "").strip()
        custom_vision_model = os.environ.get("CUSTOM_VISION_MODEL", "").strip()
        text_models = list(spec.text_models)
        vision_models = list(spec.vision_models)
        codex_status = None
        if spec.credential_type == "codex_login":
            from core.codex_sdk import get_codex_sdk_service

            service = get_codex_sdk_service()
            codex_status = service.status()
            catalog = ()
            if codex_status.get("authenticated"):
                try:
                    catalog = service.models()
                except Exception:  # noqa: BLE001 - status carries the safe failure state
                    catalog = ()
                codex_status = service.status()
            # Never fall back to the descriptive static Codex entries. The
            # account's live model/list response is the sole routing authority.
            text_models = [
                ModelSpec(
                    model.id,
                    model.label,
                    model.description,
                    model.recommended,
                    ("Codex", "订阅"),
                    _codex_effort_modes(model),
                    model.default_effort,
                )
                for model in catalog
            ]
            vision_models = [
                ModelSpec(
                    model.id,
                    model.label,
                    "通过 Codex SDK 使用本地论文图像",
                    model.recommended,
                    ("Codex", "视觉"),
                )
                for model in catalog
                if model.supports_image
            ]
        if spec.customizable:
            text_models = [
                type(spec.text_models[0])(
                    custom_text_model or "custom-model",
                    custom_text_model or "请先配置模型 ID",
                    "由中转站提供的文本模型",
                    True,
                )
            ]
            vision_models = (
                [type(spec.text_models[0])(
                    custom_vision_model,
                    custom_vision_model,
                    "由中转站提供的视觉模型",
                    True,
                )]
                if custom_vision_model
                else []
            )
        providers.append(
            {
                "id": spec.id,
                "label": provider_label(spec.id),
                "configured": provider_credential_configured(spec.id),
                "base_url": provider_base_url(spec.id),
                "key_url": spec.key_url,
                "protocol": provider_protocol(spec.id),
                "credential_type": spec.credential_type,
                "local_only": spec.local_only,
                "codex_status": codex_status,
                "customizable": spec.customizable,
                "provider_name": provider_label(spec.id),
                "supports_vision": bool(vision_models),
                "default_text_model": text_models[0].id if text_models else None,
                "default_vision_model": vision_models[0].id if vision_models else None,
                "text_models": [model.payload() for model in text_models],
                "vision_models": [model.payload() for model in vision_models],
            }
        )

    return {
        "version": PROJECT_VERSION,
        "provider": provider_label(active_text_provider),
        "api_key_configured": text_configured,
        "routing": {
            "text": {
                "provider": active_text_provider,
                "provider_label": provider_label(active_text_provider),
                "model": selected_text_model(),
                "model_label": selected_text_model_label(),
                "mode": selected_text_mode(),
                "configured": text_configured,
            },
            "vision": {
                "enabled": vision_enabled(),
                "provider": active_vision_provider,
                "provider_label": provider_label(active_vision_provider),
                "model": selected_vision_model(),
                "configured": vision_configured,
                "credential_configured": provider_credential_configured(active_vision_provider),
            },
        },
        "providers": providers,
        # Preserve the original response shape for older frontend builds.
        "models": [
            {
                "id": "text",
                "label": "文本分析",
                "name": selected_text_model(),
                "provider": provider_label(active_text_provider),
                "purpose": "论文分析与追问",
                "configured": text_configured,
            },
            {
                "id": "vision",
                "label": "图表理解",
                "name": selected_vision_model(),
                "provider": provider_label(active_vision_provider),
                "purpose": "图像、图表与公式区域",
                "configured": vision_configured,
            },
        ],
    }


def configure_provider_api_key(
    provider_id: str,
    api_key: str,
    *,
    base_url: str | None = None,
    protocol: str | None = None,
    provider_name: str | None = None,
    text_model: str | None = None,
    vision_model: str | None = None,
    env_path: Path = ENV_PATH,
) -> dict[str, Any]:
    """Validate, persist, and activate one provider credential."""
    try:
        spec = provider_spec(provider_id.strip().lower())
    except ValueError as exc:
        raise ApiKeyValidationError(str(exc)) from exc
    if spec.credential_type != "api_key":
        raise ApiKeyValidationError(
            "Codex 订阅不使用 API Key，请通过 Settings 中的 ChatGPT 登录入口连接。"
        )

    clean_key = api_key.strip()
    if not 10 <= len(clean_key) <= 4096:
        raise ApiKeyValidationError("API Key 格式不完整，请重新检查。")
    clean_base_url = _validated_base_url(base_url or provider_base_url(spec.id))
    custom_values: dict[str, str] = {}
    probe_protocol = spec.protocol
    probe_model = (
        selected_text_model()
        if text_provider_id() == spec.id
        else spec.default_text_model
    )
    if spec.customizable:
        clean_protocol = (protocol or "").strip().lower()
        if clean_protocol not in {"openai", "anthropic"}:
            raise ApiKeyValidationError("请选择 OpenAI-compatible 或 Anthropic-compatible 协议。")
        try:
            clean_text_model = _validated_model_id(text_model or "", "文本模型")
            clean_vision_model = (vision_model or "").strip()
            if clean_vision_model:
                clean_vision_model = _validated_model_id(clean_vision_model, "视觉模型")
        except ModelRoutingValidationError as exc:
            raise ApiKeyValidationError(str(exc)) from exc
        clean_name = (provider_name or "").strip() or "自定义中转站"
        if len(clean_name) > 48 or any(char in clean_name for char in "\r\n\t"):
            raise ApiKeyValidationError("中转站名称必须为不超过 48 个字符的单行文本。")
        probe_protocol = clean_protocol
        probe_model = clean_text_model
        custom_values = {
            "CUSTOM_API_PROTOCOL": clean_protocol,
            "CUSTOM_PROVIDER_NAME": clean_name,
            "CUSTOM_TEXT_MODEL": clean_text_model,
            "CUSTOM_VISION_MODEL": clean_vision_model,
        }

    try:
        discovered_models = _probe_provider_api_key(
            spec.id,
            clean_key,
            clean_base_url,
            probe_model,
            protocol=probe_protocol,
        )
    except ApiKeyValidationError:
        raise
    except Exception as exc:
        raise ApiKeyValidationError(
            f"{provider_label(spec.id)} API Key 验证失败，请检查密钥、Base URL 与网络连接。"
        ) from exc

    with _SETTINGS_LOCK:
        values = {
                spec.api_key_env: clean_key,
                spec.base_url_env: clean_base_url,
                **custom_values,
            }
        _persist_env_values(values, env_path)
        os.environ.update(values)
        reset_llm_clients()
        invalidate_model_catalog_health_cache()

    payload = application_settings_payload()
    payload["validation"] = {
        "provider": spec.id,
        "provider_label": provider_label(spec.id),
        "available_model_count": len(discovered_models),
        "available_models": discovered_models[:100],
    }
    return payload


def configure_glm_api_key(
    api_key: str,
    *,
    env_path: Path = ENV_PATH,
) -> dict[str, Any]:
    """Backward-compatible wrapper for the original GLM-only endpoint."""
    clean_key = api_key.strip()
    if not 10 <= len(clean_key) <= 4096:
        raise ApiKeyValidationError("API Key 格式不完整，请重新检查。")
    _probe_glm_api_key(clean_key)
    with _SETTINGS_LOCK:
        _persist_env_values({"GLM_API_KEY": clean_key}, env_path)
        os.environ["GLM_API_KEY"] = clean_key
        reset_llm_clients()
        invalidate_model_catalog_health_cache()
    return application_settings_payload()


def configure_model_routing(
    request: ModelRoutingSettingsRequest,
    *,
    env_path: Path = ENV_PATH,
) -> dict[str, Any]:
    """Persist one provider route and its fixed same-provider vision pairing."""
    text_provider = request.text_provider.strip().lower()
    try:
        text_spec = provider_spec(text_provider)
    except ValueError as exc:
        raise ModelRoutingValidationError(str(exc)) from exc
    text_model = _validated_model_id(request.text_model, "文本模型")
    if not text_spec.customizable and not model_is_known(text_provider, "text", text_model):
        raise ModelRoutingValidationError(
            f"{provider_label(text_provider)} 不支持文本模型 {text_model}，请从模型列表中选择。"
        )
    if not provider_credential_configured(text_provider):
        raise ModelRoutingValidationError(
            (
                "请先在 Settings 中连接本机 Codex 订阅，再应用模型配置。"
                if text_spec.credential_type == "codex_login"
                else f"请先为 {provider_label(text_provider)} 配置并验证 API Key，再应用模型配置。"
            )
        )

    available_modes = model_modes(text_provider, text_model)
    requested_mode = (request.text_mode or "").strip().lower()
    if available_modes:
        if requested_mode and not any(mode.id == requested_mode for mode in available_modes):
            raise ModelRoutingValidationError(
                f"{text_model} 不支持响应模式 {requested_mode}，请从模式列表中选择。"
            )
        text_mode = requested_mode or available_modes[0].id
    else:
        text_mode = ""

    requested_vision_provider = (request.vision_provider or text_provider).strip().lower()
    if requested_vision_provider != text_provider:
        raise ModelRoutingValidationError(
            "视觉理解必须与文本分析使用同一家厂商，不能单独切换视觉厂商。"
        )
    custom_vision_model = request.vision_model.strip() if request.vision_model else ""
    supports_selected_vision = (
        model_is_known(text_provider, "vision", text_model)
        if text_spec.credential_type == "codex_login"
        else bool(text_spec.vision_models)
    )
    if request.vision_enabled and not supports_selected_vision and not (
        text_spec.customizable and custom_vision_model
    ):
        raise ModelRoutingValidationError(
            f"{provider_label(text_provider)} 未配置可用的视觉模型，请关闭图表理解。"
        )

    if text_spec.credential_type == "codex_login":
        vision_model = text_model if supports_selected_vision else ""
    else:
        vision_model = (
            _validated_model_id(custom_vision_model, "视觉模型")
            if text_spec.customizable and custom_vision_model
            else text_spec.default_vision_model or ""
        )
    if not text_spec.customizable and request.vision_model and request.vision_model.strip() not in {vision_model, ""}:
        raise ModelRoutingValidationError(
            f"视觉模型由系统自动配对为 {vision_model or '不可用'}，不能单独修改。"
        )

    values: dict[str, str] = {
        "TEXT_PROVIDER": text_provider,
        "MODEL_NAME": text_model,
        "MODEL_MODE": text_mode,
        "ENABLE_VISION_SUMMARY": "true" if request.vision_enabled else "false",
        "VISION_PROVIDER": text_provider,
        "VISION_MODEL_NAME": vision_model,
    }
    if text_spec.customizable:
        values["CUSTOM_TEXT_MODEL"] = text_model
        values["CUSTOM_VISION_MODEL"] = vision_model

    with _SETTINGS_LOCK:
        _persist_env_values(values, env_path)
        os.environ.update(values)
        reset_llm_clients()
        invalidate_model_catalog_health_cache()

    return application_settings_payload()


def _probe_provider_api_key(
    provider_id: str,
    api_key: str,
    base_url: str,
    model: str,
    *,
    protocol: str | None = None,
) -> list[str]:
    """Verify a credential, preferring the cheap model-list endpoint."""
    timeout = min(float(os.environ.get("LLM_TIMEOUT_SECONDS", "240")), 30.0)
    wire_protocol = protocol or provider_protocol(provider_id)
    if wire_protocol == "anthropic":
        try:
            response = ChatAnthropic(
                model=model,
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,
                max_tokens=8,
            ).invoke([HumanMessage(content="只回复 OK")])
        except Exception as exc:
            raise ApiKeyValidationError(
                f"{provider_label(provider_id)} 验证请求失败，请确认 Key、Base URL、协议和模型 ID 可用。"
            ) from exc
        if not getattr(response, "content", None):
            raise ApiKeyValidationError("模型验证未返回结果，请稍后重试。")
        return []

    try:
        page = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        ).models.list()
        model_ids = sorted(
            {
                str(item.id)
                for item in getattr(page, "data", [])
                if getattr(item, "id", None)
            }
        )
        if model_ids:
            return model_ids
    except Exception as exc:
        if getattr(exc, "status_code", None) in {401, 403}:
            raise ApiKeyValidationError(
                f"{provider_label(provider_id)} 拒绝了该 API Key，请确认密钥有效且具有模型访问权限。"
            ) from exc

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout,
        "max_retries": 0,
        "max_tokens": 8,
    }
    modes = model_modes(provider_id, model)
    if modes:
        kwargs["extra_body"] = model_mode_request_body(provider_id, model, modes[0].id)
    if not (
        (provider_id == "openai" and model.startswith("gpt-5"))
        or provider_id in {"kimi", "deepseek"}
    ):
        kwargs["temperature"] = 0
    try:
        response = ChatOpenAI(**kwargs).invoke([HumanMessage(content="只回复 OK")])
    except Exception as exc:
        raise ApiKeyValidationError(
            f"{provider_label(provider_id)} 验证请求失败，请确认 Key、Base URL 和所选模型可用。"
        ) from exc
    if not getattr(response, "content", None):
        raise ApiKeyValidationError("模型验证未返回结果，请稍后重试。")
    return []


def _probe_glm_api_key(api_key: str) -> None:
    """Legacy test hook retained for downstream callers."""
    _probe_provider_api_key(
        "zhipu",
        api_key,
        provider_base_url("zhipu"),
        (
            selected_text_model()
            if text_provider_id() == "zhipu"
            else provider_spec("zhipu").default_text_model
        ),
    )


def _validated_base_url(value: str) -> str:
    clean_value = value.strip().rstrip("/")
    parsed = urlparse(clean_value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ApiKeyValidationError("Base URL 必须是完整的 http:// 或 https:// 地址。")
    if parsed.username or parsed.password:
        raise ApiKeyValidationError("Base URL 中不能包含用户名或密码。")
    return clean_value


def _codex_effort_label(effort: str) -> str:
    return {
        "low": "轻度",
        "medium": "中等",
        "high": "高",
        "xhigh": "最高",
        "max": "极高",
        "ultra": "Ultra",
    }.get(effort, effort)


def _codex_effort_modes(model: Any) -> tuple[ModelModeSpec, ...]:
    descriptions = {effort: description for effort, description in model.efforts}
    output = []
    for effort in ("low", "medium", "high", "xhigh", "max", "ultra"):
        available = effort in descriptions
        output.append(
            ModelModeSpec(
                effort,
                _codex_effort_label(effort),
                descriptions.get(effort) or (
                    "Ultra 会在相同只读沙箱和论文工具边界下按需启用受限子 Agent。"
                    if effort == "ultra"
                    else f"Codex {effort} 推理强度"
                ),
                available=available,
                disabled_reason=(
                    f"{model.label} 当前不支持 {_codex_effort_label(effort)}。"
                    if not available
                    else None
                ),
                execution_kind="multi_agent" if effort == "ultra" else "single_agent",
            )
        )
    return tuple(output)


def _validated_model_id(value: str, label: str) -> str:
    clean_value = value.strip()
    if not _MODEL_ID_PATTERN.fullmatch(clean_value):
        raise ModelRoutingValidationError(
            f"{label} ID 格式无效，仅支持字母、数字、点、短横线、下划线、斜线和冒号。"
        )
    return clean_value


def _persist_env_values(values: dict[str, str], env_path: Path) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if not env_path.exists():
        env_path.touch(mode=0o600)
    for name, value in values.items():
        set_key(str(env_path), name, value, quote_mode="always")
    try:
        env_path.chmod(0o600)
    except OSError:
        pass
