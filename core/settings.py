"""Local application settings and secure multi-provider onboarding."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import set_key
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, SecretStr

from core.model_providers import (
    PROVIDERS,
    provider_api_key,
    provider_base_url,
    provider_spec,
    selected_text_model,
    selected_text_model_label,
    selected_vision_model,
    text_provider_id,
    vision_enabled,
    vision_provider_id,
)
from utils.llm import is_llm_configured, is_vision_configured, reset_llm_clients


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = Path(os.environ.get("PAPER_READER_ENV_PATH", PROJECT_ROOT / ".env"))
PROJECT_VERSION = os.environ.get("PAPER_READER_VERSION", "V1.2.0")
_SETTINGS_LOCK = threading.Lock()
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class ApiKeySettingsRequest(BaseModel):
    """Legacy GLM request kept for old clients."""

    api_key: SecretStr


class ProviderApiKeySettingsRequest(BaseModel):
    """Receive one provider credential without exposing it in logs."""

    api_key: SecretStr
    base_url: str | None = None


class ModelRoutingSettingsRequest(BaseModel):
    """Select a text route; vision is paired to the same provider."""

    text_provider: str
    text_model: str
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
    text_spec = provider_spec(active_text_provider)
    vision_spec = provider_spec(active_vision_provider)
    text_configured = is_llm_configured()
    vision_configured = is_vision_configured()

    providers = []
    for spec in PROVIDERS.values():
        providers.append(
            {
                "id": spec.id,
                "label": spec.label,
                "configured": bool(provider_api_key(spec.id)),
                "base_url": provider_base_url(spec.id),
                "key_url": spec.key_url,
                "supports_vision": bool(spec.vision_models),
                "default_text_model": spec.default_text_model,
                "default_vision_model": spec.default_vision_model,
                "text_models": [model.payload() for model in spec.text_models],
                "vision_models": [model.payload() for model in spec.vision_models],
            }
        )

    return {
        "version": PROJECT_VERSION,
        "provider": text_spec.label,
        "api_key_configured": text_configured,
        "routing": {
            "text": {
                "provider": active_text_provider,
                "provider_label": text_spec.label,
                "model": selected_text_model(),
                "model_label": selected_text_model_label(),
                "configured": text_configured,
            },
            "vision": {
                "enabled": vision_enabled(),
                "provider": active_vision_provider,
                "provider_label": vision_spec.label,
                "model": selected_vision_model(),
                "configured": vision_configured,
                "credential_configured": bool(provider_api_key(active_vision_provider)),
            },
        },
        "providers": providers,
        # Preserve the original response shape for older frontend builds.
        "models": [
            {
                "id": "text",
                "label": "文本分析",
                "name": selected_text_model(),
                "provider": text_spec.label,
                "purpose": "论文分析与追问",
                "configured": text_configured,
            },
            {
                "id": "vision",
                "label": "图表理解",
                "name": selected_vision_model(),
                "provider": vision_spec.label,
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
    env_path: Path = ENV_PATH,
) -> dict[str, Any]:
    """Validate, persist, and activate one provider credential."""
    try:
        spec = provider_spec(provider_id.strip().lower())
    except ValueError as exc:
        raise ApiKeyValidationError(str(exc)) from exc

    clean_key = api_key.strip()
    if not 10 <= len(clean_key) <= 4096:
        raise ApiKeyValidationError("API Key 格式不完整，请重新检查。")
    clean_base_url = _validated_base_url(base_url or provider_base_url(spec.id))
    probe_model = (
        selected_text_model()
        if text_provider_id() == spec.id
        else spec.default_text_model
    )

    try:
        discovered_models = _probe_provider_api_key(
            spec.id,
            clean_key,
            clean_base_url,
            probe_model,
        )
    except ApiKeyValidationError:
        raise
    except Exception as exc:
        raise ApiKeyValidationError(
            f"{spec.label} API Key 验证失败，请检查密钥、Base URL 与网络连接。"
        ) from exc

    with _SETTINGS_LOCK:
        _persist_env_values(
            {
                spec.api_key_env: clean_key,
                spec.base_url_env: clean_base_url,
            },
            env_path,
        )
        os.environ[spec.api_key_env] = clean_key
        os.environ[spec.base_url_env] = clean_base_url
        reset_llm_clients()

    payload = application_settings_payload()
    payload["validation"] = {
        "provider": spec.id,
        "provider_label": spec.label,
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
    if not any(model.id == text_model for model in text_spec.text_models):
        raise ModelRoutingValidationError(
            f"{text_spec.label} 不支持文本模型 {text_model}，请从模型列表中选择。"
        )
    if not provider_api_key(text_provider):
        raise ModelRoutingValidationError(
            f"请先为 {text_spec.label} 配置并验证 API Key，再应用模型配置。"
        )

    requested_vision_provider = (request.vision_provider or text_provider).strip().lower()
    if requested_vision_provider != text_provider:
        raise ModelRoutingValidationError(
            "视觉理解必须与文本分析使用同一家厂商，不能单独切换视觉厂商。"
        )
    if request.vision_enabled and not text_spec.vision_models:
        raise ModelRoutingValidationError(
            f"{text_spec.label} 官方托管 API 当前不提供视觉模型，请关闭图表理解。"
        )

    vision_model = text_spec.default_vision_model or ""
    if request.vision_model and request.vision_model.strip() not in {vision_model, ""}:
        raise ModelRoutingValidationError(
            f"视觉模型由系统自动配对为 {vision_model or '不可用'}，不能单独修改。"
        )

    values: dict[str, str] = {
        "TEXT_PROVIDER": text_provider,
        "MODEL_NAME": text_model,
        "ENABLE_VISION_SUMMARY": "true" if request.vision_enabled else "false",
        "VISION_PROVIDER": text_provider,
        "VISION_MODEL_NAME": vision_model,
    }

    with _SETTINGS_LOCK:
        _persist_env_values(values, env_path)
        os.environ.update(values)
        reset_llm_clients()

    return application_settings_payload()


def _probe_provider_api_key(
    provider_id: str,
    api_key: str,
    base_url: str,
    model: str,
) -> list[str]:
    """Verify a credential, preferring the cheap model-list endpoint."""
    spec = provider_spec(provider_id)
    timeout = min(float(os.environ.get("LLM_TIMEOUT_SECONDS", "240")), 30.0)
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
                f"{spec.label} 拒绝了该 API Key，请确认密钥有效且具有模型访问权限。"
            ) from exc

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "timeout": timeout,
        "max_retries": 0,
        "max_tokens": 8,
    }
    if not (provider_id == "openai" and model.startswith("gpt-5")):
        kwargs["temperature"] = 0
    try:
        response = ChatOpenAI(**kwargs).invoke([HumanMessage(content="只回复 OK")])
    except Exception as exc:
        raise ApiKeyValidationError(
            f"{spec.label} 验证请求失败，请确认 Key、Base URL 和所选模型可用。"
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
