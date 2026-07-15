"""Cached, credential-safe health checks for the curated model catalog."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from core.model_providers import (
    PROVIDERS,
    codex_model_catalog,
    provider_api_key,
    provider_base_url,
    provider_credential_configured,
)


_CACHE_LOCK = threading.Lock()
_CACHE_PAYLOAD: dict[str, Any] | None = None
_CACHE_CREATED_AT = 0.0
_VISION_PROBE_IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAfElEQVR4nNXOQREAMAjAsK7+PTMRPLhGQd7QJnESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ESJ3ES53Vg6wNShQF/fRSLfgAAAABJRU5ErkJggg=="
)


def invalidate_model_catalog_health_cache() -> None:
    """Discard cached checks after credentials or routing endpoints change."""
    global _CACHE_PAYLOAD, _CACHE_CREATED_AT
    with _CACHE_LOCK:
        _CACHE_PAYLOAD = None
        _CACHE_CREATED_AT = 0.0


def model_catalog_health(*, force: bool = False) -> dict[str, Any]:
    """Check configured provider model lists without returning credentials."""
    global _CACHE_PAYLOAD, _CACHE_CREATED_AT
    ttl_seconds = _bounded_float("MODEL_CATALOG_HEALTH_TTL_SECONDS", 900.0, 30.0, 86400.0)
    now = time.monotonic()
    with _CACHE_LOCK:
        if (
            not force
            and _CACHE_PAYLOAD is not None
            and now - _CACHE_CREATED_AT < ttl_seconds
        ):
            return {**_CACHE_PAYLOAD, "cached": True}

    provider_ids = list(PROVIDERS)
    with ThreadPoolExecutor(max_workers=len(provider_ids)) as executor:
        results = list(executor.map(_check_provider_catalog, provider_ids))

    summary = {
        "configured": sum(item["configured"] for item in results),
        "healthy": sum(item["status"] == "ok" for item in results),
        "drifted": sum(item["status"] == "drift" for item in results),
        "unavailable": sum(item["status"] == "unavailable" for item in results),
        "unconfigured": sum(item["status"] == "unconfigured" for item in results),
    }
    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cache_ttl_seconds": int(ttl_seconds),
        "cached": False,
        "summary": summary,
        "providers": results,
    }
    with _CACHE_LOCK:
        _CACHE_PAYLOAD = payload
        _CACHE_CREATED_AT = time.monotonic()
    return payload


def _check_provider_catalog(provider_id: str) -> dict[str, Any]:
    spec = PROVIDERS[provider_id]
    base_url = provider_base_url(provider_id)
    api_key = provider_api_key(provider_id)
    configured = provider_credential_configured(provider_id)
    common: dict[str, Any] = {
        "id": provider_id,
        "label": spec.label,
        "configured": configured,
        "base_url": base_url,
        "catalog_text_model_count": len(spec.text_models),
        "catalog_vision_model_count": len(spec.vision_models),
    }
    if spec.credential_type == "codex_login":
        if not configured:
            return {
                **common,
                "status": "unconfigured",
                "message": "本机 Codex 尚未登录 ChatGPT。",
                "available_model_count": None,
                "missing_text_models": [],
                "missing_vision_models": [],
                "vision_catalog_check": "not_run",
                "vision_probe_status": "not_run",
                "vision_probe_model": None,
                "vision_http_status": None,
                "http_status": None,
            }
        try:
            models = codex_model_catalog()
        except Exception:
            models = ()
        if not models:
            return {
                **common,
                "status": "unavailable",
                "message": "Codex SDK 未返回可用模型。",
                "available_model_count": 0,
                "missing_text_models": [],
                "missing_vision_models": [],
                "vision_catalog_check": "not_run",
                "vision_probe_status": "not_run",
                "vision_probe_model": None,
                "vision_http_status": None,
                "http_status": None,
            }
        return {
            **common,
            "status": "ok",
            "message": "本机 Codex 登录与动态模型目录正常。",
            "available_model_count": len(models),
            "missing_text_models": [],
            "missing_vision_models": [],
            "vision_catalog_check": "verified" if any(item.supports_image for item in models) else "not_applicable",
            "vision_probe_status": "catalog_confirmed" if any(item.supports_image for item in models) else "not_applicable",
            "vision_probe_model": next((item.id for item in models if item.supports_image), None),
            "vision_http_status": None,
            "http_status": None,
        }
    if not api_key:
        return {
            **common,
            "status": "unconfigured",
            "message": "未配置 API Key，已跳过远端检查。",
            "available_model_count": None,
            "missing_text_models": [],
            "missing_vision_models": [],
            "vision_catalog_check": "not_run",
            "vision_probe_status": "not_run",
            "vision_probe_model": None,
            "vision_http_status": None,
            "http_status": None,
        }

    timeout = _bounded_float("MODEL_CATALOG_HEALTH_TIMEOUT_SECONDS", 15.0, 2.0, 30.0)
    try:
        page = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        ).models.list()
        available = {
            str(item.id)
            for item in getattr(page, "data", [])
            if getattr(item, "id", None)
        }
    except Exception as exc:  # noqa: BLE001 - convert provider errors to safe status
        status_code = getattr(exc, "status_code", None)
        message = (
            "厂商拒绝了当前凭据，无法检查模型目录。"
            if status_code in {401, 403}
            else "模型目录端点不可用，请检查 Base URL、网络或厂商状态。"
        )
        return {
            **common,
            "status": "unavailable",
            "message": message,
            "available_model_count": None,
            "missing_text_models": [],
            "missing_vision_models": [],
            "vision_catalog_check": "not_run",
            "vision_probe_status": "not_run",
            "vision_probe_model": None,
            "vision_http_status": None,
            "http_status": status_code if isinstance(status_code, int) else None,
        }

    if not available:
        return {
            **common,
            "status": "unavailable",
            "message": "模型目录端点未返回任何模型。",
            "available_model_count": 0,
            "missing_text_models": [],
            "missing_vision_models": [],
            "vision_catalog_check": "not_run",
            "vision_probe_status": "not_run",
            "vision_probe_model": None,
            "vision_http_status": None,
            "http_status": 200,
        }

    text_ids = [model.id for model in spec.text_models]
    vision_ids = [model.id for model in spec.vision_models]
    missing_text = sorted(model_id for model_id in text_ids if model_id not in available)

    # Some compatible providers expose text models from /models but omit their
    # vision catalog. Only report vision drift when the endpoint lists at least
    # one curated vision model, avoiding false alarms for those providers.
    vision_catalog_listed = bool(vision_ids and any(model_id in available for model_id in vision_ids))
    if not vision_ids:
        vision_check = "not_applicable"
        missing_vision: list[str] = []
    elif vision_catalog_listed:
        vision_check = "verified"
        missing_vision = sorted(model_id for model_id in vision_ids if model_id not in available)
    else:
        vision_check = "not_listed"
        missing_vision = []

    drifted = bool(missing_text or missing_vision)
    vision_model = spec.default_vision_model
    if vision_model:
        vision_probe_status, vision_http_status = _probe_vision_model(
            api_key=api_key,
            base_url=base_url,
            model=vision_model,
            timeout=timeout,
        )
    else:
        vision_probe_status, vision_http_status = "not_applicable", None

    if vision_probe_status == "unavailable":
        status = "unavailable"
        message = f"视觉模型 {vision_model} 真实调用失败，当前不可用。"
    elif drifted:
        status = "drift"
        message = (
            f"模型目录与项目配置存在差异；视觉模型 {vision_model} 已通过真实调用，可用。"
            if vision_model
            else "模型目录与项目配置存在差异，请核对缺失模型。"
        )
    elif vision_probe_status == "available":
        status = "ok"
        message = f"文本目录正常；视觉模型 {vision_model} 已通过真实调用，可用。"
    else:
        status = "ok"
        message = "文本目录正常；该厂商未配置视觉模型。"

    return {
        **common,
        "status": status,
        "message": message,
        "available_model_count": len(available),
        "missing_text_models": missing_text,
        "missing_vision_models": missing_vision,
        "vision_catalog_check": vision_check,
        "vision_probe_status": vision_probe_status,
        "vision_probe_model": vision_model,
        "vision_http_status": vision_http_status,
        "http_status": 200,
    }


def _probe_vision_model(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float,
) -> tuple[str, int | None]:
    """Return whether a configured vision model accepts and understands an image."""
    try:
        response = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        ).chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "这张纯色图片是什么颜色？只回答一种中文颜色。",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _VISION_PROBE_IMAGE_URL},
                        },
                    ],
                }
            ],
        )
        content = str(response.choices[0].message.content or "").strip().lower()
    except Exception as exc:  # noqa: BLE001 - expose status only, never provider payloads
        status_code = getattr(exc, "status_code", None)
        return "unavailable", status_code if isinstance(status_code, int) else None

    return ("available", 200) if ("红" in content or "red" in content) else ("unavailable", 200)


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))
