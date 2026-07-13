"""Cached, credential-safe health checks for the curated model catalog."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from core.model_providers import PROVIDERS, provider_api_key, provider_base_url


_CACHE_LOCK = threading.Lock()
_CACHE_PAYLOAD: dict[str, Any] | None = None
_CACHE_CREATED_AT = 0.0


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
    common: dict[str, Any] = {
        "id": provider_id,
        "label": spec.label,
        "configured": bool(api_key),
        "base_url": base_url,
        "catalog_text_model_count": len(spec.text_models),
        "catalog_vision_model_count": len(spec.vision_models),
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
    message = (
        "模型目录与项目配置存在差异，请核对缺失模型。"
        if drifted
        else (
            "文本目录已核对；厂商 /models 未列出视觉模型。"
            if vision_check == "not_listed"
            else "模型目录与项目配置一致。"
        )
    )
    return {
        **common,
        "status": "drift" if drifted else "ok",
        "message": message,
        "available_model_count": len(available),
        "missing_text_models": missing_text,
        "missing_vision_models": missing_vision,
        "vision_catalog_check": vision_check,
        "http_status": 200,
    }


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))
