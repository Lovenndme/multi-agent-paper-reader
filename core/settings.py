"""Local application settings and secure API-key onboarding."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from dotenv import set_key
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from utils.llm import (
    get_base_url,
    is_llm_configured,
    is_vision_configured,
    reset_llm_clients,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
PROJECT_VERSION = os.environ.get("PAPER_READER_VERSION", "V1.1.2")
_SETTINGS_LOCK = threading.Lock()


class ApiKeySettingsRequest(BaseModel):
    """Receive a secret without exposing it in model representations or logs."""

    api_key: SecretStr


class ApiKeyValidationError(RuntimeError):
    """Raised when the provider rejects a key or cannot be reached."""


def application_settings_payload() -> dict[str, Any]:
    """Return public configuration metadata without secret material."""
    text_configured = is_llm_configured()
    return {
        "version": PROJECT_VERSION,
        "provider": "Zhipu GLM",
        "api_key_configured": text_configured,
        "models": [
            {
                "id": "text",
                "label": "文本分析",
                "name": os.environ.get("MODEL_NAME", "glm-5.2"),
                "purpose": "论文分析与追问",
                "configured": text_configured,
            },
            {
                "id": "vision",
                "label": "图表理解",
                "name": os.environ.get("VISION_MODEL_NAME", "glm-5v-turbo"),
                "purpose": "图像、图表与公式区域",
                "configured": is_vision_configured(),
            },
        ],
    }


def configure_glm_api_key(
    api_key: str,
    *,
    env_path: Path = ENV_PATH,
) -> dict[str, Any]:
    """Validate, persist, and activate a GLM API key for this local app."""
    clean_key = api_key.strip()
    if not 10 <= len(clean_key) <= 4096:
        raise ApiKeyValidationError("API Key 格式不完整，请重新检查。")

    try:
        _probe_glm_api_key(clean_key)
    except ApiKeyValidationError:
        raise
    except Exception as exc:
        raise ApiKeyValidationError(
            "API Key 验证失败，请确认密钥有效且当前网络可以访问智谱开放平台。"
        ) from exc

    with _SETTINGS_LOCK:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        if not env_path.exists():
            env_path.touch(mode=0o600)
        set_key(str(env_path), "GLM_API_KEY", clean_key, quote_mode="always")
        try:
            env_path.chmod(0o600)
        except OSError:
            pass
        os.environ["GLM_API_KEY"] = clean_key
        reset_llm_clients()

    return application_settings_payload()


def _probe_glm_api_key(api_key: str) -> None:
    """Make a minimal real model request before accepting a credential."""
    model = os.environ.get("MODEL_NAME", "glm-5.2")
    timeout = min(float(os.environ.get("LLM_TIMEOUT_SECONDS", "240")), 30.0)
    client = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=get_base_url(),
        timeout=timeout,
        max_retries=0,
        max_tokens=8,
    )
    response = client.invoke([HumanMessage(content="只回复 OK")])
    if not getattr(response, "content", None):
        raise ApiKeyValidationError("模型验证未返回结果，请稍后重试。")
