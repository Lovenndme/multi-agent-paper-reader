"""Provider catalog and runtime model routing helpers.

All supported services expose an OpenAI-compatible API, but credentials and
model capabilities remain provider-specific. This module keeps that knowledge
in one place so the UI and LLM clients cannot drift apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


ModelCapability = Literal["text", "vision"]


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    description: str
    recommended: bool = False
    tags: tuple[str, ...] = ()

    def payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "recommended": self.recommended,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    api_key_env: str
    base_url_env: str
    default_base_url: str
    key_url: str
    text_models: tuple[ModelSpec, ...]
    vision_models: tuple[ModelSpec, ...] = ()
    api_key_aliases: tuple[str, ...] = ()

    @property
    def default_text_model(self) -> str:
        return self.text_models[0].id

    @property
    def default_vision_model(self) -> str | None:
        return self.vision_models[0].id if self.vision_models else None


PROVIDERS: dict[str, ProviderSpec] = {
    "zhipu": ProviderSpec(
        id="zhipu",
        label="Zhipu GLM",
        api_key_env="GLM_API_KEY",
        base_url_env="GLM_BASE_URL",
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        key_url="https://bigmodel.cn/usercenter/proj-mgmt/apikeys",
        text_models=(
            ModelSpec("glm-5.2", "GLM-5.2", "旗舰文本与推理模型", True, ("旗舰", "推理")),
            ModelSpec("glm-5.1", "GLM-5.1", "高能力通用文本模型", tags=("通用",)),
            ModelSpec("glm-5", "GLM-5", "上一代旗舰文本模型", tags=("通用",)),
            ModelSpec("glm-5-turbo", "GLM-5 Turbo", "低延迟 GLM-5 系列模型", tags=("高速",)),
            ModelSpec("glm-4.7", "GLM-4.7", "稳定的复杂任务模型", tags=("稳定",)),
            ModelSpec("glm-4.7-flashx", "GLM-4.7 FlashX", "高吞吐低延迟模型", tags=("高速",)),
            ModelSpec("glm-4.6", "GLM-4.6", "通用文本与工具调用模型", tags=("通用",)),
            ModelSpec("glm-4.5-air", "GLM-4.5 Air", "轻量高性价比模型", tags=("轻量",)),
            ModelSpec("glm-4-long", "GLM-4 Long", "面向长上下文任务", tags=("长上下文",)),
        ),
        vision_models=(
            ModelSpec(
                "glm-5v-turbo",
                "GLM-5V Turbo",
                "旗舰图像、图表与公式理解",
                True,
                ("旗舰", "视觉"),
            ),
            ModelSpec("glm-4.6v", "GLM-4.6V", "高精度视觉理解模型", tags=("视觉",)),
            ModelSpec("glm-4.6v-flash", "GLM-4.6V Flash", "低延迟视觉理解模型", tags=("高速",)),
            ModelSpec(
                "glm-4.1v-thinking-flashx",
                "GLM-4.1V Thinking FlashX",
                "视觉推理增强模型",
                tags=("推理",),
            ),
            ModelSpec(
                "glm-4.1v-thinking-flash",
                "GLM-4.1V Thinking Flash",
                "轻量视觉推理模型",
                tags=("轻量",),
            ),
            ModelSpec("glm-4v-flash", "GLM-4V Flash", "基础高速视觉模型", tags=("高速",)),
        ),
    ),
    "deepseek": ProviderSpec(
        id="deepseek",
        label="DeepSeek",
        api_key_env="DEEPSEEK_API_KEY",
        base_url_env="DEEPSEEK_BASE_URL",
        default_base_url="https://api.deepseek.com",
        key_url="https://platform.deepseek.com/api_keys",
        text_models=(
            ModelSpec(
                "deepseek-v4-pro",
                "DeepSeek V4 Pro",
                "高能力文本与推理模型",
                True,
                ("旗舰", "推理"),
            ),
            ModelSpec("deepseek-v4-flash", "DeepSeek V4 Flash", "低延迟高吞吐文本模型", tags=("高速",)),
        ),
    ),
    "openai": ProviderSpec(
        id="openai",
        label="OpenAI",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        default_base_url="https://api.openai.com/v1",
        key_url="https://platform.openai.com/api-keys",
        text_models=(
            ModelSpec("gpt-5.6", "GPT-5.6 Sol", "旗舰通用与推理模型", True, ("旗舰", "推理")),
            ModelSpec("gpt-5.6-terra", "GPT-5.6 Terra", "面向高难度长任务", tags=("高能力",)),
            ModelSpec("gpt-5.6-luna", "GPT-5.6 Luna", "兼顾速度与质量", tags=("均衡",)),
        ),
        vision_models=(
            ModelSpec("gpt-5.6", "GPT-5.6 Sol", "文本与图像联合理解", True, ("旗舰", "视觉")),
            ModelSpec("gpt-5.6-terra", "GPT-5.6 Terra", "复杂图表与视觉推理", tags=("高能力",)),
            ModelSpec("gpt-5.6-luna", "GPT-5.6 Luna", "高效多模态理解", tags=("均衡",)),
        ),
    ),
    "qwen": ProviderSpec(
        id="qwen",
        label="Alibaba Qwen",
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="QWEN_BASE_URL",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        key_url="https://bailian.console.aliyun.com/?tab=model#/api-key",
        api_key_aliases=("QWEN_API_KEY",),
        text_models=(
            ModelSpec("qwen3.7-max", "Qwen3.7 Max", "旗舰文本与复杂推理模型", True, ("旗舰", "推理")),
            ModelSpec("qwen3.7-plus", "Qwen3.7 Plus", "兼顾能力与成本的通用模型", tags=("均衡",)),
            ModelSpec("qwen3.6-flash", "Qwen3.6 Flash", "低延迟高吞吐模型", tags=("高速",)),
            ModelSpec("qwen-long", "Qwen Long", "面向超长文档任务", tags=("长上下文",)),
        ),
        vision_models=(
            ModelSpec("qwen3.7-plus", "Qwen3.7 Plus", "原生文本与图像联合理解", True, ("推荐", "视觉")),
            ModelSpec("qwen3.5-omni-plus", "Qwen3.5 Omni Plus", "全模态内容理解", tags=("全模态",)),
            ModelSpec("qwen3-vl-flash", "Qwen3 VL Flash", "低延迟视觉语言模型", tags=("高速",)),
            ModelSpec("qwen3-vl-plus", "Qwen3 VL Plus", "高精度图像与图表理解", tags=("视觉",)),
        ),
    ),
    "doubao": ProviderSpec(
        id="doubao",
        label="ByteDance Doubao",
        api_key_env="ARK_API_KEY",
        base_url_env="DOUBAO_BASE_URL",
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
        key_url="https://console.volcengine.com/ark/region:ark+cn-beijing/apikey",
        api_key_aliases=("VOLCENGINE_API_KEY",),
        text_models=(
            ModelSpec(
                "doubao-seed-2-0-pro-260215",
                "Doubao Seed 2.0 Pro",
                "旗舰多模态推理与长任务模型",
                True,
                ("旗舰", "推理"),
            ),
            ModelSpec(
                "doubao-seed-2-0-lite-260215",
                "Doubao Seed 2.0 Lite",
                "兼顾效果、延迟与成本的多模态模型",
                tags=("均衡",),
            ),
            ModelSpec(
                "doubao-seed-2-0-mini-260215",
                "Doubao Seed 2.0 Mini",
                "面向高并发场景的轻量多模态模型",
                tags=("高速",),
            ),
        ),
        vision_models=(
            ModelSpec(
                "doubao-seed-2-0-pro-260215",
                "Doubao Seed 2.0 Pro",
                "高精度图像、图表与公式理解",
                True,
                ("旗舰", "视觉"),
            ),
            ModelSpec(
                "doubao-seed-2-0-lite-260215",
                "Doubao Seed 2.0 Lite",
                "均衡的多模态内容理解",
                tags=("均衡",),
            ),
            ModelSpec(
                "doubao-seed-2-0-mini-260215",
                "Doubao Seed 2.0 Mini",
                "低延迟视觉理解",
                tags=("高速",),
            ),
        ),
    ),
}


def provider_spec(provider_id: str) -> ProviderSpec:
    try:
        return PROVIDERS[provider_id]
    except KeyError as exc:
        raise ValueError(f"不支持的模型厂商：{provider_id}") from exc


def infer_provider_id() -> str:
    """Infer old single-provider configurations without rewriting the file."""
    explicit = os.environ.get("TEXT_PROVIDER", "").strip().lower()
    if explicit in PROVIDERS:
        return explicit
    if os.environ.get("GLM_API_KEY"):
        return "zhipu"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY"):
        return "qwen"
    if os.environ.get("ARK_API_KEY") or os.environ.get("VOLCENGINE_API_KEY"):
        return "doubao"
    if os.environ.get("OPENAI_API_KEY"):
        legacy_url = os.environ.get("OPENAI_BASE_URL", "").lower()
        if "bigmodel" in legacy_url or "zhipu" in legacy_url:
            return "zhipu"
        if "deepseek" in legacy_url:
            return "deepseek"
        if "dashscope" in legacy_url or "aliyun" in legacy_url:
            return "qwen"
        if "volces" in legacy_url or "volcengine" in legacy_url:
            return "doubao"
        return "openai"
    return "zhipu"


def text_provider_id() -> str:
    return infer_provider_id()


def vision_provider_id() -> str:
    """Keep visual understanding on the same provider as text analysis."""
    return text_provider_id()


def provider_api_key(provider_id: str) -> str | None:
    spec = provider_spec(provider_id)
    for env_name in (spec.api_key_env, *spec.api_key_aliases):
        value = os.environ.get(env_name)
        if value:
            return value

    legacy_key = os.environ.get("OPENAI_API_KEY")
    legacy_url = os.environ.get("OPENAI_BASE_URL", "").lower()
    if not legacy_key:
        return None
    if provider_id == "zhipu" and ("bigmodel" in legacy_url or "zhipu" in legacy_url):
        return legacy_key
    if provider_id == "deepseek" and "deepseek" in legacy_url:
        return legacy_key
    if provider_id == "qwen" and ("dashscope" in legacy_url or "aliyun" in legacy_url):
        return legacy_key
    if provider_id == "doubao" and ("volces" in legacy_url or "volcengine" in legacy_url):
        return legacy_key
    if provider_id == "openai" and not any(
        marker in legacy_url
        for marker in ("bigmodel", "zhipu", "deepseek", "dashscope", "aliyun", "volces", "volcengine")
    ):
        return legacy_key
    return None


def provider_base_url(provider_id: str) -> str:
    spec = provider_spec(provider_id)
    configured = os.environ.get(spec.base_url_env)
    if configured:
        return configured.rstrip("/")

    legacy_url = os.environ.get("OPENAI_BASE_URL", "")
    if legacy_url and provider_api_key(provider_id):
        lowered = legacy_url.lower()
        matches_legacy_provider = {
            "zhipu": "bigmodel" in lowered or "zhipu" in lowered,
            "deepseek": "deepseek" in lowered,
            "qwen": "dashscope" in lowered or "aliyun" in lowered,
            "doubao": "volces" in lowered or "volcengine" in lowered,
            "openai": not any(
                marker in lowered
                for marker in (
                    "bigmodel",
                    "zhipu",
                    "deepseek",
                    "dashscope",
                    "aliyun",
                    "volces",
                    "volcengine",
                )
            ),
        }[provider_id]
        if matches_legacy_provider:
            return legacy_url.rstrip("/")
    return spec.default_base_url


def selected_text_model() -> str:
    spec = provider_spec(text_provider_id())
    configured_model = os.environ.get("MODEL_NAME", "").strip()
    if configured_model and model_is_known(spec.id, "text", configured_model):
        return configured_model
    return spec.default_text_model


def model_display_label(
    provider_id: str,
    capability: ModelCapability,
    model_id: str,
) -> str:
    """Return the user-facing catalog name while keeping API IDs internal."""
    spec = provider_spec(provider_id)
    models = spec.text_models if capability == "text" else spec.vision_models
    return next((model.label for model in models if model.id == model_id), model_id)


def selected_text_model_label() -> str:
    provider_id = text_provider_id()
    return model_display_label(provider_id, "text", selected_text_model())


def active_text_model_identity() -> str:
    """Describe the actual runtime route for prompts and user-facing status."""
    provider_id = text_provider_id()
    return f"{provider_spec(provider_id).label} / {selected_text_model_label()}"


def selected_vision_model() -> str:
    """Return the provider's fixed recommended vision model, if available."""
    return provider_spec(vision_provider_id()).default_vision_model or ""


def vision_enabled() -> bool:
    requested = os.environ.get("ENABLE_VISION_SUMMARY", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    return requested and bool(provider_spec(text_provider_id()).vision_models)


def model_is_known(provider_id: str, capability: ModelCapability, model_id: str) -> bool:
    spec = provider_spec(provider_id)
    models = spec.text_models if capability == "text" else spec.vision_models
    return any(model.id == model_id for model in models)
