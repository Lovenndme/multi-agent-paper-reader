"""Provider catalog and runtime model routing helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


ModelCapability = Literal["text", "vision"]
ApiProtocol = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class ModelModeSpec:
    id: str
    label: str
    description: str
    request_body: dict[str, object] = field(default_factory=dict)

    def payload(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label, "description": self.description}


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    description: str
    recommended: bool = False
    tags: tuple[str, ...] = ()
    modes: tuple[ModelModeSpec, ...] = ()

    def payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "recommended": self.recommended,
            "tags": list(self.tags),
            "modes": [mode.payload() for mode in self.modes],
            "default_mode": self.modes[0].id if self.modes else None,
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
    protocol: ApiProtocol = "openai"
    customizable: bool = False

    @property
    def default_text_model(self) -> str:
        return self.text_models[0].id

    @property
    def default_vision_model(self) -> str | None:
        return self.vision_models[0].id if self.vision_models else None


PROVIDERS: dict[str, ProviderSpec] = {
    "zhipu": ProviderSpec(
        id="zhipu",
        label="GLM",
        api_key_env="GLM_API_KEY",
        base_url_env="GLM_BASE_URL",
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        key_url="https://bigmodel.cn/usercenter/proj-mgmt/apikeys",
        text_models=(
            ModelSpec(
                "glm-5.2",
                "GLM-5.2",
                "旗舰文本与推理模型",
                True,
                ("旗舰", "推理"),
                (
                    ModelModeSpec("standard", "标准思考", "开启思考并使用默认推理强度", {"thinking": {"type": "enabled"}}),
                    ModelModeSpec("deep", "深度思考", "最大化推理强度，适合高难度论文任务", {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}),
                    ModelModeSpec("fast", "快速响应", "关闭思考过程，直接生成回答", {"thinking": {"type": "disabled"}}),
                ),
            ),
            ModelSpec("glm-5.1", "GLM-5.1", "高能力通用文本模型", tags=("通用",), modes=(
                ModelModeSpec("standard", "标准思考", "开启模型思考过程", {"thinking": {"type": "enabled"}}),
                ModelModeSpec("fast", "快速响应", "关闭思考过程", {"thinking": {"type": "disabled"}}),
            )),
            ModelSpec("glm-5", "GLM-5", "上一代旗舰文本模型", tags=("通用",), modes=(
                ModelModeSpec("standard", "标准思考", "开启模型思考过程", {"thinking": {"type": "enabled"}}),
                ModelModeSpec("fast", "快速响应", "关闭思考过程", {"thinking": {"type": "disabled"}}),
            )),
            ModelSpec("glm-5-turbo", "GLM-5 Turbo", "低延迟 GLM-5 系列模型", tags=("高速",), modes=(
                ModelModeSpec("standard", "标准思考", "开启模型思考过程", {"thinking": {"type": "enabled"}}),
                ModelModeSpec("fast", "快速响应", "关闭思考过程", {"thinking": {"type": "disabled"}}),
            )),
            ModelSpec("glm-4.7", "GLM-4.7", "稳定的复杂任务模型", tags=("稳定",), modes=(
                ModelModeSpec("standard", "标准思考", "开启模型思考过程", {"thinking": {"type": "enabled"}}),
                ModelModeSpec("fast", "快速响应", "关闭思考过程", {"thinking": {"type": "disabled"}}),
            )),
            ModelSpec("glm-4.6", "GLM-4.6", "通用文本与工具调用模型", tags=("通用",), modes=(
                ModelModeSpec("standard", "标准思考", "允许模型按任务自动判断是否思考", {"thinking": {"type": "enabled"}}),
                ModelModeSpec("fast", "快速响应", "关闭思考过程", {"thinking": {"type": "disabled"}}),
            )),
            ModelSpec("glm-4.5-air", "GLM-4.5 Air", "轻量高性价比模型", tags=("轻量",), modes=(
                ModelModeSpec("standard", "标准思考", "允许模型按任务自动判断是否思考", {"thinking": {"type": "enabled"}}),
                ModelModeSpec("fast", "快速响应", "关闭思考过程", {"thinking": {"type": "disabled"}}),
            )),
            ModelSpec("glm-4.5", "GLM-4.5", "上一代通用基座模型", tags=("通用",), modes=(
                ModelModeSpec("standard", "标准思考", "允许模型按任务自动判断是否思考", {"thinking": {"type": "enabled"}}),
                ModelModeSpec("fast", "快速响应", "关闭思考过程", {"thinking": {"type": "disabled"}}),
            )),
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
                "glm-4.1v-thinking",
                "GLM-4.1V Thinking",
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
                (
                    ModelModeSpec("enabled", "深度思考", "优先推理质量，适合复杂论文分析", {"thinking": {"type": "enabled"}}),
                    ModelModeSpec("disabled", "快速响应", "关闭思考过程，降低响应延迟", {"thinking": {"type": "disabled"}}),
                ),
            ),
            ModelSpec(
                "deepseek-v4-flash",
                "DeepSeek V4 Flash",
                "低延迟高吞吐文本模型",
                tags=("高速",),
                modes=(
                    ModelModeSpec("disabled", "快速响应", "关闭思考过程，发挥 Flash 低延迟优势", {"thinking": {"type": "disabled"}}),
                    ModelModeSpec("enabled", "深度思考", "开启推理过程以处理复杂问题", {"thinking": {"type": "enabled"}}),
                ),
            ),
        ),
    ),
    "anthropic": ProviderSpec(
        id="anthropic",
        label="Anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        base_url_env="ANTHROPIC_BASE_URL",
        default_base_url="https://api.anthropic.com",
        key_url="https://console.anthropic.com/settings/keys",
        protocol="anthropic",
        text_models=(
            ModelSpec("claude-fable-5", "Claude Fable 5", "长程 Agent 与复杂研究的最高能力模型", True, ("旗舰", "1M")),
            ModelSpec("claude-sonnet-5", "Claude Sonnet 5", "速度与智能兼顾的新一代主力模型", tags=("推荐", "1M")),
            ModelSpec("claude-opus-4-8", "Claude Opus 4.8", "复杂 Agent、企业任务与视觉工作流", tags=("旗舰", "1M")),
            ModelSpec("claude-opus-4-7", "Claude Opus 4.7", "稳定的高能力 Agent 模型", tags=("高能力",)),
            ModelSpec("claude-opus-4-6", "Claude Opus 4.6", "复杂研究与长程工程模型", tags=("高能力",)),
            ModelSpec("claude-sonnet-4-6", "Claude Sonnet 4.6", "成熟的均衡型生产模型", tags=("均衡",)),
            ModelSpec("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "低延迟、低成本的快速模型", tags=("高速",)),
        ),
        vision_models=(
            ModelSpec("claude-fable-5", "Claude Fable 5", "最高能力论文图像与图表理解", True, ("旗舰", "视觉")),
            ModelSpec("claude-sonnet-5", "Claude Sonnet 5", "高质量、低延迟视觉理解", tags=("推荐", "视觉")),
            ModelSpec("claude-opus-4-8", "Claude Opus 4.8", "复杂视觉推理与长文档工作流", tags=("旗舰", "视觉")),
            ModelSpec("claude-opus-4-7", "Claude Opus 4.7", "高能力视觉推理", tags=("高能力", "视觉")),
            ModelSpec("claude-opus-4-6", "Claude Opus 4.6", "稳定的复杂视觉理解", tags=("高能力", "视觉")),
            ModelSpec("claude-sonnet-4-6", "Claude Sonnet 4.6", "均衡的论文图像理解", tags=("均衡", "视觉")),
            ModelSpec("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "快速视觉理解", tags=("高速", "视觉")),
        ),
    ),
    "kimi": ProviderSpec(
        id="kimi",
        label="Kimi",
        api_key_env="MOONSHOT_API_KEY",
        base_url_env="KIMI_BASE_URL",
        default_base_url="https://api.moonshot.cn/v1",
        key_url="https://platform.kimi.com/console/api-keys",
        text_models=(
            ModelSpec(
                "kimi-k2.6",
                "Kimi K2.6",
                "最新旗舰多模态 Agent 与长上下文模型",
                True,
                ("旗舰", "256K"),
                (
                    ModelModeSpec("enabled", "深度思考", "开启长思考，适合复杂论文推理", {"thinking": {"type": "enabled"}}),
                    ModelModeSpec("disabled", "快速响应", "关闭思考过程，优先返回速度", {"thinking": {"type": "disabled"}}),
                ),
            ),
            ModelSpec(
                "kimi-k2.5",
                "Kimi K2.5",
                "上一代多模态推理模型",
                tags=("稳定", "256K"),
                modes=(
                    ModelModeSpec("enabled", "深度思考", "开启思考过程", {"thinking": {"type": "enabled"}}),
                    ModelModeSpec("disabled", "快速响应", "关闭思考过程", {"thinking": {"type": "disabled"}}),
                ),
            ),
        ),
        vision_models=(
            ModelSpec("kimi-k2.6", "Kimi K2.6", "最新原生图像、视频与图表理解", True, ("旗舰", "视觉")),
            ModelSpec("kimi-k2.5", "Kimi K2.5", "上一代原生多模态理解", tags=("稳定", "视觉")),
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
            ModelSpec("gpt-5.6-sol", "GPT-5.6 Sol", "旗舰通用与推理模型", True, ("旗舰", "推理")),
            ModelSpec("gpt-5.6-terra", "GPT-5.6 Terra", "面向高难度长任务", tags=("高能力",)),
            ModelSpec("gpt-5.6-luna", "GPT-5.6 Luna", "兼顾速度与质量", tags=("均衡",)),
        ),
        vision_models=(
            ModelSpec("gpt-5.6-sol", "GPT-5.6 Sol", "文本与图像联合理解", True, ("旗舰", "视觉")),
            ModelSpec("gpt-5.6-terra", "GPT-5.6 Terra", "复杂图表与视觉推理", tags=("高能力",)),
            ModelSpec("gpt-5.6-luna", "GPT-5.6 Luna", "高效多模态理解", tags=("均衡",)),
        ),
    ),
    "qwen": ProviderSpec(
        id="qwen",
        label="Qwen",
        api_key_env="DASHSCOPE_API_KEY",
        base_url_env="QWEN_BASE_URL",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        key_url="https://bailian.console.aliyun.com/?tab=model#/api-key",
        api_key_aliases=("QWEN_API_KEY",),
        text_models=(
            ModelSpec("qwen3.7-max", "Qwen3.7 Max", "旗舰文本与复杂推理模型", True, ("旗舰", "推理"), (
                ModelModeSpec("thinking", "深度思考", "启用思考并使用模型默认最大思考预算", {"enable_thinking": True}),
                ModelModeSpec("fast", "快速响应", "关闭思考过程，直接生成回答", {"enable_thinking": False}),
            )),
            ModelSpec("qwen3.7-plus", "Qwen3.7 Plus", "兼顾能力与成本的通用模型", tags=("均衡",), modes=(
                ModelModeSpec("thinking", "深度思考", "启用思考并使用模型默认最大思考预算", {"enable_thinking": True}),
                ModelModeSpec("fast", "快速响应", "关闭思考过程，直接生成回答", {"enable_thinking": False}),
            )),
            ModelSpec("qwen3.6-plus", "Qwen3.6 Plus", "成熟稳定的百万上下文多模态模型", tags=("稳定",), modes=(
                ModelModeSpec("thinking", "深度思考", "启用思考并使用模型默认最大思考预算", {"enable_thinking": True}),
                ModelModeSpec("fast", "快速响应", "关闭思考过程，直接生成回答", {"enable_thinking": False}),
            )),
            ModelSpec("qwen3.6-flash", "Qwen3.6 Flash", "低延迟高吞吐模型", tags=("高速",), modes=(
                ModelModeSpec("thinking", "深度思考", "启用思考并使用模型默认最大思考预算", {"enable_thinking": True}),
                ModelModeSpec("fast", "快速响应", "关闭思考过程，直接生成回答", {"enable_thinking": False}),
            )),
            ModelSpec("qwen-long", "Qwen Long", "面向超长文档任务", tags=("长上下文",)),
        ),
        vision_models=(
            ModelSpec("qwen3.7-plus", "Qwen3.7 Plus", "原生文本与图像联合理解", True, ("推荐", "视觉")),
            ModelSpec("qwen3.6-plus", "Qwen3.6 Plus", "稳定的图像、视频与文档理解", tags=("稳定", "视觉")),
            ModelSpec("qwen3.6-flash", "Qwen3.6 Flash", "百万上下文低延迟视觉理解", tags=("高速", "视觉")),
            ModelSpec("qwen3.5-omni-plus", "Qwen3.5 Omni Plus", "全模态内容理解", tags=("全模态",)),
            ModelSpec("qwen3-vl-flash", "Qwen3 VL Flash", "低延迟视觉语言模型", tags=("高速",)),
            ModelSpec("qwen3-vl-plus", "Qwen3 VL Plus", "高精度图像与图表理解", tags=("视觉",)),
        ),
    ),
    "doubao": ProviderSpec(
        id="doubao",
        label="Doubao",
        api_key_env="ARK_API_KEY",
        base_url_env="DOUBAO_BASE_URL",
        default_base_url="https://ark.cn-beijing.volces.com/api/v3",
        key_url="https://console.volcengine.com/ark/region:ark+cn-beijing/apikey",
        api_key_aliases=("VOLCENGINE_API_KEY",),
        text_models=(
            ModelSpec(
                "doubao-seed-2.1-pro",
                "Doubao Seed 2.1 Pro",
                "最新旗舰 Coding、Agent 与多模态模型",
                True,
                ("旗舰", "推理"),
            ),
            ModelSpec(
                "doubao-seed-2.1-turbo",
                "Doubao Seed 2.1 Turbo",
                "面向规模化生产的均衡多模态模型",
                tags=("均衡",),
            ),
            ModelSpec(
                "doubao-seed-evolving",
                "Doubao Seed Evolving",
                "自动跟随最强版本的周级更新模型",
                tags=("滚动更新",),
            ),
        ),
        vision_models=(
            ModelSpec(
                "doubao-seed-2.1-pro",
                "Doubao Seed 2.1 Pro",
                "最新高精度图像、视频与图表理解",
                True,
                ("旗舰", "视觉"),
            ),
            ModelSpec(
                "doubao-seed-2.1-turbo",
                "Doubao Seed 2.1 Turbo",
                "适合规模化生产的多模态理解",
                tags=("均衡",),
            ),
            ModelSpec(
                "doubao-seed-evolving",
                "Doubao Seed Evolving",
                "自动跟随最强多模态版本",
                tags=("滚动更新",),
            ),
        ),
    ),
    "custom": ProviderSpec(
        id="custom",
        label="自定义中转站",
        api_key_env="CUSTOM_API_KEY",
        base_url_env="CUSTOM_BASE_URL",
        default_base_url="",
        key_url="",
        protocol="openai",
        customizable=True,
        text_models=(ModelSpec("custom-model", "自定义文本模型", "由中转站提供的模型 ID", True),),
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
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("MOONSHOT_API_KEY"):
        return "kimi"
    if os.environ.get("CUSTOM_API_KEY") and os.environ.get("CUSTOM_BASE_URL"):
        return "custom"
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
        }.get(provider_id, False)
        if matches_legacy_provider:
            return legacy_url.rstrip("/")
    return spec.default_base_url


def selected_text_model() -> str:
    spec = provider_spec(text_provider_id())
    configured_model = os.environ.get("MODEL_NAME", "").strip()
    if spec.customizable and configured_model:
        return configured_model
    if configured_model and model_is_known(spec.id, "text", configured_model):
        return configured_model
    return spec.default_text_model


def model_modes(provider_id: str, model_id: str) -> tuple[ModelModeSpec, ...]:
    """Return the documented request modes for a catalog model."""
    spec = provider_spec(provider_id)
    if spec.customizable:
        return ()
    model = next((item for item in spec.text_models if item.id == model_id), None)
    return model.modes if model else ()


def model_mode_request_body(provider_id: str, model_id: str, mode_id: str) -> dict[str, object]:
    """Return the provider-specific request fields for one documented mode."""
    mode = next((item for item in model_modes(provider_id, model_id) if item.id == mode_id), None)
    return dict(mode.request_body) if mode else {}


def selected_text_mode() -> str:
    """Return a valid mode for the selected model, or an empty value."""
    provider_id = text_provider_id()
    modes = model_modes(provider_id, selected_text_model())
    if not modes:
        return ""
    configured = os.environ.get("MODEL_MODE", "").strip().lower()
    if any(mode.id == configured for mode in modes):
        return configured
    return modes[0].id


def model_display_label(
    provider_id: str,
    capability: ModelCapability,
    model_id: str,
) -> str:
    """Return the user-facing catalog name while keeping API IDs internal."""
    spec = provider_spec(provider_id)
    if spec.customizable:
        return model_id
    models = spec.text_models if capability == "text" else spec.vision_models
    return next((model.label for model in models if model.id == model_id), model_id)


def selected_text_model_label() -> str:
    provider_id = text_provider_id()
    return model_display_label(provider_id, "text", selected_text_model())


def active_text_model_identity() -> str:
    """Describe the actual runtime route for prompts and user-facing status."""
    provider_id = text_provider_id()
    return f"{provider_label(provider_id)} / {selected_text_model_label()}"


def selected_vision_model() -> str:
    """Return the provider's fixed recommended vision model, if available."""
    spec = provider_spec(vision_provider_id())
    if spec.customizable:
        return os.environ.get("VISION_MODEL_NAME", "").strip()
    return spec.default_vision_model or ""


def vision_enabled() -> bool:
    requested = os.environ.get("ENABLE_VISION_SUMMARY", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    spec = provider_spec(text_provider_id())
    supports_vision = bool(selected_vision_model()) if spec.customizable else bool(spec.vision_models)
    return requested and supports_vision


def model_is_known(provider_id: str, capability: ModelCapability, model_id: str) -> bool:
    spec = provider_spec(provider_id)
    if spec.customizable:
        return bool(model_id.strip())
    models = spec.text_models if capability == "text" else spec.vision_models
    return any(model.id == model_id for model in models)


def provider_protocol(provider_id: str) -> ApiProtocol:
    """Return the wire protocol used by a provider or custom relay."""
    spec = provider_spec(provider_id)
    if spec.customizable:
        configured = os.environ.get("CUSTOM_API_PROTOCOL", "openai").strip().lower()
        return "anthropic" if configured == "anthropic" else "openai"
    return spec.protocol


def provider_label(provider_id: str) -> str:
    """Return the configured relay name without allowing an empty UI label."""
    spec = provider_spec(provider_id)
    if spec.customizable:
        return os.environ.get("CUSTOM_PROVIDER_NAME", "").strip() or spec.label
    return spec.label
