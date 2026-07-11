"""Generate concise, stable titles for paper chat conversations."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from utils.llm import get_chat_llm, invoke_with_retry, is_llm_configured


MAX_TITLE_CHARS = 32


def local_conversation_title(question: str) -> str:
    """Return an immediate local summary while the model title is generated."""
    title = " ".join(question.split())
    title = re.sub(
        r"^(?:请问|请你|请帮我|帮我|麻烦你|我想问一下|我想了解一下|能否|能不能|可以|可不可以)\s*",
        "",
        title,
    )
    title = re.sub(r"[？?！!。；;]+$", "", title).strip(" \"'“”‘’")
    if not title:
        return "新对话"
    if len(title) <= MAX_TITLE_CHARS:
        return title
    boundary = max(title.rfind(mark, 0, MAX_TITLE_CHARS) for mark in ("，", ",", "：", ":", " "))
    cutoff = boundary if boundary >= 12 else MAX_TITLE_CHARS - 1
    return title[:cutoff].rstrip(" ，,:：") + "…"


def generate_conversation_title(question: str) -> str:
    """Use the configured chat model to summarize the first user question."""
    fallback = local_conversation_title(question)
    if not is_llm_configured():
        return fallback
    messages = [
        SystemMessage(
            content=(
                "你负责为论文问答会话生成简洁标题。概括用户真正询问的主题和意图，"
                "不要直接复制整句。中文标题控制在8到20个汉字左右；英文标题不超过8个单词。"
                "保留论文名、模型名、算法名、公式符号和重要缩写。"
                "不要使用引号、句号、问号、冒号，不要添加“会话标题”或解释。只返回标题。"
            )
        ),
        HumanMessage(content=f"用户的第一条问题：\n{question.strip()}"),
    ]
    try:
        response = invoke_with_retry(
            get_chat_llm().bind(max_tokens=48),
            messages,
            retries=1,
            delay=0.5,
        )
        title = _clean_model_title(_content_to_text(getattr(response, "content", response)))
        return title or fallback
    except Exception:
        return fallback


def _clean_model_title(value: str) -> str:
    title = " ".join(value.replace("```", "").split())
    title = re.sub(r"^(?:标题|会话标题|Title)\s*[:：]\s*", "", title, flags=re.IGNORECASE)
    title = title.strip(" \"'“”‘’《》")
    title = re.sub(r"[？?！!。；;]+$", "", title)
    if len(title) > MAX_TITLE_CHARS:
        title = title[: MAX_TITLE_CHARS - 1].rstrip(" ，,:：") + "…"
    return title


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)
