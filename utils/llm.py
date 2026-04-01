"""LLM call wrapper. Switch models via .env configuration."""

import os
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)

# Temperature env var lets users override without code changes.
# Default 1.0 is required by some providers (e.g. kimi-k2.5).
_DEFAULT_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "1.0"))


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return a cached ChatOpenAI instance configured from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )

    return ChatOpenAI(
        model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=_DEFAULT_TEMPERATURE,
        max_retries=3,  # built-in openai client retry for transient errors
    )


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
