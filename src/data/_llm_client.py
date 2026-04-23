"""
src/data/_llm_client.py
Shared OpenRouter LLM client and retry utility for all data generators.
"""

from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from src.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM 팩토리
# ---------------------------------------------------------------------------

def build_llm(
    model_name: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """
    OpenRouter 기반 ChatOpenAI 인스턴스를 반환합니다.

    Parameters
    ----------
    model_name   : Model name. If None, uses models.data_gen.name from configs.yaml.
    temperature  : Sampling temperature. If None, uses models.data_gen.temperature from configs.yaml.
    """
    cfg = get_config()
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Check your .env file."
        )

    name = model_name or cfg.models.data_gen.name
    temp = temperature if temperature is not None else cfg.models.data_gen.temperature

    return ChatOpenAI(
        model=name,
        temperature=temp,
        openai_api_key=api_key,
        openai_api_base=cfg.openrouter.base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/daisysooyeon/SAP-ERP-AI-Agent",
            "X-Title": "SAP ERP AI Agent - Dataset Generator",
        },
    )


# ---------------------------------------------------------------------------
# Retry 래퍼
# ---------------------------------------------------------------------------

def invoke_with_retry(
    chain,
    inputs: dict,
    *,
    max_retries: int = 5,
    initial_wait: float = 5.0,
    label: str = "",
) -> str | None:
    """
    Calls chain.invoke(inputs) with exponential backoff on 429 rate-limit errors.

    Returns
    -------
    LLM response content string, or None if all retries fail.
    """
    wait = initial_wait
    for attempt in range(1, max_retries + 1):
        try:
            response = chain.invoke(inputs)
            content = response.content if hasattr(response, "content") else str(response)
            return content
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str and attempt < max_retries:
                logger.warning(
                    "[%s] Rate limited (429). Attempt %d/%d — waiting %.0fs …",
                    label or "llm", attempt, max_retries, wait,
                )
                time.sleep(wait)
                wait = min(wait * 2, 120)
                continue
            logger.error("[%s] LLM 호출 실패 (attempt %d): %s", label or "llm", attempt, exc)
            return None
    return None
