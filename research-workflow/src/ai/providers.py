import os
from typing import Optional

import tiktoken
from openai import OpenAI

from src.ai.text_splitter import RecursiveCharacterTextSplitter

# ── Providers ──────────────────────────────────────────────────────────────
_openai_client: Optional[OpenAI] = None

# Determine which API key and endpoint to use
_api_key = os.environ.get("OPENAI_KEY") or os.environ.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORKS_KEY")
_api_endpoint = os.environ.get(
    "OPENAI_ENDPOINT",
    "https://api.fireworks.ai/inference/v1" if os.environ.get("FIREWORKS_API_KEY") or os.environ.get("FIREWORKS_KEY") else "https://api.openai.com/v1",
)

if _api_key:
    _openai_client = OpenAI(
        api_key=_api_key,
        base_url=_api_endpoint,
    )

# Model name
_custom_model: Optional[str] = os.environ.get("CUSTOM_MODEL")

_MIN_CHUNK_SIZE = 140
_encoder = tiktoken.get_encoding("o200k_base")


def get_client() -> OpenAI:
    """Return the OpenAI client instance."""
    if _openai_client is None:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_KEY.")
    return _openai_client


def get_model_id() -> str:
    """Return the model id to use (mirrors getModel() in TS)."""
    if _custom_model:
        return _custom_model
    return "o3-mini"


# ── trim_prompt ────────────────────────────────────────────────────────────

def trim_prompt(
    prompt: str,
    context_size: Optional[int] = None,
) -> str:
    """Trim prompt to fit within the context size (tokens)."""
    if context_size is None:
        context_size = int(os.environ.get("CONTEXT_SIZE", "128000"))
    if not prompt:
        return ""

    length = len(_encoder.encode(prompt))
    if length <= context_size:
        return prompt

    overflow_tokens = length - context_size
    # ~3 characters per token
    chunk_size = len(prompt) - overflow_tokens * 3
    if chunk_size < _MIN_CHUNK_SIZE:
        return prompt[:_MIN_CHUNK_SIZE]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )
    trimmed = splitter.split_text(prompt)
    trimmed_prompt = trimmed[0] if trimmed else ""

    # Hard-cut fallback
    if len(trimmed_prompt) == len(prompt):
        return trim_prompt(prompt[:chunk_size], context_size)

    # Recursively trim
    return trim_prompt(trimmed_prompt, context_size)
