from __future__ import annotations

import json
from typing import Any

from dynamic_cloud.config import LlmSettings, load_llm_settings


def create_openai_client(settings: LlmSettings | None = None) -> Any:
    from model import make_model, model1

    return make_model(settings).client if settings else model1.client


def generate_json(system_prompt: str, user_payload: Any, settings: LlmSettings | None = None) -> dict[str, Any]:
    settings = settings or load_llm_settings()
    from model import make_model

    return generate_json_with_model(make_model(settings), system_prompt, user_payload)


def generate_json_with_client(
    client: Any,
    settings: LlmSettings,
    system_prompt: str,
    user_payload: Any,
) -> dict[str, Any]:
    from model import make_model

    return generate_json_with_model(make_model(settings), system_prompt, user_payload)


def generate_json_with_model(model_handle: Any, system_prompt: str, user_payload: Any) -> dict[str, Any]:
    content = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, indent=2)
    response_text = model_handle.complete_json(system_prompt, content)
    return parse_json_response(response_text)


def parse_json_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        payload = _load_first_balanced_json_object(text, exc)
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON must be an object.")
    return payload


def _load_first_balanced_json_object(text: str, original_error: json.JSONDecodeError) -> dict[str, Any]:
    in_string = False
    escaped = False
    depth = 0
    start: int | None = None

    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    payload = json.loads(candidate)
                except json.JSONDecodeError:
                    # LLMs often embed real newlines inside JSON string values
                    # instead of \\n escape sequences.  Try to salvage by
                    # escaping unescaped control characters inside strings.
                    sanitized = _sanitize_json_strings(candidate)
                    try:
                        payload = json.loads(sanitized)
                    except json.JSONDecodeError:
                        start = None
                        continue
                if isinstance(payload, dict):
                    return payload
                start = None

    raise ValueError(f"LLM did not return valid JSON: {original_error}\n{text}") from original_error


def _sanitize_json_strings(text: str) -> str:
    """Return a version of *text* where unescaped control characters (newlines,
    tabs, etc.) inside JSON string literals are replaced with their two-character
    escape sequences (\\n, \\t, etc.).  This is a best-effort repair for LLM
    responses that embed real newlines in JSON string values."""
    result: list[str] = []
    in_string = False
    escape_map = {
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
        "\f": "\\f",
        "\b": "\\b",
    }
    i = 0
    while i < len(text):
        ch = text[i]
        if not in_string:
            result.append(ch)
            if ch == '"':
                in_string = True
            i += 1
        else:
            if ch == "\\" and i + 1 < len(text):
                result.append(ch)
                result.append(text[i + 1])
                i += 2
            elif ch == '"':
                result.append(ch)
                in_string = False
                i += 1
            elif ch in escape_map:
                result.append(escape_map[ch])
                i += 1
            else:
                result.append(ch)
                i += 1
    return "".join(result)
