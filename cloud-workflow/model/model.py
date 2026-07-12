from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from dynamic_cloud.config import LlmSettings, load_environment


FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/gemma-4-26b-a4b-it"


@dataclass(frozen=True)
class ModelConfig:
    model: str
    api_key: str
    base_url: str


@dataclass(frozen=True)
class CompatResponse:
    content: str


class CentralModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    @property
    def model_name(self) -> str:
        return self.config.model

    def complete_json(self, system_prompt: str, user_payload: Any) -> str:
        content = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, indent=2)
        response = self._chat_completion(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return response.choices[0].message.content or ""

    def invoke(self, messages: list[tuple[str, str]] | list[dict[str, Any]]) -> CompatResponse:
        response = self._chat_completion(self._normalize_messages(messages))
        return CompatResponse(content=response.choices[0].message.content or "")

    def _chat_completion(self, messages: list[dict[str, str]]) -> Any:
        return self.client.chat.completions.create(model=self.config.model, messages=messages)

    @staticmethod
    def _normalize_messages(messages: list[tuple[str, str]] | list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, dict):
                role = str(message.get("role") or "user")
                content = message.get("content") or ""
            else:
                role, content = message
            if role == "human":
                role = "user"
            normalized.append({"role": role, "content": content if isinstance(content, str) else json.dumps(content)})
        return normalized


def model_config_from_env(slot: str = "MODEL1") -> ModelConfig:
    load_environment()
    prefix = slot.upper()
    model_name = os.getenv(prefix) or os.getenv(f"{prefix}_MODEL") or DEFAULT_MODEL
    api_key = os.getenv("FIREWORKS_API_KEY") or os.getenv(f"{prefix}_API_KEY") or ""
    base_url = os.getenv("FIREWORKS_BASE_URL") or FIREWORKS_BASE_URL
    return ModelConfig(model=model_name, api_key=api_key, base_url=base_url)


def model_config_from_settings(settings: LlmSettings | None, slot: str = "MODEL1") -> ModelConfig:
    if settings is None:
        return model_config_from_env(slot)

    load_environment()
    prefix = slot.upper()
    base_url = settings.base_url or os.getenv("FIREWORKS_BASE_URL") or FIREWORKS_BASE_URL
    model_name = settings.model or os.getenv(prefix) or os.getenv(f"{prefix}_MODEL") or DEFAULT_MODEL
    api_key = settings.api_key or os.getenv("FIREWORKS_API_KEY") or os.getenv(f"{prefix}_API_KEY") or ""
    return ModelConfig(model=model_name, api_key=api_key, base_url=base_url)


def make_model(settings: LlmSettings | None = None, slot: str = "MODEL1") -> CentralModel:
    return CentralModel(model_config_from_settings(settings, slot))


model1 = CentralModel(model_config_from_env("MODEL1"))
model2 = CentralModel(model_config_from_env("MODEL2"))
model = model1
