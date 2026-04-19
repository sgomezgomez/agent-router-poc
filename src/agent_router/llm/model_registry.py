"""Model registry for provider capabilities and costs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal

import yaml
from pydantic import BaseModel, ConfigDict

class ModelCosts(BaseModel):
    """Cost metadata for a model."""

    model_config = ConfigDict(extra="forbid")

    input_per_1m_usd: float | None = None
    output_per_1m_usd: float | None = None
    thinking_per_1m_usd: float | None = None

class ModelCapabilities(BaseModel):
    """Capabilities for a model."""

    model_config = ConfigDict(extra="forbid")

    supports_temperature: bool | None = None
    supports_top_p: bool | None = None
    supports_top_k: bool | None = None
    supports_tools: bool | None = None
    supports_response_format: bool | None = None
    supports_thinking_effort: bool | None = None
    supports_thinking_budget: bool | None = None
    supports_enable_thinking: bool | None = None
    tool_schema: Literal["chat_completions", "responses"] | None = None
    tools_api_mode: Literal["chat_completions", "responses"] | None = None # openai-compatible only
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] | None = None
    api_mode: Literal["chat_completions", "responses"] | None = None
    costs: ModelCosts | None = None

class ProviderModels(BaseModel):
    """Models for a provider."""

    model_config = ConfigDict(extra="forbid")

    models: Dict[str, ModelCapabilities] = {}

class LLMModelRegistry(BaseModel):
    """Registry for provider model capabilities."""

    model_config = ConfigDict(extra="forbid")

    providers: Dict[str, ProviderModels] = {}

    @classmethod
    def from_yaml(cls, path: Path) -> "LLMModelRegistry":
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls(**raw)

    def get_model_capabilities(
        self,
        provider: str,
        model: str
    ) -> ModelCapabilities | None:
        provider_models = self.providers.get(provider)
        if not provider_models:
            return None
        return provider_models.models.get(model)

def load_model_registry() -> LLMModelRegistry:
    """Load registry from repo-root config/llm_models.yaml."""
    root = Path(__file__).resolve().parents[3]
    return LLMModelRegistry.from_yaml(root / "config" / "llm_models.yaml")
