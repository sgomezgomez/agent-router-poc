"""Agent configuration models and loader."""

from __future__ import annotations

from pathlib import Path
from agent_router.core.types import JsonObject

import yaml
from pydantic import BaseModel, Field, ConfigDict


class AgentConfig(BaseModel):
    """Configuration for a single agent persona."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(description="Agent name")
    background: str = Field(description="Agent background/persona")
    goal: str = Field(description="Primary goal")
    task: str = Field(description="Primary task description")
    guidelines: str = Field(description="Behavioral guidelines")
    examples: str = Field(description="Examples for the agent")
    guardrails: str = Field(description="Safety or operational guardrails")

    # Optional LLM defaults (agent-specific overrides)
    provider: str | None = None
    model: str | None = None
    api_mode: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    thinking_budget: int | None = None
    thinking_effort: str | None = None

    @classmethod
    def from_yaml(cls, path: Path, agent_name: str) -> "AgentConfig":
        """Load agent config from a YAML file under config/agents."""
        with path.open("r", encoding="utf-8") as f:
            data: dict[str, JsonObject] = yaml.safe_load(f) or {}

        if agent_name not in data:
            raise ValueError(f"Agent '{agent_name}' not found in {path}")

        payload = data[agent_name] | {"name": agent_name}
        return cls(**payload)
