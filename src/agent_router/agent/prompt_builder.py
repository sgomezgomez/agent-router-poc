"""Prompt building utilities for agents."""

from __future__ import annotations

from agent_router.core.types import JsonObject

from agent_router.llm.utils import clean_encoded_text

from .config import AgentConfig


class PromptBuilder:
    """Build system prompts from agent configuration."""

    def __init__(self, config: AgentConfig):
        self.config = config

    def build_system_prompt(self, input_variables: JsonObject | None = None) -> str:
        """Build the system prompt by formatting config blocks."""
        variables = dict(input_variables or {})
        variables["agent_name"] = self.config.name

        def fmt(value: str) -> str:
            try:
                return value.format(**variables)
            except KeyError:
                return value

        prompt_parts = [
            f"<background>{fmt(self.config.background)}</background>",
            f"<goal>{fmt(self.config.goal)}</goal>",
            f"<task>{fmt(self.config.task)}</task>",
            f"<guidelines>{fmt(self.config.guidelines)}</guidelines>",
            f"<examples>{fmt(self.config.examples)}</examples>",
            f"<guardrails>{fmt(self.config.guardrails)}</guardrails>",
        ]

        return clean_encoded_text("\n".join(prompt_parts))
