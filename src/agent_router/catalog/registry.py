"""Catalog registry for concrete runtime workers."""

from __future__ import annotations

from pathlib import Path

from agent_router.agent import Agent, AgentConfig
from agent_router.llm.service import LLMService
from agent_router.orchestrator import Orchestrator
from agent_router.storage import ConversationRepository, LLMCallRepository


CATALOG_CONFIG = {
    "router": {
        "path": Path("config/agents/router.yaml"),
        "key": "router",
    },
    "tool_formatter": {
        "path": Path("config/agents/tool_formatter.yaml"),
        "key": "tool_formatter",
    },
}


def list_catalog_agents() -> list[str]:
    """Return available catalog agent names."""
    return sorted(CATALOG_CONFIG.keys())


def build_catalog_agent(
    *,
    name: str,
    config_path: Path | None,
    llm_service: LLMService,
    conversation_repo: ConversationRepository,
    llm_call_repo: LLMCallRepository,
    orchestrator: Orchestrator,
) -> Agent:
    """Build a concrete catalog agent by name."""
    catalog_entry = CATALOG_CONFIG.get(name)
    if catalog_entry is None:
        available = ", ".join(list_catalog_agents())
        raise ValueError(f"Unknown catalog agent '{name}'. Available: {available}")

    resolved_path = config_path or catalog_entry["path"]
    config_key = catalog_entry["key"]
    agent_config = AgentConfig.from_yaml(resolved_path, config_key)
    return Agent(
        config=agent_config,
        llm_service=llm_service,
        conversation_repo=conversation_repo,
        llm_call_repo=llm_call_repo,
        orchestrator=orchestrator,
    )
