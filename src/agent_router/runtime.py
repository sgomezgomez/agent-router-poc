"""Routing agent wiring for the POC."""

from __future__ import annotations

from pathlib import Path
import asyncio

from agent_router.agent import Agent
from agent_router.catalog import build_catalog_agent
from agent_router.connectors.mcp import (
    create_mcp_clients_with_status,
    load_mcp_server_configs,
    MCPClient,
)
from agent_router.core.config import Settings
from agent_router.llm.service import LLMService
from agent_router.orchestrator import Orchestrator
from agent_router.storage import (
    ConversationRepository,
    LLMCallRepository,
    MongoDBConnection,
    ToolExecutionRepository,
)


class RouterRuntime:
    """Runtime wiring for the routing agent."""

    def __init__(
        self,
        *,
        agent: Agent,
        mcp_clients: dict[str, MCPClient],
        mcp_status: list[dict[str, str]] | None = None,
    ) -> None:
        self.agent = agent
        self.mcp_clients = mcp_clients
        self.mcp_status = mcp_status or []

    @classmethod
    async def create(
        cls,
        *,
        agent_name: str = "router",
        agent_config_path: Path | None = None,
        mcp_config_path: Path | None = None,
        create_indexes: bool = True,
    ) -> "RouterRuntime":
        settings = Settings()

        await MongoDBConnection.initialize(settings.mongodb)
        if create_indexes:
            await MongoDBConnection.create_indexes()

        llm_service = LLMService(settings)
        conversation_repo = ConversationRepository()
        llm_call_repo = LLMCallRepository()
        tool_execution_repo = ToolExecutionRepository()

        mcp_configs = load_mcp_server_configs(mcp_config_path)
        mcp_clients, mcp_status = await create_mcp_clients_with_status(mcp_configs)

        orchestrator = Orchestrator(
            tool_execution_repo=tool_execution_repo,
            mcp_clients=mcp_clients,
        )

        config_path = agent_config_path or Path("config/agents/router.yaml")
        agent = build_catalog_agent(
            name=agent_name,
            config_path=config_path,
            llm_service=llm_service,
            conversation_repo=conversation_repo,
            llm_call_repo=llm_call_repo,
            orchestrator=orchestrator,
        )

        try:
            formatter_agent = build_catalog_agent(
                name="tool_formatter",
                config_path=None,
                llm_service=llm_service,
                conversation_repo=conversation_repo,
                llm_call_repo=llm_call_repo,
                orchestrator=orchestrator,
            )
        except ValueError:
            formatter_agent = None
        agent.tool_result_formatter = formatter_agent

        if mcp_clients:
            await agent.load_mcp_tools(mcp_clients.values())

        return cls(agent=agent, mcp_clients=mcp_clients, mcp_status=mcp_status)

    async def process_query(self, query: str, *, stream: bool = False):
        return await self.agent.process_query(query, stream=stream)

    async def close(self) -> None:
        for name, client in self.mcp_clients.items():
            try:
                await asyncio.wait_for(client.close(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            except (RuntimeError, asyncio.CancelledError, BaseExceptionGroup):
                pass
        try:
            await asyncio.wait_for(MongoDBConnection.close(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


__all__ = ["RouterRuntime"]
