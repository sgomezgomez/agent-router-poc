"""Manual test for Agent tool calling via Orchestrator + MCP."""

import asyncio
from pathlib import Path

import yaml

from agent_router.agent import Agent, AgentConfig
from agent_router.core.config import Settings
from agent_router.llm.service import LLMService
from agent_router.orchestrator import Orchestrator
from agent_router.connectors.mcp.client import MCPClient
from agent_router.storage import (
    MongoDBConnection,
    ConversationRepository,
    LLMCallRepository,
    ToolExecutionRepository,
)


async def main() -> int:
    settings = Settings()
    await MongoDBConnection.initialize(settings.mongodb)

    repo_root = Path(__file__).resolve().parents[2]
    config = AgentConfig.from_yaml(
        path=repo_root / "config" / "agents" / "router.yaml",
        agent_name="router",
    ).model_copy(update={
        "provider": settings.fallback_llm_provider,
        "model": settings.fallback_llm_model,
        "temperature": settings.fallback_temperature,
        "max_tokens": settings.fallback_max_tokens,
        "top_p": settings.fallback_top_p,
        "top_k": settings.fallback_top_k,
        "thinking_budget": settings.fallback_thinking_budget,
        "thinking_effort": settings.fallback_thinking_effort,
        "api_mode": settings.fallback_llm_api_mode,
    })

    mcp_config_path = repo_root / "config" / "mcp_servers.yaml"
    config_data = yaml.safe_load(mcp_config_path.read_text()) or {}
    fs_config = config_data.get("servers", {}).get("filesystem")
    if not fs_config:
        print("[FAIL] filesystem MCP server not configured")
        return 1

    mcp_client = MCPClient(server_name="filesystem", server_config=fs_config)
    await mcp_client.connect(fs_config)

    tool_repo = ToolExecutionRepository()
    orchestrator = Orchestrator(
        tool_execution_repo=tool_repo,
        mcp_clients={"filesystem": mcp_client},
    )

    llm_service = LLMService(settings)
    agent = Agent(
        config=config,
        llm_service=llm_service,
        conversation_repo=ConversationRepository(),
        llm_call_repo=LLMCallRepository(),
        orchestrator=orchestrator,
        tools=[{
            "type": "function",
            "function": {
                "name": "filesystem::read_file",
                "description": "Read a file from the local workspace",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }],
    )

    test_file = repo_root / "tests" / "manual" / "mcp_tmp.txt"
    test_file.write_text("Hello Agent Tool Call", encoding="utf-8")

    response = await agent.process_query(
        f"Use the filesystem::read_file tool to read {test_file} and summarize it.",
        message_window="none",
    )
    print(f"[OK] Agent response: {response.content[:120]}")

    await mcp_client.close()
    await MongoDBConnection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
