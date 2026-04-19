"""Manual streaming test for Agent tool calling via Orchestrator + MCP."""

import asyncio
import os
from pathlib import Path

import yaml

from agent_router.agent import Agent, AgentConfig, MessageWindow
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
    base_config = AgentConfig.from_yaml(
        path=repo_root / "config" / "agents" / "router.yaml",
        agent_name="router",
    )

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

    test_file = repo_root / "tests" / "manual" / "mcp_tmp.txt"
    test_file.write_text("Hello Agent Tool Call (stream)", encoding="utf-8")

    for api_mode in ["chat_completions", "responses"]:
        os.environ["FALLBACK_LLM_API_MODE"] = api_mode
        settings = Settings()
        llm_service = LLMService(settings)

        for model in ["openai/gpt-oss-20b", "zai-org/glm-4.7-flash"]:
            config = base_config.model_copy(update={
                "provider": "lm_studio",
                "model": model,
                "temperature": settings.fallback_temperature,
                "max_tokens": settings.fallback_max_tokens,
                "top_p": settings.fallback_top_p,
                "top_k": settings.fallback_top_k,
                "thinking_budget": None,
                "thinking_effort": None,
                "api_mode": api_mode,
                "stream_tool_planning": True,
            })

            agent = Agent(
                config=config,
                llm_service=llm_service,
                conversation_repo=ConversationRepository(),
                llm_call_repo=LLMCallRepository(),
                orchestrator=orchestrator,
                tools=None,
            )
            await agent.load_mcp_tools([mcp_client])
            tool_names = {
                tool.get("function", {}).get("name")
                for tool in agent.tools or []
            }
            if tool_names != {"filesystem::read_file"}:
                raise RuntimeError(f"Unexpected tool contracts: {sorted(tool_names)}")

            chunks = []
            stream = await agent.process_query(
                f"Use the filesystem::read_file tool to read {test_file} and summarize it.",
                message_window=MessageWindow.NONE,
                stream=True,
            )
            async for chunk in stream:
                chunks.append(chunk.content)

            print(f"[OK] {model} ({api_mode}) streamed {len(chunks)} chunks")

    await mcp_client.close()
    await MongoDBConnection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
