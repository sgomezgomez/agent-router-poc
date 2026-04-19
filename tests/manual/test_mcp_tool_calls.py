"""Manual test for MCP tool execution via Orchestrator.

Run with: python tests/manual/test_mcp_tool_calls.py
"""

import asyncio
from pathlib import Path

import yaml

from agent_router.core.config import Settings
from agent_router.orchestrator import Orchestrator
from agent_router.connectors.mcp.client import MCPClient
from agent_router.storage import (
    MongoDBConnection,
    ConversationRepository,
    ToolExecutionRepository,
    Conversation,
)


def _pick_read_tool(tools: list[dict]) -> str | None:
    for tool in tools:
        name = tool.get("name", "")
        schema = tool.get("parameters") or {}
        props = schema.get("properties") or {}
        if ("read" in name or "file" in name) and ("path" in props or "file_path" in props):
            return name
    for tool in tools:
        name = tool.get("name", "")
        if "read" in name or "file" in name:
            return name
    return None


def _build_read_params(schema: dict | None, file_path: str) -> dict:
    if not schema:
        return {"path": file_path}
    props = (schema.get("properties") or {})
    for key in ("path", "file_path", "filename"):
        if key in props:
            return {key: file_path}
    first_key = next(iter(props.keys()), "path")
    return {first_key: file_path}


async def main() -> int:
    settings = Settings()
    await MongoDBConnection.initialize(settings.mongodb)

    repo_root = Path(__file__).resolve().parents[2]
    tmp_file = repo_root / "tests" / "manual" / "mcp_tmp.txt"
    tmp_file.write_text("Hello MCP", encoding="utf-8")

    config_path = repo_root / "config" / "mcp_servers.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    server_config = config.get("servers", {}).get("filesystem")
    if not server_config:
        print("[FAIL] filesystem MCP server not configured")
        return 1

    client = MCPClient(server_name="filesystem", server_config=server_config)
    await client.connect(server_config)
    tools = await client.list_tools()
    print("[INFO] MCP tools:", [tool.get("name") for tool in tools])
    tool_name = _pick_read_tool(tools)
    if not tool_name:
        print("[FAIL] No tools found on filesystem server")
        await client.close()
        return 1

    tool_schema = None
    for tool in tools:
        if tool.get("name") == tool_name:
            tool_schema = tool.get("parameters")
            break

    conversation_repo = ConversationRepository()
    tool_repo = ToolExecutionRepository()
    conversation = Conversation()
    await conversation_repo.create(conversation)

    orchestrator = Orchestrator(
        tool_execution_repo=tool_repo,
        mcp_clients={"filesystem": client},
    )

    tool_calls = [{
        "function": {
            "name": f"filesystem::{tool_name}",
            "arguments": _build_read_params(tool_schema, str(tmp_file)),
        }
    }]
    results = await orchestrator.execute_tool_calls(
        conversation_id=conversation.uuid,
        tool_calls=tool_calls,
    )
    print(f"[OK] Tool results: {results}")

    await client.close()
    await MongoDBConnection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
