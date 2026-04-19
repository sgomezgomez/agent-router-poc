"""Orchestrator for executing tool calls via connectors."""

from __future__ import annotations

import ast
import json
from typing import Iterable, List
from uuid import UUID

from agent_router.core.errors import ToolExecutionError
from agent_router.core.types import JsonObject, JsonValue
from agent_router.storage.builders import build_tool_execution
from agent_router.storage.tool_execution_repository import ToolExecutionRepository
from agent_router.connectors.mcp.client import MCPClient


class Orchestrator:
    """Dumb execution engine for tool calls.

    Executes tool calls via connectors and logs tool executions.
    """

    def __init__(
        self,
        *,
        tool_execution_repo: ToolExecutionRepository,
        mcp_clients: dict[str, MCPClient] | None = None,
    ) -> None:
        self.tool_execution_repo = tool_execution_repo
        self.mcp_clients = mcp_clients or {}

    async def execute_tool_calls(
        self,
        *,
        conversation_id: UUID,
        tool_calls: Iterable[JsonObject],
    ) -> List[JsonObject]:
        """Execute tool calls and return results."""
        results: List[JsonObject] = []

        for tool_call in tool_calls:
            try:
                tool_name, parameters = self._normalize_tool_call(tool_call)
                server_name, connector_tool_name = self._split_tool_name(tool_name)
                client = self._get_mcp_client(server_name)

                result = await client.execute(connector_tool_name, parameters)
                results.append(result)

                execution = build_tool_execution(
                    conversation_id=conversation_id,
                    tool_name=tool_name,
                    tool_type="mcp_tool",
                    parameters=parameters,
                    result=result,
                    error=None,
                )
                await self.tool_execution_repo.create(execution)
            except Exception as e:
                error_message = str(e)
                results.append({"error": error_message})
                execution = build_tool_execution(
                    conversation_id=conversation_id,
                    tool_name=tool_name if "tool_name" in locals() else "unknown",
                    tool_type="mcp_tool",
                    parameters=parameters if "parameters" in locals() else {},
                    result=None,
                    error=error_message,
                )
                await self.tool_execution_repo.create(execution)

        return results

    def _normalize_tool_call(self, tool_call: JsonObject) -> tuple[str, JsonObject]:
        """Normalize tool call into (tool_name, parameters)."""
        function = tool_call.get("function")
        if isinstance(function, dict):
            tool_name = function.get("name")
            raw_args = function.get("arguments")
        else:
            tool_name = tool_call.get("name")
            raw_args = tool_call.get("arguments")

        if not tool_name:
            raise ToolExecutionError("unknown", "Missing tool name")

        if isinstance(raw_args, str) and raw_args.strip():
            try:
                parameters = json.loads(raw_args)
            except json.JSONDecodeError as e:
                # Some providers emit Python-literal dict strings (single quotes).
                try:
                    parsed = ast.literal_eval(raw_args)
                    if isinstance(parsed, dict):
                        parameters = parsed
                    else:
                        raise ToolExecutionError(
                            tool_name,
                            "Invalid tool arguments JSON: parsed literal is not an object",
                        )
                except (SyntaxError, ValueError, ToolExecutionError) as literal_error:
                    raise ToolExecutionError(tool_name, f"Invalid tool arguments JSON: {e}") from literal_error
        elif isinstance(raw_args, dict):
            parameters = raw_args
        else:
            parameters = {}

        return tool_name, parameters

    def _split_tool_name(self, tool_name: str) -> tuple[str, str]:
        """Split tool name into (server_name, connector_tool_name)."""
        if "::" in tool_name:
            server, name = tool_name.split("::", 1)
            return server, name
        if "/" in tool_name:
            server, name = tool_name.split("/", 1)
            return server, name
        if len(self.mcp_clients) == 1:
            server = next(iter(self.mcp_clients.keys()))
            return server, tool_name
        raise ToolExecutionError(tool_name, "Tool name missing server prefix")

    def _get_mcp_client(self, server_name: str) -> MCPClient:
        client = self.mcp_clients.get(server_name)
        if not client:
            raise ToolExecutionError(server_name, "MCP client not configured")
        return client
