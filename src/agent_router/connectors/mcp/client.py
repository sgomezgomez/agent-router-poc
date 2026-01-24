"""MCP client using the official SDK."""

from __future__ import annotations

from typing import List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_router.core.interfaces import ToolConnector
from agent_router.core.errors import ToolExecutionError

from .models import MCPTool, MCPToolResult


class MCPClient(ToolConnector):
    """Client for communicating with MCP servers via stdio transport."""

    def __init__(self, server_name: str, server_config: dict):
        self.server_name = server_name
        self.server_config = server_config
        self._session_cm = None
        self._stdio_cm = None
        self.session: ClientSession | None = None
        self.tools: List[MCPTool] = []

    async def connect(self, config: dict) -> None:
        """Start MCP server process and discover tools via stdio transport."""
        command = config["command"]
        args = config.get("args", [])
        env = config.get("env")

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

        self._stdio_cm = stdio_client(server_params)
        read, write = await self._stdio_cm.__aenter__()

        self._session_cm = ClientSession(read, write)
        self.session = await self._session_cm.__aenter__()
        await self.session.initialize()

        await self._discover_tools()

    async def execute(self, tool_name: str, parameters: dict) -> dict:
        """Execute a tool call."""
        if not self.session:
            raise ToolExecutionError(tool_name, "MCP client not connected")

        try:
            result = await self.session.call_tool(tool_name, parameters)
            return MCPToolResult(success=True, result=result).model_dump(mode="json")
        except Exception as e:
            return MCPToolResult(success=False, error=str(e)).model_dump(mode="json")

    async def list_tools(self) -> list:
        """List tools discovered from the MCP server."""
        return [tool.model_dump(mode="json") for tool in self.tools]

    async def close(self) -> None:
        """Close the MCP session and stdio transport."""
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
            self._session_cm = None
            self.session = None
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(None, None, None)
            self._stdio_cm = None

    async def _discover_tools(self) -> None:
        if not self.session:
            return
        tools = await self.session.list_tools()
        normalized: List[MCPTool] = []
        for tool in tools:
            if hasattr(tool, "name"):
                name = tool.name
                description = getattr(tool, "description", None)
                parameters = getattr(tool, "inputSchema", None)
            elif isinstance(tool, dict):
                name = tool.get("name")
                description = tool.get("description")
                parameters = tool.get("inputSchema") or tool.get("input_schema")
            elif isinstance(tool, tuple) and len(tool) >= 2:
                name = getattr(tool[1], "name", None) or tool[0]
                description = getattr(tool[1], "description", None)
                parameters = getattr(tool[1], "inputSchema", None)
            else:
                continue
            if not name:
                continue
            normalized.append(
                MCPTool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    server=self.server_name,
                )
            )
        self.tools = normalized
