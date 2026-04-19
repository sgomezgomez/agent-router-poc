"""MCP client using the official SDK."""

from __future__ import annotations

import asyncio
import time
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
        self._rate_lock = asyncio.Lock()
        self._execute_lock = asyncio.Lock()
        self._last_call_at: float | None = None

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
            return MCPToolResult(
                success=False,
                error="MCP client not connected",
                error_type="technical",
            ).model_dump(mode="json")

        max_attempts, delay, max_delay, backoff = self._get_retry_config()
        last_error: str | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                result = await self._execute_single_call(tool_name, parameters)
                if hasattr(result, "model_dump"):
                    result_payload = result.model_dump(mode="json")
                elif hasattr(result, "dict"):
                    result_payload = result.dict()
                elif isinstance(result, (dict, list, str, int, float, bool)) or result is None:
                    result_payload = result
                else:
                    result_payload = {"value": str(result)}

                is_error = bool(
                    isinstance(result_payload, dict)
                    and (
                        result_payload.get("isError")
                        or result_payload.get("is_error")
                        or result_payload.get("error")
                    )
                )
                if is_error:
                    detailed_error = self._extract_tool_error_message(result_payload)
                    return MCPToolResult(
                        success=False,
                        result=result_payload,
                        error=detailed_error,
                        error_type="functional",
                    ).model_dump(mode="json")

                return MCPToolResult(
                    success=True,
                    result=result_payload,
                ).model_dump(mode="json")
            except Exception as e:
                last_error = str(e)
                if attempt >= max_attempts:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * backoff, max_delay)

        return MCPToolResult(
            success=False,
            error=last_error or "Tool execution failed",
            error_type="technical",
        ).model_dump(mode="json")

    async def _execute_single_call(self, tool_name: str, parameters: dict):
        """Execute one tool call with optional per-server serialization and cooldown."""
        if self._is_sequential_execution_enabled():
            async with self._execute_lock:
                return await self._execute_call_with_timing(tool_name, parameters)
        return await self._execute_call_with_timing(tool_name, parameters)

    async def _execute_call_with_timing(self, tool_name: str, parameters: dict):
        await self._apply_rate_limit()
        call_timeout = self._get_call_timeout_seconds()
        try:
            return await asyncio.wait_for(
                self.session.call_tool(tool_name, parameters),
                timeout=call_timeout,
            )
        finally:
            delay_seconds = self._get_post_call_delay_seconds()
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

    async def list_tools(self) -> list:
        """List tools discovered from the MCP server."""
        return [tool.model_dump(mode="json") for tool in self.tools]

    async def close(self) -> None:
        """Close the MCP session and stdio transport."""
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except (RuntimeError, asyncio.CancelledError, BaseExceptionGroup):
                pass
            finally:
                self._session_cm = None
                self.session = None
        if self._stdio_cm is not None:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except (RuntimeError, asyncio.CancelledError, BaseExceptionGroup):
                pass
            finally:
                self._stdio_cm = None

    async def _discover_tools(self) -> None:
        if not self.session:
            return
        tools = await self.session.list_tools()
        if hasattr(tools, "model_dump"):
            tools = tools.model_dump(mode="json")
        if hasattr(tools, "tools"):
            tools = tools.tools
        if isinstance(tools, dict):
            tools = tools.get("tools", tools)
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

    def _get_retry_config(self) -> tuple[int, float, float, float]:
        max_attempts = int(self.server_config.get("retry_max_retries", 3))
        base_delay = float(self.server_config.get("retry_base_delay", 1.0))
        max_delay = float(self.server_config.get("retry_max_delay", 10.0))
        backoff = float(self.server_config.get("retry_exponential_base", 2.0))
        return max_attempts, base_delay, max_delay, backoff

    def _get_min_request_interval(self) -> float:
        min_interval = float(self.server_config.get("min_request_interval_seconds", 0) or 0)
        rate = self.server_config.get("rate_limit_per_second")
        if rate:
            try:
                per_second = float(rate)
                if per_second > 0:
                    min_interval = max(min_interval, 1.0 / per_second)
            except (TypeError, ValueError):
                pass
        return min_interval

    def _get_call_timeout_seconds(self) -> float:
        raw = self.server_config.get("call_timeout_seconds", 45)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 45.0
        return max(1.0, value)

    def _is_sequential_execution_enabled(self) -> bool:
        return bool(self.server_config.get("sequential_calls", False))

    def _get_post_call_delay_seconds(self) -> float:
        raw = self.server_config.get("post_call_delay_seconds", 0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0
        return max(0.0, value)

    async def _apply_rate_limit(self) -> None:
        min_interval = self._get_min_request_interval()
        if min_interval <= 0:
            return
        async with self._rate_lock:
            now = time.monotonic()
            if self._last_call_at is not None:
                elapsed = now - self._last_call_at
                remaining = min_interval - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
                    now = time.monotonic()
            self._last_call_at = now

    def _extract_tool_error_message(self, result_payload: object) -> str:
        """Extract a detailed functional error message from MCP result payload."""
        if not isinstance(result_payload, dict):
            return "Tool returned error"

        direct_error = result_payload.get("error")
        if isinstance(direct_error, str) and direct_error.strip():
            return direct_error.strip()

        content = result_payload.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

        nested_result = result_payload.get("result")
        if isinstance(nested_result, dict):
            nested_error = nested_result.get("error")
            if isinstance(nested_error, str) and nested_error.strip():
                return nested_error.strip()
            nested_content = nested_result.get("content")
            if isinstance(nested_content, list):
                for item in nested_content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        return text.strip()

        return "Tool returned error"
