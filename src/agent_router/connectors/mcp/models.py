"""MCP connector models."""

from __future__ import annotations

from agent_router.core.types import JsonObject, JsonValue
from typing import Literal
from pydantic import BaseModel, Field


class MCPTool(BaseModel):
    """Tool metadata discovered from MCP servers."""

    name: str = Field(description="Tool name")
    description: str | None = Field(default=None, description="Tool description")
    parameters: JsonObject | None = Field(
        default=None,
        description="JSON schema for tool parameters",
    )
    server: str = Field(description="MCP server name")


class MCPToolCall(BaseModel):
    """Tool invocation request."""

    tool_name: str
    parameters: JsonObject


class MCPToolResult(BaseModel):
    """Tool invocation result."""

    success: bool
    result: JsonValue | None = None
    error: str | None = None
    error_type: Literal["technical", "functional"] | None = None