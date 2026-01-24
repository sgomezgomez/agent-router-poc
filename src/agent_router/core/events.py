"""Event models for streaming mode."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal
from agent_router.core.types import JsonObject, JsonValue


class AgentEvent(BaseModel):
    """Base event for agent streaming."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ThinkingEvent(AgentEvent):
    """Event emitted when agent is thinking/analyzing."""

    type: Literal["thinking"] = "thinking"
    content: str


class ToolCallEvent(AgentEvent):
    """Event emitted when agent calls a tool."""

    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    parameters: JsonObject


class ToolResultEvent(AgentEvent):
    """Event emitted when tool execution completes."""

    type: Literal["tool_result"] = "tool_result"
    tool_name: str
    result: JsonValue
    error: str | None = None


class ResponseEvent(AgentEvent):
    """Event emitted with final response."""

    type: Literal["response"] = "response"
    content: str
