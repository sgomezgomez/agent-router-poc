"""Storage models for MongoDB collections."""

from datetime import datetime
from typing import List, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict

from agent_router.core.types import JsonObject, JsonValue

class LLMCall(BaseModel):
    """Model for storing LLM call records.

    Stored by Agent Layer whenever an LLM call is made.
    """

    uuid: UUID = Field(default_factory=uuid4)
    conversation_id: UUID

    # Request details
    prompt: List[JsonObject] | None = None  # Full message payload
    provider: str
    model: str

    # Sampling parameters used
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    thinking_budget: int | None = None

    # Response
    response: str | None = None
    raw_response: str | None = None
    parsed_response: JsonValue | None = None
    tool_calls: List[JsonObject] | None = None
    reasoning: str | None = None
    finish_reason: str | None = None

    # Usage metrics
    usage: dict[str, int] | None = None  # {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}

    # Performance
    latency_ms: int | None = None

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: str | None = None
    streaming: bool = False
    used_fallback: bool = False  # True if fallback config was used
    retry_count: int = 0  # Number of retries before success/failure

    # Model config
    model_config = ConfigDict(
        json_encoders={UUID: str, datetime: lambda v: v.isoformat()}
    )

class ToolExecution(BaseModel):
    """Model for storing tool execution records.

    Stored by Orchestrator whenever a tool is executed.
    """

    uuid: UUID = Field(default_factory=uuid4)
    conversation_id: UUID

    # Tool details
    tool_name: str
    tool_type: Literal["mcp_tool", "mcp_agent", "a2a_agent"]
    parameters: JsonObject

    # Result
    result: JsonValue | None = None
    error: str | None = None

    # Performance
    latency_ms: int | None = None

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Model config
    model_config = ConfigDict(
        json_encoders={UUID: str, datetime: lambda v: v.isoformat()}
    )


class Message(BaseModel):
    """Single message in conversation history."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Optional metadata
    tool_calls: List[JsonObject] | None = None
    reasoning: str | None = None
    llm_call_id: UUID | None = None  # Reference to LLMCall if this was an LLM response
    tool_call_id: str | None = None

    # Model config
    model_config = ConfigDict(
        json_encoders={UUID: str, datetime: lambda v: v.isoformat()}
    )

class Conversation(BaseModel):
    """Model for storing conversations.

    Maintains conversation history and context for multi-turn interactions.
    """

    uuid: UUID = Field(default_factory=uuid4)

    # Conversation history
    message_history: List[Message] = Field(default_factory=list)

    # Conversation metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    # Optional context
    user_id: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    # Model config
    model_config = ConfigDict(
        json_encoders={UUID: str, datetime: lambda v: v.isoformat()}
    )

    def add_message(self, role: str, content: str, **kwargs) -> Message:
        """Add a message to the conversation history.

        Args:
            role: Message role (user/assistant/system)
            content: Message content
            **kwargs: Additional message metadata

        Returns:
            The created Message object
        """
        message = Message(role=role, content=content, **kwargs)
        self.message_history.append(message)
        self.last_updated = datetime.utcnow()
        return message

    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get the most recent N messages.

        Args:
            count: Number of messages to retrieve

        Returns:
            List of recent messages
        """
        return self.message_history[-count:] if count > 0 else self.message_history

    def get_context_window(self, max_tokens: int = 4000) -> List[Message]:
        """Get messages that fit within a token budget.

        This is a simple implementation that estimates tokens.
        For production, use tiktoken or similar for accurate counting.

        Args:
            max_tokens: Maximum tokens to include

        Returns:
            List of messages within token budget
        """
        # Simple estimation: ~4 characters per token
        chars_per_token = 4
        max_chars = max_tokens * chars_per_token

        result = []
        total_chars = 0

        # Walk backwards through history
        for message in reversed(self.message_history):
            message_chars = len(message.content)
            if total_chars + message_chars > max_chars:
                break
            result.insert(0, message)
            total_chars += message_chars

        return result
