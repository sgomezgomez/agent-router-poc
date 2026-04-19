"""LLM request and response models."""

from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime
from agent_router.core.types import JsonObject, JsonValue


class Message(BaseModel):
    """Single message in a conversation."""
    role: str = Field(description="Message role: system, user, or assistant")
    content: str | None = Field(description="Message content (can be null for tool-call assistant messages)")
    tool_calls: List[JsonObject] | None = Field(
        default=None,
        description="Tool calls requested by the assistant"
    )
    tool_call_id: str | None = Field(
        default=None,
        description="ID of the tool call this message responds to"
    )


class LLMRequest(BaseModel):
    """Request to LLM service."""
    messages: List[Message] = Field(description="Conversation messages")

    # Model selection
    provider: str | None = Field(default=None, description="LLM provider to use")
    model: str | None = Field(default=None, description="Specific model to use")

    # Sampling parameters
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0-2)"
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold"
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-k sampling parameter"
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate"
    )
    thinking_budget: int | None = Field(
        default=None,
        ge=-1,
        description="Thinking budget for reasoning models (-1 dynamic, 0 disable where supported)"
    )
    thinking_effort: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description="Reasoning effort level for models that support it"
    )
    enable_thinking: bool | None = Field(
        default=None,
        description="Tri-state thinking override: true/false to force, null to use model defaults"
    )

    # Tool use
    tools: List[JsonObject] | None = Field(
        default=None,
        description="Available tools for function calling"
    )
    tool_choice: str | JsonObject | None = Field(
        default=None,
        description="Tool choice strategy (auto, none, required, or specific tool)"
    )

    # Response format (Pydantic model class for structured output)
    response_format: type[BaseModel] | None = Field(
        default=None,
        description="Pydantic model class for structured output (e.g., MyResponseModel)"
    )

    # Streaming
    stream: bool = Field(default=False, description="Enable streaming response")


class LLMResponse(BaseModel):
    """Response from LLM service."""
    content: str = Field(description="Generated content (cleaned)")
    raw_content: str | None = Field(
        default=None,
        description="Raw content before cleaning (if different)"
    )

    # Structured output
    parsed_response: JsonValue | None = Field(
        default=None,
        description="Parsed response object (if response_format was provided)"
    )
    parse_error: str | None = Field(
        default=None,
        description="Parse/validation error when response_format is provided"
    )

    # Request parameters (following reference implementation pattern)
    system_prompt: str | None = Field(
        default=None,
        description="System prompt used in the request"
    )
    prompt: List[JsonObject] | None = Field(
        default=None,
        description="Full prompt/messages sent to the LLM"
    )

    # Metadata
    provider: str = Field(description="Provider that generated the response")
    model: str = Field(description="Model that generated the response")

    # Sampling parameters used
    temperature: float | None = Field(
        default=None,
        description="Temperature parameter used"
    )
    max_tokens: int | None = Field(
        default=None,
        description="Max tokens parameter used"
    )

    # Usage statistics
    usage: dict[str, int] | None = Field(
        default=None,
        description="Token usage statistics",
        examples=[{
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200
        }]
    )

    # Performance
    latency_ms: float | None = Field(
        default=None,
        description="Response latency in milliseconds"
    )

    # Tool calls
    tool_calls: List[JsonObject] | None = Field(
        default=None,
        description="Tool calls requested by the model"
    )

    # Reasoning (for o1/o3 and Gemini 2.5)
    reasoning: str | None = Field(
        default=None,
        description="Reasoning/thinking content from reasoning models"
    )

    # Metadata
    finish_reason: str | None = Field(
        default=None,
        description="Reason for completion (stop, length, tool_calls, etc.)"
    )

    # Fallback indicator
    used_fallback: bool = Field(
        default=False,
        description="Whether fallback configuration was used"
    )

    # Timestamp
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )


class LLMStreamChunk(BaseModel):
    """Chunk of streaming LLM response."""
    content: str = Field(description="Chunk content (output)")
    reasoning: str | None = Field(
        default=None,
        description="Reasoning/thinking content (for reasoning models)"
    )
    is_reasoning: bool = Field(
        default=False,
        description="True if this chunk contains reasoning, False if output"
    )
    finish_reason: str | None = Field(
        default=None,
        description="Reason for completion if this is the final chunk"
    )
    tool_calls: List[JsonObject] | None = Field(
        default=None,
        description="Tool calls if present in this chunk"
    )
