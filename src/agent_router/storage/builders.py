"""Helper builders for storage models."""

from __future__ import annotations

from typing import List
from uuid import UUID

from agent_router.llm.models import LLMRequest, LLMResponse, Message as LLMMessage
from agent_router.core.types import JsonObject, JsonValue

from .models import LLMCall, ToolExecution, Message


def _messages_to_prompt(messages: List[LLMMessage]) -> List[JsonObject]:
    """Convert LLM messages to dicts suitable for storage."""
    return [msg.model_dump(mode="json") for msg in messages]


def build_llm_call(
    *,
    conversation_id: UUID,
    request: LLMRequest | None = None,
    response: LLMResponse | None = None,
    error: str | None = None,
    retry_count: int = 0,
    streaming: bool | None = None,
) -> LLMCall:
    """Build an LLMCall record from a request/response pair."""
    messages = request.messages if request else []
    prompt = response.prompt if response and response.prompt else (
        _messages_to_prompt(messages) if messages else None
    )
    return LLMCall(
        conversation_id=conversation_id,
        prompt=prompt,
        provider=response.provider if response else (request.provider if request else ""),
        model=response.model if response else (request.model if request else ""),
        temperature=request.temperature if request else None,
        top_p=request.top_p if request else None,
        top_k=request.top_k if request else None,
        max_tokens=request.max_tokens if request else (response.max_tokens if response else None),
        thinking_budget=request.thinking_budget if request else None,
        response=response.content if response else None,
        raw_response=response.raw_content if response else None,
        parsed_response=response.parsed_response if response else None,
        tool_calls=response.tool_calls if response else None,
        reasoning=response.reasoning if response else None,
        finish_reason=response.finish_reason if response else None,
        usage=response.usage if response else None,
        latency_ms=int(response.latency_ms) if response and response.latency_ms else None,
        error=error,
        streaming=streaming if streaming is not None else (request.stream if request else False),
        used_fallback=response.used_fallback if response else False,
        retry_count=retry_count,
    )


def build_tool_execution(
    *,
    conversation_id: UUID,
    tool_name: str,
    tool_type: str,
    parameters: JsonObject,
    result: JsonValue | None = None,
    error: str | None = None,
    latency_ms: int | None = None,
) -> ToolExecution:
    """Build a ToolExecution record."""
    return ToolExecution(
        conversation_id=conversation_id,
        tool_name=tool_name,
        tool_type=tool_type,  # validated by model Literal
        parameters=parameters,
        result=result,
        error=error,
        latency_ms=latency_ms,
    )


def build_conversation_message(
    *,
    role: str,
    content: str,
    tool_calls: List[JsonObject] | None = None,
    reasoning: str | None = None,
    llm_call_id: UUID | None = None,
    tool_call_id: str | None = None,
) -> Message:
    """Build a conversation message for UI display."""
    return Message(
        role=role,
        content=content,
        tool_calls=tool_calls,
        reasoning=reasoning,
        llm_call_id=llm_call_id,
        tool_call_id=tool_call_id,
    )
