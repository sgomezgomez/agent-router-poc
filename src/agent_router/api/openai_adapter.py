"""OpenAI-compatible adapter API for external chat UIs."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, NamedTuple
from uuid import UUID, NAMESPACE_URL, uuid4, uuid5

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from agent_router.agent import MessageWindow
from agent_router.core.config import Settings
from agent_router.llm.model_registry import load_model_registry
from agent_router.runtime import RouterRuntime
from agent_router.storage import ToolExecutionRepository

_runtime: RouterRuntime | None = None
_runtime_lock = asyncio.Lock()
_agent_config_lock = asyncio.Lock()
_responses_store: OrderedDict[str, dict] = OrderedDict()
_responses_lock = asyncio.Lock()
_lineage_map: OrderedDict[str, UUID] = OrderedDict()
_lineage_lock = asyncio.Lock()
logger = logging.getLogger(__name__)
_MAX_RESPONSES_STORE_SIZE = 1000
_MAX_LINEAGE_MAP_SIZE = 5000

def _configure_app_logging() -> None:
    """Ensure agent_router loggers honor configured level in adapter mode."""
    settings = Settings()
    level_name = settings.logging.level.upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

    logging.getLogger("agent_router").setLevel(level)

class OpenAIMessage(BaseModel):
    """OpenAI-compatible message payload."""

    role: str
    content: str | list[dict] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None

class ChatCompletionsRequest(BaseModel):
    """OpenAI-compatible chat completions payload."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[OpenAIMessage] = Field(default_factory=list)
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    user: str | None = None
    conversation_id: str | None = None
    metadata: dict | None = None

class ResponsesRequest(BaseModel):
    """OpenAI-compatible responses payload."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: str | list | dict | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    user: str | None = None
    conversation_id: str | None = None
    metadata: dict | None = None

def _extract_ui_chat_key(extra_fields: dict | None) -> str | None:
    """Prefer explicit UI chat/session identifiers when present."""
    if not isinstance(extra_fields, dict):
        return None
    for key in ("chat_id", "session_id", "conversation_id", "thread_id"):
        value = extra_fields.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

async def _get_runtime() -> RouterRuntime:
    global _runtime
    if _runtime is not None:
        return _runtime
    async with _runtime_lock:
        if _runtime is None:
            _runtime = await RouterRuntime.create()
    return _runtime


def _bounded_put(mapping: OrderedDict[str, Any], key: str, value: Any, max_size: int) -> None:
    mapping[key] = value
    mapping.move_to_end(key)
    while len(mapping) > max_size:
        mapping.popitem(last=False)

def _message_content_to_text(content: str | list[dict] | None) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""

def _extract_latest_user_prompt(messages: list[OpenAIMessage]) -> str:
    for message in reversed(messages):
        if message.role != "user":
            continue
        text = _message_content_to_text(message.content)
        if text:
            return text
    return ""


def _coerce_session_key(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _extract_session_key(
    *,
    conversation_id: str | None,
    metadata: dict | None,
    user: str | None,
    extra_fields: dict | None = None,
) -> str | None:
    preferred_keys = ("conversation_id", "chat_id", "session_id", "thread_id")

    def _extract_from_mapping(mapping: dict | None) -> str | None:
        if not mapping:
            return None
        for key in preferred_keys:
            coerced = _coerce_session_key(mapping.get(key))
            if coerced:
                return coerced
        return None

    explicit = _coerce_session_key(conversation_id)
    if explicit:
        return explicit
    key_from_metadata = _extract_from_mapping(metadata)
    if key_from_metadata:
        return key_from_metadata
    key_from_extra = _extract_from_mapping(extra_fields)
    if key_from_extra:
        return key_from_extra
    del user
    return None


def _extract_lineage_ids(extra_fields: dict | None, metadata: dict | None) -> tuple[str | None, str | None]:
    """Extract request/message lineage IDs (OpenWebUI-style) for fallback conversation continuity."""
    request_id: str | None = None
    parent_id: str | None = None

    def _as_str(value: object) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    for mapping in (extra_fields, metadata):
        if not isinstance(mapping, dict):
            continue
        request_id = request_id or _as_str(mapping.get("id"))
        parent_id = parent_id or _as_str(mapping.get("parent_id"))

        parent_message = mapping.get("parent_message")
        if isinstance(parent_message, dict):
            parent_id = parent_id or _as_str(parent_message.get("id")) or _as_str(parent_message.get("parentId"))

    return request_id, parent_id

def _extract_responses_input_text(value: str | list | dict | None) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if value.get("role") == "user":
            content = value.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                return "\n".join(parts).strip()
    if isinstance(value, list):
        for item in reversed(value):
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                role = item.get("role")
                if role not in (None, "user"):
                    continue
                content = item.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts: list[str] = []
                    for block in content:
                        if isinstance(block, dict):
                            text = block.get("text")
                            if isinstance(text, str):
                                parts.append(text)
                    if parts:
                        return "\n".join(parts).strip()
    return ""

async def _prepare_runtime_for_request(
    *,
    model: str | None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> tuple[RouterRuntime, str | None, str | None, str]:
    del temperature, top_p, max_tokens
    runtime = await _get_runtime()
    provider, resolved_model = _resolve_provider_for_model(model)
    response_model = resolved_model or model or runtime.agent.config.model or "unknown"
    return runtime, provider, resolved_model, response_model

def _require_prompt(prompt: str, *, detail: str) -> str:
    if not prompt:
        raise HTTPException(status_code=400, detail=detail)
    return prompt

def _resolve_conversation_id(session_key: str | None) -> UUID:
    if not session_key:
        return uuid4()
    try:
        return UUID(session_key)
    except (ValueError, TypeError):
        return uuid5(NAMESPACE_URL, session_key)


async def _resolve_conversation_id_with_lineage(
    *,
    session_key: str | None,
    request_id: str | None,
    parent_id: str | None,
) -> UUID:
    if session_key:
        return _resolve_conversation_id(session_key)

    async with _lineage_lock:
        conversation_id: UUID | None = None

        if parent_id:
            conversation_id = _lineage_map.get(parent_id)
        if conversation_id is None and request_id:
            conversation_id = _lineage_map.get(request_id)
        if conversation_id is None:
            conversation_id = uuid4()

        if request_id:
            _bounded_put(_lineage_map, request_id, conversation_id, _MAX_LINEAGE_MAP_SIZE)
        if parent_id:
            _bounded_put(_lineage_map, parent_id, conversation_id, _MAX_LINEAGE_MAP_SIZE)

        return conversation_id

def _sse_response(stream: AsyncGenerator[bytes, None]) -> StreamingResponse:
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

async def _run_non_stream_query(
    *,
    runtime: RouterRuntime,
    prompt: str,
    conversation_id: UUID,
) -> tuple[object, int]:
    started = time.perf_counter()
    response = await runtime.agent.process_query(
        prompt,
        stream=False,
        conversation_id=conversation_id,
        message_window=MessageWindow.ALL,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    return response, latency_ms

class _RequestContext(NamedTuple):
    runtime: RouterRuntime
    provider_override: str | None
    model_override: str | None
    temperature_override: float | None
    top_p_override: float | None
    max_tokens_override: int | None
    response_model: str
    conversation_id: UUID
    op_id: str

async def _build_request_context(
    *,
    model: str | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    conversation_id: str | None,
    metadata: dict | None,
    user: str | None,
    extra_fields: dict | None,
    op_prefix: str,
) -> _RequestContext:
    runtime, provider, resolved_model, response_model = await _prepare_runtime_for_request(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    session_key = _extract_session_key(
        conversation_id=conversation_id,
        metadata=metadata,
        user=user,
        extra_fields=extra_fields,
    )
    request_id, parent_id = _extract_lineage_ids(extra_fields, metadata)
    resolved_conversation_id = await _resolve_conversation_id_with_lineage(
        session_key=session_key,
        request_id=request_id,
        parent_id=parent_id,
    )
    if bool(Settings().llm_debug_logging):
        logger.info(
            "[ADAPTER-SESSION] op_prefix=%s session_key_present=%s session_key=%s request_id=%s parent_id=%s conversation_id=%s",
            op_prefix,
            bool(session_key),
            session_key if session_key else "<none>",
            request_id if request_id else "<none>",
            parent_id if parent_id else "<none>",
            str(resolved_conversation_id),
        )
    return _RequestContext(
        runtime=runtime,
        provider_override=provider,
        model_override=resolved_model,
        temperature_override=temperature,
        top_p_override=top_p,
        max_tokens_override=max_tokens,
        response_model=response_model,
        conversation_id=resolved_conversation_id,
        op_id=f"{op_prefix}{uuid4().hex}",
    )


@asynccontextmanager
async def _apply_runtime_overrides(context: _RequestContext):
    async with _agent_config_lock:
        agent_config = context.runtime.agent.config
        prev_provider = agent_config.provider
        prev_model = agent_config.model
        prev_temperature = agent_config.temperature
        prev_top_p = agent_config.top_p
        prev_max_tokens = agent_config.max_tokens
        try:
            if context.provider_override is not None:
                agent_config.provider = context.provider_override
            if context.model_override is not None:
                agent_config.model = context.model_override
            if context.temperature_override is not None:
                agent_config.temperature = context.temperature_override
            if context.top_p_override is not None:
                agent_config.top_p = context.top_p_override
            if context.max_tokens_override is not None:
                agent_config.max_tokens = context.max_tokens_override
            yield
        finally:
            agent_config.provider = prev_provider
            agent_config.model = prev_model
            agent_config.temperature = prev_temperature
            agent_config.top_p = prev_top_p
            agent_config.max_tokens = prev_max_tokens

async def _run_endpoint_pipeline(
    *,
    context: _RequestContext,
    prompt: str,
    stream: bool,
    include_debug_trace: bool,
    stream_builder,
    non_stream_builder,
):
    async def _stream_with_overrides() -> AsyncGenerator[bytes, None]:
        async with _apply_runtime_overrides(context):
            async for chunk in stream_builder(
                runtime=context.runtime,
                prompt=prompt,
                conversation_id=context.conversation_id,
                op_id=context.op_id,
                model=context.response_model,
                include_debug_trace=include_debug_trace,
            ):
                yield chunk

    if stream:
        return _sse_response(_stream_with_overrides())

    async with _apply_runtime_overrides(context):
        response, latency_ms = await _run_non_stream_query(
            runtime=context.runtime,
            prompt=prompt,
            conversation_id=context.conversation_id,
        )
        if include_debug_trace:
            trace_markdown = await _build_tool_trace_markdown(context.conversation_id)
            if trace_markdown:
                base_content = response.content or ""
                response.content = f"{base_content}\n\n{trace_markdown}" if base_content else trace_markdown
        body = non_stream_builder(
            op_id=context.op_id,
            model=context.response_model,
            response=response,
            latency_ms=latency_ms,
        )
        if inspect.isawaitable(body):
            body = await body
        return JSONResponse(body)

async def _handle_openai_endpoint(
    *,
    raw_prompt: str,
    prompt_error_detail: str,
    model: str | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    conversation_id: str | None,
    metadata: dict | None,
    user: str | None,
    extra_fields: dict | None,
    stream: bool,
    op_prefix: str,
    stream_builder,
    non_stream_builder,
):
    prompt = _require_prompt(raw_prompt, detail=prompt_error_detail)
    context = await _build_request_context(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        conversation_id=conversation_id,
        metadata=metadata,
        user=user,
        extra_fields=extra_fields,
        op_prefix=op_prefix,
    )
    return await _run_endpoint_pipeline(
        context=context,
        prompt=prompt,
        stream=stream,
        include_debug_trace=_is_debug_trace_enabled(metadata),
        stream_builder=stream_builder,
        non_stream_builder=non_stream_builder,
    )

def _resolve_provider_for_model(model: str | None) -> tuple[str | None, str | None]:
    if not model:
        return None, None
    if ":" in model:
        provider_hint, model_name = model.split(":", 1)
        registry = load_model_registry()
        provider_cfg = registry.providers.get(provider_hint)
        if provider_cfg and model_name in provider_cfg.models:
            return provider_hint, model_name
    registry = load_model_registry()
    matches: list[str] = []
    for provider_name, provider_cfg in registry.providers.items():
        if model in provider_cfg.models:
            matches.append(provider_name)
    if len(matches) == 1:
        return matches[0], model
    return None, model

def _now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())

def _build_models_response() -> dict:
    registry = load_model_registry()
    data: list[dict] = []
    seen: set[str] = set()

    for provider_name, provider_cfg in registry.providers.items():
        for model_name in provider_cfg.models:
            object_id = f"{provider_name}:{model_name}"
            data.append(
                {
                    "id": f"{provider_name}:{model_name}",
                    "object": "model",
                    "created": _now_epoch(),
                    "owned_by": provider_name,
                    "root": model_name,
                    "parent": None,
                    "adapter_id": object_id,
                }
            )
            seen.add(f"{provider_name}:{model_name}")

    if not data:
        settings = Settings()
        fallback_id = settings.fallback_llm_model
        fallback_composite = f"{settings.fallback_llm_provider}:{fallback_id}"
        if fallback_composite not in seen:
            data.append(
                {
                    "id": fallback_composite,
                    "object": "model",
                    "created": _now_epoch(),
                    "owned_by": settings.fallback_llm_provider,
                    "root": fallback_id,
                    "parent": None,
                    "adapter_id": f"{settings.fallback_llm_provider}:{fallback_id}",
                }
            )

    return {"object": "list", "data": data}

def _usage_or_default(usage: dict[str, int] | None) -> dict[str, int]:
    if usage:
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def _completion_response(
    *,
    request_id: str,
    model: str,
    content: str,
    tool_calls: list[dict] | None,
    reasoning: str | None,
    finish_reason: str | None,
    usage: dict[str, int] | None,
) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning:
        message["reasoning"] = reasoning

    inferred_finish = finish_reason or ("tool_calls" if tool_calls else "stop")
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": _now_epoch(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": inferred_finish,
            }
        ],
        "usage": _usage_or_default(usage),
    }

def _responses_object(
    *,
    response_id: str,
    model: str,
    content: str,
    usage: dict[str, int] | None,
) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "created_at": _now_epoch(),
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": content,
        "usage": _usage_or_default(usage),
    }

async def _store_response(response_obj: dict) -> None:
    response_id = response_obj.get("id")
    if not isinstance(response_id, str):
        return
    async with _responses_lock:
        _bounded_put(_responses_store, response_id, response_obj, _MAX_RESPONSES_STORE_SIZE)

async def _get_stored_response(response_id: str) -> dict | None:
    async with _responses_lock:
        found = _responses_store.get(response_id)
        if found is not None:
            _responses_store.move_to_end(response_id)
        return found

def _is_debug_trace_enabled(metadata: dict | None) -> bool:
    default_enabled = _debug_trace_default_enabled()
    if not metadata:
        return default_enabled

    value = metadata.get("debug_trace")
    if value is None:
        return default_enabled

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default_enabled


@lru_cache(maxsize=1)
def _debug_trace_default_enabled() -> bool:
    return bool(Settings().adapter_debug_trace_default)

def _json_preview(value, *, max_chars: int = 600) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n... [truncated {omitted} chars] ..."


async def _build_tool_trace_markdown(conversation_id: UUID) -> str | None:
    repo = ToolExecutionRepository()
    executions = await repo.get_by_conversation(conversation_id, limit=25)
    if not executions:
        return "\n".join(
            [
                "### Tool Trace",
                "",
                "- No tool executions recorded for this conversation yet.",
            ]
        )

    lines: list[str] = ["", "### Tool Trace", ""]
    for item in reversed(executions):
        status = "ERROR" if item.error else "OK"
        lines.append(f"- `{item.tool_name}` ({status})")
        if item.latency_ms is not None:
            lines.append(f"  - latency_ms: `{item.latency_ms}`")
        if item.parameters:
            lines.extend(
                [
                    "  - parameters:",
                    "```json",
                    _json_preview(item.parameters, max_chars=500),
                    "```",
                ]
            )
        if item.error:
            lines.extend(
                [
                    "  - error:",
                    "```text",
                    str(item.error),
                    "```",
                ]
            )
        elif item.result is not None:
            lines.extend(
                [
                    "  - result:",
                    "```json",
                    _json_preview(item.result, max_chars=900),
                    "```",
                ]
            )
        lines.append("")

    return "\n".join(lines).strip()

async def _stream_chunks(
    *,
    runtime: RouterRuntime,
    prompt: str,
    conversation_id: UUID,
    op_id: str,
    model: str,
    include_debug_trace: bool = False,
) -> AsyncGenerator[bytes, None]:
    role_event = {
        "id": op_id,
        "object": "chat.completion.chunk",
        "created": _now_epoch(),
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_event, ensure_ascii=False)}\n\n".encode("utf-8")

    async for chunk in await runtime.agent.process_query(
        prompt,
        stream=True,
        debug_tool_timeline=include_debug_trace,
        conversation_id=conversation_id,
        message_window=MessageWindow.ALL,
    ):
        delta: dict = {}
        if chunk.content:
            delta["content"] = chunk.content
        if chunk.reasoning:
            delta["reasoning_content"] = chunk.reasoning
            delta["reasoning"] = chunk.reasoning
        if chunk.tool_calls:
            delta["tool_calls"] = chunk.tool_calls

        if delta:
            payload = {
                "id": op_id,
                "object": "chat.completion.chunk",
                "created": _now_epoch(),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

        if chunk.finish_reason:
            finish_payload = {
                "id": op_id,
                "object": "chat.completion.chunk",
                "created": _now_epoch(),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": chunk.finish_reason,
                    }
                ],
            }
            yield f"data: {json.dumps(finish_payload, ensure_ascii=False)}\n\n".encode("utf-8")

    yield b"data: [DONE]\n\n"

async def _stream_responses(
    *,
    runtime: RouterRuntime,
    prompt: str,
    conversation_id: UUID,
    op_id: str,
    model: str,
    include_debug_trace: bool = False,
) -> AsyncGenerator[bytes, None]:
    content_parts: list[str] = []

    created_event = {
        "type": "response.created",
        "response": {
            "id": op_id,
            "object": "response",
            "created_at": _now_epoch(),
            "status": "in_progress",
            "model": model,
        },
    }
    yield f"data: {json.dumps(created_event, ensure_ascii=False)}\n\n".encode("utf-8")

    async for chunk in await runtime.agent.process_query(
        prompt,
        stream=True,
        debug_tool_timeline=include_debug_trace,
        conversation_id=conversation_id,
        message_window=MessageWindow.ALL,
    ):
        if chunk.content:
            content_parts.append(chunk.content)
            delta_event = {
                "type": "response.output_text.delta",
                "response_id": op_id,
                "delta": chunk.content,
            }
            yield f"data: {json.dumps(delta_event, ensure_ascii=False)}\n\n".encode("utf-8")

    completed_event = {
        "type": "response.completed",
        "response": {
            "id": op_id,
            "object": "response",
            "created_at": _now_epoch(),
            "status": "completed",
            "model": model,
        },
    }
    await _store_response(
        _responses_object(
            response_id=op_id,
            model=model,
            content="".join(content_parts),
            usage=None,
        )
    )
    yield f"data: {json.dumps(completed_event, ensure_ascii=False)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"

@asynccontextmanager
async def _lifespan(_: FastAPI):
    global _runtime
    _configure_app_logging()
    await _get_runtime()
    try:
        yield
    finally:
        if _runtime is not None:
            close_task = asyncio.create_task(_runtime.close())
            try:
                done, pending = await asyncio.wait({close_task}, timeout=8.0)
                if pending:
                    close_task.cancel()
                    logger.warning("Adapter shutdown timed out while closing runtime; forcing exit for reload")
            except (RuntimeError, asyncio.CancelledError, BaseExceptionGroup) as exc:
                logger.warning("Adapter shutdown encountered an error while closing runtime: %s", exc)
            _runtime = None

app = FastAPI(title="Agent Router OpenAI Adapter", version="0.1.0", lifespan=_lifespan)

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse(_build_models_response())

@app.get("/v1/openapi.json")
async def v1_openapi() -> JSONResponse:
    return JSONResponse(app.openapi())

def _chat_non_stream_body(*, op_id: str, model: str, response, latency_ms: int) -> dict:
    body = _completion_response(
        request_id=op_id,
        model=model,
        content=response.content or "",
        tool_calls=response.tool_calls,
        reasoning=response.reasoning,
        finish_reason=response.finish_reason,
        usage=response.usage,
    )
    body["latency_ms"] = latency_ms
    return body

async def _responses_non_stream_body(*, op_id: str, model: str, response, latency_ms: int) -> dict:
    del latency_ms
    body = _responses_object(
        response_id=op_id,
        model=model,
        content=response.content or "",
        usage=response.usage,
    )
    await _store_response(body)
    return body

@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(payload: ChatCompletionsRequest):
    explicit_ui_key = _extract_ui_chat_key(payload.model_extra)
    effective_conversation_id = payload.conversation_id or explicit_ui_key
    return await _handle_openai_endpoint(
        raw_prompt=_extract_latest_user_prompt(payload.messages),
        prompt_error_detail="No user message content found in payload.messages",
        model=payload.model,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
        conversation_id=effective_conversation_id,
        metadata=payload.metadata,
        user=payload.user,
        extra_fields=payload.model_extra,
        stream=payload.stream,
        op_prefix="chatcmpl-",
        stream_builder=_stream_chunks,
        non_stream_builder=_chat_non_stream_body,
    )

@app.post("/v1/responses", response_model=None)
async def responses(payload: ResponsesRequest):
    return await _handle_openai_endpoint(
        raw_prompt=_extract_responses_input_text(payload.input),
        prompt_error_detail="No user input text found in payload.input",
        model=payload.model,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_output_tokens,
        conversation_id=payload.conversation_id,
        metadata=payload.metadata,
        user=payload.user,
        extra_fields=payload.model_extra,
        stream=payload.stream,
        op_prefix="resp_",
        stream_builder=_stream_responses,
        non_stream_builder=_responses_non_stream_body,
    )

@app.get("/v1/responses/{response_id}", response_model=None)
async def get_response(response_id: str):
    stored = await _get_stored_response(response_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"Response '{response_id}' not found")
    return JSONResponse(stored)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agent_router.api.openai_adapter:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        timeout_graceful_shutdown=8,
    )
