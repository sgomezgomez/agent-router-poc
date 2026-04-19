"""Base agent implementation."""

from __future__ import annotations

import json
import ast
import logging
from enum import Enum
from typing import Iterable, List, Optional
from uuid import UUID

from agent_router.llm.models import Message as LLMMessage, LLMResponse, LLMStreamChunk, LLMRequest
from agent_router.llm.service import LLMService
from agent_router.storage.models import Conversation, Message as ConversationMessage
from agent_router.core.types import JsonObject
from agent_router.storage import (
    ConversationRepository,
    LLMCallRepository,
    build_conversation_message,
    build_llm_call,
)
from agent_router.orchestrator import Orchestrator
from agent_router.connectors.mcp.client import MCPClient
from agent_router.connectors.mcp.contracts import build_tool_contracts

from .config import AgentConfig
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class MessageWindow(str, Enum):
    ALL = "all"
    LAST_BOTH = "last_both"
    LAST_USER = "last_user"
    LAST_ASSISTANT = "last_assistant"
    LAST_N_TURNS = "last_n_turns"
    NONE = "none"


class Agent:
    """Base Agent: builds prompts, filters messages, calls LLM service."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        llm_service: LLMService,
        conversation_repo: ConversationRepository,
        llm_call_repo: LLMCallRepository,
        orchestrator: Orchestrator | None = None,
        tools: Optional[list[dict]] = None,
        response_format: type[object] | None = None,
        tool_result_formatter: "Agent | None" = None,
    ) -> None:
        self.config = config
        self.prompt_builder = PromptBuilder(config)
        self.llm_service = llm_service
        self.conversation_repo = conversation_repo
        self.llm_call_repo = llm_call_repo
        self.orchestrator = orchestrator
        self.tools = tools
        self.response_format = response_format
        self.tool_result_formatter = tool_result_formatter

    def _group_by_role(self, messages: List[ConversationMessage]) -> List[List[ConversationMessage]]:
        groups: List[List[ConversationMessage]] = []
        current_group: List[ConversationMessage] = []
        last_role: Optional[str] = None

        for msg in messages:
            if last_role is None or msg.role == last_role:
                current_group.append(msg)
            else:
                groups.append(current_group)
                current_group = [msg]
            last_role = msg.role
        if current_group:
            groups.append(current_group)
        return groups

    def _filter_messages(
        self,
        messages: List[ConversationMessage],
        window: MessageWindow,
        n_turns: Optional[int] = None,
    ) -> List[ConversationMessage]:
        if not messages or window == MessageWindow.ALL:
            return messages
        if window == MessageWindow.NONE:
            return []

        turns = self._group_by_role(messages)

        if window == MessageWindow.LAST_N_TURNS:
            if not n_turns or n_turns < 1:
                raise ValueError("n_turns must be a positive integer")
            max_groups = n_turns * 2
            selected_turns = turns[-max_groups:] if max_groups <= len(turns) else turns
        else:
            last_user_turn: List[ConversationMessage] = []
            last_assistant_turn: List[ConversationMessage] = []
            if turns[-1][0].role == "user":
                last_user_turn = turns[-1]
                if len(turns) > 1:
                    last_assistant_turn = turns[-2]
            else:
                last_assistant_turn = turns[-1]
                if len(turns) > 1:
                    last_user_turn = turns[-2]

            if window == MessageWindow.LAST_USER:
                selected_turns = last_user_turn
            elif window == MessageWindow.LAST_ASSISTANT:
                selected_turns = last_assistant_turn
            else:
                selected_turns = last_user_turn + last_assistant_turn

        return [msg for group in selected_turns for msg in group]

    def _to_llm_messages(self, messages: Iterable[ConversationMessage]) -> List[LLMMessage]:
        llm_messages: List[LLMMessage] = []
        for msg in messages:
            if not hasattr(msg, "role"):
                if isinstance(msg, dict):
                    msg = ConversationMessage.model_validate(msg)
                elif isinstance(msg, tuple):
                    try:
                        msg = ConversationMessage.model_validate(dict(msg))
                    except Exception:
                        continue
                else:
                    continue
            llm_messages.append(
                LLMMessage(
                    role=msg.role,
                    content=msg.content,
                    tool_calls=msg.tool_calls,
                    tool_call_id=msg.tool_call_id,
                )
            )
        return llm_messages

    def _build_message_list(
        self,
        query: str,
        conversation: Conversation | None,
        window: MessageWindow,
        n_turns: Optional[int],
        input_variables: JsonObject | None,
    ) -> List[LLMMessage]:
        system_prompt = self.prompt_builder.build_system_prompt(input_variables=input_variables)
        messages: List[LLMMessage] = [LLMMessage(role="system", content=system_prompt)]

        if conversation and conversation.message_history:
            history = self._filter_messages(conversation.message_history, window, n_turns)
            messages.extend(self._to_llm_messages(history))

        messages.append(LLMMessage(role="user", content=query))
        return messages

    async def load_mcp_tools(self, mcp_clients: Iterable[MCPClient]) -> None:
        """Load MCP tool contracts into this agent based on config allowlists."""
        all_tools: list[JsonObject] = []
        for client in mcp_clients:
            tools = await client.list_tools()
            all_tools.extend(tools)

        allowed_servers = (
            set(self.config.allowed_mcp_servers)
            if self.config.allowed_mcp_servers is not None
            else None
        )
        allowed_tools = (
            set(self.config.allowed_mcp_tools)
            if self.config.allowed_mcp_tools is not None
            else None
        )

        self.tools = build_tool_contracts(
            all_tools,
            allowed_servers=allowed_servers,
            allowed_tools=allowed_tools,
        )

    async def process_query(
        self,
        query: str,
        *,
        conversation: Conversation | None = None,
        conversation_id: UUID | None = None,
        stream: bool = False,
        debug_tool_timeline: bool = False,
        message_window: MessageWindow = MessageWindow.ALL,
        n_turns: Optional[int] = None,
        input_variables: JsonObject | None = None,
    ) -> LLMResponse | Iterable[LLMStreamChunk]:
        conversation = await self._get_or_create_conversation(
            conversation=conversation,
            conversation_id=conversation_id,
        )

        messages = self._build_message_list(
            query=query,
            conversation=conversation,
            window=message_window,
            n_turns=n_turns,
            input_variables=input_variables,
        )

        if stream:
            return self._run_tool_loop(
                conversation=conversation,
                messages=messages,
                stream=True,
                debug_tool_timeline=debug_tool_timeline,
            )

        result_holder: dict[str, LLMResponse] = {}
        async for _ in self._run_tool_loop(
            conversation=conversation,
            messages=messages,
            stream=False,
            debug_tool_timeline=False,
            result_holder=result_holder,
        ):
            pass
        return result_holder["response"]

    async def _get_or_create_conversation(
        self,
        *,
        conversation: Conversation | None,
        conversation_id: UUID | None,
    ) -> Conversation:
        if conversation is not None:
            existing = await self.conversation_repo.get_by_id(conversation.uuid)
            if existing is not None:
                return existing
            await self.conversation_repo.create(conversation)
            return conversation

        if conversation_id is not None:
            existing = await self.conversation_repo.get_by_id(conversation_id)
            if existing is not None:
                return existing
            new_conv = Conversation(uuid=conversation_id)
            await self.conversation_repo.create(new_conv)
            return new_conv

        new_conv = Conversation()
        await self.conversation_repo.create(new_conv)
        return new_conv

    async def _persist_conversation(
        self,
        *,
        conversation: Conversation,
        query: str,
        response: LLMResponse,
        messages: List[LLMMessage],
        streaming: bool,
    ) -> None:
        settings = self.llm_service.settings
        provider = self.config.provider or response.provider or self.llm_service.fallback_provider
        model = self.config.model or response.model or self.llm_service.fallback_model
        temperature = (
            self.config.temperature
            if self.config.temperature is not None
            else response.temperature if response.temperature is not None
            else settings.fallback_temperature
        )
        top_p = self.config.top_p if self.config.top_p is not None else settings.fallback_top_p
        top_k = self.config.top_k if self.config.top_k is not None else settings.fallback_top_k
        max_tokens = (
            self.config.max_tokens
            if self.config.max_tokens is not None
            else response.max_tokens if response.max_tokens is not None
            else settings.fallback_max_tokens
        )
        thinking_budget = (
            self.config.thinking_budget
            if self.config.thinking_budget is not None
            else settings.fallback_thinking_budget
        )
        thinking_effort = (
            self.config.thinking_effort
            if self.config.thinking_effort is not None
            else settings.fallback_thinking_effort
        )
        enable_thinking = (
            self.config.enable_thinking
            if self.config.enable_thinking is not None
            else settings.fallback_enable_thinking
        )

        capabilities = self.llm_service.model_registry.get_model_capabilities(provider, model)
        if capabilities:
            if capabilities.supports_thinking_budget is False:
                thinking_budget = None
            if capabilities.supports_thinking_effort is False:
                thinking_effort = None
            if capabilities.supports_enable_thinking is False:
                enable_thinking = None

        request = LLMRequest(
            messages=messages,
            provider=provider,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            thinking_effort=thinking_effort,
            enable_thinking=enable_thinking,
            tools=self.tools,
            response_format=self.response_format,
            stream=streaming,
        )
        llm_call = build_llm_call(
            conversation_id=conversation.uuid,
            request=request,
            response=response,
            streaming=streaming,
        )
        await self.llm_call_repo.create(llm_call)

        if query:
            user_msg = build_conversation_message(role="user", content=query)
            await self.conversation_repo.add_message(
                conversation.uuid,
                user_msg.role,
                user_msg.content,
                tool_calls=user_msg.tool_calls,
                reasoning=user_msg.reasoning,
                llm_call_id=user_msg.llm_call_id,
                tool_call_id=user_msg.tool_call_id,
            )

        assistant_msg = build_conversation_message(
            role="assistant",
            content=response.content or "",
            tool_calls=response.tool_calls,
            reasoning=response.reasoning,
            llm_call_id=llm_call.uuid,
        )
        await self.conversation_repo.add_message(
            conversation.uuid,
            assistant_msg.role,
            assistant_msg.content,
            tool_calls=assistant_msg.tool_calls,
            reasoning=assistant_msg.reasoning,
            llm_call_id=assistant_msg.llm_call_id,
            tool_call_id=assistant_msg.tool_call_id,
        )

    async def _stream_response(
        self,
        *,
        messages: List[LLMMessage],
        tools: Optional[list[dict]],
        result_holder: dict[str, LLMResponse],
    ) -> Iterable[LLMStreamChunk]:
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls_map: dict[str, dict] = {}
        finish_reason: str | None = None

        async for chunk in self.llm_service.generate_stream(
            messages=messages,
            provider=self.config.provider,
            model=self.config.model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_tokens,
            thinking_budget=self.config.thinking_budget,
            thinking_effort=self.config.thinking_effort,
            enable_thinking=self.config.enable_thinking,
            tools=tools,
        ):
            if chunk.content:
                content_parts.append(chunk.content)
            if chunk.reasoning:
                reasoning_parts.append(chunk.reasoning)
            if tools and chunk.tool_calls:
                tool_calls_map = self._merge_stream_tool_calls(
                    tool_calls_map,
                    chunk.tool_calls,
                )
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            yield chunk

        tool_calls = list(tool_calls_map.values()) if tool_calls_map else None
        result_holder["response"] = LLMResponse(
            content="".join(content_parts),
            provider=self.config.provider or "",
            model=self.config.model or "",
            tool_calls=tool_calls,
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
            finish_reason=finish_reason,
        )

    async def _run_tool_loop(
        self,
        *,
        conversation: Conversation,
        messages: List[LLMMessage],
        stream: bool,
        debug_tool_timeline: bool,
        result_holder: dict[str, LLMResponse] | None = None,
    ) -> Iterable[LLMStreamChunk]:
        iterations = 0
        current_query = (messages[-1].content or "") if messages else ""
        stream_tool_planning_enabled = self._is_stream_tool_planning_enabled()
        self._log_loop_debug(
            "loop_start "
            f"stream={stream} provider={self.config.provider} model={self.config.model} "
            f"stream_tool_planning={stream_tool_planning_enabled} "
            f"max_tool_iterations={self.config.max_tool_iterations}"
        )

        while True:
            if stream: # Streaming enabled
                # When streaming is disabled for tool planning
                if self.tools and self.orchestrator and not stream_tool_planning_enabled:
                    # Non-stream planning call determines whether tools are needed.
                    response = await self.llm_service.generate(
                        messages=messages,
                        provider=self.config.provider,
                        model=self.config.model,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        top_k=self.config.top_k,
                        max_tokens=self.config.max_tokens,
                        thinking_budget=self.config.thinking_budget,
                        thinking_effort=self.config.thinking_effort,
                        enable_thinking=self.config.enable_thinking,
                        tools=self.tools,
                        response_format=self.response_format,
                    )
                    await self._persist_conversation(
                        conversation=conversation,
                        query=current_query,
                        response=response,
                        messages=messages,
                        streaming=False,
                    )
                    if result_holder is not None:
                        result_holder["response"] = response
                    current_query = ""

                    # Stream every non-stream planning response so the UI timeline
                    # matches all LLM calls, not only tool traces/final answer.
                    async for planning_chunk in self._stream_chunks_from_response(
                        response=response,
                        final=not bool(response.tool_calls),
                        prefix_newline=debug_tool_timeline,
                    ):
                        yield planning_chunk

                    if response.tool_calls:
                        self._log_loop_debug(
                            f"continue_reason=tool_calls_detected planning_non_stream tool_calls_count={len(response.tool_calls)}"
                        )
                        iterations += 1
                        if iterations > self.config.max_tool_iterations:
                            self._log_loop_debug(
                                f"stop_reason=max_tool_iterations iterations={iterations} limit={self.config.max_tool_iterations}"
                            )
                            return
                        tool_calls, tool_results, messages = await self._execute_tool_calls_with_trace(
                            conversation=conversation,
                            messages=messages,
                            response=response,
                        )
                        if debug_tool_timeline:
                            yield LLMStreamChunk(
                                content=self._build_tool_timeline_markdown(
                                    tool_calls=tool_calls,
                                    tool_results=tool_results,
                                )
                            )
                        continue

                    # No tool calls means the model is done; stream the final answer.
                    self._log_loop_debug("stop_reason=no_tool_calls_in_planning using_planning_response_as_final_stream")
                    return
                else:
                    # Streaming enabled for tool calling
                    stream_holder: dict[str, LLMResponse] = {}
                    async for chunk in self._stream_response(
                        messages=messages,
                        tools=self.tools,
                        result_holder=stream_holder,
                    ):
                        yield chunk
                    response = stream_holder["response"]
            else:
                response = await self.llm_service.generate(
                    messages=messages,
                    provider=self.config.provider,
                    model=self.config.model,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    max_tokens=self.config.max_tokens,
                    thinking_budget=self.config.thinking_budget,
                    thinking_effort=self.config.thinking_effort,
                    enable_thinking=self.config.enable_thinking,
                    tools=self.tools,
                    response_format=self.response_format,
                )

            await self._persist_conversation(
                conversation=conversation,
                query=current_query,
                response=response,
                messages=messages,
                streaming=stream,
            )
            if result_holder is not None:
                result_holder["response"] = response
            current_query = ""

            if not response.tool_calls or not self.orchestrator:
                if not self.orchestrator:
                    self._log_loop_debug("stop_reason=no_orchestrator")
                else:
                    self._log_loop_debug("stop_reason=no_tool_calls")
                return

            self._log_loop_debug(
                f"continue_reason=tool_calls_detected tool_calls_count={len(response.tool_calls)}"
            )
            iterations += 1
            if iterations > self.config.max_tool_iterations:
                self._log_loop_debug(
                    f"stop_reason=max_tool_iterations iterations={iterations} limit={self.config.max_tool_iterations}"
                )
                return

            tool_calls, tool_results, messages = await self._execute_tool_calls_with_trace(
                conversation=conversation,
                messages=messages,
                response=response,
            )
            if stream and debug_tool_timeline:
                yield LLMStreamChunk(
                    content=self._build_tool_timeline_markdown(
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                    )
                )

    def _log_loop_debug(self, message: str) -> None:
        if bool(self.llm_service.settings.llm_debug_logging):
            logger.info(f"[AGENT-LOOP] {message}")

    def _is_stream_tool_planning_enabled(self) -> bool:
        """Provider-specific guard for streamed tool-call planning reliability."""
        provider = (self.config.provider or "").strip().lower()
        if provider == "gemini":
            return False
        return bool(self.config.stream_tool_planning)

    async def _stream_chunks_from_response(
        self,
        *,
        response: LLMResponse,
        final: bool,
        prefix_newline: bool,
    ) -> Iterable[LLMStreamChunk]:
        if response.reasoning:
            yield LLMStreamChunk(
                content="",
                reasoning=response.reasoning,
                is_reasoning=True,
                finish_reason=None,
            )

        text = (response.content or "").strip()
        if not text:
            text = (response.raw_content or "").strip()
        if not text and response.reasoning:
            text = response.reasoning.strip()
        if text and prefix_newline:
            text = f"\n\n{text}"

        if text or final:
            finish_reason = (response.finish_reason or "stop") if final else None
            if final:
                self._log_loop_debug(
                    f"final_stream_payload content_len={len(text)} "
                    f"reasoning_len={len(response.reasoning or '')} "
                    f"finish_reason={finish_reason}"
                )
            yield LLMStreamChunk(
                content=text,
                reasoning=None,
                is_reasoning=False,
                finish_reason=finish_reason,
                tool_calls=None,
            )

    async def _execute_tool_calls_with_trace(
        self,
        *,
        conversation: Conversation,
        messages: List[LLMMessage],
        response: LLMResponse,
    ) -> tuple[List[dict], List[dict], List[LLMMessage]]:
        tool_calls = self._ensure_tool_call_ids(response.tool_calls or [])
        tool_calls = self._repair_tool_call_arguments(
            tool_calls=tool_calls,
            messages=messages,
        )
        tool_results = await self.orchestrator.execute_tool_calls(
            conversation_id=conversation.uuid,
            tool_calls=tool_calls,
        )
        formatted_tool_results = await self._format_tool_results(
            messages=messages,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
        # Persist tool messages to conversation
        await self._persist_tool_messages(
            conversation=conversation,
            tool_calls=tool_calls,
            tool_results=tool_results,
            formatted_tool_results=formatted_tool_results,
        )

        assistant_call_message = LLMMessage(
            role="assistant",
            content=response.content or "",
            tool_calls=tool_calls,
        )
        tool_messages = self._build_tool_messages(tool_calls, formatted_tool_results)
        return tool_calls, tool_results, messages + [assistant_call_message] + tool_messages

    def _build_tool_timeline_markdown(
        self,
        *,
        tool_calls: List[dict],
        tool_results: List[dict],
    ) -> str:
        lines: List[str] = [""]

        for tool_call, tool_result in zip(tool_calls, tool_results):
            function = tool_call.get("function") if isinstance(tool_call, dict) else {}
            name = function.get("name") if isinstance(function, dict) else "unknown_tool"
            args = function.get("arguments") if isinstance(function, dict) else None
            parsed_args = self._parse_tool_arguments_for_timeline(args)

            lines.append(f"`{name}`")
            payload = {
                "query": parsed_args,
                "result": tool_result,
            }
            lines.append("```json")
            lines.append(self._to_compact_json(payload))
            lines.append("```")
            lines.append("")
            lines.append("")

        return "\n".join(lines).strip()

    def _to_compact_json(self, value: object) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except TypeError:
            return json.dumps({"value": str(value)}, ensure_ascii=False, separators=(",", ":"))

    def _parse_tool_arguments_for_timeline(self, args: object) -> object:
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            text = args.strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except Exception:
                pass
            try:
                return ast.literal_eval(text)
            except Exception:
                return args
        return args

    async def _format_tool_results(
        self,
        *,
        messages: List[LLMMessage],
        tool_calls: List[dict],
        tool_results: List[dict],
    ) -> List[str]:
        original_user_request = self._latest_user_request(messages)
        formatted: List[str] = []

        for tool_call, result in zip(tool_calls, tool_results):
            deterministic_text = self._format_tool_result_for_prompt(result)
            formatted_text: str | None = deterministic_text

            if self.tool_result_formatter is not None and self._needs_llm_tool_formatting(
                result=result,
                deterministic_text=deterministic_text,
            ):
                try:
                    formatted_text = await self._format_tool_result_with_agent(
                        formatter_agent=self.tool_result_formatter,
                        original_user_request=original_user_request,
                        tool_call=tool_call,
                        tool_result=result,
                    )
                except Exception:
                    formatted_text = deterministic_text
            if not formatted_text:
                formatted_text = deterministic_text
            formatted.append(formatted_text)

        return formatted

    def _needs_llm_tool_formatting(
        self,
        *,
        result: dict,
        deterministic_text: str,
    ) -> bool:
        """Only invoke formatter agent for oversized/noisy payloads."""
        max_deterministic_chars = 2800
        if len(deterministic_text) > max_deterministic_chars:
            return True

        try:
            raw_json = json.dumps(result, ensure_ascii=False)
        except TypeError:
            raw_json = str(result)
        if len(raw_json) > 6000:
            return True

        # If deterministic formatter already flagged heavy truncation, LLM compaction helps.
        if "[truncated " in deterministic_text:
            return True

        return False

    async def _format_tool_result_with_agent(
        self,
        *,
        formatter_agent: "Agent",
        original_user_request: str | None,
        tool_call: dict,
        tool_result: dict,
    ) -> str:
        tool_name = (
            (tool_call.get("function") or {}).get("name")
            if isinstance(tool_call, dict)
            else None
        ) or "unknown_tool"
        tool_args = (
            (tool_call.get("function") or {}).get("arguments")
            if isinstance(tool_call, dict)
            else None
        )
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                pass

        payload = {
            "original_user_request": original_user_request or "",
            "tool_name": tool_name,
            "tool_arguments": tool_args,
            "tool_result": tool_result,
        }
        query = (
            "Reformat this tool execution into readable markdown for another LLM call.\n"
            "- Preserve facts, numbers, IDs, URLs, dates, quotes, and explicit errors.\n"
            "- Prefer lossless cleanup over summarization.\n"
            "- Only compact content when it is excessively long; mark any truncation clearly.\n"
            "- Do not invent, paraphrase away, or reinterpret factual details.\n"
            "- Output markdown only.\n\n"
            f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        # Keep formatter aligned with the active primary model/provider for this request.
        prior_provider = formatter_agent.config.provider
        prior_model = formatter_agent.config.model
        if self.config.provider:
            formatter_agent.config.provider = self.config.provider
        if self.config.model:
            formatter_agent.config.model = self.config.model
        try:
            response = await formatter_agent.process_query(
                query,
                stream=False,
                message_window=MessageWindow.NONE,
            )
        finally:
            formatter_agent.config.provider = prior_provider
            formatter_agent.config.model = prior_model
        return response.content.strip()

    def _latest_user_request(self, messages: List[LLMMessage]) -> str | None:
        for message in reversed(messages):
            if message.role == "user" and message.content:
                return message.content
        return None

    async def _persist_tool_messages(
        self,
        *,
        conversation: Conversation,
        tool_calls: List[dict],
        tool_results: List[dict],
        formatted_tool_results: List[str],
    ) -> None:
        for tool_call, tool_result, formatted_result in zip(tool_calls, tool_results, formatted_tool_results):
            tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
            is_error = False
            if isinstance(tool_result, dict):
                nested_result = tool_result.get("result")
                is_error = bool(
                    tool_result.get("error")
                    or tool_result.get("success") is False
                    or (isinstance(nested_result, dict) and nested_result.get("isError"))
                )
            tool_metadata = [
                {
                    "id": tool_call_id,
                    "type": "tool_result",
                    "function": (tool_call.get("function") if isinstance(tool_call, dict) else {}) or {},
                    "result": tool_result,
                    "is_error": is_error,
                }
            ]
            tool_msg = build_conversation_message(
                role="tool",
                content=formatted_result,
                tool_calls=tool_metadata,
                tool_call_id=tool_call_id,
            )
            await self.conversation_repo.add_message(
                conversation.uuid,
                tool_msg.role,
                tool_msg.content,
                tool_calls=tool_msg.tool_calls,
                reasoning=tool_msg.reasoning,
                llm_call_id=tool_msg.llm_call_id,
                tool_call_id=tool_msg.tool_call_id,
            )

    def _build_tool_messages(
        self,
        tool_calls: List[dict],
        formatted_tool_results: List[str],
    ) -> List[LLMMessage]:
        tool_messages: List[LLMMessage] = []
        for tool_call, formatted_result in zip(tool_calls, formatted_tool_results):
            tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
            tool_messages.append(
                LLMMessage(
                    role="tool",
                    content=formatted_result,
                    tool_call_id=tool_call_id,
                )
            )
        return tool_messages

    def _format_tool_result_for_prompt(self, result: dict) -> str:
        """Convert raw tool output into compact markdown for LLM context."""
        safe_result = self._sanitize_tool_result(result)
        status = "error" if isinstance(result, dict) and result.get("error") else "success"

        lines: list[str] = [f"Tool execution status: {status}"]
        if status == "error":
            error_text = str(result.get("error", "unknown error")) if isinstance(result, dict) else "unknown error"
            lines.extend(
                [
                    "",
                    "Error:",
                    "```text",
                    self._truncate_for_prompt(error_text, max_chars=1200),
                    "```",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Structured result (sanitized):",
                    "```json",
                    self._truncate_for_prompt(
                        json.dumps(safe_result, indent=2, ensure_ascii=False),
                        max_chars=3200,
                    ),
                    "```",
                ]
            )
        return "\n".join(lines)

    def _sanitize_tool_result(self, value, depth: int = 0):
        """Remove noisy/bulky fields so tool context stays useful."""
        if depth > 4:
            return "<omitted: depth_limit>"

        noisy_key_tokens = {
            "html",
            "snapshot",
            "screenshot",
            "base64",
            "image",
            "dom",
            "accessibility",
            "raw",
            "trace",
            "binary",
            "bytes",
        }

        if isinstance(value, dict):
            out: dict = {}
            for key, item in value.items():
                key_text = str(key).lower()
                if any(token in key_text for token in noisy_key_tokens):
                    out[str(key)] = "<omitted: bulky field>"
                    continue
                out[str(key)] = self._sanitize_tool_result(item, depth + 1)
            return out

        if isinstance(value, list):
            max_items = 20
            compact = [self._sanitize_tool_result(item, depth + 1) for item in value[:max_items]]
            if len(value) > max_items:
                compact.append(f"<truncated: {len(value) - max_items} additional items>")
            return compact

        if isinstance(value, str):
            return self._truncate_for_prompt(value, max_chars=600)

        return value

    def _truncate_for_prompt(self, text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        omitted = len(text) - max_chars
        return f"{text[:max_chars]}\n...[truncated {omitted} chars]..."

    def _ensure_tool_call_ids(self, tool_calls: List[dict]) -> List[dict]:
        normalized: List[dict] = []
        for idx, tool_call in enumerate(tool_calls):
            if isinstance(tool_call, dict) and tool_call.get("id"):
                normalized.append(tool_call)
                continue
            call_id = f"tool-call-{idx}"
            if isinstance(tool_call, dict):
                updated = dict(tool_call)
                updated["id"] = call_id
                normalized.append(updated)
            else:
                normalized.append({"id": call_id})
        return normalized

    def _repair_tool_call_arguments(
        self,
        *,
        tool_calls: List[dict],
        messages: List[LLMMessage],
    ) -> List[dict]:
        """Repair common malformed tool arguments produced by some providers."""
        repaired: List[dict] = []
        latest_user_request = self._latest_user_request(messages)

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                repaired.append(tool_call)
                continue

            function = tool_call.get("function")
            if not isinstance(function, dict):
                repaired.append(tool_call)
                continue

            name = function.get("name")
            args_raw = function.get("arguments")
            args_obj = self._parse_tool_arguments_for_timeline(args_raw)
            if not isinstance(args_obj, dict):
                args_obj = {}

            if name == "brave_search::brave_web_search":
                query = args_obj.get("query")
                if not isinstance(query, str) or not query.strip():
                    if latest_user_request and latest_user_request.strip():
                        args_obj["query"] = latest_user_request.strip()
                        self._log_loop_debug(
                            "tool_args_repaired tool=brave_search::brave_web_search source=latest_user_request"
                        )

            updated_tool_call = dict(tool_call)
            updated_function = dict(function)
            updated_function["arguments"] = json.dumps(args_obj, ensure_ascii=False)
            updated_tool_call["function"] = updated_function
            repaired.append(updated_tool_call)

        return repaired

    def _merge_stream_tool_calls(
        self,
        tool_calls_map: dict[str, dict],
        delta_calls: List[dict],
    ) -> dict[str, dict]:
        for idx, tc in enumerate(delta_calls):
            call_key = f"idx-{idx}"
            entry = tool_calls_map.get(call_key) or {
                "id": None,
                "type": None,
                "function": {"name": None, "arguments": ""},
            }

            if tc.get("id"):
                entry["id"] = tc.get("id")
            if tc.get("type") and not entry.get("type"):
                entry["type"] = tc.get("type")
            func = tc.get("function") or {}
            if func.get("name"):
                entry["function"]["name"] = func.get("name")
            args = func.get("arguments")
            if args is not None:
                if isinstance(args, str):
                    entry["function"]["arguments"] += args
                else:
                    entry["function"]["arguments"] = json.dumps(args)
            tool_calls_map[call_key] = entry
        return tool_calls_map
