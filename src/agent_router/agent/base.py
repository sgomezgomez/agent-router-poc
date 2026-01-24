"""Base agent implementation."""

from __future__ import annotations

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

from .config import AgentConfig
from .prompt_builder import PromptBuilder


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
        tools: Optional[list[dict]] = None,
        response_format: type[object] | None = None,
    ) -> None:
        self.config = config
        self.prompt_builder = PromptBuilder(config)
        self.llm_service = llm_service
        self.conversation_repo = conversation_repo
        self.llm_call_repo = llm_call_repo
        self.tools = tools
        self.response_format = response_format

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

    async def process_query(
        self,
        query: str,
        *,
        conversation: Conversation | None = None,
        conversation_id: UUID | None = None,
        stream: bool = False,
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
            return self._stream_with_persistence(
                conversation=conversation,
                messages=messages,
            )

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
            tools=self.tools,
            response_format=self.response_format,
        )
        await self._persist_conversation(
            conversation=conversation,
            query=query,
            response=response,
            messages=messages,
            streaming=False,
        )
        return response

    async def _get_or_create_conversation(
        self,
        *,
        conversation: Conversation | None,
        conversation_id: UUID | None,
    ) -> Conversation:
        if conversation is not None:
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
        request = LLMRequest(
            messages=messages,
            provider=self.config.provider,
            model=self.config.model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_tokens=self.config.max_tokens,
            thinking_budget=self.config.thinking_budget,
            thinking_effort=self.config.thinking_effort,
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

    async def _stream_with_persistence(
        self,
        *,
        conversation: Conversation,
        messages: List[LLMMessage],
    ) -> Iterable[LLMStreamChunk]:
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: List[dict] = []
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
            tools=self.tools,
        ):
            if chunk.content:
                content_parts.append(chunk.content)
            if chunk.reasoning:
                reasoning_parts.append(chunk.reasoning)
            if chunk.tool_calls:
                tool_calls.extend(chunk.tool_calls)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
            yield chunk

        response = LLMResponse(
            content="".join(content_parts),
            provider=self.config.provider or "",
            model=self.config.model or "",
            tool_calls=tool_calls or None,
            reasoning="".join(reasoning_parts) if reasoning_parts else None,
            finish_reason=finish_reason,
        )
        await self._persist_conversation(
            conversation=conversation,
            query=messages[-1].content,
            response=response,
            messages=messages,
            streaming=True,
        )
