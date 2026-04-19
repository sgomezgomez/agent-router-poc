"""OpenAI-compatible provider base implementation.

This provider works with any OpenAI-compatible API including:
- OpenAI (official)
- LM Studio (local)
- Grok (xAI)
- Together AI
- Azure OpenAI
- And others

All configuration (API keys, base URLs) comes from Settings/environment variables.
"""

import logging
import os
import time
from typing import AsyncIterator
from openai import AsyncOpenAI

from agent_router.llm.providers.base import BaseLLMProvider
from agent_router.llm.models import LLMRequest, LLMResponse, LLMStreamChunk
from agent_router.core.errors import LLMProviderError
from agent_router.llm.model_registry import LLMModelRegistry, ModelCapabilities

logger = logging.getLogger(__name__)

class OpenAICompatibleProvider(BaseLLMProvider):
    """Base provider for OpenAI-compatible APIs.

    Handles standard OpenAI chat completions API with support for:
    - Temperature, top_p sampling
    - max_tokens parameter
    - Tool calling (function calling)
    - Streaming responses

    Supports reasoning parameters and structured output for OpenAI-compatible APIs.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        api_mode: str = "chat_completions",
        model_registry: LLMModelRegistry | None = None,
        **kwargs
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            model: Model name
            api_key: API key for authentication
            base_url: Custom base URL (optional, for non-OpenAI endpoints)
            **kwargs: Additional configuration
        """
        super().__init__(model=model, api_key=api_key, base_url=base_url, **kwargs)
        self.api_mode = api_mode
        self.model_registry = model_registry

    def _setup_client(self):
        """Set up OpenAI-compatible client (lazy initialization).

        Returns:
            AsyncOpenAI client
        """
        client_kwargs = {"api_key": self.api_key}

        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        return AsyncOpenAI(**client_kwargs)

    def _build_responses_input(self, request: LLMRequest) -> list[dict]:
        """Build Responses API input from request messages."""
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in request.messages
        ]

    def _get_model_capabilities(self, request: LLMRequest) -> ModelCapabilities | None:
        if not self.model_registry:
            return None
        provider = self.provider_name
        model = self._get_model(request)
        return self.model_registry.get_model_capabilities(provider, model)

    def _apply_reasoning_params(self, request: LLMRequest, api_kwargs: dict) -> None:
        """Apply reasoning parameters when thinking settings are provided."""
        capabilities = self._get_model_capabilities(request)
        if capabilities and capabilities.supports_thinking_effort is False:
            if request.thinking_effort:
                logger.warning(
                    f"{self.provider_name}:{self._get_model(request)} does not support thinking_effort"
                )
            return
        if capabilities and capabilities.supports_thinking_budget is False:
            if request.thinking_budget is not None:
                logger.warning(
                    f"{self.provider_name}:{self._get_model(request)} does not support thinking_budget"
                )
            return

        if request.thinking_effort:
            api_kwargs["reasoning_effort"] = request.thinking_effort
            return

        if request.thinking_budget is None:
            return

        if request.thinking_budget <= 0:
            # No direct mapping for dynamic/disabled thinking on OpenAI-compatible APIs
            return

        if request.thinking_budget < 3000:
            api_kwargs["reasoning_effort"] = "low"
        elif request.thinking_budget < 10000:
            api_kwargs["reasoning_effort"] = "medium"
        else:
            api_kwargs["reasoning_effort"] = "high"

    def _apply_provider_toggles(self, request: LLMRequest, api_kwargs: dict) -> None:
        """Apply provider-specific boolean toggles from the model registry."""
        capabilities = self._get_model_capabilities(request)
        if capabilities and capabilities.supports_enable_thinking is not True:
            if request.enable_thinking is not None or capabilities.supports_enable_thinking is False:
                logger.warning(
                    f"{self.provider_name}:{self._get_model(request)} does not explicitly support enable_thinking"
                )
            return
        enable_thinking = request.enable_thinking
        if enable_thinking is not None:
            extra_body = api_kwargs.get("extra_body") or {}
            extra_body["enableThinking"] = bool(enable_thinking)
            api_kwargs["extra_body"] = extra_body

    def _normalize_tools_for_api(self, request: LLMRequest, api_mode: str) -> None:
        """Normalize tool schema based on model registry preferences."""
        if not request.tools:
            return
        capabilities = self._get_model_capabilities(request)
        tool_schema = capabilities.tool_schema if capabilities else None

        def _as_dict(tool: dict) -> dict:
            if isinstance(tool, dict):
                return tool
            if hasattr(tool, "model_dump"):
                return tool.model_dump(mode="json")
            if hasattr(tool, "dict"):
                return tool.dict()
            return {}

        def to_chat_schema(tool: dict) -> dict:
            tool = _as_dict(tool)
            if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                return tool
            return {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "parameters": tool.get("parameters"),
                },
            }

        def to_responses_schema(tool: dict) -> dict:
            tool = _as_dict(tool)
            if "name" in tool and "parameters" in tool and "function" not in tool:
                return tool
            function = tool.get("function") or {}
            return {
                "type": "function",
                "name": function.get("name"),
                "description": function.get("description"),
                "parameters": function.get("parameters"),
            }

        force_chat_schema = self.provider_name == "lm_studio"
        if api_mode == "chat_completions" or force_chat_schema:
            normalized = []
            for tool in request.tools:
                tool = to_chat_schema(tool)
                if not isinstance(tool.get("function"), dict):
                    name = tool.get("name") or (tool.get("function") or {}).get("name")
                    parameters = tool.get("parameters") or (tool.get("function") or {}).get("parameters")
                    tool = {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": tool.get("description"),
                            "parameters": parameters,
                        },
                    }
                if "function" not in tool:
                    tool["function"] = {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("parameters"),
                    }
                normalized.append(tool)
            request.tools = normalized
        else:
            if tool_schema == "chat_completions":
                request.tools = [to_chat_schema(tool) for tool in request.tools]
            else:
                request.tools = [to_responses_schema(tool) for tool in request.tools]

        if request.tool_choice and isinstance(request.tool_choice, dict):
            if api_mode == "responses" and "name" in request.tool_choice and "function" not in request.tool_choice:
                request.tool_choice = {
                    "type": "function",
                    "function": {"name": request.tool_choice.get("name")},
                }

    def _apply_sampling_params(self, request: LLMRequest, api_kwargs: dict) -> None:
        """Apply sampling and token limit parameters with model-specific guards."""
        capabilities = self._get_model_capabilities(request)
        model = self._get_model(request)

        # gpt-5 family has stricter parameter support
        supports_temperature = capabilities.supports_temperature if capabilities else None
        supports_top_p = capabilities.supports_top_p if capabilities else None
        supports_top_k = capabilities.supports_top_k if capabilities else None
        max_tokens_param = capabilities.max_tokens_param if capabilities else None

        if supports_top_k is False and request.top_k is not None:
            logger.warning(
                f"{self.provider_name}:{model} does not support top_k"
            )

        if not model.startswith("gpt-5"):
            if supports_temperature is False and request.temperature is not None:
                logger.warning(
                    f"{self.provider_name}:{model} does not support temperature"
                )
            elif request.temperature is not None:
                api_kwargs["temperature"] = request.temperature

            if supports_top_p is False and request.top_p is not None:
                logger.warning(
                    f"{self.provider_name}:{model} does not support top_p"
                )
            elif request.top_p is not None:
                api_kwargs["top_p"] = request.top_p
        else:
            # gpt-5-nano disallows temperature/top_p overrides
            pass

        if request.max_tokens is not None:
            if max_tokens_param == "max_completion_tokens":
                api_kwargs["max_completion_tokens"] = request.max_tokens
            elif max_tokens_param == "max_tokens":
                api_kwargs["max_tokens"] = request.max_tokens
            elif model.startswith("gpt-5") or request.thinking_effort or (request.thinking_budget and request.thinking_budget > 0):
                api_kwargs["max_completion_tokens"] = request.max_tokens
            else:
                api_kwargs["max_tokens"] = request.max_tokens

    def _apply_responses_overrides(self, api_kwargs: dict) -> None:
        """Normalize parameters for Responses API."""
        if "max_completion_tokens" in api_kwargs:
            api_kwargs["max_output_tokens"] = api_kwargs.pop("max_completion_tokens")
        if "max_tokens" in api_kwargs:
            api_kwargs["max_output_tokens"] = api_kwargs.pop("max_tokens")

    def _extract_responses_reasoning(self, response_obj) -> str | None:
        """Extract reasoning text from Responses API payloads when available."""
        try:
            payload = response_obj.model_dump() if hasattr(response_obj, "model_dump") else {}
            output = payload.get("output") or []
            parts: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "reasoning":
                    continue
                content = item.get("content") or []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    text = block.get("text")
                    if text:
                        parts.append(str(text))
            return "".join(parts) if parts else None
        except Exception:
            return None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from OpenAI-compatible API.

        Args:
            request: LLM request

        Returns:
            LLM response

        Raises:
            LLMProviderError: If generation fails
        """
        try:
            api_mode = self.api_mode
            capabilities = self._get_model_capabilities(request)
            if not api_mode and capabilities and capabilities.api_mode:
                api_mode = capabilities.api_mode
            if request.tools and capabilities and capabilities.tools_api_mode:
                api_mode = capabilities.tools_api_mode
            start_time = time.time()
            client = self._get_client()

            # Build API request
            if api_mode == "responses":
                api_kwargs = {
                    "model": self._get_model(request),
                    "input": self._build_responses_input(request),
                }
            else:
                api_kwargs = {
                    "model": self._get_model(request),
                    "messages": self._build_messages(request),
                }

            # Add optional parameters if provided
            self._apply_reasoning_params(request, api_kwargs)
            self._apply_sampling_params(request, api_kwargs)
            if api_mode == "responses":
                self._apply_responses_overrides(api_kwargs)
            self._apply_provider_toggles(request, api_kwargs)
            self._normalize_tools_for_api(request, api_mode)

            # Add response_format if provided (not supported on OpenAI reasoning models)
            if request.response_format:
                if capabilities and capabilities.supports_response_format is False:
                    logger.warning(
                        f"{self.provider_name}:{self._get_model(request)} does not support response_format"
                    )
                else:
                    api_kwargs["response_format"] = request.response_format

            # Add tools if provided
            if request.tools:
                if capabilities and capabilities.supports_tools is False:
                    logger.warning(
                        f"{self.provider_name}:{self._get_model(request)} does not support tools"
                    )
                    request.tools = None
                    request.tool_choice = None
                else:
                    api_kwargs["tools"] = request.tools
                    if request.tool_choice:
                        api_kwargs["tool_choice"] = request.tool_choice

            # Call API
            if api_mode == "responses":
                response = await client.responses.create(**api_kwargs)
                content = getattr(response, "output_text", "") or ""
                usage_obj = getattr(response, "usage", None)
                tool_calls = None
                reasoning = self._extract_responses_reasoning(response)
            else:
                response = await client.chat.completions.create(**api_kwargs)
                choice = response.choices[0]
                message = choice.message
                content = message.content or ""
                tool_calls = self._extract_tool_calls(message.tool_calls)
                usage_obj = response.usage
                reasoning = (
                    getattr(message, "reasoning", None)
                    or getattr(message, "reasoning_content", None)
                )

            # Use base class helpers for extraction
            usage = self._extract_usage(usage_obj)

            # Use base class helper to build response
            return self._build_response(
                request=request,
                content=content,
                usage=usage,
                latency_ms=self._measure_latency(start_time),
                tool_calls=tool_calls,
                reasoning=reasoning,
            )

        except Exception as e:
            raise LLMProviderError(
                f"{self.provider_name} generation failed: {str(e)}"
            ) from e

    async def generate_stream(
        self,
        request: LLMRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming response from OpenAI-compatible API.

        Args:
            request: LLM request

        Yields:
            Stream chunks

        Raises:
            LLMProviderError: If generation fails
        """
        try:
            api_mode = self.api_mode
            capabilities = self._get_model_capabilities(request)
            if not api_mode and capabilities and capabilities.api_mode:
                api_mode = capabilities.api_mode
            if request.tools and capabilities and capabilities.tools_api_mode:
                api_mode = capabilities.tools_api_mode
            client = self._get_client()

            # Build API request
            if api_mode == "responses":
                api_kwargs = {
                    "model": self._get_model(request),
                    "input": self._build_responses_input(request),
                    "stream": True,
                }
            else:
                api_kwargs = {
                    "model": self._get_model(request),
                    "messages": self._build_messages(request),
                    "stream": True,
                }

            # Add optional parameters if provided
            self._apply_reasoning_params(request, api_kwargs)
            self._apply_sampling_params(request, api_kwargs)
            if api_mode == "responses":
                self._apply_responses_overrides(api_kwargs)
            self._apply_provider_toggles(request, api_kwargs)
            self._normalize_tools_for_api(request, api_mode)

            # Add response_format if provided (not supported on OpenAI reasoning models)
            if request.response_format:
                if capabilities and capabilities.supports_response_format is False:
                    logger.warning(
                        f"{self.provider_name}:{self._get_model(request)} does not support response_format"
                    )
                else:
                    api_kwargs["response_format"] = request.response_format

            # Add tools if provided
            if request.tools:
                if capabilities and capabilities.supports_tools is False:
                    logger.warning(
                        f"{self.provider_name}:{self._get_model(request)} does not support tools"
                    )
                    request.tools = None
                    request.tool_choice = None
                else:
                    api_kwargs["tools"] = request.tools
                    if request.tool_choice:
                        api_kwargs["tool_choice"] = request.tool_choice

            # Stream from API
            if api_mode == "responses":
                stream = await client.responses.create(**api_kwargs)
                async for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            yield LLMStreamChunk(content=delta)
                    elif event_type == "response.reasoning_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            yield LLMStreamChunk(
                                content="",
                                reasoning=delta,
                                is_reasoning=True,
                            )
                return
            stream = await client.chat.completions.create(**api_kwargs)

            async for chunk in stream:
                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Extract reasoning and output content
                reasoning_content = ""
                output_content = ""
                is_reasoning = False

                # Check if this chunk has reasoning content
                if hasattr(delta, 'reasoning') and delta.reasoning:
                    reasoning_content = delta.reasoning
                    is_reasoning = True
                elif hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_content = delta.reasoning_content
                    is_reasoning = True

                # Extract regular output content
                if delta.content:
                    output_content = delta.content

                # Extract tool calls if present
                tool_calls = self._extract_stream_tool_calls(delta.tool_calls)

                yield LLMStreamChunk(
                    content=output_content,
                    reasoning=reasoning_content if reasoning_content else None,
                    is_reasoning=is_reasoning,
                    finish_reason=choice.finish_reason,
                    tool_calls=tool_calls
                )

        except Exception as e:
            raise LLMProviderError(
                f"{self.provider_name} streaming failed: {str(e)}"
            ) from e

class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider using OpenAI-compatible API."""

    @property
    def provider_name(self) -> str:
        return "openai"
