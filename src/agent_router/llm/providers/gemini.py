"""Google Gemini provider implementation using modern google-genai SDK."""

import time
from typing import AsyncIterator
from google import genai
from google.genai import types

from agent_router.llm.providers.base import BaseLLMProvider
from agent_router.llm.models import LLMRequest, LLMResponse, LLMStreamChunk
from agent_router.core.errors import LLMProviderError
from agent_router.llm.utils import clean_encoded_text


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider using modern google-genai SDK.

    Uses the unified Google GenAI SDK (google-genai) which replaced
    the legacy google-generativeai package.

    Supports:
    - Gemini 2.0 Flash, Gemini 2.5 Flash
    - Gemini Pro models
    - Tool calling (function calling)
    - Streaming responses
    - Thinking budget for reasoning models
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        **kwargs
    ):
        """Initialize Gemini provider.

        Args:
            model: Gemini model name (e.g., gemini-2.0-flash-exp, gemini-2.5-flash)
            api_key: Google AI API key (optional, uses environment variable if not provided)
            **kwargs: Additional configuration
        """
        super().__init__(model=model, api_key=api_key, **kwargs)

    def _setup_client(self):
        """Set up Gemini client (lazy initialization).

        Returns:
            genai.Client instance
        """
        return genai.Client(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "gemini"

    def _format_messages(
        self,
        request: LLMRequest
    ) -> tuple[str | None, list[types.Content]]:
        """Format messages for Gemini API.

        Gemini uses:
        - System prompt as system_instruction (separate parameter)
        - History as list of types.Content objects
        - Roles: "user" and "model" (not "assistant")

        Args:
            request: LLM request

        Returns:
            Tuple of (system_instruction, contents)
        """
        system_instruction = None
        contents = []

        for msg in request.messages:
            if msg.role == "system":
                # Extract system message
                system_instruction = clean_encoded_text(msg.content)
            elif msg.role == "user":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=clean_encoded_text(msg.content))]
                ))
            elif msg.role == "assistant":
                # Gemini uses "model" role instead of "assistant"
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=clean_encoded_text(msg.content))]
                ))
            elif msg.role == "tool":
                # Tool response message
                if msg.tool_call_id and msg.content:
                    contents.append(types.Content(
                        role="function",
                        parts=[types.Part(
                            function_response=types.FunctionResponse(
                                name=msg.tool_call_id,
                                response={"result": msg.content}
                            )
                        )]
                    ))

        return system_instruction, contents

    def _build_generation_config(
        self,
        request: LLMRequest
    ) -> types.GenerateContentConfig:
        """Build generation configuration for Gemini.

        Args:
            request: LLM request

        Returns:
            GenerateContentConfig object
        """
        config_kwargs = {}

        # Temperature
        if request.temperature is not None:
            config_kwargs["temperature"] = request.temperature

        # Top-p
        if request.top_p is not None:
            config_kwargs["top_p"] = request.top_p

        # Top-k
        if request.top_k is not None:
            config_kwargs["top_k"] = request.top_k

        # Max output tokens
        if request.max_tokens is not None:
            config_kwargs["max_output_tokens"] = request.max_tokens

        # Response format (JSON mode)
        if request.response_format:
            config_kwargs["response_mime_type"] = "application/json"
            # If response_format is a Pydantic model, extract schema
            if hasattr(request.response_format, 'model_json_schema'):
                config_kwargs["response_schema"] = request.response_format

        # Tools (function calling)
        if request.tools:
            normalized_tools = self._normalize_tools(request.tools)
            if normalized_tools:
                config_kwargs["tools"] = normalized_tools
            # Disable automatic function calling (agent layer handles it)
            config_kwargs["automatic_function_calling"] = \
                types.AutomaticFunctionCallingConfig(disable=True)

        # Thinking budget (for reasoning models like Gemini 2.5)
        if request.thinking_budget is not None:
            # Gemini 2.5+ supports thinking_config
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=request.thinking_budget
            )

        return types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    def _normalize_tools(self, tools: list) -> list[types.Tool] | None:
        """Normalize OpenAI-style tool dicts into Gemini Tool objects."""
        if not tools:
            return None

        if isinstance(tools[0], types.Tool):
            return tools

        function_decls = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            function = tool.get("function") or {}
            if not function.get("name"):
                continue
            function_decls.append(
                types.FunctionDeclaration(
                    name=function.get("name"),
                    description=function.get("description"),
                    parametersJsonSchema=function.get("parameters"),
                )
            )

        if not function_decls:
            return None

        return [types.Tool(function_declarations=function_decls)]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from Gemini.

        Args:
            request: LLM request

        Returns:
            LLM response

        Raises:
            LLMProviderError: If generation fails
        """
        try:
            start_time = time.time()
            client = self._get_client()
            model_name = self._get_model(request)

            # Format messages
            system_instruction, contents = self._format_messages(request)

            # Build config
            config = self._build_generation_config(request)

            # Add system instruction to config if present
            if system_instruction and config:
                config.system_instruction = system_instruction
            elif system_instruction:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction
                )

            # Generate content
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )

            # Extract content
            content = response.text if hasattr(response, 'text') else ""

            # Extract usage metadata
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                metadata = response.usage_metadata
                usage = {
                    "prompt_tokens": getattr(metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(metadata, 'total_token_count', 0),
                }

            # Extract reasoning (thinking) for Gemini 2.5+
            reasoning = None
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    # Extract thoughts from parts
                    thoughts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            thoughts.append(part.text if hasattr(part, 'text') else str(part.thought))
                    if thoughts:
                        reasoning = "\n".join(f"Thought {i+1}: {t}" for i, t in enumerate(thoughts))

            # Extract tool calls (function calls)
            tool_calls = None
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    function_calls = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            function_calls.append({
                                "id": getattr(fc, 'id', fc.name),  # Use name as fallback
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": str(fc.args) if hasattr(fc, 'args') else "{}"
                                }
                            })
                    if function_calls:
                        tool_calls = function_calls

            # Use base class helper to build response
            return self._build_response(
                request=request,
                content=content,
                usage=usage,
                latency_ms=self._measure_latency(start_time),
                tool_calls=tool_calls,
                reasoning=reasoning
            )

        except Exception as e:
            raise LLMProviderError(
                f"Gemini generation failed: {str(e)}"
            ) from e

    async def generate_stream(
        self,
        request: LLMRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming response from Gemini.

        Args:
            request: LLM request

        Yields:
            Stream chunks

        Raises:
            LLMProviderError: If generation fails
        """
        try:
            client = self._get_client()
            model_name = self._get_model(request)

            # Format messages
            system_instruction, contents = self._format_messages(request)

            # Build config
            config = self._build_generation_config(request)

            # Add system instruction to config if present
            if system_instruction and config:
                config.system_instruction = system_instruction
            elif system_instruction:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction
                )

            # Generate content stream
            stream = await client.aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=config
            )
            async for chunk in stream:
                # Extract content
                content = chunk.text if hasattr(chunk, 'text') else ""

                # Extract finish reason
                finish_reason = None
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = str(candidate.finish_reason)

                # Extract tool calls from chunk
                tool_calls = None
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        function_calls = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                fc = part.function_call
                                function_calls.append({
                                    "id": getattr(fc, 'id', fc.name),
                                    "type": "function",
                                    "function": {
                                        "name": fc.name,
                                        "arguments": str(fc.args) if hasattr(fc, 'args') else "{}"
                                    }
                                })
                        if function_calls:
                            tool_calls = function_calls

                yield LLMStreamChunk(
                    content=content,
                    finish_reason=finish_reason,
                    tool_calls=tool_calls
                )

        except Exception as e:
            raise LLMProviderError(
                f"Gemini streaming failed: {str(e)}"
            ) from e
