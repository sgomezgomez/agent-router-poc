"""Google Gemini provider implementation using modern google-genai SDK."""

import hashlib
import ast
import json
import re
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
        tool_call_id_to_name: dict[str, str] = {}

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
                model_parts: list[types.Part] = []
                text_content = clean_encoded_text(msg.content)
                if text_content:
                    model_parts.append(types.Part(text=text_content))

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        tc_id = tc.get("id")
                        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                        fn_name = fn.get("name")
                        if not isinstance(fn_name, str) or not fn_name:
                            continue

                        if isinstance(tc_id, str) and tc_id:
                            tool_call_id_to_name[tc_id] = fn_name

                        raw_args = fn.get("arguments")
                        parsed_args = self._parse_function_args_object(raw_args)
                        model_parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    name=fn_name,
                                    args=parsed_args,
                                )
                            )
                        )

                if model_parts:
                    contents.append(types.Content(
                        role="model",
                        parts=model_parts,
                    ))
            elif msg.role == "tool":
                # Tool response message
                if msg.tool_call_id and msg.content:
                    function_name = tool_call_id_to_name.get(msg.tool_call_id, msg.tool_call_id)
                    contents.append(types.Content(
                        role="function",
                        parts=[types.Part(
                            function_response=types.FunctionResponse(
                                name=function_name,
                                response={"result": msg.content}
                            )
                        )]
                    ))

        return system_instruction, contents

    def _build_generation_config(
        self,
        request: LLMRequest
    ) -> tuple[types.GenerateContentConfig | None, dict[str, str]]:
        """Build generation configuration for Gemini.

        Args:
            request: LLM request

        Returns:
            GenerateContentConfig object
        """
        config_kwargs = {}
        tool_name_map: dict[str, str] = {}

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
            normalized_tools, tool_name_map = self._normalize_tools(request.tools)
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

        return (
            types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
            tool_name_map,
        )

    def _sanitize_function_name(self, name: str) -> str:
        """Gemini requires conservative function names; MCP uses namespace separators."""
        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name or "")
        if sanitized and sanitized[0].isdigit():
            sanitized = f"f_{sanitized}"
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized[:64] or "tool_fn"

    def _normalize_tools(self, tools: list) -> tuple[list[types.Tool] | None, dict[str, str]]:
        """Normalize OpenAI-style tool dicts into Gemini Tool objects."""
        if not tools:
            return None, {}

        if isinstance(tools[0], types.Tool):
            return tools, {}

        function_decls = []
        sanitized_to_original: dict[str, str] = {}
        used_sanitized: set[str] = set()
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue
            function = tool.get("function") or {}
            original_name = function.get("name")
            if not original_name:
                continue
            sanitized_name = self._sanitize_function_name(original_name)
            if sanitized_name in used_sanitized and sanitized_to_original.get(sanitized_name) != original_name:
                suffix = hashlib.md5(original_name.encode("utf-8")).hexdigest()[:8]
                base = sanitized_name[: max(1, 64 - 9)]
                sanitized_name = f"{base}_{suffix}"
            used_sanitized.add(sanitized_name)
            sanitized_to_original[sanitized_name] = original_name
            function_decls.append(
                types.FunctionDeclaration(
                    name=sanitized_name,
                    description=function.get("description"),
                    parametersJsonSchema=function.get("parameters"),
                )
            )

        if not function_decls:
            return None, {}

        return [types.Tool(function_declarations=function_decls)], sanitized_to_original

    def _serialize_function_args(self, args: object) -> str:
        """Return valid JSON string for function arguments."""
        if args is None:
            return "{}"
        if isinstance(args, str):
            text = args.strip()
            if not text:
                return "{}"
            # Already JSON
            try:
                parsed = json.loads(text)
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                pass
            # Python literal dict/list -> JSON
            try:
                parsed = ast.literal_eval(text)
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                return text

        try:
            return json.dumps(args, ensure_ascii=False)
        except Exception:
            try:
                return json.dumps(dict(args), ensure_ascii=False)  # type: ignore[arg-type]
            except Exception:
                return "{}"

    def _parse_function_args_object(self, args: object) -> dict:
        """Return arguments as dict for Gemini FunctionCall args."""
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        if isinstance(args, str):
            text = args.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                pass
            try:
                parsed = ast.literal_eval(text)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        try:
            maybe_dict = dict(args)  # type: ignore[arg-type]
            return maybe_dict if isinstance(maybe_dict, dict) else {}
        except Exception:
            return {}

    def _safe_int(self, value: object) -> int:
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    def _iter_candidate_parts(self, response_obj):
        candidates = getattr(response_obj, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                yield part

    def _extract_reasoning_text(self, response_obj, *, numbered: bool) -> str | None:
        thoughts: list[str] = []
        for part in self._iter_candidate_parts(response_obj):
            if hasattr(part, "thought") and part.thought:
                text = getattr(part, "text", None)
                thoughts.append(text if isinstance(text, str) else str(part.thought))
        if not thoughts:
            return None
        if numbered:
            return "\n".join(f"Thought {i+1}: {t}" for i, t in enumerate(thoughts))
        return "\n".join(thoughts)

    def _extract_visible_text(self, response_obj) -> str:
        """Best-effort visible assistant text from candidate parts."""
        texts: list[str] = []
        for part in self._iter_candidate_parts(response_obj):
            text = getattr(part, "text", None)
            if not isinstance(text, str) or not text.strip():
                continue
            # Keep reasoning extraction separate from visible output text.
            if hasattr(part, "thought") and part.thought:
                continue
            texts.append(text)
        return "\n".join(texts).strip()

    def _extract_tool_calls(self, response_obj, tool_name_map: dict[str, str]) -> list[dict] | None:
        function_calls: list[dict] = []
        for part in self._iter_candidate_parts(response_obj):
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                function_calls.append(
                    {
                        "id": getattr(fc, "id", fc.name),
                        "type": "function",
                        "function": {
                            "name": tool_name_map.get(fc.name, fc.name),
                            "arguments": self._serialize_function_args(fc.args) if hasattr(fc, "args") else "{}",
                        },
                    }
                )
        return function_calls or None

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
            config, tool_name_map = self._build_generation_config(request)

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
            if content is None:
                content = ""
            if not content.strip():
                content = self._extract_visible_text(response)

            # Extract usage metadata
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                metadata = response.usage_metadata
                usage = {
                    "prompt_tokens": self._safe_int(getattr(metadata, 'prompt_token_count', 0)),
                    "completion_tokens": self._safe_int(getattr(metadata, 'candidates_token_count', 0)),
                    "total_tokens": self._safe_int(getattr(metadata, 'total_token_count', 0)),
                }

            # Extract reasoning/tool calls across all candidates.
            reasoning = self._extract_reasoning_text(response, numbered=True)
            tool_calls = self._extract_tool_calls(response, tool_name_map)

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
            config, tool_name_map = self._build_generation_config(request)

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
                if content is None:
                    content = ""
                if not content.strip():
                    content = self._extract_visible_text(chunk)

                # Extract finish reason (first available candidate finish reason)
                finish_reason = None
                candidates = getattr(chunk, "candidates", None) or []
                for candidate in candidates:
                    value = getattr(candidate, "finish_reason", None)
                    if value is not None:
                        finish_reason = str(value)
                        break

                # Extract reasoning/tool calls across all candidates.
                reasoning = self._extract_reasoning_text(chunk, numbered=False)
                tool_calls = self._extract_tool_calls(chunk, tool_name_map)

                yield LLMStreamChunk(
                    content=content,
                    reasoning=reasoning,
                    is_reasoning=bool(reasoning and not content),
                    finish_reason=finish_reason,
                    tool_calls=tool_calls
                )

        except Exception as e:
            raise LLMProviderError(
                f"Gemini streaming failed: {str(e)}"
            ) from e
