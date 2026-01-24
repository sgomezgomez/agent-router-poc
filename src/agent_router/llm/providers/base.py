"""Base LLM provider abstract class."""

from abc import ABC, abstractmethod
from typing import AsyncIterator
from agent_router.core.types import JsonObject
import time

from agent_router.llm.models import LLMRequest, LLMResponse, LLMStreamChunk
from agent_router.core.interfaces import LLMProvider
from agent_router.core.errors import LLMProviderError
from agent_router.llm.utils import clean_encoded_text


class BaseLLMProvider(LLMProvider, ABC):
    """Abstract base class for all LLM providers.

    All providers must implement this interface to ensure consistent behavior
    across different LLM APIs (OpenAI, Gemini, LM Studio, Grok, etc.).
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs
    ):
        """Initialize provider.

        Args:
            model: Model name for this provider (used when agent doesn't specify)
            api_key: API key for authentication (if required)
            base_url: Base URL for API endpoint (if custom)
            **kwargs: Additional provider-specific configuration
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.config = kwargs
        self._client = None  # Lazy initialization

    @abstractmethod
    def _setup_client(self):
        """Set up and return the client for this provider.

        This is called lazily when needed, not in __init__.
        Subclasses must implement this to create their specific client.

        Returns:
            Client instance for this provider
        """
        pass

    def _get_client(self):
        """Get or create client (lazy initialization).

        Returns:
            Client instance
        """
        if self._client is None:
            self._client = self._setup_client()
        return self._client

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            request: LLM request with messages and parameters

        Returns:
            LLM response with generated content and metadata

        Raises:
            LLMProviderError: If generation fails
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        request: LLMRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Generate a streaming response from the LLM.

        Args:
            request: LLM request with messages and parameters

        Yields:
            Stream chunks with content and metadata

        Raises:
            LLMProviderError: If generation fails
        """
        pass

    def _get_model(self, request: LLMRequest) -> str:
        """Get model name from request or use default.

        Args:
            request: LLM request

        Returns:
            Model name to use
        """
        return request.model or self.model

    def _build_messages(self, request: LLMRequest) -> list:
        """Convert request messages to provider-specific format.

        Args:
            request: LLM request

        Returns:
            List of messages in provider format
        """
        return [
            {
                "role": msg.role,
                "content": msg.content,
                **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
                **({"tool_call_id": msg.tool_call_id} if msg.tool_call_id else {})
            }
            for msg in request.messages
        ]

    def _measure_latency(self, start_time: float) -> float:
        """Calculate latency in milliseconds.

        Args:
            start_time: Start time from time.time()

        Returns:
            Latency in milliseconds
        """
        return (time.time() - start_time) * 1000

    def _extract_tool_calls(self, tool_calls_obj) -> list[JsonObject] | None:
        """Extract tool calls in standard format.

        Handles OpenAI-compatible tool call objects.

        Args:
            tool_calls_obj: Tool calls object from provider API response

        Returns:
            List of standardized tool call dicts, or None if no tool calls
        """
        if not tool_calls_obj:
            return None

        return [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in tool_calls_obj
        ]

    def _extract_stream_tool_calls(self, tool_calls_obj) -> list[JsonObject] | None:
        """Extract tool calls in standard format for streaming deltas."""
        if not tool_calls_obj:
            return None

        return [
            {
                "id": tc.id if hasattr(tc, 'id') else None,
                "type": tc.type if hasattr(tc, 'type') else None,
                "function": {
                    "name": tc.function.name if tc.function and hasattr(tc.function, 'name') else None,
                    "arguments": tc.function.arguments if tc.function and hasattr(tc.function, 'arguments') else None
                } if tc.function else None
            }
            for tc in tool_calls_obj
        ]

    def _extract_usage(self, usage_obj) -> dict[str, int] | None:
        """Extract usage statistics in standard format.

        Handles OpenAI-compatible usage objects with safe attribute access.

        Args:
            usage_obj: Usage object from provider API response

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens, or None
        """
        if not usage_obj:
            return None

        return {
            "prompt_tokens": getattr(usage_obj, 'prompt_tokens', 0),
            "completion_tokens": getattr(usage_obj, 'completion_tokens', 0),
            "total_tokens": getattr(usage_obj, 'total_tokens', 0),
        }

    def _build_response(
        self,
        request: LLMRequest,
        content: str,
        usage: dict[str, int] | None,
        latency_ms: float,
        raw_content: str | None = None,
        tool_calls: list | None = None,
        reasoning: str | None = None
    ) -> LLMResponse:
        """Build a standardized LLM response.

        This centralizes response building logic to reduce duplication across providers.

        Args:
            request: Original request
            content: Response content
            usage: Token usage statistics
            latency_ms: Latency in milliseconds
            raw_content: Raw response before cleaning (optional)
            tool_calls: Tool calls from response (optional)
            reasoning: Reasoning/thinking content (optional)

        Returns:
            LLMResponse object
        """
        # Clean content
        cleaned_content = clean_encoded_text(content) if content else ""

        # Extract system prompt from messages
        system_prompt = None
        if request.messages:
            for msg in request.messages:
                if msg.role == "system":
                    system_prompt = msg.content
                    break

        # Convert messages to dict format for prompt field
        prompt = [
            {
                "role": msg.role,
                "content": msg.content,
                **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
                **({"tool_call_id": msg.tool_call_id} if msg.tool_call_id else {})
            }
            for msg in request.messages
        ]

        return LLMResponse(
            provider=self.provider_name,
            model=self._get_model(request),
            content=cleaned_content,
            raw_content=raw_content or content,
            usage=usage,
            latency_ms=int(latency_ms),
            tool_calls=tool_calls,
            reasoning=reasoning,
            used_fallback=False,
            # Request parameters (following reference implementation)
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'gemini').

        Returns:
            Provider name string
        """
        pass
