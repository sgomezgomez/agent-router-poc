"""LLM Service orchestrator with multi-provider support and fallback."""

import json
import logging
import asyncio
from typing import AsyncIterator
from agent_router.core.config import Settings
from agent_router.core.errors import LLMProviderError
from agent_router.llm.models import LLMRequest, LLMResponse, LLMStreamChunk, Message
from agent_router.llm.providers.base import BaseLLMProvider
from agent_router.llm.utils import clean_encoded_text, calculate_retry_delay
from agent_router.llm.model_registry import load_model_registry

logger = logging.getLogger(__name__)

class LLMService:
    """Multi-provider LLM service with automatic retry and fallback.

    Architecture follows reference implementation from mentalth-backend:
    - Centralized message formatting with text cleaning
    - Retry decorator for consistent error handling
    - Clean fallback strategy
    - Provider-agnostic interface
    - Lazy provider loading (providers created on first use)

    Fallback Strategy:
    1. Primary provider: 3 attempts with exponential backoff (via decorator)
    2. Fallback provider: If primary fails and fallback configured
    3. Error: Raise exception for Agent Layer to handle
    """

    def __init__(self, settings: Settings):
        """Initialize LLM service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.providers: dict[str, BaseLLMProvider] = {}
        self.provider_configs = settings.get_llm_providers_config()
        self.model_registry = load_model_registry()

        # Fallback configuration
        self.fallback_provider = settings.fallback_llm_provider
        self.fallback_model = settings.fallback_llm_model
        self.fallback_api_mode = settings.fallback_llm_api_mode
        self.fallback_temperature = settings.fallback_temperature
        self.fallback_max_tokens = settings.fallback_max_tokens
        self.fallback_top_p = settings.fallback_top_p
        self.fallback_top_k = settings.fallback_top_k
        self.fallback_thinking_budget = settings.fallback_thinking_budget
        self.fallback_thinking_effort = settings.fallback_thinking_effort

        logger.info("LLMService initialized with lazy provider loading")

    def _get_provider(self, provider_name: str) -> BaseLLMProvider:
        """Get or create provider (lazy loading).

        Providers are only instantiated when first requested, not at service initialization.
        This improves startup time and avoids errors for unconfigured providers.

        Args:
            provider_name: Provider name (e.g., "openai", "lm_studio", "gemini", "grok")

        Returns:
            Provider instance

        Raises:
            LLMProviderError: If provider not available or configuration missing
        """
        # Return cached provider if already loaded
        if provider_name in self.providers:
            return self.providers[provider_name]

        # Load provider on-demand
        if provider_name == "lm_studio":
            provider_config = self.provider_configs.get("lm_studio")
            from agent_router.llm.providers.lm_studio import LMStudioProvider
            self.providers["lm_studio"] = LMStudioProvider(
                model=self.fallback_model,
                base_url=(
                    provider_config.base_url if provider_config and provider_config.base_url
                    else self.settings.lm_studio_base_url
                ),
                api_mode=provider_config.api_mode if provider_config else self.fallback_api_mode,
                model_registry=self.model_registry
            )
            logger.info(f"Loaded LM Studio provider: {self.settings.lm_studio_base_url}")

        elif provider_name == "openai":
            if not self.settings.openai_api_key:
                raise LLMProviderError(
                    "OpenAI API key not configured. Set OPENAI_API_KEY in .env"
                )
            provider_config = self.provider_configs.get("openai")
            from agent_router.llm.providers.openai_compatible import OpenAIProvider
            self.providers["openai"] = OpenAIProvider(
                model=self.fallback_model,
                api_key=self.settings.openai_api_key,
                api_mode=provider_config.api_mode if provider_config else self.fallback_api_mode,
                model_registry=self.model_registry
            )
            logger.info("Loaded OpenAI provider")

        elif provider_name == "gemini":
            if not self.settings.gemini_api_key:
                raise LLMProviderError(
                    "Gemini API key not configured. Set GEMINI_API_KEY in .env"
                )
            provider_config = self.provider_configs.get("gemini")
            from agent_router.llm.providers.gemini import GeminiProvider
            self.providers["gemini"] = GeminiProvider(
                model=self.fallback_model,
                api_key=self.settings.gemini_api_key
            )
            logger.info("Loaded Gemini provider")

        elif provider_name == "grok":
            if not self.settings.grok_api_key:
                raise LLMProviderError(
                    "Grok API key not configured. Set GROK_API_KEY in .env"
                )
            provider_config = self.provider_configs.get("grok")
            from agent_router.llm.providers.grok import GrokProvider
            self.providers["grok"] = GrokProvider(
                model=self.fallback_model,
                api_key=self.settings.grok_api_key,
                api_mode=provider_config.api_mode if provider_config else self.fallback_api_mode,
                model_registry=self.model_registry
            )
            logger.info("Loaded Grok provider")

        else:
            raise LLMProviderError(
                f"Unknown provider '{provider_name}'. "
                f"Available: lm_studio, openai, gemini, grok"
            )

        return self.providers[provider_name]


    def _build_messages(
        self,
        messages: list[Message] | None,
        system_prompt: str | None,
        user_prompt: str | None
    ) -> list[Message]:
        """Build message list from various input formats.

        Follows reference implementation pattern of centralized message building.

        Args:
            messages: Existing messages (if provided)
            system_prompt: System prompt (if messages not provided)
            user_prompt: User prompt (if messages not provided)

        Returns:
            List of messages with cleaned content

        Raises:
            LLMProviderError: If no valid input provided
        """
        if messages:
            # Clean existing messages
            return [
                Message(
                    role=msg.role,
                    content=clean_encoded_text(msg.content),
                    tool_calls=msg.tool_calls,
                    tool_call_id=msg.tool_call_id
                )
                for msg in messages
            ]

        # Build from prompts
        built_messages = []

        if system_prompt:
            built_messages.append(
                Message(role="system", content=clean_encoded_text(system_prompt))
            )

        if user_prompt:
            built_messages.append(
                Message(role="user", content=clean_encoded_text(user_prompt))
            )

        if not built_messages:
            raise LLMProviderError(
                "Must provide either messages or user_prompt/system_prompt"
            )

        return built_messages

    async def generate(
        self,
        messages: list[Message] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        thinking_budget: int | None = None,
        thinking_effort: str | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
        response_format: type | None = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from LLM with retry and fallback.

        Follows reference implementation's clean separation of concerns:
        - Message building
        - Primary generation with retries
        - Fallback on failure
        - Structured output parsing (if response_format provided)

        Args:
            messages: Conversation messages (if not provided, built from prompts)
            system_prompt: System prompt (if messages not provided)
            user_prompt: User prompt (if messages not provided)
            provider: LLM provider to use (defaults to fallback provider)
            model: Specific model to use (optional, provider uses default)
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            thinking_budget: Thinking budget for reasoning models (Gemini 2.5, o1, etc.)
            tools: Available tools for function calling
            tool_choice: Tool choice strategy
            response_format: Pydantic model class for structured output (e.g., MyResponseModel)
            **kwargs: Additional provider-specific parameters

        Returns:
            LLM response (with parsed_response populated if response_format provided)

        Raises:
            LLMProviderError: If all attempts fail
        """
        # Build and clean messages
        built_messages = self._build_messages(messages, system_prompt, user_prompt)

        # Determine provider
        primary_provider = provider or self.fallback_provider
        model = model or self.fallback_model

        # Apply fallback defaults when agent config omits values
        temperature = (
            temperature if temperature is not None else self.settings.fallback_temperature
        )
        top_p = top_p if top_p is not None else self.settings.fallback_top_p
        top_k = top_k if top_k is not None else self.settings.fallback_top_k
        max_tokens = (
            max_tokens if max_tokens is not None else self.settings.fallback_max_tokens
        )
        thinking_budget = (
            thinking_budget
            if thinking_budget is not None
            else self.settings.fallback_thinking_budget
        )
        thinking_effort = (
            thinking_effort
            if thinking_effort is not None
            else self.settings.fallback_thinking_effort
        )

        # Build request for primary provider
        request = LLMRequest(
            messages=built_messages,
            provider=primary_provider,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            thinking_effort=thinking_effort,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            stream=False
        )

        try:
            # Try primary provider with retries (via decorator)
            logger.info(
                f"Generating with {primary_provider}, model={model or 'default'}"
            )
            return await self._call_provider(request, primary_provider)

        except Exception as primary_error:
            # Check if we're already using fallback configuration
            using_fallback_already = (
                primary_provider == self.fallback_provider and
                (model or self.fallback_model) == self.fallback_model
            )

            if using_fallback_already:
                # Already using fallback, can't fall back further
                raise LLMProviderError(
                    f"Provider {primary_provider} failed: {primary_error}"
                ) from primary_error

            # Try fallback provider
            logger.warning(
                f"Primary {primary_provider} failed, trying fallback {self.fallback_provider}"
            )

            try:
                return await self._try_fallback_provider(
                    built_messages=built_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format
                )

            except Exception as fallback_error:
                # Both primary and fallback failed
                raise LLMProviderError(
                    f"Primary ({primary_provider}) and fallback "
                    f"({self.fallback_provider}) both failed"
                ) from fallback_error

    async def generate_stream(
        self,
        messages: list[Message] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        thinking_budget: int | None = None,
        thinking_effort: str | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None
    ) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming response from LLM.

        Note: Streaming does not use retry or fallback - single attempt only.
        This matches reference implementation behavior.

        Args:
            messages: Conversation messages
            system_prompt: System prompt (if messages not provided)
            user_prompt: User prompt (if messages not provided)
            provider: LLM provider to use
            model: Specific model to use
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate
            thinking_budget: Thinking budget for reasoning models
            tools: Available tools
            tool_choice: Tool choice strategy

        Yields:
            Stream chunks

        Raises:
            LLMProviderError: If streaming fails
        """
        # Build and clean messages
        built_messages = self._build_messages(messages, system_prompt, user_prompt)

        # Determine provider
        provider_name = provider or self.fallback_provider
        model = model or self.fallback_model

        # Apply fallback defaults when agent config omits values
        temperature = (
            temperature if temperature is not None else self.settings.fallback_temperature
        )
        top_p = top_p if top_p is not None else self.settings.fallback_top_p
        top_k = top_k if top_k is not None else self.settings.fallback_top_k
        max_tokens = (
            max_tokens if max_tokens is not None else self.settings.fallback_max_tokens
        )
        thinking_budget = (
            thinking_budget
            if thinking_budget is not None
            else self.settings.fallback_thinking_budget
        )
        thinking_effort = (
            thinking_effort
            if thinking_effort is not None
            else self.settings.fallback_thinking_effort
        )

        # Build request
        request = LLMRequest(
            messages=built_messages,
            provider=provider_name,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            thinking_effort=thinking_effort,
            tools=tools,
            tool_choice=tool_choice,
            stream=True
        )

        logger.info(
            f"Streaming from {provider_name}, model={model or 'default'}"
        )

        # Stream from provider with retry policy
        max_retries = self.settings.llm_retry_max_retries
        base_delay = self.settings.llm_retry_base_delay
        max_delay = self.settings.llm_retry_max_delay
        exponential_base = self.settings.llm_retry_exponential_base

        for attempt in range(max_retries):
            yielded_any = False
            try:
                provider = self._get_provider(provider_name)
                async for chunk in provider.generate_stream(request):
                    yielded_any = True
                    yield chunk
                return
            except Exception as e:
                if yielded_any or attempt >= max_retries - 1:
                    raise LLMProviderError(
                        f"Streaming failed after {attempt + 1} attempts: {e}"
                    ) from e
                delay = calculate_retry_delay(
                    attempt,
                    base_delay,
                    max_delay,
                    exponential_base
                )
                logger.warning(
                    f"Streaming attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

    def get_available_providers(self) -> list[str]:
        """Get list of available providers based on configuration.

        Note: With lazy loading, providers are only instantiated when used.
        This returns providers that CAN be loaded based on configuration.

        Returns:
            List of provider names that can be loaded
        """
        available = ["lm_studio"]  # Always available (local)

        if self.settings.openai_api_key:
            available.append("openai")

        if self.settings.gemini_api_key:
            available.append("gemini")

        if self.settings.grok_api_key:
            available.append("grok")

        return available
    
    async def _call_provider(
        self,
        request: LLMRequest,
        provider_name: str
    ) -> LLMResponse:
        """Call provider with automatic retry via decorator.

        Follows reference implementation pattern of decorated provider calls.
        Uses lazy loading - provider is only instantiated on first use.

        Args:
            request: LLM request
            provider_name: Provider name

        Returns:
            LLM response

        Raises:
            LLMProviderError: If provider not found or generation fails
        """
        max_retries = self.settings.llm_retry_max_retries
        base_delay = self.settings.llm_retry_base_delay
        max_delay = self.settings.llm_retry_max_delay
        exponential_base = self.settings.llm_retry_exponential_base

        last_exception: Exception | None = None
        for attempt in range(max_retries):
            try:
                # Get provider (lazy loads if not already cached)
                provider = self._get_provider(provider_name)

                response = await provider.generate(request)

                # Parse structured response if response_format was provided
                if request.response_format and response.content:
                    try:
                        # Parse JSON response and instantiate Pydantic model
                        json_data = json.loads(response.content)
                        parsed_response = request.response_format(**json_data)
                        response.parsed_response = parsed_response
                        logger.debug(
                            f"Parsed structured response: {type(parsed_response).__name__}"
                        )
                    except json.JSONDecodeError as e:
                        response.parse_error = f"JSONDecodeError: {e}"
                        logger.warning(f"Failed to parse JSON response: {e}")
                        raise LLMProviderError(
                            f"Structured response parse failed: {e}"
                        ) from e
                    except Exception as e:
                        response.parse_error = f"ModelParseError: {e}"
                        logger.warning(f"Failed to instantiate response model: {e}")
                        raise LLMProviderError(
                            f"Structured response model failed: {e}"
                        ) from e

                logger.info(
                    f"Generated: provider={response.provider}, "
                    f"model={response.model}, "
                    f"latency={response.latency_ms}ms, "
                    f"tokens={response.usage.get('total_tokens', 0) if response.usage else 0}"
                )

                return response
            except Exception as e:
                last_exception = e
                if attempt >= max_retries - 1:
                    raise LLMProviderError(
                        f"Failed after {max_retries} attempts: {e}"
                    ) from e

                delay = calculate_retry_delay(
                    attempt,
                    base_delay,
                    max_delay,
                    exponential_base
                )
                logger.warning(
                    f"_call_provider attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)

        if last_exception is not None:
            raise LLMProviderError(
                f"Failed after {max_retries} attempts: {last_exception}"
            ) from last_exception
        raise LLMProviderError("Failed after retries with unknown error")
    
    async def _try_fallback_provider(
        self,
        built_messages: list[Message],
        tools: list | None,
        tool_choice: str | dict | None,
        response_format: type | None
    ) -> LLMResponse:
        """Try fallback provider after primary provider fails.

        Args:
            built_messages: Cleaned messages to send
            tools: Available tools
            tool_choice: Tool choice strategy
            response_format: Pydantic model class for structured output

        Returns:
            LLM response from fallback provider

        Raises:
            LLMProviderError: If fallback also fails
        """
        fallback_request = LLMRequest(
            messages=built_messages,
            provider=self.fallback_provider,
            model=self.fallback_model,
            temperature=self.fallback_temperature,
            max_tokens=self.fallback_max_tokens,
            top_p=self.settings.fallback_top_p,
            top_k=self.settings.fallback_top_k,
            thinking_budget=self.settings.fallback_thinking_budget,
            thinking_effort=self.settings.fallback_thinking_effort,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            stream=False
        )

        response = await self._call_provider(
            fallback_request,
            self.fallback_provider
        )
        response.used_fallback = True
        logger.info(f"Fallback {self.fallback_provider} succeeded")
        return response
