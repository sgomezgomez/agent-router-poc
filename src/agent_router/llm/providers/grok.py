"""Grok provider implementation.

Grok (by xAI) provides an OpenAI-compatible API.
This is a thin wrapper around OpenAICompatibleProvider with Grok-specific configuration.

All configuration (API key, base URL) comes from Settings/environment variables.
"""

from agent_router.llm.providers.openai_compatible import OpenAICompatibleProvider


class GrokProvider(OpenAICompatibleProvider):
    """Grok provider using OpenAI-compatible API.

    Grok (by xAI) provides an OpenAI-compatible API.
    Default endpoint: https://api.x.ai/v1

    This is a thin wrapper that:
    - Configures the xAI base URL
    - Uses all standard OpenAI-compatible features from parent class
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        **kwargs
    ):
        """Initialize Grok provider.

        Args:
            model: Grok model name
            api_key: xAI API key (from settings)
            base_url: Grok API endpoint (from settings)
            **kwargs: Additional configuration
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "grok"
