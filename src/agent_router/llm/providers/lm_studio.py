"""LM Studio provider implementation.

LM Studio provides a local OpenAI-compatible API for running models locally.
This is a thin wrapper around OpenAICompatibleProvider with LM Studio-specific configuration.

All configuration (base URL) comes from Settings/environment variables.
"""

from agent_router.llm.providers.openai_compatible import OpenAICompatibleProvider


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio provider using OpenAI-compatible API.

    LM Studio provides a local OpenAI-compatible API for running models locally.
    Default endpoint: http://localhost:1234/v1

    This is a thin wrapper that:
    - Sets a dummy API key (LM Studio doesn't require authentication)
    - Configures the local base URL
    - Uses all standard OpenAI-compatible features from parent class
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        **kwargs
    ):
        """Initialize LM Studio provider.

        Args:
            model: Model name (loaded in LM Studio)
            base_url: LM Studio API endpoint (from settings)
            **kwargs: Additional configuration
        """
        # LM Studio doesn't require authentication, use dummy key
        super().__init__(
            model=model,
            api_key="lm-studio",
            base_url=base_url,
            **kwargs
        )

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "lm_studio"
