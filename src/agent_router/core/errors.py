"""Custom exception classes for agent router."""


class AgentRouterError(Exception):
    """Base exception for all agent router errors."""

    pass


class LLMProviderError(AgentRouterError):
    """LLM provider specific errors."""

    pass


class ToolExecutionError(AgentRouterError):
    """Tool execution errors."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ConfigurationError(AgentRouterError):
    """Configuration errors."""

    pass
