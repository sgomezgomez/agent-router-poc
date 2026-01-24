"""Abstract interfaces for all major components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, TYPE_CHECKING
from agent_router.core.types import JsonObject, JsonValue

if TYPE_CHECKING:
    from agent_router.llm.models import LLMRequest, LLMResponse, LLMStreamChunk


class LLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM.

        Args:
            request: Structured LLM request payload

        Returns:
            LLM response with metadata
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        request: LLMRequest
    ) -> AsyncIterator[LLMStreamChunk]:
        """Generate streaming response from LLM.

        Args:
            request: Structured LLM request payload

        Yields:
            Streaming chunks with metadata
        """
        ...


class ToolConnector(ABC):
    """Abstract base for tool connectors (MCP, A2A)."""

    @abstractmethod
    async def connect(self, config: JsonObject) -> None:
        """Establish connection to tool/agent service.

        Args:
            config: Connection configuration dictionary
        """
        ...

    @abstractmethod
    async def execute(self, tool_name: str, parameters: JsonObject) -> JsonObject:
        """Execute a tool or agent call.

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool

        Returns:
            Execution result dictionary
        """
        ...

    @abstractmethod
    async def list_tools(self) -> list:
        """List available tools/agents.

        Returns:
            List of available tools with their descriptions
        """
        ...


class Repository(ABC):
    """Abstract base repository for data persistence."""

    @abstractmethod
    async def create(self, data: JsonObject) -> str:
        """Create a new document.

        Args:
            data: Document data

        Returns:
            UUID of created document
        """
        ...

    @abstractmethod
    async def find_by_id(self, id: str) -> JsonObject | None:
        """Find document by ID.

        Args:
            id: Document UUID

        Returns:
            Document data or None if not found
        """
        ...
