"""Storage layer for MongoDB persistence.

This module provides:
- Models: Pydantic models for MongoDB documents
- Connection: MongoDB connection management
- Repositories: CRUD operations for each collection
"""

from .models import LLMCall, ToolExecution, Conversation, Message
from .builders import build_llm_call, build_tool_execution, build_conversation_message
from .db import MongoDBConnection, get_db
from .repository import BaseRepository
from .llm_call_repository import LLMCallRepository
from .tool_execution_repository import ToolExecutionRepository
from .conversation_repository import ConversationRepository

__all__ = [
    # Models
    "LLMCall",
    "ToolExecution",
    "Conversation",
    "Message",
    # Connection
    "MongoDBConnection",
    "get_db",
    # Repositories
    "BaseRepository",
    "LLMCallRepository",
    "ToolExecutionRepository",
    "ConversationRepository",
    # Builders
    "build_llm_call",
    "build_tool_execution",
    "build_conversation_message",
]
