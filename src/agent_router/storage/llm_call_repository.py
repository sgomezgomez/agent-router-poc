"""Repository for LLM call records."""

from typing import List, Optional
from uuid import UUID
from datetime import datetime

from .repository import BaseRepository
from .models import LLMCall


class LLMCallRepository(BaseRepository[LLMCall]):
    """Repository for LLM call storage and retrieval.

    Provides specialized queries for analyzing LLM usage patterns,
    performance metrics, and fallback behavior.
    """

    @property
    def collection_name(self) -> str:
        return "llm_calls"

    @property
    def model_class(self) -> type[LLMCall]:
        return LLMCall

    async def get_by_conversation(
        self,
        conversation_id: UUID,
        limit: int = 100,
        skip: int = 0,
    ) -> List[LLMCall]:
        """Get all LLM calls for a conversation, ordered by timestamp.

        Args:
            conversation_id: Conversation UUID
            limit: Maximum number of calls to return
            skip: Number of calls to skip

        Returns:
            List of LLM calls for the conversation
        """
        return await self.get_by_filter(
            filter_dict={"conversation_id": conversation_id},
            limit=limit,
            skip=skip,
            sort=[("timestamp", -1)],
        )

    async def get_by_provider(
        self,
        provider: str,
        limit: int = 100,
        skip: int = 0,
    ) -> List[LLMCall]:
        """Get all LLM calls for a specific provider.

        Args:
            provider: Provider name (e.g., "openai", "gemini")
            limit: Maximum number of calls to return
            skip: Number of calls to skip

        Returns:
            List of LLM calls for the provider
        """
        return await self.get_by_filter(
            filter_dict={"provider": provider},
            limit=limit,
            skip=skip,
            sort=[("timestamp", -1)],
        )

    async def get_failed_calls(
        self,
        limit: int = 100,
        skip: int = 0,
    ) -> List[LLMCall]:
        """Get all failed LLM calls (with errors).

        Args:
            limit: Maximum number of calls to return
            skip: Number of calls to skip

        Returns:
            List of failed LLM calls
        """
        return await self.get_by_filter(
            filter_dict={"error": {"$ne": None}},
            limit=limit,
            skip=skip,
            sort=[("timestamp", -1)],
        )

    async def get_fallback_calls(
        self,
        limit: int = 100,
        skip: int = 0,
    ) -> List[LLMCall]:
        """Get all LLM calls that used fallback configuration.

        Args:
            limit: Maximum number of calls to return
            skip: Number of calls to skip

        Returns:
            List of LLM calls that used fallback
        """
        return await self.get_by_filter(
            filter_dict={"used_fallback": True},
            limit=limit,
            skip=skip,
            sort=[("timestamp", -1)],
        )

    async def get_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
        skip: int = 0,
    ) -> List[LLMCall]:
        """Get LLM calls within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of calls to return
            skip: Number of calls to skip

        Returns:
            List of LLM calls in the time range
        """
        return await self.get_by_filter(
            filter_dict={
                "timestamp": {
                    "$gte": start_time,
                    "$lte": end_time,
                }
            },
            limit=limit,
            skip=skip,
            sort=[("timestamp", -1)],
        )

    async def get_usage_stats(
        self,
        conversation_id: Optional[UUID] = None,
        provider: Optional[str] = None,
    ) -> dict:
        """Get usage statistics for LLM calls.

        Args:
            conversation_id: Optional conversation UUID to filter by
            provider: Optional provider name to filter by

        Returns:
            Dictionary with usage statistics:
            - total_calls: Total number of calls
            - successful_calls: Number of successful calls
            - failed_calls: Number of failed calls
            - fallback_calls: Number of calls using fallback
            - total_tokens: Total tokens used
            - avg_latency_ms: Average latency
        """
        collection = await self._get_collection()

        # Build filter
        filter_dict = {}
        if conversation_id:
            filter_dict["conversation_id"] = str(conversation_id)
        if provider:
            filter_dict["provider"] = provider

        # Aggregate statistics
        pipeline = [
            {"$match": filter_dict} if filter_dict else {"$match": {}},
            {
                "$group": {
                    "_id": None,
                    "total_calls": {"$sum": 1},
                    "successful_calls": {
                        "$sum": {"$cond": [{"$eq": ["$error", None]}, 1, 0]}
                    },
                    "failed_calls": {
                        "$sum": {"$cond": [{"$ne": ["$error", None]}, 1, 0]}
                    },
                    "fallback_calls": {
                        "$sum": {"$cond": [{"$eq": ["$used_fallback", True]}, 1, 0]}
                    },
                    "total_tokens": {
                        "$sum": {"$ifNull": ["$usage.total_tokens", 0]}
                    },
                    "avg_latency_ms": {"$avg": "$latency_ms"},
                }
            },
        ]

        result = await collection.aggregate(pipeline).to_list(length=1)

        if not result:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "fallback_calls": 0,
                "total_tokens": 0,
                "avg_latency_ms": 0,
            }

        stats = result[0]
        del stats["_id"]
        return stats
