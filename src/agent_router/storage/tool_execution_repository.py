"""Repository for tool execution records."""

from typing import List, Optional
from uuid import UUID
from datetime import datetime

from .repository import BaseRepository
from .models import ToolExecution


class ToolExecutionRepository(BaseRepository[ToolExecution]):
    """Repository for tool execution storage and retrieval.

    Provides specialized queries for analyzing tool usage patterns,
    performance metrics, and error rates.
    """

    @property
    def collection_name(self) -> str:
        return "tool_executions"

    @property
    def model_class(self) -> type[ToolExecution]:
        return ToolExecution

    async def get_by_conversation(
        self,
        conversation_id: UUID,
        limit: int = 100,
        skip: int = 0,
    ) -> List[ToolExecution]:
        """Get all tool executions for a conversation, ordered by timestamp.

        Args:
            conversation_id: Conversation UUID
            limit: Maximum number of executions to return
            skip: Number of executions to skip

        Returns:
            List of tool executions for the conversation
        """
        return await self.get_by_filter(
            filter_dict={"conversation_id": conversation_id},
            limit=limit,
            skip=skip,
            sort=[("timestamp", -1)],
        )

    async def get_by_tool_name(
        self,
        tool_name: str,
        limit: int = 100,
        skip: int = 0,
    ) -> List[ToolExecution]:
        """Get all executions for a specific tool.

        Args:
            tool_name: Tool name
            limit: Maximum number of executions to return
            skip: Number of executions to skip

        Returns:
            List of executions for the tool
        """
        return await self.get_by_filter(
            filter_dict={"tool_name": tool_name},
            limit=limit,
            skip=skip,
            sort=[("timestamp", -1)],
        )

    async def get_by_tool_type(
        self,
        tool_type: str,
        limit: int = 100,
        skip: int = 0,
    ) -> List[ToolExecution]:
        """Get all executions for a specific tool type.

        Args:
            tool_type: Tool type (e.g., "mcp_tool", "mcp_agent", "a2a_agent")
            limit: Maximum number of executions to return
            skip: Number of executions to skip

        Returns:
            List of executions for the tool type
        """
        return await self.get_by_filter(
            filter_dict={"tool_type": tool_type},
            limit=limit,
            skip=skip,
            sort=[("timestamp", -1)],
        )

    async def get_failed_executions(
        self,
        limit: int = 100,
        skip: int = 0,
    ) -> List[ToolExecution]:
        """Get all failed tool executions (with errors).

        Args:
            limit: Maximum number of executions to return
            skip: Number of executions to skip

        Returns:
            List of failed tool executions
        """
        return await self.get_by_filter(
            filter_dict={"error": {"$ne": None}},
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
    ) -> List[ToolExecution]:
        """Get tool executions within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of executions to return
            skip: Number of executions to skip

        Returns:
            List of tool executions in the time range
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

    async def get_execution_stats(
        self,
        conversation_id: Optional[UUID] = None,
        tool_name: Optional[str] = None,
        tool_type: Optional[str] = None,
    ) -> dict:
        """Get execution statistics for tool calls.

        Args:
            conversation_id: Optional conversation UUID to filter by
            tool_name: Optional tool name to filter by
            tool_type: Optional tool type to filter by

        Returns:
            Dictionary with execution statistics:
            - total_executions: Total number of executions
            - successful_executions: Number of successful executions
            - failed_executions: Number of failed executions
            - avg_latency_ms: Average latency
            - tools_used: List of tool names used
        """
        collection = await self._get_collection()

        # Build filter
        filter_dict = {}
        if conversation_id:
            filter_dict["conversation_id"] = str(conversation_id)
        if tool_name:
            filter_dict["tool_name"] = tool_name
        if tool_type:
            filter_dict["tool_type"] = tool_type

        # Aggregate statistics
        pipeline = [
            {"$match": filter_dict} if filter_dict else {"$match": {}},
            {
                "$group": {
                    "_id": None,
                    "total_executions": {"$sum": 1},
                    "successful_executions": {
                        "$sum": {"$cond": [{"$eq": ["$error", None]}, 1, 0]}
                    },
                    "failed_executions": {
                        "$sum": {"$cond": [{"$ne": ["$error", None]}, 1, 0]}
                    },
                    "avg_latency_ms": {"$avg": "$latency_ms"},
                    "tools_used": {"$addToSet": "$tool_name"},
                }
            },
        ]

        result = await collection.aggregate(pipeline).to_list(length=1)

        if not result:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "avg_latency_ms": 0,
                "tools_used": [],
            }

        stats = result[0]
        del stats["_id"]
        return stats

    async def get_tool_usage_breakdown(
        self,
        conversation_id: Optional[UUID] = None,
    ) -> List[dict]:
        """Get breakdown of tool usage by tool name.

        Args:
            conversation_id: Optional conversation UUID to filter by

        Returns:
            List of dictionaries with tool usage breakdown:
            - tool_name: Name of the tool
            - count: Number of times used
            - avg_latency_ms: Average latency for this tool
            - error_rate: Percentage of failed executions
        """
        collection = await self._get_collection()

        # Build filter
        filter_dict = {}
        if conversation_id:
            filter_dict["conversation_id"] = str(conversation_id)

        # Aggregate by tool name
        pipeline = [
            {"$match": filter_dict} if filter_dict else {"$match": {}},
            {
                "$group": {
                    "_id": "$tool_name",
                    "count": {"$sum": 1},
                    "avg_latency_ms": {"$avg": "$latency_ms"},
                    "total": {"$sum": 1},
                    "errors": {
                        "$sum": {"$cond": [{"$ne": ["$error", None]}, 1, 0]}
                    },
                }
            },
            {
                "$project": {
                    "tool_name": "$_id",
                    "count": 1,
                    "avg_latency_ms": 1,
                    "error_rate": {
                        "$multiply": [{"$divide": ["$errors", "$total"]}, 100]
                    },
                    "_id": 0,
                }
            },
            {"$sort": {"count": -1}},
        ]

        result = await collection.aggregate(pipeline).to_list(length=None)
        return result
