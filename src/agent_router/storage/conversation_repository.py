"""Repository for conversation history."""

from typing import List, Optional
from uuid import UUID
from datetime import datetime

from .repository import BaseRepository
from .models import Conversation, Message


class ConversationRepository(BaseRepository[Conversation]):
    """Repository for conversation storage and history management."""

    @property
    def collection_name(self) -> str:
        return "conversations"

    @property
    def model_class(self) -> type[Conversation]:
        return Conversation

    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        **kwargs,
    ) -> Optional[Conversation]:
        """Add a message to a conversation's history.

        Args:
            conversation_id: Conversation UUID
            role: Message role (user/assistant/system)
            content: Message content
            **kwargs: Additional message metadata

        Returns:
            Updated conversation or None if not found
        """
        conversation = await self.get_by_id(conversation_id)
        if conversation is None:
            return None

        # Add message using the session's helper method
        conversation.add_message(role, content, **kwargs)

        # Update in database
        await self.update(
            conversation_id,
            {
                "message_history": [
                    msg.model_dump(mode="json") for msg in conversation.message_history
                ],
                "last_updated": conversation.last_updated,
            },
        )

        return conversation

    async def get_recent_messages(
        self,
        conversation_id: UUID,
        count: int = 10,
    ) -> List[Message]:
        """Get the most recent messages from a conversation.

        Args:
            conversation_id: Conversation UUID
            count: Number of messages to retrieve

        Returns:
            List of recent messages (empty if conversation not found)
        """
        conversation = await self.get_by_id(conversation_id)
        if conversation is None:
            return []

        return conversation.get_recent_messages(count)

    async def get_context_window(
        self,
        conversation_id: UUID,
        max_tokens: int = 4000,
    ) -> List[Message]:
        """Get messages that fit within a token budget.

        Args:
            conversation_id: Conversation UUID
            max_tokens: Maximum tokens to include

        Returns:
            List of messages within token budget (empty if conversation not found)
        """
        conversation = await self.get_by_id(conversation_id)
        if conversation is None:
            return []

        return conversation.get_context_window(max_tokens)

    async def get_by_user(
        self,
        user_id: str,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Conversation]:
        """Get all conversations for a user.

        Args:
            user_id: User ID
            limit: Maximum number of conversations to return
            skip: Number of conversations to skip

        Returns:
            List of conversations for the user
        """
        return await self.get_by_filter(
            filter_dict={"user_id": user_id},
            limit=limit,
            skip=skip,
            sort=[("last_updated", -1)],
        )

    async def get_active_conversations(
        self,
        since: datetime,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Conversation]:
        """Get conversations active since a specific time.

        Args:
            since: Get conversations updated after this time
            limit: Maximum number of conversations to return
            skip: Number of conversations to skip

        Returns:
            List of active conversations
        """
        return await self.get_by_filter(
            filter_dict={"last_updated": {"$gte": since}},
            limit=limit,
            skip=skip,
            sort=[("last_updated", -1)],
        )

    async def get_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Conversation]:
        """Get conversations created within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of conversations to return
            skip: Number of conversations to skip

        Returns:
            List of conversations in the time range
        """
        return await self.get_by_filter(
            filter_dict={
                "created_at": {
                    "$gte": start_time,
                    "$lte": end_time,
                }
            },
            limit=limit,
            skip=skip,
            sort=[("created_at", -1)],
        )

    async def clear_old_conversations(
        self,
        older_than: datetime,
    ) -> int:
        """Delete conversations older than a specific time.

        Useful for cleanup and GDPR compliance.

        Args:
            older_than: Delete conversations last updated before this time

        Returns:
            Number of conversations deleted
        """
        collection = await self._get_collection()
        result = await collection.delete_many(
            {"last_updated": {"$lt": older_than}}
        )
        return result.deleted_count

    async def get_conversation_stats(
        self,
        user_id: Optional[str] = None,
    ) -> dict:
        """Get statistics about conversations.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            Dictionary with conversation statistics:
            - total_conversations: Total number of conversations
            - avg_messages: Average messages per conversation
            - total_messages: Total messages across all conversations
        """
        collection = await self._get_collection()

        # Build filter
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = user_id

        # Aggregate statistics
        pipeline = [
            {"$match": filter_dict} if filter_dict else {"$match": {}},
            {
                "$project": {
                    "message_count": {"$size": "$message_history"}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_conversations": {"$sum": 1},
                    "total_messages": {"$sum": "$message_count"},
                    "avg_messages": {"$avg": "$message_count"},
                }
            },
        ]

        result = await collection.aggregate(pipeline).to_list(length=1)

        if not result:
            return {
                "total_conversations": 0,
                "avg_messages": 0,
                "total_messages": 0,
            }

        stats = result[0]
        del stats["_id"]
        return stats
