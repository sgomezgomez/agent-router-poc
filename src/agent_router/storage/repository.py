"""Abstract base repository for MongoDB operations."""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional
from uuid import UUID
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import ReturnDocument
from pydantic import BaseModel

from .db import get_db
from agent_router.core.types import JsonObject, JsonValue

T = TypeVar("T", bound=BaseModel)


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository for MongoDB CRUD operations.

    Provides common database operations for all storage models.
    Subclasses must implement the model property and collection_name property.
    """

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Name of the MongoDB collection."""
        pass

    @property
    @abstractmethod
    def model_class(self) -> type[T]:
        """Pydantic model class for this repository."""
        pass

    async def _get_collection(self) -> AsyncIOMotorCollection:
        """Get the MongoDB collection for this repository.

        Returns:
            AsyncIOMotorCollection instance
        """
        db = await get_db()
        return db[self.collection_name]

    async def create(self, item: T) -> T:
        """Create a new document.

        Args:
            item: Model instance to create

        Returns:
            Created model instance with assigned ID

        Raises:
            Exception: If creation fails
        """
        collection = await self._get_collection()
        document = item.model_dump(mode="json")

        # Convert UUID to string for MongoDB
        if "uuid" in document:
            document["_id"] = str(document["uuid"])
            del document["uuid"]

        await collection.insert_one(document)
        return item

    async def get_by_id(self, uuid: UUID) -> Optional[T]:
        """Get a document by UUID.

        Args:
            uuid: Document UUID

        Returns:
            Model instance or None if not found
        """
        collection = await self._get_collection()
        document = await collection.find_one({"_id": str(uuid)})

        if document is None:
            return None

        # Convert _id back to uuid
        document["uuid"] = document["_id"]
        del document["_id"]

        return self.model_class(**document)

    async def get_by_filter(
        self,
        filter_dict: JsonObject,
        limit: int = 100,
        skip: int = 0,
        sort: Optional[List[tuple]] = None,
    ) -> List[T]:
        """Get documents matching a filter.

        Args:
            filter_dict: MongoDB filter dictionary
            limit: Maximum number of documents to return
            skip: Number of documents to skip
            sort: List of (field, direction) tuples for sorting

        Returns:
            List of model instances
        """
        collection = await self._get_collection()

        # Convert UUID values in filter to strings
        filter_dict = self._convert_uuids_to_strings(filter_dict)

        cursor = collection.find(filter_dict).skip(skip).limit(limit)

        if sort:
            cursor = cursor.sort(sort)

        documents = await cursor.to_list(length=limit)

        results = []
        for doc in documents:
            # Convert _id back to uuid
            doc["uuid"] = doc["_id"]
            del doc["_id"]
            results.append(self.model_class(**doc))

        return results

    async def update(self, uuid: UUID, updates: JsonObject) -> Optional[T]:
        """Update a document.

        Args:
            uuid: Document UUID
            updates: Dictionary of fields to update

        Returns:
            Updated model instance or None if not found
        """
        collection = await self._get_collection()

        # Convert UUID values to strings
        updates = self._convert_uuids_to_strings(updates)
        updates.pop("uuid", None)

        result = await collection.find_one_and_update(
            {"_id": str(uuid)},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )

        if result is None:
            return None

        # Convert _id back to uuid
        result["uuid"] = result["_id"]
        del result["_id"]

        return self.model_class(**result)

    async def delete(self, uuid: UUID) -> bool:
        """Delete a document.

        Args:
            uuid: Document UUID

        Returns:
            True if deleted, False if not found
        """
        collection = await self._get_collection()
        result = await collection.delete_one({"_id": str(uuid)})
        return result.deleted_count > 0

    async def count(self, filter_dict: Optional[JsonObject] = None) -> int:
        """Count documents matching a filter.

        Args:
            filter_dict: MongoDB filter dictionary (None for all documents)

        Returns:
            Number of matching documents
        """
        collection = await self._get_collection()
        filter_dict = filter_dict or {}
        filter_dict = self._convert_uuids_to_strings(filter_dict)
        return await collection.count_documents(filter_dict)

    def _convert_uuids_to_strings(self, data: JsonObject) -> JsonObject:
        """Convert UUID values in dictionary to strings for MongoDB.

        Args:
            data: Dictionary that may contain UUID values

        Returns:
            Dictionary with UUIDs converted to strings
        """
        return {key: self._convert_value(value) for key, value in data.items()}

    def _convert_value(self, value: JsonValue) -> JsonValue:
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, dict):
            return self._convert_uuids_to_strings(value)
        if isinstance(value, list):
            return [self._convert_value(item) for item in value]
        return value
