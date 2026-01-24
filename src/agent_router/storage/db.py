"""MongoDB connection manager."""

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional

from ..core.config import MongoDBConfig

logger = logging.getLogger(__name__)


class MongoDBConnection:
    """Singleton MongoDB connection manager.

    Manages a single MongoDB connection throughout the application lifecycle.
    Uses Motor for async MongoDB operations.
    """

    _instance: Optional["MongoDBConnection"] = None
    _client: Optional[AsyncIOMotorClient] = None
    _database: Optional[AsyncIOMotorDatabase] = None

    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    async def initialize(cls, config: MongoDBConfig) -> "MongoDBConnection":
        """Initialize the MongoDB connection.

        Args:
            config: MongoDB configuration

        Returns:
            MongoDBConnection instance

        Raises:
            Exception: If connection fails
        """
        instance = cls()

        if instance._client is None:
            logger.info(f"Connecting to MongoDB: {config.uri}")

            # Build client arguments
            client_args = {
                "maxPoolSize": config.max_pool_size,
                "minPoolSize": config.min_pool_size,
            }

            # Add authentication if provided
            if config.username and config.password:
                from urllib.parse import quote_plus
                logger.info(f"Using authentication with username: {config.username}")

                # URL encode credentials and build connection string
                username = quote_plus(config.username)
                password = quote_plus(config.password)

                # Parse base URI and add credentials
                if "://" in config.uri:
                    protocol, rest = config.uri.split("://", 1)
                    # Use the target database as authSource (where user was created)
                    auth_uri = f"{protocol}://{username}:{password}@{rest.split('?')[0]}?authSource={config.database}&retryWrites=false"
                    instance._client = AsyncIOMotorClient(auth_uri, **client_args)
                else:
                    # Fallback: use parameters (though URI method is preferred)
                    client_args["username"] = config.username
                    client_args["password"] = config.password
                    client_args["authSource"] = config.database
                    instance._client = AsyncIOMotorClient(config.uri, **client_args)
            else:
                instance._client = AsyncIOMotorClient(config.uri, **client_args)
            instance._database = instance._client[config.database]

            # Test connection
            try:
                await instance._client.admin.command("ping")
                logger.info("MongoDB connection established successfully")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                instance._client = None
                instance._database = None
                raise

        return instance

    @classmethod
    def get_database(cls) -> AsyncIOMotorDatabase:
        """Get the MongoDB database instance.

        Returns:
            AsyncIOMotorDatabase instance

        Raises:
            RuntimeError: If not initialized
        """
        instance = cls()
        if instance._database is None:
            raise RuntimeError(
                "MongoDB connection not initialized. Call initialize() first."
            )
        return instance._database

    @classmethod
    async def close(cls):
        """Close the MongoDB connection."""
        instance = cls()
        if instance._client is not None:
            logger.info("Closing MongoDB connection")
            instance._client.close()
            instance._client = None
            instance._database = None

    @classmethod
    async def create_indexes(cls):
        """Create necessary indexes for performance.

        Should be called once during application startup after initialize().
        """
        db = cls.get_database()

        # LLM calls indexes
        await db.llm_calls.create_index("conversation_id")
        await db.llm_calls.create_index("timestamp")
        await db.llm_calls.create_index("provider")
        await db.llm_calls.create_index("used_fallback")
        await db.llm_calls.create_index([("conversation_id", 1), ("timestamp", -1)])

        # Tool executions indexes
        await db.tool_executions.create_index("conversation_id")
        await db.tool_executions.create_index("timestamp")
        await db.tool_executions.create_index("tool_name")
        await db.tool_executions.create_index("tool_type")
        await db.tool_executions.create_index([("conversation_id", 1), ("timestamp", -1)])

        # Conversation indexes
        await db.conversations.create_index("created_at")
        await db.conversations.create_index("last_updated")
        await db.conversations.create_index("user_id")

        logger.info("MongoDB indexes created successfully")


# Convenience function for getting database
async def get_db() -> AsyncIOMotorDatabase:
    """Get the MongoDB database instance.

    Returns:
        AsyncIOMotorDatabase instance

    Raises:
        RuntimeError: If not initialized
    """
    return MongoDBConnection.get_database()
