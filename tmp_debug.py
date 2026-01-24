import asyncio
from agent_router.core.config import Settings
from agent_router.storage import MongoDBConnection, ConversationRepository, Conversation

async def main():
    settings = Settings()
    await MongoDBConnection.initialize(settings.mongodb)
    repo = ConversationRepository()
    conv = Conversation()
    await repo.create(conv)
    await repo.add_message(conv.uuid, "user", "First question?")
    await repo.add_message(conv.uuid, "assistant", "First answer.")
    conv = await repo.get_by_id(conv.uuid)
    print('type', type(conv.message_history), 'len', len(conv.message_history))
    print('elem type', type(conv.message_history[0]), conv.message_history[0])
    await MongoDBConnection.close()

asyncio.run(main())
