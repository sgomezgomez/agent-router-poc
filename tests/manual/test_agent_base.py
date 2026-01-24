"""Manual smoke test for Agent Base (Phase 3).

Run with: python tests/manual/test_agent_base.py
"""

import asyncio
from pathlib import Path

import httpx

from agent_router.core.config import Settings
from agent_router.llm.service import LLMService
from agent_router.storage import (
    MongoDBConnection,
    ConversationRepository,
    LLMCallRepository,
    Conversation,
)
from agent_router.agent import Agent, AgentConfig, MessageWindow


async def main() -> int:
    settings = Settings()
    await MongoDBConnection.initialize(settings.mongodb)

    llm_service = LLMService(settings)
    conversation_repo = ConversationRepository()
    llm_call_repo = LLMCallRepository()

    repo_root = Path(__file__).resolve().parents[2]
    config = AgentConfig.from_yaml(
        path=repo_root / "config" / "agents" / "router.yaml",
        agent_name="router",
    )
    config = config.model_copy(update={
        "provider": settings.fallback_llm_provider,
        "model": settings.fallback_llm_model,
        "temperature": settings.fallback_temperature,
        "max_tokens": settings.fallback_max_tokens,
        "top_p": settings.fallback_top_p,
        "top_k": settings.fallback_top_k,
        "thinking_budget": settings.fallback_thinking_budget,
        "thinking_effort": settings.fallback_thinking_effort,
        "api_mode": settings.fallback_llm_api_mode,
    })

    if config.provider == "lm_studio":
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{settings.lm_studio_base_url}/models")
                resp.raise_for_status()
            except Exception:
                print("[SKIP] LM Studio not reachable; start local server to run test.")
                await MongoDBConnection.close()
                return 0

    agent = Agent(
        config=config,
        llm_service=llm_service,
        conversation_repo=conversation_repo,
        llm_call_repo=llm_call_repo,
    )

    # Non-streaming
    response = await agent.process_query(
        "Say hello in one sentence.",
        message_window=MessageWindow.NONE,
    )
    print(f"[OK] Non-stream response: {response.content[:60]}")

    # Message window checks (history included)
    conv = Conversation()
    await conversation_repo.create(conv)
    await conversation_repo.add_message(conv.uuid, "user", "First question?")
    await conversation_repo.add_message(conv.uuid, "assistant", "First answer.")
    await conversation_repo.add_message(conv.uuid, "user", "Second question?")
    await conversation_repo.add_message(conv.uuid, "assistant", "Second answer.")
    conv = await conversation_repo.get_by_id(conv.uuid) or conv

    response_last_user = await agent.process_query(
        "Window test",
        conversation=conv,
        message_window=MessageWindow.LAST_USER,
    )
    print(f"[OK] LAST_USER window response: {response_last_user.content[:60]}")

    response_last_both = await agent.process_query(
        "Window test",
        conversation=conv,
        message_window=MessageWindow.LAST_BOTH,
    )
    print(f"[OK] LAST_BOTH window response: {response_last_both.content[:60]}")

    # Streaming
    chunks = []
    stream_iter = await agent.process_query(
        "Say goodbye in one sentence.",
        stream=True,
        message_window=MessageWindow.NONE,
    )
    async for chunk in stream_iter:
        chunks.append(chunk.content)
    print(f"[OK] Streamed {len(chunks)} chunks")

    # Verify persistence
    conversations = await conversation_repo.get_by_filter(
        filter_dict={},
        limit=10,
        skip=0,
    )
    print(f"[OK] Conversations stored: {len(conversations)}")

    # Close DB
    await MongoDBConnection.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
