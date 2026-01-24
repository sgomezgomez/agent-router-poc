"""Manual test script for Phase 1: MongoDB Storage Layer.

Run this script to verify that the storage layer works correctly.

Requirements:
- MongoDB running locally (see MONGODB-SETUP.md)
- .env file configured with MONGODB__URI

Usage:
    python tests/manual/test_storage.py
"""

import asyncio
from uuid import uuid4
from datetime import datetime, timedelta, timezone

from agent_router.core.config import Settings
from agent_router.llm.models import LLMRequest, LLMResponse, Message as LLMMessage
from agent_router.storage import (
    MongoDBConnection,
    Conversation,
    LLMCallRepository,
    ToolExecutionRepository,
    ConversationRepository,
    build_llm_call,
    build_tool_execution,
    build_conversation_message,
)


async def test_mongodb_connection():
    """Test 1: MongoDB connection and initialization."""
    print("\n" + "=" * 60)
    print("Test 1: MongoDB Connection")
    print("=" * 60)

    settings = Settings()
    print(f"MongoDB URI: {settings.mongodb.uri}")
    print(f"Database: {settings.mongodb.database}")

    try:
        await MongoDBConnection.initialize(settings.mongodb)
        print("[OK] MongoDB connection successful")

        try:
            await MongoDBConnection.create_indexes()
            print("[OK] Indexes created successfully")
        except Exception as e:
            print(f"[WARN] Could not create indexes (auth issue): {e}")
            print("[WARN] Continuing without indexes (slower queries)")

        return True
    except Exception as e:
        print(f"[FAIL] MongoDB connection failed: {e}")
        return False


async def test_llm_call_repository():
    """Test 2: LLM call storage and retrieval."""
    print("\n" + "=" * 60)
    print("Test 2: LLM Call Repository")
    print("=" * 60)

    repo = LLMCallRepository()
    conversation_id = uuid4()

    # Create test LLM calls
    calls = []
    for i in range(3):
        provider = "openai" if i % 2 == 0 else "gemini"
        model = "gpt-4o-mini" if i % 2 == 0 else "gemini-2.0-flash-exp"
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content="You are a helpful assistant"),
                LLMMessage(role="user", content=f"Test prompt {i + 1}"),
            ],
            provider=provider,
            model=model,
            temperature=0.7,
            stream=False,
        )
        response = LLMResponse(
            content=f"Test response {i + 1}",
            provider=provider,
            model=model,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            latency_ms=100 + i * 50,
            used_fallback=i == 2,
        )
        call = build_llm_call(
            conversation_id=conversation_id,
            request=request,
            response=response,
            retry_count=0,
            streaming=False,
        )
        await repo.create(call)
        calls.append(call)
        print(f"[OK] Created LLM call {i + 1}: {call.uuid}")

    # Test retrieval by session
    session_calls = await repo.get_by_conversation(conversation_id)
    assert len(session_calls) == 3, f"Expected 3 calls, got {len(session_calls)}"
    print(f"[OK] Retrieved {len(session_calls)} calls by session")

    # Test retrieval by provider
    openai_calls = await repo.get_by_provider("openai")
    print(f"[OK] Retrieved {len(openai_calls)} OpenAI calls")

    # Test fallback calls
    fallback_calls = await repo.get_fallback_calls()
    assert len(fallback_calls) >= 1, "Expected at least 1 fallback call"
    print(f"[OK] Retrieved {len(fallback_calls)} fallback calls")

    # Test usage stats
    stats = await repo.get_usage_stats(conversation_id=conversation_id)
    print(f"[OK] Usage stats: {stats}")
    assert stats["total_calls"] == 3, f"Expected 3 total calls, got {stats['total_calls']}"
    assert stats["total_tokens"] == 90, f"Expected 90 tokens, got {stats['total_tokens']}"

    return True


async def test_tool_execution_repository():
    """Test 3: Tool execution storage and retrieval."""
    print("\n" + "=" * 60)
    print("Test 3: Tool Execution Repository")
    print("=" * 60)

    repo = ToolExecutionRepository()
    conversation_id = uuid4()

    # Create test tool executions
    tools = ["read_file", "write_file", "search_web", "read_file"]
    executions = []
    for i, tool_name in enumerate(tools):
        execution = build_tool_execution(
            conversation_id=conversation_id,
            tool_name=tool_name,
            tool_type="mcp_tool" if i < 2 else "mcp_agent",
            parameters={"param1": f"value{i}"},
            result={"success": True} if i != 1 else None,
            error="File not found" if i == 1 else None,
            latency_ms=50 + i * 25,
        )
        await repo.create(execution)
        executions.append(execution)
        print(f"[OK] Created tool execution {i + 1}: {tool_name}")

    # Test retrieval by session
    session_executions = await repo.get_by_conversation(conversation_id)
    assert len(session_executions) == 4, f"Expected 4 executions, got {len(session_executions)}"
    print(f"[OK] Retrieved {len(session_executions)} executions by session")

    # Test retrieval by tool name
    read_file_executions = await repo.get_by_tool_name("read_file")
    assert len(read_file_executions) >= 2, f"Expected at least 2 read_file executions"
    print(f"[OK] Retrieved {len(read_file_executions)} read_file executions")

    # Test failed executions
    failed = await repo.get_failed_executions()
    assert len(failed) >= 1, "Expected at least 1 failed execution"
    print(f"[OK] Retrieved {len(failed)} failed executions")

    # Test execution stats
    stats = await repo.get_execution_stats(conversation_id=conversation_id)
    print(f"[OK] Execution stats: {stats}")
    assert stats["total_executions"] == 4, f"Expected 4 executions, got {stats['total_executions']}"
    assert stats["failed_executions"] == 1, f"Expected 1 failure, got {stats['failed_executions']}"

    # Test tool usage breakdown
    breakdown = await repo.get_tool_usage_breakdown(conversation_id=conversation_id)
    print(f"[OK] Tool usage breakdown: {breakdown}")

    return True


async def test_conversation_repository():
    """Test 4: Conversation and message history management."""
    print("\n" + "=" * 60)
    print("Test 4: Conversation Repository")
    print("=" * 60)

    repo = ConversationRepository()

    # Create test conversation
    conversation = Conversation(
        user_id="test_user_123",
        metadata={"source": "test_script"},
    )
    await repo.create(conversation)
    print(f"[OK] Created conversation: {conversation.uuid}")

    # Add messages (via builder)
    user_msg = build_conversation_message(
        role="user",
        content="Hello, how are you?"
    )
    await repo.add_message(
        conversation.uuid,
        user_msg.role,
        user_msg.content,
        tool_calls=user_msg.tool_calls,
        reasoning=user_msg.reasoning,
        llm_call_id=user_msg.llm_call_id,
        tool_call_id=user_msg.tool_call_id,
    )
    print("[OK] Added user message")

    assistant_msg = build_conversation_message(
        role="assistant",
        content="I'm doing well! How can I help you today?",
        reasoning="Respond politely and ask how I can help.",
        llm_call_id=uuid4(),
    )
    await repo.add_message(
        conversation.uuid,
        assistant_msg.role,
        assistant_msg.content,
        tool_calls=assistant_msg.tool_calls,
        reasoning=assistant_msg.reasoning,
        llm_call_id=assistant_msg.llm_call_id,
        tool_call_id=assistant_msg.tool_call_id,
    )
    print("[OK] Added assistant message")

    tool_call_msg = build_conversation_message(
        role="assistant",
        content="",
        tool_calls=[{
            "id": "toolcall-1",
            "type": "function",
            "function": {"name": "search_web", "arguments": "{\"q\": \"python help\"}"},
        }],
    )
    await repo.add_message(
        conversation.uuid,
        tool_call_msg.role,
        tool_call_msg.content,
        tool_calls=tool_call_msg.tool_calls,
        reasoning=tool_call_msg.reasoning,
        llm_call_id=tool_call_msg.llm_call_id,
        tool_call_id=tool_call_msg.tool_call_id,
    )
    print("[OK] Added tool call message")

    tool_result_msg = build_conversation_message(
        role="tool",
        content="{\"results\": [\"python.org\"]}",
        tool_call_id="toolcall-1",
    )
    await repo.add_message(
        conversation.uuid,
        tool_result_msg.role,
        tool_result_msg.content,
        tool_calls=tool_result_msg.tool_calls,
        reasoning=tool_result_msg.reasoning,
        llm_call_id=tool_result_msg.llm_call_id,
        tool_call_id=tool_result_msg.tool_call_id,
    )
    print("[OK] Added tool response message")

    await repo.add_message(conversation.uuid, "user", "Can you help me with Python?")
    print("[OK] Added another user message")

    # Test message retrieval
    recent_messages = await repo.get_recent_messages(conversation.uuid, count=2)
    assert len(recent_messages) == 2, f"Expected 2 messages, got {len(recent_messages)}"
    print(f"[OK] Retrieved {len(recent_messages)} recent messages")

    context_window = await repo.get_context_window(conversation.uuid, max_tokens=1000)
    print(f"[OK] Retrieved context window: {len(context_window)} messages")

    # Test session retrieval by user
    user_conversations = await repo.get_by_user("test_user_123")
    assert len(user_conversations) >= 1, "Expected at least 1 conversation for user"
    print(f"[OK] Retrieved {len(user_conversations)} conversations for user")

    # Test active conversations
    active_conversations = await repo.get_active_conversations(
        since=datetime.now(timezone.utc) - timedelta(minutes=5)
    )
    print(f"[OK] Retrieved {len(active_conversations)} active conversations")

    # Test conversation stats
    stats = await repo.get_conversation_stats(user_id="test_user_123")
    print(f"[OK] Conversation stats: {stats}")

    return True


async def cleanup_test_data():
    """Clean up test data after tests."""
    print("\n" + "=" * 60)
    print("Cleanup: Removing test data")
    print("=" * 60)

    # Note: In a real test suite, you'd want to use a separate test database
    # For manual testing, we'll just note that data was created
    print("Note: Test data remains in database for inspection")
    print("To clean up manually, use MongoDB Compass or:")
    print("  db.llm_calls.deleteMany({})")
    print("  db.tool_executions.deleteMany({})")
    print("  db.conversations.deleteMany({})")


async def main():
    """Run all storage layer tests."""
    print("\n" + "=" * 80)
    print("Agent Router POC - Phase 1: MongoDB Storage Layer Tests")
    print("=" * 80)

    try:
        # Test 1: Connection
        if not await test_mongodb_connection():
            print("\n[FAIL] MongoDB connection failed. Make sure:")
            print("  1. MongoDB service is running (see MONGODB-SETUP.md)")
            print("  2. .env file is configured with MONGODB__URI")
            return

        # Test 2: LLM Calls
        await test_llm_call_repository()

        # Test 3: Tool Executions
        await test_tool_execution_repository()

        # Test 4: Conversations
        await test_conversation_repository()

        # Summary
        print("\n" + "=" * 80)
        print("[OK] All Phase 1 tests passed!")
        print("=" * 80)
        print("\nPhase 1 Complete: MongoDB Storage Layer")
        print("- Models: LLMCall, ToolExecution, Conversation")
        print("- Connection: MongoDBConnection with async Motor")
        print("- Repositories: CRUD operations with specialized queries")
        print("\nNext: Phase 2 - LLM Service Implementation")

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Close MongoDB connection
        await MongoDBConnection.close()
        print("\n[OK] MongoDB connection closed")


if __name__ == "__main__":
    asyncio.run(main())
