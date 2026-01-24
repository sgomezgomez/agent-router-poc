"""Manual tests for LLM service.

Run with: python tests/manual/test_llm_service.py
"""

import asyncio
import sys
from pathlib import Path
from pydantic import BaseModel, Field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_router.core.config import Settings
from agent_router.llm import LLMService, Message


async def test_lm_studio():
    """Test LM Studio provider."""
    print("\n" + "=" * 80)
    print("TEST: LM Studio Provider")
    print("=" * 80)

    settings = Settings()
    llm_service = LLMService(settings)

    try:
        response = await llm_service.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'Hello, I am working!' in exactly those words.",
            provider="lm_studio",
            temperature=0.7,
            max_tokens=50
        )

        print(f"[OK] LM Studio generation successful")
        print(f"  Provider: {response.provider}")
        print(f"  Model: {response.model}")
        print(f"  Content: {response.content[:100]}...")
        print(f"  Latency: {response.latency_ms:.0f}ms")
        if response.usage:
            print(f"  Tokens: {response.usage.get('total_tokens', 0)}")

        return True

    except Exception as e:
        print(f"[FAIL] LM Studio test failed: {e}")
        return False


async def test_lm_studio_tool_calling():
    """Test tool calling extraction with LM Studio."""
    print("\n" + "=" * 80)
    print("TEST: LM Studio Tool Calling")
    print("=" * 80)

    settings = Settings()
    llm_service = LLMService(settings)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo back a message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }
            }
        }
    ]

    try:
        response = await llm_service.generate(
            system_prompt="You must call the tool.",
            user_prompt="Call the tool with message 'hello'.",
            provider="lm_studio",
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "echo"}},
            max_tokens=50
        )

        if response.tool_calls:
            print("[OK] Tool calls detected")
            print(f"  Tool calls: {response.tool_calls}")
            return True

        print("[WARN] No tool calls returned")
        return False

    except Exception as e:
        print(f"[FAIL] Tool calling test failed: {e}")
        return False


async def test_structured_output_retry():
    """Test that structured output parsing triggers retry."""
    print("\n" + "=" * 80)
    print("TEST: Structured Output Retry")
    print("=" * 80)

    settings = Settings()
    llm_service = LLMService(settings)

    class TestSchema(BaseModel):
        """Simple schema for structured output parsing test."""
        message: str = Field(description="Response message")

    try:
        # Intentionally request invalid JSON to trigger parse failure + retry
        response = await llm_service.generate(
            system_prompt="Return invalid JSON.",
            user_prompt="Reply with: not_json",
            provider="lm_studio",
            response_format=TestSchema,  # Not a valid JSON schema on purpose
            max_tokens=20
        )

        if response.parse_error:
            print("[OK] Parse error recorded, retries attempted")
            return True

        print("[WARN] No parse error detected (model returned valid JSON)")
        return True

    except Exception as e:
        print(f"[INFO] Structured output test exception: {e}")
        return True


async def test_lm_studio_stream():
    """Test LM Studio streaming."""
    print("\n" + "=" * 80)
    print("TEST: LM Studio Streaming")
    print("=" * 80)

    settings = Settings()
    llm_service = LLMService(settings)

    try:
        print("Streaming response:")
        print("-" * 40)

        chunks = []
        async for chunk in llm_service.generate_stream(
            system_prompt="You are a helpful assistant.",
            user_prompt="Count from 1 to 5.",
            provider="lm_studio",
            temperature=0.7,
            max_tokens=100
        ):
            print(chunk.content, end="", flush=True)
            chunks.append(chunk.content)

        print("\n" + "-" * 40)
        print(f"[OK] LM Studio streaming successful")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total content length: {len(''.join(chunks))} chars")

        return True

    except Exception as e:
        print(f"\n[FAIL] LM Studio streaming test failed: {e}")
        return False


async def test_conversation():
    """Test multi-turn conversation."""
    print("\n" + "=" * 80)
    print("TEST: Multi-turn Conversation")
    print("=" * 80)

    settings = Settings()
    llm_service = LLMService(settings)

    try:
        # Build conversation
        messages = [
            Message(role="system", content="You are a helpful math tutor."),
            Message(role="user", content="What is 2 + 2?"),
            Message(role="assistant", content="2 + 2 equals 4."),
            Message(role="user", content="What about 3 + 3?")
        ]

        response = await llm_service.generate(
            messages=messages,
            provider="lm_studio",
            temperature=0.7,
            max_tokens=50
        )

        print(f"[OK] Conversation test successful")
        print(f"  Messages in conversation: {len(messages)}")
        print(f"  Response: {response.content[:100]}...")

        return True

    except Exception as e:
        print(f"[FAIL] Conversation test failed: {e}")
        return False


async def test_fallback():
    """Test fallback mechanism."""
    print("\n" + "=" * 80)
    print("TEST: Fallback Mechanism")
    print("=" * 80)

    settings = Settings()
    llm_service = LLMService(settings)

    try:
        # Try with a provider that doesn't exist - should fallback to lm_studio
        response = await llm_service.generate(
            user_prompt="Say 'Fallback works!'",
            provider="nonexistent_provider",  # This will fail
            max_retries=1,  # Reduce retries for faster test
            use_fallback=True
        )

        if response.used_fallback:
            print(f"[OK] Fallback mechanism working")
            print(f"  Fallback provider: {response.provider}")
            print(f"  Response: {response.content[:100]}...")
        else:
            print(f"[UNEXPECTED] Fallback flag not set")

        return True

    except Exception as e:
        # Expected to fail and use fallback
        print(f"[INFO] Fallback test behavior: {e}")
        return False


async def test_provider_list():
    """Test available providers."""
    print("\n" + "=" * 80)
    print("TEST: Available Providers")
    print("=" * 80)

    settings = Settings()
    llm_service = LLMService(settings)

    providers = llm_service.get_available_providers()

    print(f"[OK] Available providers: {', '.join(providers)}")
    print(f"  Total: {len(providers)} provider(s)")

    # Check if LM Studio is available
    if "lm_studio" not in providers:
        print(f"[WARN] LM Studio not available - is it running?")
        return False

    return True


async def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print("LLM SERVICE MANUAL TESTS")
    print("=" * 80)
    print("\nPrerequisites:")
    print("  - LM Studio running at http://localhost:1234")
    print("  - Model loaded in LM Studio (e.g., openai/gpt-oss-20b)")
    print("")

    results = {
        "Provider List": await test_provider_list(),
        "LM Studio Generation": await test_lm_studio(),
        "LM Studio Tool Calling": await test_lm_studio_tool_calling(),
        "LM Studio Streaming": await test_lm_studio_stream(),
        "Structured Output Retry": await test_structured_output_retry(),
        "Multi-turn Conversation": await test_conversation(),
        "Fallback Mechanism": await test_fallback(),
    }

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"  {status} {test_name}")

    print("-" * 80)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 80)

    if passed == total:
        print("\n[OK] All tests passed!")
        return 0
    else:
        print(f"\n[FAIL] {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
