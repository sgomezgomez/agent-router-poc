"""Manual tests for Gemini provider features (streaming + tools).

Run with: python tests/manual/test_gemini_features.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_router.core.config import Settings
from agent_router.llm.models import LLMRequest, Message
from agent_router.llm.providers.gemini import GeminiProvider


def _build_tools() -> list[dict]:
    """Build an OpenAI-style tool definition (Gemini normalizes it)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo back a message",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            },
        }
    ]


def _build_request(
    model: str,
    max_tokens: int = 64,
    tools: bool = False
) -> LLMRequest:
    user_content = "Reply with 'OK'."
    if tools:
        # Encourage a valid tool call with enough tokens to serialize JSON arguments.
        user_content = "Call the echo tool with message 'OK'."
        max_tokens = max(max_tokens, 128)

    return LLMRequest(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content=user_content),
        ],
        provider="gemini",
        model=model,
        max_tokens=max_tokens,
        tools=_build_tools() if tools else None,
    )


async def main() -> int:
    settings = Settings()
    if not settings.gemini_api_key:
        print("[SKIP] Gemini not configured (GEMINI_API_KEY missing)")
        return 0

    provider = GeminiProvider(
        model="gemini-2.5-flash-lite",
        api_key=settings.gemini_api_key,
    )

    # Non-streaming
    try:
        response = await provider.generate(_build_request("gemini-2.5-flash-lite"))
        print(f"[OK] gemini:generate -> {response.content[:50]}...")
    except Exception as e:
        print(f"[FAIL] gemini:generate -> {e}")
        return 1

    # Streaming
    try:
        chunks = []
        async for chunk in provider.generate_stream(_build_request("gemini-2.5-flash-lite")):
            chunks.append(chunk.content)
        print(f"[OK] gemini:stream -> {len(chunks)} chunks")
    except Exception as e:
        print(f"[FAIL] gemini:stream -> {e}")
        return 1

    # Tool calling
    try:
        response = await provider.generate(
            _build_request("gemini-2.5-flash-lite", tools=True)
        )
        if response.tool_calls:
            print(f"[OK] gemini:tools -> {response.tool_calls}")
        else:
            print("[FAIL] gemini:tools -> no tool_calls returned")
            return 1
    except Exception as e:
        print(f"[FAIL] gemini:tools -> {e}")
        return 1

    print("\n" + "=" * 80)
    print("GEMINI FEATURE TEST SUMMARY")
    print("=" * 80)
    print("  Passed: 3/3")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
