"""Smoke test for one-call per provider/model.

Run with: python tests/manual/test_provider_smoke.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_router.core.config import Settings
from agent_router.llm import LLMService
from agent_router.core.errors import LLMProviderError


TEST_CASES = [
    ("openai", "gpt-5-nano"),
    ("gemini", "gemini-2.5-flash-lite"),
    ("grok", "grok-4-1-fast-reasoning"),
    ("lm_studio", "openai/gpt-oss-20b"),
    ("lm_studio", "zai-org/glm-4.7-flash")
]


async def run_test(provider: str, model: str) -> bool:
    """Run a single call for a provider/model."""
    settings = Settings()
    llm_service = LLMService(settings)

    try:
        response = await llm_service.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Reply with 'OK'.",
            provider=provider,
            model=model,
            max_tokens=50
        )
        print(f"[OK] {provider}::{model} -> {response.content[:50]}...")
        return True
    except LLMProviderError as e:
        print(f"[SKIP] {provider}::{model} -> {e}")
        return False
    except Exception as e:
        print(f"[FAIL] {provider}::{model} -> {e}")
        return False


async def main() -> int:
    print("\n" + "=" * 80)
    print("PROVIDER SMOKE TESTS")
    print("=" * 80)

    results = []
    for provider, model in TEST_CASES:
        results.append(await run_test(provider, model))

    passed = sum(1 for v in results if v)
    total = len(results)

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"  Passed: {passed}/{total}")
    print("=" * 80)

    return 0 if passed > 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
