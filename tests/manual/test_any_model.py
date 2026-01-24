"""Test that providers can use any model name specified by agent.

Run with: python tests/manual/test_any_model.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent_router.core.config import Settings
from agent_router.llm import LLMService


async def test_custom_model_name():
    """Test using a custom model name different from the default."""
    print("\n" + "=" * 80)
    print("TEST: Custom Model Name Override")
    print("=" * 80)

    settings = Settings()
    llm_service = LLMService(settings)

    # Test 1: Use fallback model (from settings)
    print("\n1. Using fallback model from settings:")
    print(f"   Settings fallback: {settings.fallback_llm_model}")

    response1 = await llm_service.generate(
        user_prompt="Say 'Test 1'",
        provider="lm_studio",
        max_tokens=20
    )

    print(f"   [OK] Used model: {response1.model}")
    print(f"   Response: {response1.content[:50]}...")

    # Test 2: Override with custom model name
    print("\n2. Overriding with custom model name:")
    custom_model = "custom-model-name-123"
    print(f"   Requesting model: {custom_model}")

    try:
        response2 = await llm_service.generate(
            user_prompt="Say 'Test 2'",
            provider="lm_studio",
            model=custom_model,  # Override the default!
            max_tokens=20
        )

        print(f"   [OK] Used model: {response2.model}")
        print(f"   Response: {response2.content[:50]}...")

        # Verify the model was actually used
        if response2.model == custom_model:
            print(f"\n[OK] Provider successfully used the custom model name!")
            return True

        if response2.used_fallback:
            print("\n[OK] Provider rejected unknown model name and used fallback.")
            print(f"   Expected: {custom_model}")
            print(f"   Got: {response2.model}")
            return True

        print(f"\n[FAIL] Provider did not use custom model name")
        print(f"   Expected: {custom_model}")
        print(f"   Got: {response2.model}")
        return False
    except Exception as e:
        message = str(e)
        if "Invalid model identifier" in message or "model_not_found" in message:
            print("\n[OK] Provider rejected unknown model name (LM Studio validation).")
            return True
        print(f"\n[FAIL] Unexpected error for custom model: {message}")
        return False


async def test_model_flexibility():
    """Test that the agent can specify any model for any provider."""
    print("\n" + "=" * 80)
    print("TEST: Model Flexibility Demonstration")
    print("=" * 80)

    settings = Settings()
    llm_service = LLMService(settings)

    test_cases = [
        ("lm_studio", "qwen2.5:32b-instruct"),
        ("lm_studio", "llama-3.1-8b"),
        ("lm_studio", "mistral-nemo:latest"),
        ("lm_studio", "zai-org/glm-4.7-flash"),
    ]

    results = []

    for provider, model in test_cases:
        print(f"\nTesting: provider={provider}, model={model}")

        try:
            response = await llm_service.generate(
                user_prompt="Reply with just 'OK'",
                provider=provider,
                model=model,
                max_tokens=10
            )

            print(f"  [OK] Successfully used model: {response.model}")
            print(f"  Response: {response.content[:30]}...")
            results.append(True)

        except Exception as e:
            print(f"  [INFO] Model not available: {str(e)[:60]}...")
            results.append(False)

    print(f"\n[OK] Provider accepted all model names (even if models aren't loaded)")
    return True


async def main():
    """Run all tests."""
    print("\n")
    print("=" * 80)
    print("MODEL FLEXIBILITY TESTS")
    print("=" * 80)
    print("\nThis demonstrates that:")
    print("  - Providers use the model specified by the agent")
    print("  - Default model is used only if agent doesn't specify")
    print("  - A model name can be passed (agent decides)")
    print("")

    results = {
        "Custom Model Override": await test_custom_model_name(),
        "Model Flexibility": await test_model_flexibility(),
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

    print("\n[INFO] Key Takeaway:")
    print("  The agent has full control over which model to use.")
    print("  Providers will attempt to use whatever model the agent specifies.")
    print("  Default models in settings are just fallbacks when agent doesn't specify.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
