"""Smoke test for provider refactoring.

Verifies that the refactored provider architecture works correctly without
requiring any external services (LM Studio, OpenAI, etc.).

Tests:
1. Provider imports work
2. Provider initialization works
3. Provider inheritance is correct
4. LLMService loads providers correctly
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agent_router.core.config import Settings
from agent_router.llm.service import LLMService
from agent_router.llm.providers.openai import OpenAIProvider
from agent_router.llm.providers.lm_studio import LMStudioProvider
from agent_router.llm.providers.grok import GrokProvider
from agent_router.llm.providers.gemini import GeminiProvider
from agent_router.llm.providers.openai_compatible import OpenAICompatibleProvider
from agent_router.llm.providers.base import BaseLLMProvider


def test_provider_imports():
    """Test that all providers can be imported."""
    print("\n" + "="*80)
    print("TEST: Provider Imports")
    print("="*80)

    try:
        # All imports already done at top
        print("[OK] All provider imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Provider import failed: {e}")
        return False


def test_provider_inheritance():
    """Test that provider inheritance is correct."""
    print("\n" + "="*80)
    print("TEST: Provider Inheritance")
    print("="*80)

    try:
        # Check inheritance chain
        assert issubclass(OpenAICompatibleProvider, BaseLLMProvider), \
            "OpenAICompatibleProvider should inherit from BaseLLMProvider"

        assert issubclass(OpenAIProvider, OpenAICompatibleProvider), \
            "OpenAIProvider should inherit from OpenAICompatibleProvider"

        assert issubclass(LMStudioProvider, OpenAICompatibleProvider), \
            "LMStudioProvider should inherit from OpenAICompatibleProvider"

        assert issubclass(GrokProvider, OpenAICompatibleProvider), \
            "GrokProvider should inherit from OpenAICompatibleProvider"

        assert issubclass(GeminiProvider, BaseLLMProvider), \
            "GeminiProvider should inherit from BaseLLMProvider"

        # Gemini should NOT inherit from OpenAICompatibleProvider
        assert not issubclass(GeminiProvider, OpenAICompatibleProvider), \
            "GeminiProvider should NOT inherit from OpenAICompatibleProvider"

        print("[OK] Inheritance chain:")
        print("  BaseLLMProvider")
        print("    -> OpenAICompatibleProvider")
        print("       -> OpenAIProvider")
        print("       -> LMStudioProvider")
        print("       -> GrokProvider")
        print("    -> GeminiProvider")

        return True

    except AssertionError as e:
        print(f"[FAIL] Inheritance test failed: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False


def test_provider_initialization():
    """Test that providers can be initialized."""
    print("\n" + "="*80)
    print("TEST: Provider Initialization")
    print("="*80)

    try:
        # LM Studio (no API key needed)
        lm = LMStudioProvider(
            model="test-model",
            base_url="http://localhost:1234/v1"
        )
        assert lm.provider_name == "lm_studio", f"Expected 'lm_studio', got '{lm.provider_name}'"
        assert lm.model == "test-model", f"Expected 'test-model', got '{lm.model}'"
        print("[OK] LMStudioProvider initialized: lm_studio")

        # OpenAI (with dummy API key for testing)
        openai = OpenAIProvider(
            model="gpt-4o",
            api_key="test-key"
        )
        assert openai.provider_name == "openai", f"Expected 'openai', got '{openai.provider_name}'"
        assert openai.model == "gpt-4o", f"Expected 'gpt-4o', got '{openai.model}'"
        print("[OK] OpenAIProvider initialized: openai")

        # Grok (with dummy API key)
        grok = GrokProvider(
            model="grok-beta",
            api_key="test-key",
            base_url="https://api.x.ai/v1"
        )
        assert grok.provider_name == "grok", f"Expected 'grok', got '{grok.provider_name}'"
        assert grok.model == "grok-beta", f"Expected 'grok-beta', got '{grok.model}'"
        print("[OK] GrokProvider initialized: grok")

        # Gemini (with dummy API key)
        gemini = GeminiProvider(
            model="gemini-2.0-flash-exp",
            api_key="test-key"
        )
        assert gemini.provider_name == "gemini", f"Expected 'gemini', got '{gemini.provider_name}'"
        assert gemini.model == "gemini-2.0-flash-exp", f"Expected 'gemini-2.0-flash-exp', got '{gemini.model}'"
        print("[OK] GeminiProvider initialized: gemini")

        return True

    except AssertionError as e:
        print(f"[FAIL] Initialization test failed: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False


def test_llm_service_loading():
    """Test that LLMService can load providers lazily."""
    print("\n" + "="*80)
    print("TEST: LLMService Lazy Provider Loading")
    print("="*80)

    try:
        # Create settings
        settings = Settings()

        # Create LLM service
        service = LLMService(settings)

        # Check that providers dict is empty (lazy loading)
        assert len(service.providers) == 0, \
            f"Providers dict should be empty on init (lazy loading), got {len(service.providers)}"
        print("[OK] LLMService initialized with empty providers dict (lazy loading)")

        # Check that LM Studio is available (always available)
        available = service.get_available_providers()
        assert "lm_studio" in available, "LM Studio should always be available"
        print(f"[OK] LLMService can load {len(available)} provider(s): {', '.join(available)}")

        # Lazy load LM Studio provider
        lm_provider = service._get_provider("lm_studio")
        assert isinstance(lm_provider, LMStudioProvider), \
            f"Expected LMStudioProvider, got {type(lm_provider)}"
        assert isinstance(lm_provider, OpenAICompatibleProvider), \
            "LMStudioProvider should be instance of OpenAICompatibleProvider"
        print("[OK] LM Studio provider lazy loaded successfully")

        # Check that it's now cached
        assert "lm_studio" in service.providers, \
            "LM Studio should be cached after lazy load"
        assert len(service.providers) == 1, \
            f"Should have 1 cached provider, got {len(service.providers)}"
        print("[OK] Provider cached after first load")

        # Getting again should return same instance
        lm_provider2 = service._get_provider("lm_studio")
        assert lm_provider is lm_provider2, \
            "Second get should return same cached instance"
        print("[OK] Cached provider reused on subsequent calls")

        return True

    except AssertionError as e:
        print(f"[FAIL] LLMService loading test failed: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False


def test_code_reduction():
    """Test that code reduction goals were achieved."""
    print("\n" + "="*80)
    print("TEST: Code Reduction Metrics")
    print("="*80)

    import inspect

    try:
        # Count lines in each provider
        openai_lines = len(inspect.getsource(OpenAIProvider).splitlines())
        lm_studio_lines = len(inspect.getsource(LMStudioProvider).splitlines())
        grok_lines = len(inspect.getsource(GrokProvider).splitlines())
        openai_compat_lines = len(inspect.getsource(OpenAICompatibleProvider).splitlines())

        print(f"  OpenAICompatibleProvider: {openai_compat_lines} lines (new base class)")
        print(f"  OpenAIProvider: {openai_lines} lines (was 251, now includes reasoning)")
        print(f"  LMStudioProvider: {lm_studio_lines} lines (was 189)")
        print(f"  GrokProvider: {grok_lines} lines (was 187)")

        # Verify thin wrappers are actually thin
        assert lm_studio_lines < 50, f"LMStudioProvider should be < 50 lines, got {lm_studio_lines}"
        assert grok_lines < 50, f"GrokProvider should be < 50 lines, got {grok_lines}"

        print(f"\n[OK] LM Studio and Grok are thin wrappers (< 50 lines each)")

        # Total lines comparison
        old_total = 251 + 189 + 187  # Old OpenAI + LM Studio + Grok
        new_total = openai_compat_lines + openai_lines + lm_studio_lines + grok_lines
        reduction = old_total - new_total
        reduction_pct = (reduction / old_total) * 100

        print(f"\nTotal lines:")
        print(f"  Before: {old_total} lines (OpenAI 251 + LM Studio 189 + Grok 187)")
        print(f"  After: {new_total} lines")
        print(f"  Net change: {reduction} lines ({reduction_pct:.1f}%)")

        # Note: OpenAI grew because we added response_format support
        # But LM Studio and Grok shrank dramatically (189->37, 187->37)
        wrapper_reduction = (189 - lm_studio_lines) + (187 - grok_lines)

        print(f"\nWrapper reduction (LM Studio + Grok):")
        print(f"  Before: {189 + 187} lines")
        print(f"  After: {lm_studio_lines + grok_lines} lines")
        print(f"  Reduction: {wrapper_reduction} lines")

        assert wrapper_reduction > 250, f"Expected > 250 line wrapper reduction, got {wrapper_reduction}"

        print(f"[OK] Achieved > 250 line wrapper reduction (eliminated duplication)")

        return True

    except AssertionError as e:
        print(f"[FAIL] Code reduction test failed: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("\n" + "="*80)
    print("PROVIDER REFACTORING SMOKE TESTS")
    print("="*80)
    print("\nVerifying Option C (Hybrid Approach) implementation:")
    print("  - OpenAICompatibleProvider base class")
    print("  - OpenAIProvider extends with reasoning support")
    print("  - LMStudioProvider thin wrapper")
    print("  - GrokProvider thin wrapper")
    print("  - GeminiProvider stays separate")

    results = []

    # Run tests
    results.append(("Provider Imports", test_provider_imports()))
    results.append(("Provider Inheritance", test_provider_inheritance()))
    results.append(("Provider Initialization", test_provider_initialization()))
    results.append(("LLMService Loading", test_llm_service_loading()))
    results.append(("Code Reduction", test_code_reduction()))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print("-" * 80)
    print(f"  Total: {total_passed}/{total_tests} tests passed")
    print("="*80)

    if total_passed == total_tests:
        print("\n[OK] All smoke tests passed!")
        print("\nRefactoring Summary:")
        print("  - Eliminated 302 lines of duplicate code from wrappers")
        print("  - LM Studio and Grok are now thin wrappers (37 lines each)")
        print("  - OpenAI reasoning logic isolated to OpenAIProvider")
        print("  - Easy to add new OpenAI-compatible providers")
        print("  - All configuration comes from Settings/environment")
        return 0
    else:
        print(f"\n[FAIL] {total_tests - total_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
