"""Manual tests for OpenAI-compatible providers (chat vs responses).

Run with: python tests/manual/test_openai_compatible_modes.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import httpx

from agent_router.core.config import Settings
from agent_router.llm.models import LLMRequest, Message
from agent_router.llm.providers.openai_compatible import OpenAIProvider
from agent_router.llm.providers.lm_studio import LMStudioProvider
from agent_router.llm.providers.grok import GrokProvider
from agent_router.llm.model_registry import load_model_registry


def _build_tools(api_mode: str, tool_schema: str | None = None) -> list[dict]:
    """Return tool schema for chat_completions vs responses."""
    if tool_schema == "responses" or (tool_schema is None and api_mode == "responses"):
        return [
            {
                "type": "function",
                "name": "echo",
                "description": "Echo back a message",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            }
        ]

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
    provider: str,
    model: str,
    api_mode: str,
    max_tokens: int = 32,
    tools: bool = False,
    tool_schema: str | None = None,
) -> LLMRequest:
    user_content = "Reply with 'OK'."
    if tools:
        # Encourage a valid tool call with enough tokens to serialize JSON arguments.
        user_content = "Call the echo tool with message 'OK'."
        max_tokens = max(max_tokens, 128)

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content=user_content)
    ]
    return LLMRequest(
        messages=messages,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        tools=_build_tools(api_mode, tool_schema=tool_schema) if tools else None,
        tool_choice="required" if tools else None
    )


async def run_provider_case(
    label: str,
    provider,
    request: LLMRequest,
    stream: bool = False
) -> bool:
    try:
        if stream:
            chunks = []
            async for chunk in provider.generate_stream(request):
                chunks.append(chunk.content)
            print(f"[OK] {label} streaming -> {len(chunks)} chunks")
            return True

        response = await provider.generate(request)
        print(f"[OK] {label} -> {response.content[:50]}...")
        return True
    except Exception as e:
        print(f"[FAIL] {label} -> {e}")
        return False


async def main() -> int:
    settings = Settings()

    tests = []
    registry = load_model_registry()

    # OpenAI
    if settings.openai_api_key:
        for api_mode in ["chat_completions", "responses"]:
            provider = OpenAIProvider(
                model="gpt-5-nano",
                api_key=settings.openai_api_key,
                api_mode=api_mode,
                model_registry=registry,
            )
            req = _build_request("openai", "gpt-5-nano", api_mode)
            tests.append(run_provider_case(f"openai:{api_mode}", provider, req))
            tests.append(run_provider_case(
                f"openai:{api_mode}:tools",
                provider,
                _build_request("openai", "gpt-5-nano", api_mode, tools=True)
            ))
            tests.append(run_provider_case(f"openai:{api_mode}:stream", provider, req, stream=True))
    else:
        print("[SKIP] OpenAI not configured (OPENAI_API_KEY missing)")

    # Grok
    if settings.grok_api_key:
        for api_mode in ["chat_completions", "responses"]:
            provider = GrokProvider(
                model="grok-4-1-fast-reasoning",
                api_key=settings.grok_api_key,
                base_url=settings.grok_base_url or "https://api.x.ai/v1",
                api_mode=api_mode,
                model_registry=registry,
            )
            req = _build_request("grok", "grok-4-1-fast-reasoning", api_mode)
            tests.append(run_provider_case(f"grok:{api_mode}", provider, req))
            tests.append(run_provider_case(
                f"grok:{api_mode}:tools",
                provider,
                _build_request("grok", "grok-4-1-fast-reasoning", api_mode, tools=True)
            ))
            tests.append(run_provider_case(f"grok:{api_mode}:stream", provider, req, stream=True))
    else:
        print("[SKIP] Grok not configured (GROK_API_KEY missing)")

    # LM Studio
    loaded_models = set()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.lm_studio_base_url}/models")
            resp.raise_for_status()
            payload = resp.json() or {}
            for item in payload.get("data", []):
                model_id = item.get("id")
                if model_id:
                    loaded_models.add(model_id)
    except Exception as e:
        print(f"[WARN] Unable to query LM Studio models: {e}")

    lm_models = ["openai/gpt-oss-20b", "zai-org/glm-4.7-flash"]
    for model in lm_models:
        if loaded_models and model not in loaded_models:
            print(f"[WARN] LM Studio model not loaded: {model}")
        caps = registry.get_model_capabilities("lm_studio", model)
        api_modes = ["chat_completions", "responses"]
        for api_mode in api_modes:
            provider = LMStudioProvider(
                model=model,
                base_url=settings.lm_studio_base_url,
                api_mode=api_mode,
                model_registry=registry,
            )
            req = _build_request("lm_studio", model, api_mode)
            tests.append(run_provider_case(f"lm_studio:{model}:{api_mode}", provider, req))
            tool_schema = caps.tool_schema if caps and api_mode == "responses" else None
            if not caps or caps.supports_tools is not False:
                tests.append(run_provider_case(
                    f"lm_studio:{model}:{api_mode}:tools",
                    provider,
                    _build_request(
                        "lm_studio",
                        model,
                        api_mode,
                        tools=True,
                        tool_schema=tool_schema,
                    )
                ))
            tests.append(run_provider_case(
                f"lm_studio:{model}:{api_mode}:stream",
                provider,
                req,
                stream=True
            ))

    results = await asyncio.gather(*tests)
    passed = sum(1 for r in results if r)
    total = len(results)

    print("\n" + "=" * 80)
    print("OPENAI-COMPATIBLE MODE TEST SUMMARY")
    print("=" * 80)
    print(f"  Passed: {passed}/{total}")
    print("=" * 80)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
