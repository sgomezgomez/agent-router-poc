"""Manual test for RouterRuntime wiring.

Run with: python tests/manual/test_router_runtime.py
"""

from __future__ import annotations

import argparse
import asyncio

from agent_router.runtime import RouterRuntime


async def main() -> int:
    parser = argparse.ArgumentParser(description="RouterRuntime manual test")
    parser.add_argument(
        "--query",
        default="read tests/manual/mcp_tmp.txt",
        help="User query to send",
    )
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    args = parser.parse_args()

    runtime = await RouterRuntime.create()
    try:
        result = await runtime.process_query(args.query, stream=args.stream)
        if args.stream:
            async for chunk in result:
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            print()
        else:
            print(result.content or "")
    finally:
        await runtime.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
