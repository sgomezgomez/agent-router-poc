"""CLI entrypoint for the agent router."""

from __future__ import annotations

import argparse
import asyncio

from agent_router.runtime import RouterRuntime

async def run(query: str, stream: bool) -> None:
    runtime = await RouterRuntime.create()
    try:
        result = await runtime.process_query(query, stream=stream)
        if stream:
            async for chunk in result:
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            print()
        else:
            print(result.content or "")
    finally:
        await runtime.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Router POC")
    parser.add_argument("--query", required=True, help="User query to process")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response as tokens arrive",
    )
    args = parser.parse_args()

    asyncio.run(run(args.query, args.stream))

if __name__ == "__main__":
    main()
