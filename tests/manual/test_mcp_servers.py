"""Manual test for MCP servers (Phase 4).

Run with: python tests/manual/test_mcp_servers.py
"""

import asyncio
from pathlib import Path

import yaml

from agent_router.connectors.mcp import MCPClient


async def main() -> int:
    config_path = Path(__file__).resolve().parents[2] / "config" / "mcp_servers.yaml"
    if not config_path.exists():
        print("[FAIL] config/mcp_servers.yaml not found")
        return 1

    config = yaml.safe_load(config_path.read_text()) or {}
    servers = config.get("servers", {})
    if not servers:
        print("[FAIL] No MCP servers configured")
        return 1

    failures = 0
    for name, server_config in servers.items():
        client = MCPClient(server_name=name, server_config=server_config)
        try:
            await client.connect(server_config)
            tools = await client.list_tools()
            print(f"[OK] {name} tools: {len(tools)}")
        except Exception as e:
            failures += 1
            print(f"[FAIL] {name} MCP server error: {e}")
        finally:
            await client.close()

    if failures:
        print(f"[FAIL] MCP servers failed: {failures}/{len(servers)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
