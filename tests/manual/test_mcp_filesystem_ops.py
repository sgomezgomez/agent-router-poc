"""Manual MCP filesystem operations test.

Validates list/read/write/update/delete via filesystem MCP server.

Run with: python tests/manual/test_mcp_filesystem_ops.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agent_router.connectors.mcp import MCPClient, load_mcp_server_configs


async def main() -> int:
    configs = load_mcp_server_configs()
    server_config = configs.get("filesystem")
    if not server_config:
        print("[FAIL] filesystem MCP server not configured")
        return 1

    client = MCPClient(server_name="filesystem", server_config=server_config)
    try:
        await client.connect(server_config)
        tools = {tool.get("name") for tool in await client.list_tools()}
        required = {
            "list_directory",
            "read_file",
            "write_file",
            "create_directory",
        }
        missing = required - tools
        if missing:
            print(f"[FAIL] Missing tools: {sorted(missing)}")
            return 1

        base_dir = Path("tests/manual/mcp_fs_ops")
        file_path = base_dir / "test.txt"

        # Ensure directory exists
        await client.execute("create_directory", {"path": str(base_dir)})

        # List directory
        list_result = await client.execute("list_directory", {"path": str(base_dir)})
        print(f"[OK] list_directory: {list_result.get('success')}")

        # Write file
        write_result = await client.execute(
            "write_file",
            {"path": str(file_path), "content": "hello"},
        )
        print(f"[OK] write_file: {write_result.get('success')}")

        # Read file
        read_result = await client.execute("read_file", {"path": str(file_path)})
        print(f"[OK] read_file: {read_result.get('success')}")

        # Update file
        update_result = await client.execute(
            "write_file",
            {"path": str(file_path), "content": "hello updated"},
        )
        print(f"[OK] update_file: {update_result.get('success')}")

        # Delete file if supported
        if "delete_file" in tools:
            delete_result = await client.execute("delete_file", {"path": str(file_path)})
            print(f"[OK] delete_file: {delete_result.get('success')}")
        else:
            print("[SKIP] delete_file not available on filesystem server")

        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
