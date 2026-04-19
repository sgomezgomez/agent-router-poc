"""Helpers for building tool contracts from MCP tool metadata."""

from __future__ import annotations

from typing import Iterable

from agent_router.core.types import JsonObject


def build_tool_contracts(
    tools: Iterable[JsonObject],
    *,
    allowed_servers: set[str] | None = None,
    allowed_tools: set[str] | None = None,
) -> list[JsonObject]:
    """Build tool contracts from MCP tool metadata.

    Tools are expected to include "name" and "server" fields (from MCPClient.list_tools()).
    Providers can normalize this schema as needed.
    """
    contracts: list[JsonObject] = []
    for tool in tools:
        name = tool.get("name")
        server = tool.get("server")
        if not name or not server:
            continue
        if allowed_servers and server not in allowed_servers:
            continue

        qualified_name = f"{server}::{name}"
        if allowed_tools and qualified_name not in allowed_tools:
            continue

        parameters = tool.get("parameters") or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        contracts.append(
            {
                "type": "function",
                "function": {
                    "name": qualified_name,
                    "description": tool.get("description") or "",
                    "parameters": parameters,
                },
            }
        )
    return contracts
