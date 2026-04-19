"""Helpers for loading MCP server configs and clients."""

from __future__ import annotations

import logging
from pathlib import Path
import os
import yaml
from dotenv import dotenv_values

from .client import MCPClient

logger = logging.getLogger(__name__)

MCPStatus = dict[str, str]

def load_mcp_server_configs(path: Path | None = None) -> dict[str, dict]:
    """Load MCP server configurations from YAML."""
    config_path = path or Path("config/mcp_servers.yaml")
    if not config_path.exists():
        return {}

    env_values = dotenv_values(Path(".env"))
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    servers = payload.get("servers", {})
    return {
        name: _expand_server_config(cfg, env_values)
        for name, cfg in servers.items()
    }

async def create_mcp_clients_with_status(
    server_configs: dict[str, dict]
) -> tuple[dict[str, MCPClient], list[MCPStatus]]:
    """Create MCP clients and return per-server startup status."""
    clients: dict[str, MCPClient] = {}
    statuses: list[MCPStatus] = []
    for name, config in server_configs.items():
        if config.get("enabled") is False:
            statuses.append(
                {
                    "server": name,
                    "status": "disabled",
                    "detail": "disabled by config",
                }
            )
            continue
        client = MCPClient(server_name=name, server_config=config)
        try:
            await client.connect(config)
            clients[name] = client
            statuses.append(
                {
                    "server": name,
                    "status": "connected",
                    "detail": "",
                }
            )
        except Exception as exc:
            logger.warning("Skipping MCP server '%s': %s", name, exc)
            statuses.append(
                {
                    "server": name,
                    "status": "skipped",
                    "detail": str(exc),
                }
            )
    return clients, statuses

async def create_mcp_clients(server_configs: dict[str, dict]) -> dict[str, MCPClient]:
    """Create and connect MCP clients from server configs."""
    clients, _ = await create_mcp_clients_with_status(server_configs)
    return clients

def _expand_server_config(config: dict, env_values: dict) -> dict:
    expanded = dict(config)
    env = expanded.get("env")
    if isinstance(env, dict):
        expanded["env"] = _expand_env_map(env, env_values)
    return expanded

def _expand_env_map(env: dict, env_values: dict) -> dict:
    expanded: dict[str, str] = {}
    for key, value in env.items():
        expanded[key] = _expand_env_value(value, env_values)
    return expanded

def _expand_env_value(value: object, env_values: dict) -> str | object:
    if not isinstance(value, str):
        return value
    name = None
    if value.startswith("${") and value.endswith("}"):
        name = value[2:-1]
    elif value.startswith("$"):
        name = value[1:]
    if not name:
        return value
    resolved = os.environ.get(name) or env_values.get(name)
    return resolved if resolved is not None else value
