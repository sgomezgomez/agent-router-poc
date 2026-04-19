# agent-router-poc

Multi-provider LLM routing PoC with MCP/A2A support.

**Stack:** Python / FastAPI / MongoDB (Motor) / Poetry

**Entry points:** `src/agent_router/main.py`, `src/agent_router/runtime.py`
**Config:** `config/agents/router.yaml`, `config/mcp_servers.yaml`, `config/llm_models.yaml`
**A2A (Agent-to-Agent Protocol):** `src/agent_router/connectors/a2a/`

**Build/test:**
```bash
poetry install
poetry run pytest
poetry run ruff check .
poetry run mypy src/
```

See [README.md](README.md) for more.
