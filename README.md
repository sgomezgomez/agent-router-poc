# Agent Router POC

Proof of concept for an autonomous agent that routes user queries to tools/agents
using LLM-based decision-making.

## Features
- Multi-provider LLM support (OpenAI, Gemini, LM Studio, Grok)
- MCP (Model Context Protocol) tool integration
- A2A (Agent-to-Agent Protocol) support
- MongoDB audit logging (LLM calls + tool executions)
- Streaming and non-streaming modes

## Requirements
- Python 3.10+
- Poetry
- MongoDB (local)
- Node.js/npm (for MCP servers)
- LM Studio running (OpenAI-compatible endpoint)

## Setup
```bash
poetry install
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-everything
npm install -g @modelcontextprotocol/server-brave-search
npm install -g @modelcontextprotocol/server-sequential-thinking
npm install -g @playwright/mcp
# Optional: you can skip global installs and rely on npx downloads.
```

Create a `.env` in the repo root:
```bash
# MongoDB (local)
MONGODB__URI=mongodb://localhost:27017/?retryWrites=false
MONGODB__DATABASE=agent_router_db

# LM Studio
LM_STUDIO_BASE_URL=http://localhost:1234/v1

# Optional paid providers
OPENAI_API_KEY=
GEMINI_API_KEY=
GROK_API_KEY=
GROK_BASE_URL=

# Brave Search (MCP)
BRAVE_API_KEY=

# Fallback config (used when agent omits params or retries exhaust)
FALLBACK_LLM_PROVIDER=lm_studio
FALLBACK_LLM_MODEL=zai-org/glm-4.7-flash
FALLBACK_TEMPERATURE=0.5
FALLBACK_MAX_TOKENS=2048
FALLBACK_TOP_P=
FALLBACK_TOP_K=
FALLBACK_THINKING_BUDGET=
FALLBACK_THINKING_EFFORT=
ADAPTER_DEBUG_TRACE_DEFAULT=false

# LLM retry configuration
LLM_RETRY_MAX_RETRIES=3
LLM_RETRY_BASE_DELAY=1.0
LLM_RETRY_MAX_DELAY=10.0
LLM_RETRY_EXPONENTIAL_BASE=2.0

# Logging
LOG_LEVEL=INFO
```

## Manual Tests
```bash
python tests/manual/test_mcp_servers.py
python tests/manual/test_mcp_tool_calls.py
python tests/manual/test_agent_tool_calls_stream.py
```

## OpenAI-Compatible Adapter (for OSS Chat UIs)
Run a local OpenAI-compatible API that proxies into this router runtime:
```bash
python -m agent_router.api.openai_adapter
```

Available endpoints:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions` (streaming + non-streaming)
- `POST /v1/responses` (streaming + non-streaming)

Use this base URL in LibreChat/Open WebUI/LobeChat:
- `http://127.0.0.1:8000/v1`

Notes:
- Conversation continuity is keyed by `conversation_id`, `metadata.conversation_id`, or `user` in the request body.
- If none are sent, each request is treated as a new conversation.
- The adapter keeps tool orchestration in the backend (your existing agent + MCP flow).

## MCP Retry Settings
Per-server retry parameters live in `config/mcp_servers.yaml`:
`retry_max_retries`, `retry_base_delay`, `retry_max_delay`, `retry_exponential_base`.

## Tool Contracts
Tools are loaded from MCP and filtered by allowlists in `config/agents/router.yaml`.
Call `Agent.load_mcp_tools()` before invoking `process_query()` **only if**
the agent has MCP tool access. Agents without tool access (or ones that decide
they don't need tools) can run without loading any tool contracts.

## Project Status
- Phase 0 (Setup): ✅ Complete
- Phase 1 (Storage): ✅ Complete
- Phase 2 (LLM Service): ✅ Complete
- Phase 3 (Agent Base): ✅ Complete
- Phase 4 (MCP Integration): ✅ Complete
- Phase 5A (Orchestrator): ✅ Complete
- Phase 5B (Routing Agent): ✅ Complete
- Phase 6 (MCP Tools Integration): ⚠️ In Progress (existing MCP servers)
- Phase 7 (External UI Adapter): Complete
- Phase 8 (Custom MCP Tools): Waiting

## Architecture & Plan
- `(.cursor/rules/architecture.mdc)` for architecture
- `(.cursor/rules/agent-router-plan.mdc)` for the implementation plan
