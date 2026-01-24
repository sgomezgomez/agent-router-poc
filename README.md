# Agent Router POC

Proof of concept for an autonomous agent that routes user queries to appropriate tools and agents using LLM-based decision-making.

## Features

- Multi-provider LLM support (OpenAI, Gemini, LM Studio, Grok)
- MCP (Model Context Protocol) integration
- A2A (Agent-to-Agent Protocol) support
- MongoDB-based audit logging
- Streaming and non-streaming modes
- Self-healing error recovery

## Quick Start

See [SETUP.md](SETUP.md) for complete manual setup instructions.

## Project Status

Phase 2 complete: LLM Service implemented and verified

## Architecture

See [.cursor/rules/architecture.mdc](.cursor/rules/architecture.mdc) for complete architecture documentation.

## Implementation Plan

See [.cursor/rules/agent-router-plan.mdc](.cursor/rules/agent-router-plan.mdc) for the full implementation plan.
