# Claude Agent SDK

- Repo: https://github.com/anthropics/claude-agent-sdk-python
- Documentation: https://platform.claude.com/docs/en/agent-sdk/overview

## Claude Agent SDK Examples

| # | Example | Feature | Doc Page |
|---|---------|---------|----------|
| 00 | [Hello World](00_hello_world.py) | One-shot query | [Quickstart](https://platform.claude.com/docs/en/agent-sdk/quickstart) |
| 01 | [Built-in Tools](01_built_in_tools.py) | Read, Bash, Glob, Grep | [Overview](https://platform.claude.com/docs/en/agent-sdk/overview#capabilities) |
| 02 | [Custom Tools](02_custom_tools.py) | @tool + MCP server | [Custom Tools](https://platform.claude.com/docs/en/agent-sdk/custom-tools) |
| 03 | [Structured Outputs](03_structured_outputs.py) | JSON Schema / Pydantic | [Structured Outputs](https://platform.claude.com/docs/en/agent-sdk/structured-outputs) |
| 04 | [System Prompts](04_system_prompts.py) | Custom, preset, append | [System Prompts](https://platform.claude.com/docs/en/agent-sdk/modifying-system-prompts) |
| 05 | [Permissions](05_permissions.py) | Modes, allow/deny, can_use_tool | [Permissions](https://platform.claude.com/docs/en/agent-sdk/permissions) |
| 06 | [Hooks](06_hooks.py) | PreToolUse / PostToolUse | [Hooks](https://platform.claude.com/docs/en/agent-sdk/hooks) |
| 07 | [Sessions](07_sessions.py) | Resume, fork, continue | [Sessions](https://platform.claude.com/docs/en/agent-sdk/sessions) |
| 08 | [Multi-turn](08_multi_turn.py) | ClaudeSDKClient | [Streaming vs Single](https://platform.claude.com/docs/en/agent-sdk/streaming-vs-single-mode) |
| 09 | [Subagents](09_subagents.py) | AgentDefinition, delegation | [Subagents](https://platform.claude.com/docs/en/agent-sdk/subagents) |
| 10 | [MCP Servers](10_mcp_servers.py) | External stdio MCP | [MCP](https://platform.claude.com/docs/en/agent-sdk/mcp) |
| 11 | [Streaming](11_streaming.py) | Real-time StreamEvent | [Streaming Output](https://platform.claude.com/docs/en/agent-sdk/streaming-output) |
| 12 | [Cost Tracking](12_cost_tracking.py) | Budget, turns, effort | [Cost Tracking](https://platform.claude.com/docs/en/agent-sdk/cost-tracking) |
| 13 | [File Checkpointing](13_file_checkpointing.py) | Track & rewind files | [File Checkpointing](https://platform.claude.com/docs/en/agent-sdk/file-checkpointing) |

### How to setup

#### Virtual environment

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

Or create a standard virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install claude-agent-sdk python-dotenv pydantic pydantic-settings
```

#### Prerequisites

The Claude Agent SDK requires the **Claude Code CLI** to be installed:

```bash
npm install -g @anthropic-ai/claude-code
```

#### .env

See `.env.example` and create a `.env` file.
You need an Anthropic API key (`ANTHROPIC_API_KEY`).

#### Run examples

```bash
uv run 00_hello_world.py
uv run 02_custom_tools.py
# etc.
```
