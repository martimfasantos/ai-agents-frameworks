# OpenAI Agents SDK

- Repo: https://github.com/openai/openai-agents-python
- Documentation: https://openai.github.io/openai-agents-python/
- SDK Version: >= 0.12.5

## Key Features

The OpenAI Agents SDK provides a lightweight framework for building multi-agent workflows. Features that make it **distinct from other frameworks**:

- **Sessions** — Built-in persistent memory across turns (SQLite, Redis, SQLAlchemy, etc.) with no manual history management
- **Human-in-the-Loop** — Pause/resume execution for tool approval with serializable `RunState`
- **MCP Tools** — First-class Model Context Protocol support, including `HostedMCPTool` (OpenAI-hosted, no local server needed)
- **Realtime Agents** — WebSocket-based voice agents with sub-second latency (`RealtimeAgent` + `RealtimeRunner`)
- **Agent Visualization** — Generate Graphviz diagrams of agent architectures with `draw_graph()`
- **Handoffs & Orchestration** — Native agent-to-agent delegation with streaming support
- **Guardrails** — Input/output validation and LLM-as-a-judge patterns
- **Tracing** — Built-in OpenAI trace integration for debugging

## Examples

| # | File | Feature |
|---|------|---------|
| 0 | `0_hello_world.py` | Basic agent setup and execution |
| 1 | `1_tools_and_metrics.py` | Function tools with structured tool results |
| 2 | `2_structured_outputs.py` | Pydantic-based structured output parsing |
| 3 | `3_parallelization_in_workflow.py` | Running multiple agents in parallel |
| 4 | `4_handoffs_and_streaming.py` | Agent handoffs with streaming output |
| 5 | `5_agents_as_tools.py` | Using agents as callable tools |
| 6 | `6_output_guardrails.py` | Output validation guardrails |
| 7 | `7_llm_as_a_judge.py` | LLM-as-a-judge evaluation pattern |
| 8 | `8_tracing.py` | OpenAI trace integration |
| 9 | `9_sessions.py` | **Sessions** — persistent multi-turn memory with SQLiteSession |
| 10 | `10_human_in_the_loop.py` | **Human-in-the-Loop** — tool approval, RunState serialize/resume |
| 11 | `11_mcp_tools.py` | **MCP Tools** — HostedMCPTool with streaming and approval |
| 12 | `12_realtime_agent.py` | **Realtime Agents** — WebSocket voice agents with RealtimeRunner |
| 13 | `13_agent_visualization.py` | **Agent Visualization** — draw_graph() for agent architecture |

## How to Setup

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- For visualization: [Graphviz](https://graphviz.org/) installed (`brew install graphviz` on macOS)

### Install dependencies

```bash
uv sync
```

### .env

Copy `.env.example` to `.env` and fill in your OpenAI API key:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL_NAME=gpt-4o-mini
```

### Run an example

```bash
uv run python 0_hello_world.py
uv run python 9_sessions.py
uv run python 13_agent_visualization.py
```
