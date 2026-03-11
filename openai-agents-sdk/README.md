# OpenAI Agents SDK

- Repo: https://github.com/openai/openai-agents-python
- Documentation: https://openai.github.io/openai-agents-python/
- Version: **0.11.1**

## About OpenAI Agents SDK

The OpenAI Agents SDK is a lightweight, production-ready framework for building agentic AI applications. It provides primitives for creating agents with tools, handoffs, guardrails, and tracing — all with minimal abstraction over the OpenAI API.

Key features:
- **Agents** - LLM-powered units with instructions, tools, and handoffs
- **Tools** - Function tools, agents-as-tools, and built-in tools
- **Handoffs** - Agent-to-agent delegation for multi-agent workflows
- **Guardrails** - Input, output, and tool guardrails for safety validation
- **Structured output** - Pydantic model responses via `output_type`
- **Tracing** - Built-in tracing with custom spans and exporters
- **Streaming** - Real-time streamed responses
- **Sessions** - Multi-turn conversation memory with SQLiteSession
- **Human-in-the-loop** - Tool approval workflows with RunState

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Install dependencies

```bash
uv sync
```

### Configure environment

Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
# Edit .env with your API key
```

You need an OpenAI API key from https://platform.openai.com/api-keys.

### Run examples

```bash
uv run python 00_hello_world.py
uv run python 01_tools_and_metrics.py
uv run python 02_structured_outputs.py
uv run python 03_parallelization_in_workflow.py
uv run python 04_handoffs_and_streaming.py
uv run python 05_agents_as_tools.py
uv run python 06_output_guardrails.py
uv run python 07_llm_as_a_judge.py
uv run python 08_tracing.py
uv run python 09_input_guardrails.py
uv run python 10_lifecycle_hooks.py
uv run python 11_context_management.py
uv run python 12_dynamic_instructions.py
uv run python 13_sessions.py
uv run python 14_human_in_the_loop.py
uv run python 15_tool_guardrails.py
```

## Examples

| # | File | Topics |
|---|------|--------|
| 0 | `00_hello_world.py` | Basic agent creation, `Runner.run_sync()` |
| 1 | `01_tools_and_metrics.py` | `@function_tool`, token usage metrics |
| 2 | `02_structured_outputs.py` | Pydantic `output_type` for typed responses |
| 3 | `03_parallelization_in_workflow.py` | Parallel agent execution with `asyncio.gather` |
| 4 | `04_handoffs_and_streaming.py` | Agent handoffs and `Runner.run_streamed()` |
| 5 | `05_agents_as_tools.py` | `Agent.as_tool()` for multi-agent orchestration |
| 6 | `06_output_guardrails.py` | `@output_guardrail` for response validation |
| 7 | `07_llm_as_a_judge.py` | LLM-based evaluation with guardrail agents |
| 8 | `08_tracing.py` | `trace()` context manager, custom spans |
| 9 | `09_input_guardrails.py` | `@input_guardrail` for input validation |
| 10 | `10_lifecycle_hooks.py` | `RunHooks` for agent lifecycle events |
| 11 | `11_context_management.py` | `RunContextWrapper` with typed dataclass context |
| 12 | `12_dynamic_instructions.py` | Dynamic instructions as functions |
| 13 | `13_sessions.py` | `SQLiteSession` for multi-turn memory |
| 14 | `14_human_in_the_loop.py` | Tool approval with `needs_approval`, `RunState` |
| 15 | `15_tool_guardrails.py` | `@tool_input_guardrail`, `@tool_output_guardrail` |

## Key dependencies

- `openai-agents>=0.11.1` - OpenAI Agents SDK
- `pydantic>=2.12.5` - Data validation
- `pydantic-settings>=2.13.1` - Settings management from .env
