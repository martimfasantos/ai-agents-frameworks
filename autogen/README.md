# Autogen

- Repo: https://github.com/microsoft/autogen
- Documentation: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html
- Version: **0.7.5**

## About AutoGen

AutoGen `v0.4+` was built from the ground up with an asynchronous, event-driven architecture. The API is layered: the **Core** API provides a scalable actor framework for agentic workflows, and the **AgentChat** API (built on Core) offers a task-driven, high-level framework for interactive multi-agent applications.

Key features:
- **AssistantAgent** - Model-powered agent with tool calling, structured output, and streaming
- **Team presets** - RoundRobin, Selector, MagenticOne, Swarm, and GraphFlow
- **Tools** - Function tools, agents-as-tools, teams-as-tools
- **Structured outputs** - Pydantic model responses with chain-of-thought
- **Human-in-the-loop** - Handoff pattern for pausing and resuming with user input
- **Memory** - ListMemory and custom memory stores for persistent context
- **GraphFlow** - Directed graph workflows with sequential, parallel, and conditional routing
- **Termination conditions** - MaxMessage, TextMention, Timeout, combinable with `|` and `&`
- **State management** - save_state() / load_state() for checkpointing agents and teams
- **Custom agents** - Subclass BaseChatAgent for arbitrary logic

> **Note:** Microsoft recommends new users also check out [Microsoft Agent Framework](https://github.com/microsoft/agent-framework). AutoGen will continue to be maintained.

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Install dependencies

```bash
uv sync
```

### Configure environment

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Run examples

```bash
uv run python 00_hello_world.py
uv run python 01_tools.py
uv run python 02_streaming_and_metrics.py
uv run python 03_structured_outputs.py
uv run python 04_human_in_the_loop.py
uv run python 05_multi_agent_teams.py
uv run python 06_agents_as_tool.py
uv run python 07_memory.py
uv run python 08_graph_flow.py
uv run python 09_termination_conditions.py
uv run python 10_state_management.py
uv run python 11_custom_agents.py
```

## Examples

| # | File | Topics |
|---|------|--------|
| 0 | `00_hello_world.py` | Basic agent creation, streaming output with Console |
| 1 | `01_tools.py` | Function tools, tool schema generation, tool call execution |
| 2 | `02_streaming_and_metrics.py` | Streaming responses, token usage statistics |
| 3 | `03_structured_outputs.py` | Pydantic model responses, chain-of-thought reasoning |
| 4 | `04_human_in_the_loop.py` | Handoff pattern, pausing and resuming with user input |
| 5 | `05_multi_agent_teams.py` | RoundRobin, Selector, MagenticOne, Swarm team presets |
| 6 | `06_agents_as_tool.py` | AgentTool, TeamTool, disabling parallel tool calls |
| 7 | `07_memory.py` | ListMemory, user preferences, memory-aware tool usage |
| 8 | `08_graph_flow.py` | DiGraphBuilder, sequential/parallel/conditional workflows |
| 9 | `09_termination_conditions.py` | MaxMessage, TextMention, Timeout, OR/AND combinators |
| 10 | `10_state_management.py` | save_state/load_state for agents and teams, checkpointing |
| 11 | `11_custom_agents.py` | Subclassing BaseChatAgent, CountDownAgent, EchoAgent |

## Key dependencies

- `autogen-agentchat>=0.7.5` - AgentChat high-level API
- `autogen-core>=0.7.5` - Core actor framework
- `autogen-ext[openai]>=0.7.5` - Extensions (OpenAI model client)
- `pydantic-settings` - Settings management from .env
