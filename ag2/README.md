# AG2

- Repo: https://github.com/ag2ai/ag2
- Documentation: https://docs.ag2.ai/
- Version: **1.2.1**

## About AG2

AG2 1.0 is a ground-up rewrite. The `autogen.beta` package was promoted to the top-level `ag2` package and the classic `ConversableAgent` framework moved out of this distribution entirely:

| Install | Import | What you get |
|---------|--------|--------------|
| `pip install ag2` | `import ag2` | The 1.0 framework (these examples) |
| `pip install autogen` | `import autogen` | [AG2 Classic](https://github.com/ag2ai/ag2-classic), maintenance mode only ([docs](https://classic.docs.ag2.ai/latest/)) |

`ag2>=1.0` no longer ships the `autogen` import name at all, so classic code does not run against it.

Key features:
- **Agent** - the single core primitive; async-only `ask()` / `run()` / `resume()`
- **Typed model configs** - `OpenAIConfig`, `AnthropicConfig`, `GeminiConfig`, and friends
- **Tools** - plain callables or `@tool`, plus builtins for code execution, search, and MCP
- **Structured output** - `response_schema=` with Pydantic models, dataclasses, or plain types
- **Human-in-the-loop** - `context.input()` inside tools, answered by an agent `hitl_hook`
- **Subagents** - `Agent.as_tool()` for LLM-driven delegation with isolated sub-task streams
- **ag2.network** - hub, channels and `TransitionGraph` for deterministic multi-agent workflows
- **Middleware** - intercept turns and LLM calls; builtins for logging, retries, limits, metrics
- **Observers & policies** - `BaseObserver` + `AlertPolicy` guardrails that can halt an agent
- **MemoryStream** - the structured event log behind history, observation, and resume
- **MCP** - `MCPToolkit` over stdio or HTTP servers
- **A2A protocol** - `A2AServer` to publish an agent, `A2AConfig` to consume a remote one
- **ACP** - drive Claude Code, Codex, or OpenCode as first-class agents
- **Metrics** - `MetricsMiddleware` emitting Prometheus counters and histograms
- **Decision models** - `OpenAIDecisionsConfig` answering typed classify/detect questions with probabilities

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Install dependencies

```bash
uv sync
```

### Configure environment

Copy `.env.example` to `.env` and add your credentials:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Run examples

```bash
uv run python 00_simple_agent.py
uv run python 01_agent_with_tools.py
uv run python 02_structured_outputs.py
uv run python 03_human_in_the_loop.py
uv run python 04_multi_agent.py
uv run python 05_sequential_chat.py
uv run python 06_nested_chat.py
uv run python 07_code_execution.py
uv run python 08_guardrails.py
uv run python 09_mcp_tools.py
uv run python 10_observability.py
uv run python 11_a2a.py
uv run python 12_middleware.py
uv run python 13_memory_stream.py
uv run python 14_observable_run.py
uv run python 15_resume.py
uv run python 16_metrics.py
uv run python 17_cli_agents_acp.py
uv run python 18_invocation_plugins.py
uv run python 19_decision_models.py
```

## Examples

| # | File | Topics |
|---|------|--------|
| 0 | `00_simple_agent.py` | `Agent`, `OpenAIConfig`, async `ask()`, `reply.ask()` continuation |
| 1 | `01_agent_with_tools.py` | `@tool` decorator, `tools=`, tool call/result events |
| 2 | `02_structured_outputs.py` | `response_schema=`, `await reply.content()`, per-turn schema override |
| 3 | `03_human_in_the_loop.py` | `context.input()` inside a tool, `hitl_hook`, approval gating |
| 4 | `04_multi_agent.py` | `Agent.as_tool()` subagents, LLM-driven delegation |
| 5 | `05_sequential_chat.py` | `ag2.network` `Hub`, `TransitionGraph.sequence()`, WAL replay |
| 6 | `06_nested_chat.py` | Two-level subagent nesting, shared stream to observe inner work |
| 7 | `07_code_execution.py` | `SandboxCodeTool` + `LocalEnvironment`, sandbox state persistence |
| 8 | `08_guardrails.py` | `on_tool_execution` veto, `BaseObserver`, `ObserverAlert`, `AlertPolicy`, `HaltEvent` |
| 9 | `09_mcp_tools.py` | `MCPToolkit`, `MCPStdioServerConfig`, local `MCPServer` (mcp 2.x) |
| 10 | `10_observability.py` | `MemoryStream.subscribe()`, `stream.where()`, event log + token usage |
| 11 | `11_a2a.py` | `A2AServer`, `build_card` with a tagged `AgentSkill`, `A2AConfig`, server-side tools |
| 12 | `12_middleware.py` | `BaseMiddleware`, `on_llm_call` / `on_turn`, middleware stacks |
| 13 | `13_memory_stream.py` | `MemoryStream` shared across turns for conversation memory |
| 14 | `14_observable_run.py` | `agent.run()`, `run.start()`, `run.stream.join()`, `run.enqueue()` |
| 15 | `15_resume.py` | `agent.resume()` from a stored trajectory and from a tool result |
| 16 | `16_metrics.py` | `MetricsMiddleware`, Prometheus counters/histograms, exposition text |
| 17 | `17_cli_agents_acp.py` | `ag2.acp` `ClaudeCodeConfig` driving Claude Code as an agent |
| 18 | `18_invocation_plugins.py` | `Plugin(prompt=, tools=, variables=)`, `ask(plugins=...)`, `Variable()` injection |
| 19 | `19_decision_models.py` | `OpenAIDecisionsConfig`, `Enum` classifier, `bool` detector with `boolean_threshold`, `reply.response.metadata` |

> `17_cli_agents_acp.py` needs Node.js (the Claude Code ACP adapter runs through `npx`) and an existing Claude Code login under `~/.claude`.
>
> `19_decision_models.py` always uses OpenAI's `gpt-6-luna` decision model (public beta), not `OPENAI_MODEL_NAME`; your OpenAI key needs access to the Decisions API.

## Key dependencies

- `ag2[openai,mcp,a2a,acp,metrics]>=1.2.1` - AG2 with OpenAI, MCP, A2A, ACP and Prometheus extras (pulls `openai>=3.27` and `a2a-sdk>=1.2`)
- `a2a-sdk[http-server]` - required by `A2AServer.build_jsonrpc()` (add `grpc` only for `build_grpc()`)
- `mcp>=2.0,<3` - Model Context Protocol SDK (ag2 1.0.3+ requires mcp 2.x)
- `pydantic-settings` - settings management from .env
- `uvicorn` - ASGI server for the A2A example
