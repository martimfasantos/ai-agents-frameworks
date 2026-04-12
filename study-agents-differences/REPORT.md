# Comprehensive Comparison Report: AI Agent Frameworks

This report provides a detailed analysis of 14 AI agent frameworks (15 basic agent implementations, 5 RAG+API agents), comparing their architecture, implementation complexity, capabilities, and performance characteristics. All agents were implemented following a consistent interface to enable fair comparison.

---

## Table of Contents

1. [Methodology](#methodology)
2. [Framework Overview](#framework-overview)
3. [Implementation Complexity](#implementation-complexity)
4. [Architecture Comparison](#architecture-comparison)
5. [Tool Integration Patterns](#tool-integration-patterns)
6. [Memory Management](#memory-management)
7. [Token Tracking](#token-tracking)
8. [Per-Framework Metrics Capabilities](#per-framework-metrics-capabilities)
9. [Execution Model Deep-Dive](#execution-model-deep-dive)
10. [Error Handling Patterns](#error-handling-patterns)
11. [Provider Support](#provider-support)
12. [RAG Capabilities](#rag-capabilities)
13. [Multi-Agent Support](#multi-agent-support)
14. [Historical Benchmark Results](#historical-benchmark-results)
    - [Tool-Calling Benchmark (All 15 Frameworks)](#tool-calling-benchmark-all-15-frameworks)
    - [Token Tracking Research](#token-tracking-research)
15. [Key Findings](#key-findings)
16. [Recommendations](#recommendations)

---

## Methodology

### Agent Interface Standard

Every agent implementation follows a consistent Python interface to allow fair comparison:

```python
class Agent:
    def __init__(self, provider="azure", memory=True, verbose=True, tokens=True):
        ...
    def chat(self, message: str) -> tuple[str, float, dict]:
        # Returns: (response_text, execution_time_seconds, tokens_dict)
        ...
    def clear_chat(self) -> bool:
        ...
```

The tokens dictionary follows a standard format:

```python
{
    "total_embedding_token_count": int,
    "prompt_llm_token_count": int,
    "completion_llm_token_count": int,
    "total_llm_token_count": int,
}
```

### Shared Infrastructure

All agents use:
- **`settings.py`**: Centralized configuration via `pydantic-settings` reading from `.env`
- **`prompts.py`**: Shared system prompts (role, goal, instructions, knowledge)
- **`utils.py`**: Common CLI argument parsing (`parse_args()`) and REPL execution (`execute_agent()`)
- **`shared_functions/`**: F1API and MetroAPI tool implementations for API-based agents

### Agent Tiers

Agents are organized into two tiers:
- **Basic agents**: Web search tool + date utility. Tests general conversation and tool-calling ability.
- **RAG+API agents**: Knowledge base retrieval + F1 API + Metro API. Tests retrieval accuracy and multi-tool coordination.

### Knowledge Base

RAG agents use two markdown files in `knowledge_base/cl_matches/` containing UEFA Champions League 2025 match data (Benfica, Bayern, PSG, Dortmund, Arsenal, Real Madrid, Club Brugge, Feyenoord results).

---

## Framework Overview

| # | Framework | Version | Type | Origin | License |
|---|-----------|---------|------|--------|---------|
| 1 | **Agno** | >=2.5.0 | Full agent framework | Agno (formerly Phidata) | Apache 2.0 |
| 2 | **AG2** | >=0.11.4 | Conversable agent framework | AG2 (AutoGen fork) | Apache 2.0 |
| 3 | **Claude Agent SDK** | >=0.1.52 | Anthropic agent toolkit | Anthropic | MIT |
| 4 | **CrewAI** | >=1.12.2 | Task-orchestration framework | CrewAI Inc. | MIT |
| 5 | **Google ADK** | >=1.26.0 | Agent Development Kit | Google | Apache 2.0 |
| 6 | **LangChain** | >=1.0.0 | LLM application framework | LangChain Inc. | MIT |
| 7 | **LangGraph** | >=1.1.3 | Graph-based agent framework | LangChain Inc. | MIT |
| 8 | **LlamaIndex** | >=0.12.19 | Data framework for LLMs | LlamaIndex Inc. | MIT |
| 9 | **Microsoft Agent Framework** | >=1.0.0rc5 | Agent framework | Microsoft | MIT |
| 10 | **OpenAI (raw API)** | >=1.63.2 | Direct API usage | OpenAI | N/A |
| 11 | **OpenAI Agents SDK** | >=0.12.5 | Agent orchestration SDK | OpenAI | MIT |
| 12 | **PydanticAI** | >=1.70.0 | Type-safe agent framework | Pydantic/Samuel Colvin | MIT |
| 13 | **Smolagents** | >=1.24.0 | Lightweight agent library | HuggingFace | Apache 2.0 |
| 14 | **Strands Agents SDK** | >=1.32.0 | Agent SDK | AWS | Apache 2.0 |

---

## Implementation Complexity

### Lines of Code (Basic Agents)

| Agent File | Lines | Complexity Rating |
|---|---|---|
| `openai_agents_sdk_agent.py` | 137 | Low |
| `crewai_agent.py` | 144 | Low |
| `strands_agent.py` | 158 | Low |
| `langchain_agent.py` | 161 | Low-Medium |
| `ag2_agent.py` | 167 | Low-Medium |
| `microsoft_agent.py` | 167 | Low-Medium |
| `google_adk_agent.py` | 171 | Medium |
| `claude_sdk_agent.py` | 176 | Medium |
| `smolagents_agent.py` | 177 | Medium |
| `pydantic_ai_agent.py` | 186 | Medium |
| `agno_agent.py` | 198 | Medium |
| `llama_index_fc_agent.py` | 205 | Medium-High |
| `llama_index_agent.py` | 206 | Medium-High |
| `langgraph_agent.py` | 232 | High |
| `openai_agent.py` | 280 | High |

**Key observations:**
- The **OpenAI Agents SDK** requires the least code (137 lines), demonstrating the advantage of a higher-level abstraction
- **Raw OpenAI API** requires the most code (280 lines) since it needs manual tool dispatch, JSON schema definitions, and a two-pass completion flow
- **Claude SDK** grew from 140 to 176 lines due to the MCP server pattern (`create_sdk_mcp_server()`) and async event streaming — more complex than the previous simple `@tool` approach
- **LangGraph** decreased from 243 to 232 lines after the rewrite from `create_react_agent` to manual `StateGraph` — the explicit graph construction is actually slightly more concise
- Both LlamaIndex agents converged to ~205 lines after migration to `FunctionAgent` (no longer a ReAct vs FC distinction)

### Lines of Code (RAG+API Agents)

| Agent File | Lines |
|---|---|
| `crewai_rag_api_agent.py` | 191 |
| `langchain_rag_api_agent.py` | 209 |
| `llama_index_rag_api_agent.py` | 238 |
| `agno_rag_api_agent.py` | 266 |
| `langgraph_rag_api_agent.py` | 279 |

---

## Architecture Comparison

### Execution Models

| Framework | Execution Pattern | Async Native | Agent Creation Pattern |
|---|---|---|---|
| Agno | `agent.run(message)` | No | `AgnoAgent(model=..., tools=..., instructions=...)` |
| AG2 | `agent.run(message)` | No | `ConversableAgent(llm_config=LLMConfig({...}), functions=[...])` |
| Claude SDK | `query()` async generator | **Yes** | `query(prompt=..., options=ClaudeAgentOptions(mcp_servers=...))` |
| CrewAI | `crew.kickoff()` | No | `CrewAgent(role=..., llm=model_name) + Task + Crew` |
| Google ADK | `runner.run_async()` | **Yes** | `LlmAgent(model=..., tools=[FunctionTool(func=...)])` |
| LangChain | `agent.invoke()` | No | `create_agent(model=llm, tools=..., system_prompt=...)` |
| LangGraph | `graph.stream()` | No | `StateGraph(MessagesState)` + `ToolNode` + `tools_condition` |
| LlamaIndex | `agent.run(message)` | **Yes** | `FunctionAgent(llm=..., tools=..., system_prompt=...)` |
| Microsoft AF | `agent.run()` | **Yes** | `OpenAIChatClient(model_id=...).as_agent(tools=...)` |
| OpenAI (raw) | `completions.create()` | No | Manual setup |
| OpenAI Agents SDK | `Runner.run_sync()` | No* | `Agent(name=..., model=..., tools=...)` |
| PydanticAI | `agent.run_sync(message)` | **Yes** | `PydanticAgent(model=model_name, tools=..., instructions=...)` |
| Smolagents | `agent.run(message)` | No | `CodeAgent(model=OpenAIModel(...), tools=...)` |
| Strands | `agent(message)` | No | `StrandsAgent(model=..., tools=...)` |

*OpenAI Agents SDK is async internally but provides `run_sync()` wrapper.
*LlamaIndex FunctionAgent is async internally; wrapped via `asyncio.new_event_loop()`.
*PydanticAI provides `run_sync()` as a convenience wrapper over the async `run()`.

### Agent Paradigms

| Paradigm | Frameworks | Description |
|---|---|---|
| **Tool-calling agent** | Agno, AG2, LangChain, LlamaIndex, Microsoft AF, OpenAI Agents SDK, PydanticAI, Strands | Agent decides when to call tools based on the conversation |
| **Code-generating agent** | Smolagents | Agent writes and executes Python code to call tools (CodeAgent) |
| **Function-calling agent** | OpenAI (raw) | Direct JSON function-calling via the LLM's native function-calling API |
| **Task-orchestration** | CrewAI | Agent + Task + Crew pattern; designed for workflow orchestration |
| **Graph-based** | LangGraph | Manual `StateGraph` with `ToolNode` + `tools_condition` edges; streaming execution |
| **Session-based** | Google ADK, Claude SDK | Session-managed conversations with event streaming |

**Note:** LlamaIndex previously had two paradigms (ReAct via `ReActAgent` and Function-Calling via `FunctionCallingAgentWorker`), but both now use the unified `FunctionAgent` from `llama_index.core.agent.workflow`. LangGraph previously used the prebuilt `create_react_agent()` helper but now uses explicit `StateGraph(MessagesState)` construction with `ToolNode` and `tools_condition`.

---

## Tool Integration Patterns

Each framework has a different approach to registering tools:

### Decorator-based

```python
# Agno — bare @tool, description via docstring
@tool
def web_search(query: str) -> str:
    """Search the web for a query."""
    ...

# CrewAI
@crewai_tool
def web_search(query: str) -> str: ...

# Smolagents (requires Args/Returns in docstring)
@smolagent_tool
def web_search(query: str) -> str: ...

# Strands (requires Args in docstring)
@strands_tool
def web_search(query: str) -> str: ...

# OpenAI Agents SDK
@function_tool
def web_search(query: str) -> str: ...

# Microsoft AF — @tool from agent_framework
from agent_framework import tool
@tool
def web_search(query: str) -> str: ...

# LangChain / LangGraph — @tool from langchain.tools
from langchain.tools import tool
@tool
def web_search(query: str) -> str: ...
```

### Constructor-based

```python
# PydanticAI — tools passed to Agent constructor, @agent.tool_plain for inline
agent = PydanticAgent(model="openai:gpt-4o", tools=[...], instructions="...")

# LlamaIndex — FunctionTool.from_defaults()
FunctionTool.from_defaults(fn=web_search, name="web_search", description="...")

# Google ADK — FunctionTool(func=) wrapping
FunctionTool(func=web_search)
```

### Plain Functions

```python
# AG2 — functions passed as list to ConversableAgent
ConversableAgent(..., functions=[web_search, get_date])
```

### MCP Server Pattern

```python
# Claude SDK — tools exposed via MCP server, not decorators
async def handle_tool_call(name, arguments):
    if name == "web_search":
        return {"type": "text", "text": web_search(arguments["query"])}

server = create_sdk_mcp_server(tools=[...])
result = query(prompt=msg, options=ClaudeAgentOptions(mcp_servers=[server]))
```

### Raw JSON Schema

```python
# OpenAI (raw API)
tools = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "...",
        "parameters": {"type": "object", "properties": {...}}
    }
}]
```

### Toolkit Integration

Some frameworks support bundled tool collections:
- **Agno**: `TavilyTools()` toolkit class
- **CrewAI**: `@crewai_tool` wrappers (no native toolkit)
- **Smolagents**: `smolagents[toolkit]` extra

### Tool Registration Summary

| Framework | Registration | Docstring Required | Type Hints Required | Notes |
|---|---|---|---|---|
| Agno | `@tool` decorator | **Yes** (for description) | Yes | Bare decorator; description from docstring |
| AG2 | Plain functions list | No | Yes | Functions auto-registered |
| Claude SDK | MCP server pattern | No | Yes | Tools exposed via `create_sdk_mcp_server()`, async handlers return MCP dicts |
| CrewAI | `@crewai_tool` decorator | Yes (for description) | Yes | Must re-wrap existing functions |
| Google ADK | `FunctionTool(func=)` | No | Yes | Explicit wrapping with `FunctionTool` |
| LangChain | `@tool` from `langchain.tools` | No | Yes | Also supports `Tool()` constructor |
| LangGraph | `@tool` from `langchain.tools` | No | Yes | Same decorator, used with `ToolNode` + `tools_condition` |
| LlamaIndex | `FunctionTool.from_defaults()` | No | No | Constructor-based only |
| Microsoft AF | `@tool` from `agent_framework` | No | Yes | Async support |
| OpenAI (raw) | JSON schema | N/A | N/A | Most verbose, full control |
| OpenAI Agents SDK | `@function_tool` | No | Yes | Cleanest decorator API |
| PydanticAI | Tools list in constructor | No | Yes | Type-safe with Pydantic; also `@agent.tool_plain` |
| Smolagents | `@smolagent_tool` | **Yes** (Args/Returns) | Yes | Strict docstring format |
| Strands | `@strands_tool` | **Yes** (Args) | Yes | AWS-style conventions |

---

## Memory Management

| Framework | Memory Mechanism | Stateful | Clear Method |
|---|---|---|---|
| Agno | `AgentMemory()` + `add_history_to_messages` | Yes | `agent.memory.clear()` |
| AG2 | Built-in history | Yes | `agent.clear_history()` |
| Claude SDK | Manual `messages` list | Manual | Reset list |
| CrewAI | Framework-level `memory=True/False` | Partial | No-op (task-based) |
| Google ADK | `InMemorySessionService` | Yes | New session ID |
| LangChain | `InMemorySaver` + thread_id | Yes | Increment thread_id |
| LangGraph | `MemorySaver` + thread_id | Yes | Increment thread_id |
| LlamaIndex | Manual `Context` + `memory=` param | Manual | Recreate context/agent |
| Microsoft AF | None (stateless) | No | No-op |
| OpenAI (raw) | Manual messages list | Manual | Reset list (buggy) |
| OpenAI Agents SDK | None (stateless) | No | No-op |
| PydanticAI | Manual `messages` list + `message_history=` | Manual | Reset list |
| Smolagents | `agent.memory.steps` | Yes | Clear steps list |
| Strands | `agent.messages` | Yes | Reset list |

**Key finding:** Only Agno, LangGraph, LlamaIndex, and Google ADK provide robust built-in memory management. Most other frameworks require manual message history tracking.

---

## Token Tracking

| Framework | Token Source | Accuracy | Notes |
|---|---|---|---|
| Agno | `response.metrics` dict (list-based) | High | Per-step metrics lists, aggregated with `sum()` |
| AG2 | None | **N/A** | No documented token API; `runtime_logging` captures metadata but not tokens |
| Claude SDK | `ResultMessage.usage` dict | Partial | Only `input_tokens`/`output_tokens` from final `ResultMessage`; input count is minimal (SDK doesn't expose full prompt tokenization) |
| CrewAI | `result.token_usage` | High | Aggregated by framework |
| Google ADK | None | **N/A** | No token API when proxying through LiteLLM to non-Gemini models |
| LangChain | `response_metadata["token_usage"]` | High | From message metadata |
| LangGraph | `response_metadata["token_usage"]` | High | Same as LangChain; from last streamed event |
| LlamaIndex | Global `TokenCountingHandler` | Medium-High | Client-side tiktoken (gpt-4 encoding), not server-reported |
| Microsoft AF | None | **N/A** | Framework is RC; no token usage API documented |
| OpenAI (raw) | `completion.usage` | High | Direct from API; multi-call accumulation |
| OpenAI Agents SDK | `RunResult.raw_responses[].usage` | High | `input_tokens`/`output_tokens`/`total_tokens` per response |
| PydanticAI | `result.usage()` | High | Clean API: `.request_tokens`, `.response_tokens` |
| Smolagents | `ActionStep.token_usage` in `agent.memory.steps` | High | Summed across all steps; `input_tokens`/`output_tokens`/`total_tokens` per step |
| Strands | `result.metrics.get_summary()["accumulated_usage"]` | High | camelCase keys: `inputTokens`/`outputTokens`/`totalTokens` |

### Token Tracking Accuracy Deep-Dive

#### Extraction Patterns

Five distinct approaches are used across all 20 agents:

| Pattern | Agents | Source Type |
|---|---|---|
| **LangChain `response_metadata`** | LangChain, LangGraph, LangChain RAG, LangGraph RAG | Server-reported via message metadata |
| **Framework result object** | CrewAI, AG2, Microsoft AF, Strands, CrewAI RAG | `result.token_usage` / `agent.get_total_usage()` / `result.usage_details` / `result.metrics` |
| **Response metrics (dataclass)** | Agno, Agno RAG | `response.metrics.input_tokens` / `.output_tokens` / `.total_tokens` (RunMetrics) |
| **Raw API usage object** | OpenAI, OpenAI Agents SDK, Claude SDK, Google ADK | `completion.usage.*` or `event.usage_metadata.*` |
| **Client-side tiktoken counting** | LlamaIndex (×3), LlamaIndex RAG | `TokenCountingHandler` callback — NOT server-reported |

#### Field Name Mismatches

Each framework uses different names for token counts, requiring per-framework mapping to the standard output format:

| Framework | Prompt Field | Completion Field | Total Field |
|---|---|---|---|
| OpenAI / LangChain / LangGraph / CrewAI | `prompt_tokens` | `completion_tokens` | `total_tokens` |
| PydanticAI | `request_tokens` | `response_tokens` | `total_tokens` |
| Claude SDK | `input_tokens` (dict key) | `output_tokens` (dict key) | computed |
| OpenAI Agents SDK | `input_tokens` (attr) | `output_tokens` (attr) | `total_tokens` (attr) |
| Strands | `inputTokens` | `outputTokens` | `totalTokens` |
| Smolagents | `token_usage.input_tokens` | `token_usage.output_tokens` | `token_usage.total_tokens` |
| Agno | `input_tokens` (attr) | `output_tokens` (attr) | `total_tokens` (attr) |
| LlamaIndex | `prompt_llm_token_count` | `completion_llm_token_count` | `total_llm_token_count` |
| AG2 | `prompt_tokens` (nested dict) | `completion_tokens` (nested dict) | `total_tokens` (nested dict) |
| Google ADK | `prompt_token_count` | `candidates_token_count` | `total_token_count` |
| Microsoft AF | `input_token_count` | `output_token_count` | `total_token_count` |

#### Safety Levels

| Safety Level | Agents | Pattern |
|---|---|---|
| **Most defensive** | LangGraph, Agno, PydanticAI, Microsoft AF, AG2, LangGraph RAG, Agno RAG | Inner `try/except` + `getattr` with defaults + conditional checks |
| **Moderately safe** | LangChain, CrewAI, OpenAI Agents SDK, Strands, Google ADK, Claude SDK, Smolagents + all their RAG variants | `hasattr`/`.get()`/`getattr` with defaults, no inner `try/except` |
| **Least safe** | OpenAI raw, LlamaIndex (×3 + RAG) | Direct attribute access with `or 0` or no guard at all |

#### Embedding Token Tracking

| Tracks embeddings? | Agents |
|---|---|
| **Yes** (non-zero possible) | LlamaIndex, LlamaIndex FC, LlamaIndex RAG |
| **No** (hardcoded 0) | All other 17 agents |

Only LlamaIndex agents track embedding tokens because they use a global `TokenCountingHandler` callback that intercepts all LLM and embedding calls through the callback manager. All other frameworks — including RAG agents that actively use embeddings for vector search — report 0 for `total_embedding_token_count`.

#### Known Limitations

1. **Claude SDK** only reports tokens from the final `ResultMessage.usage` — the input token count is minimal (~3) because the SDK wraps the Claude Code CLI and doesn't expose full prompt tokenization. Output tokens (~59) are accurate for the final response.
2. **LlamaIndex** uses client-side tiktoken counting hardcoded to gpt-4 encoding — may diverge from actual server-reported usage, especially for non-OpenAI models.
3. **Google ADK** exposes `usage_metadata` on events only when `usage_metadata` is populated by the model backend. When proxying through LiteLLM to OpenAI, `event.usage_metadata.prompt_token_count` and `candidates_token_count` are now correctly captured.
4. **AG2** exposes token usage via `ConversableAgent.get_total_usage()` which returns a nested dict keyed by model name. This was not documented in the main README but is available in the `autogen` source and test code.
5. **Agno** changed from a dict-based `response.metrics.get("prompt_tokens", [0])` pattern (pre-2.5) to a `RunMetrics` dataclass on `response.metrics` with `input_tokens`, `output_tokens`, `total_tokens` fields.
6. **Microsoft AF** exposes `usage_details` (a `UsageDetails` TypedDict with `input_token_count`, `output_token_count`, `total_token_count`) on `AgentResponse`, or alternatively as `Content` items with `type == "usage"` in response messages.
5. **Smolagents** CodeAgent uses 8x more tokens (~5,776 total) than direct tool-calling frameworks (~700) because it generates Python code, often with retry cycles on format/parsing errors.
6. **OpenAI raw API** makes two completion calls (initial + post-tool-call) and manually sums tokens across both. The total for the second call is computed as `prompt + completion` rather than reading `total_tokens`, introducing a subtle inconsistency.
7. **All RAG agents except LlamaIndex RAG** ignore embedding token costs, meaning the true cost of RAG queries is underreported.

---

## Per-Framework Metrics Capabilities

Beyond token tracking, each framework exposes different levels of observability. This table summarizes what metrics are natively available vs. what must be computed by the wrapper.

| Framework | Response Time | Token Counts | Tool Call Count | Cost Estimation | Streaming Events | Step-Level Metrics |
|---|---|---|---|---|---|---|
| Agno | Wrapper | **Native** (RunMetrics dataclass) | Via metrics | No | No | **Yes** (per-step lists) |
| AG2 | Wrapper | **Native** (get_total_usage() dict) | No | No | No | No |
| Claude SDK | Wrapper | **Partial** (ResultMessage.usage, output-heavy) | Via events | **Yes** (`total_cost_usd`) | **Yes** (async generator) | **Yes** (per-message) |
| CrewAI | Wrapper | **Native** (aggregated) | No | No | No | No |
| Google ADK | Wrapper | **Native** (event.usage_metadata) | Via events | No | **Yes** (async events) | **Yes** (per-event) |
| LangChain | Wrapper | **Native** (response metadata) | No | No | No | No |
| LangGraph | Wrapper | **Native** (response metadata) | Via state | No | **Yes** (graph streaming) | **Yes** (per-node events) |
| LlamaIndex | Wrapper | **Callback** (tiktoken) | Via handler | No | No | **Yes** (callback events) |
| Microsoft AF | Wrapper | **Native** (AgentResponse.usage_details) | No | No | No | No |
| OpenAI (raw) | Wrapper | **Native** (API response) | Manual | Manual | No | No |
| OpenAI Agents SDK | Wrapper | **Native** (raw_responses[].usage) | No | No | No | No |
| PydanticAI | Wrapper | **Native** (usage method) | No | **Yes** (via usage) | No | No |
| Smolagents | Wrapper | **Native** (ActionStep.token_usage per step) | No | No | No | **Yes** (step memory) |
| Strands | Wrapper | **Native** (metrics.get_summary()) | Via metrics | No | No | **Yes** (tool_usage, traces) |

**Key takeaways:**
- **Response time** is always computed by the wrapper (`time.perf_counter()` before/after `chat()`), never by the framework itself.
- **Token tracking** is now available in **15/15 frameworks** after two rounds of targeted fixes. All frameworks expose token usage through at least one API.
- **Cost estimation** is only natively provided by Claude SDK (`ResultMessage.total_cost_usd`) and PydanticAI (structured usage data that can be multiplied by known rates).
- **Step-level metrics** are available in Agno (per-step token lists), Claude SDK (per-message events), Google ADK (per-event usage), LangGraph (per-node state), LlamaIndex (callback events), Smolagents (ActionStep memory with per-step token counts), and Strands (tool_usage stats, execution traces).
- **Streaming frameworks** (Claude SDK, Google ADK, LangGraph) can provide real-time token counts during execution, but the wrapper collapses this to a single aggregated count.

---

## Execution Model Deep-Dive

### Sync vs. Async Landscape

| Category | Frameworks | Wrapper Strategy |
|---|---|---|
| **Sync-native** | AG2, CrewAI, LangChain, Smolagents, Strands | Direct call — no async handling needed |
| **Async-native with sync helper** | OpenAI Agents SDK (`run_sync()`), PydanticAI (`run_sync()`) | Use the provided sync wrapper |
| **Async-native, manual loop** | LlamaIndex (`FunctionAgent.run()`), Claude SDK (`query()`), Google ADK (`runner.run_async()`), Microsoft AF (`agent.run()`) | Create `asyncio.new_event_loop()` per call |
| **Streaming (sync iteration)** | LangGraph (`graph.stream()`) | Iterate over stream synchronously |
| **Mixed** | Agno (`agent.run()`) | Sync call, but may use async internals |
| **Two-pass** | OpenAI raw | Two synchronous `completions.create()` calls (initial + post-tool) |

### Event Loop Management Patterns

Frameworks that are async-native but wrapped in a sync `chat()` method use one of two strategies:

**Strategy 1: New event loop per call** (LlamaIndex, Claude SDK, Google ADK, Microsoft AF)
```python
loop = asyncio.new_event_loop()
try:
    result = loop.run_until_complete(agent.run(message))
finally:
    loop.close()
```
This is safe for single-threaded CLI usage but creates overhead from loop creation/teardown on each `chat()` call.

**Strategy 2: Framework-provided sync wrapper** (PydanticAI, OpenAI Agents SDK)
```python
result = agent.run_sync(message)  # Framework manages the event loop
```
Cleaner and avoids loop management. PydanticAI and OpenAI Agents SDK both provide this pattern.

### Streaming vs. Batch Execution

| Model | Frameworks | Behavior |
|---|---|---|
| **Batch** (single response) | AG2, Agno, CrewAI, LangChain, LlamaIndex, Microsoft AF, OpenAI Agents SDK, PydanticAI, Smolagents, Strands | Agent runs to completion, returns full response |
| **Event stream** | Claude SDK, Google ADK | Async generator yields events; wrapper collects until `ResultMessage` / final event |
| **Graph stream** | LangGraph | `graph.stream()` yields per-node events; wrapper extracts from last event's messages |
| **Two-pass** | OpenAI raw | First call may return tool calls; second call returns final response |

### Multi-Step Tool Execution

When an LLM decides to call tools, each framework handles the tool call → result → next LLM call cycle differently:

| Framework | Tool Loop | Developer Visibility |
|---|---|---|
| **Fully automatic** | Agno, CrewAI, LangChain, LlamaIndex, Microsoft AF, OpenAI Agents SDK, PydanticAI, Smolagents, Strands | Zero — framework handles all tool dispatch internally |
| **Graph-controlled** | LangGraph | Full — developer defines the `tools_condition` edge and `ToolNode`; can add custom routing logic |
| **Event-visible** | Claude SDK, Google ADK | Partial — events reveal tool calls in the stream, but dispatch is automatic |
| **Fully manual** | OpenAI raw | Full — developer writes the tool dispatch `if/elif` chain and re-calls the API |
| **Conversation-based** | AG2 | Agent generates reply including tool results; `user_input=False` prevents human-in-the-loop |

---

## Error Handling Patterns

### Wrapper-Level Error Handling

Every agent's `chat()` method follows the same error contract: **always return a 3-tuple**, even on failure.

```python
def chat(self, message: str) -> tuple[str, float, dict]:
    try:
        # ... framework-specific logic ...
        return (response_text, elapsed_time, tokens_dict)
    except Exception as e:
        if self.verbose:
            print(f"Error: {e}")
        return ("Sorry, I couldn't process that request.", 0.0, {})
```

This ensures the benchmark runner (`utils.py` / `benchmark_runner.py`) never crashes due to a framework error.

### Framework-Specific Error Behaviors

| Framework | Common Failure Mode | Framework Error Type | Recovery |
|---|---|---|---|
| Agno | API timeout | `AgnoError` | Wrapper catch → 3-tuple |
| AG2 | Conversation loop stuck | `ConversableAgent` returns empty | Check for empty string |
| Claude SDK | CLI not installed | `subprocess.CalledProcessError` | Wrapper catch → 3-tuple |
| CrewAI | Verbose logging interferes | `CrewAIError` | Wrapper catch → 3-tuple |
| Google ADK | LiteLLM proxy misconfiguration | Various LiteLLM errors | Wrapper catch → 3-tuple |
| LangChain | Tool execution failure | `ToolException` | Agent retries, then wrapper catches |
| LangGraph | Graph state corruption | `GraphRecursionError` | Wrapper catch → 3-tuple |
| LlamaIndex | Async event loop conflict | `RuntimeError` (nested loop) | New loop per call avoids this |
| Microsoft AF | RC API instability | Various | `getattr` guards + wrapper catch |
| OpenAI (raw) | Tool schema mismatch | `openai.BadRequestError` | Wrapper catch → 3-tuple |
| OpenAI Agents SDK | Model not available | `openai.APIError` | Wrapper catch → 3-tuple |
| PydanticAI | Type validation failure | `pydantic.ValidationError` | Wrapper catch → 3-tuple |
| Smolagents | Code execution failure | `AgentError` | CodeAgent retries internally |
| Strands | AWS credentials missing | `botocore.exceptions` | Wrapper catch → 3-tuple |

### Token Extraction Error Isolation

Seven agents have **inner try/except blocks** specifically around token extraction, preventing token-related errors from affecting the response:

```python
# Pattern used by LangGraph, Agno, PydanticAI, Microsoft AF, AG2, LangGraph RAG, Agno RAG
try:
    usage = getattr(result, "usage", None)
    tokens = { ... }
except Exception:
    tokens = {}  # Silently degrade — response still returned
```

The remaining agents rely on the outer `chat()` try/except. If token extraction fails in those agents, the entire response is lost and replaced with the error 3-tuple. This is a design trade-off: inner try/except adds resilience but makes token extraction failures invisible.

---

## Provider Support

| Framework | OpenAI | Azure OpenAI | HuggingFace/OSS | Other |
|---|---|---|---|---|
| Agno | Yes | Yes | Yes | - |
| AG2 | Yes | Yes | Yes | - |
| Claude SDK | - | - | - | Anthropic only |
| CrewAI | Yes | Yes | Yes | LiteLLM-based |
| Google ADK | Yes | Yes | - | Gemini default |
| LangChain | Yes | Yes | Yes | - |
| LangGraph | Yes | Yes | Yes | - |
| LlamaIndex | Yes | Yes | Yes | - |
| Microsoft AF | Yes | Yes | Yes | - |
| OpenAI (raw) | Yes | Yes | - | - |
| OpenAI Agents SDK | Yes | - | - | OpenAI only |
| PydanticAI | Yes | Yes | Yes | Via OpenAIModel |
| Smolagents | Yes | Yes | Yes | LiteLLM-based |
| Strands | Yes | Yes | Yes | Via client_args |

**Notable limitations:**
- **Claude SDK** only works with Anthropic's Claude models (requires Claude Code CLI)
- **OpenAI Agents SDK** only supports OpenAI's API directly (no Azure, no OSS)
- **Google ADK** defaults to Gemini models but supports OpenAI via LiteLLM prefixes

---

## RAG Capabilities

Five frameworks have RAG+API agent implementations:

### RAG Architecture Comparison

| Framework | Vector Store | Embedding Model | Chunking Strategy | RAG Integration |
|---|---|---|---|---|
| **Agno** | ChromaDb (native) | AzureOpenAIEmbedder / local | Fixed size: 1024 tokens, 50 overlap | Native `knowledge=` parameter (agentic RAG) |
| **CrewAI** | Internal (managed) | Internal (managed) | Internal (managed) | `StringKnowledgeSource` passed to agent |
| **LangChain** | Chroma | AzureOpenAIEmbeddings / FastEmbed | Raw documents (no splitting) | Retriever wrapped as `@langchain_tool` |
| **LangGraph** | Chroma | AzureOpenAIEmbeddings / FastEmbed | Commented-out RecursiveTextSplitter | `create_retriever_tool()` native integration |
| **LlamaIndex** | VectorStoreIndex (in-memory) | OpenAIEmbedding / local | SentenceSplitter: 1024, 50 overlap | `RetrieverTool` (raw retrieval, no LLM reasoning) |

### RAG Integration Approaches

1. **Native knowledge** (Agno): RAG is a first-class concept. Pass `knowledge=knowledge_base` to the agent constructor. The agent automatically decides when to search the knowledge base.

2. **Knowledge sources** (CrewAI): `StringKnowledgeSource` objects are passed to the agent. CrewAI manages the vectorization and retrieval internally.

3. **Tool-wrapped retriever** (LangChain, LangGraph): The retriever is wrapped as a regular tool. The agent calls it like any other tool. This gives the agent explicit control over when to retrieve.

4. **RetrieverTool** (LlamaIndex): Similar to the tool approach but uses LlamaIndex's native `RetrieverTool` class. Deliberately uses `RetrieverTool` instead of `QueryEngineTool` to return raw retrieved text without additional LLM reasoning.

---

## Multi-Agent Support

The `benchmark_multi_agent.py` script tests three frameworks with multi-agent capabilities:

### CrewAI Multi-Agent

- **Pattern**: Researcher agent + Analyst agent in a sequential `Crew`
- **Execution**: `crew.kickoff()` with `Process.sequential`
- **Communication**: Task output chaining (output of one task feeds into the next)
- **Strengths**: Natural workflow orchestration, clear role separation
- **Limitations**: Sequential only in basic mode, verbose logging

### LangGraph Multi-Agent

- **Pattern**: Single `create_react_agent` with tools (simulates multi-agent via tool calls)
- **Execution**: `graph.invoke()` with tool-augmented reasoning
- **Communication**: Graph state passing between nodes
- **Strengths**: Fine-grained control, graph-based state management
- **Limitations**: Not truly multi-agent in the basic implementation

### OpenAI Agents SDK Multi-Agent

- **Pattern**: Analyst agent with `handoffs=[researcher]` for delegation
- **Execution**: `Runner.run_sync()` with automatic handoff routing
- **Communication**: Native handoff mechanism between agents
- **Strengths**: Cleanest multi-agent API, automatic routing
- **Limitations**: OpenAI-only, limited customization of handoff logic

### Other Frameworks with Multi-Agent Potential

| Framework | Multi-Agent Support | Notes |
|---|---|---|
| AG2 | Yes (native) | `ConversableAgent` group chats, not benchmarked |
| Google ADK | Yes (sub-agents) | `LlmAgent` with `sub_agents=[]` |
| Strands | Yes (swarm) | Agent-to-agent delegation |
| LlamaIndex | Yes (workflows) | Workflow-based orchestration |
| Microsoft AF | Planned | Framework is in RC |

---

## Historical Benchmark Results

The following results were collected prior to the framework expansion, using Azure OpenAI with GPT-4o-mini. They remain valid as baselines for the original four frameworks.

### Response Time with Memory (Web Search)

**Prompt:** _search the web for who won the Champions League final in 2024?_

| Iterations | Agno | LangGraph | LlamaIndex |
|---|---|---|---|
| 20x | 5.41 +/- 1.19s | 6.04 +/- 2.61s | 5.36 +/- 2.02s |
| 30x | 5.84 +/- 1.01s | 6.17 +/- 1.14s | 5.32 +/- 2.26s |
| 50x | 4.24 +/- 0.78s | 8.48 +/- 2.56s | 3.00 +/- 3.24s |
| 100x | 4.39 +/- 0.73s | 9.45 +/- 4.73s | 2.64 +/- 2.29s |

**Notable:** LangGraph shows increasing response times at higher iterations due to memory accumulation. LlamaIndex maintains consistent performance.

### Response Time without Memory (Web Search)

**Prompt:** _search the web for who won the Champions League final in 2024?_

| Iterations | Agno | LangGraph | LlamaIndex | OpenAI (raw) |
|---|---|---|---|---|
| 50x | 4.58 +/- 1.03s | 4.22 +/- 1.11s | 4.12 +/- 1.01s | 3.83 +/- 0.99s |
| 100x | 4.28 +/- 0.76s | 3.31 +/- 0.59s | 3.63 +/- 0.66s | 3.61 +/- 0.83s |

### Token Usage (100x, without Memory)

| Metric | Agno | LangGraph | LlamaIndex | OpenAI (raw) |
|---|---|---|---|---|
| Prompt Tokens | 1999.2 | 1946.1 | 2121.7 | 1888.5 |
| Completion Tokens | 65.3 | 53.5 | 76.9 | 58.3 |
| Total Tokens | 2064.5 | 1999.7 | 2198.6 | 1946.7 |

### RAG Performance (100x)

**Prompt:** _Ball possession in Benfica's game?_

| Metric | Agno | LangGraph | LlamaIndex |
|---|---|---|---|
| Response Time | 3.30 +/- 0.75s | 2.68 +/- 1.35s | 2.86 +/- 1.05s |
| Total Tokens | 4439.3 | 4877.2 | 3279.9 |
| Misses | 2/100 | 4/100 | 2/100 |

**Prompt:** _Benfica's UCL match score?_

| Metric | Agno | LangGraph | LlamaIndex |
|---|---|---|---|
| Response Time | 3.17 +/- 0.74s | 2.43 +/- 1.09s | 2.74 +/- 0.79s |
| Total Tokens | 4515.9 | 5053.3 | 3341.0 |
| Misses | 0/100 | 0/100 | 0/100 |

### API Performance (100x)

**Prompt:** _Tell me the waiting time at the CG station and the status of the red line, and also give me information about Formula 1 driver number 44!_

| Metric | Agno | LangGraph | LlamaIndex |
|---|---|---|---|
| Response Time | 5.49 +/- 1.40s | 4.24 +/- 1.35s | 6.41 +/- 2.47s |
| Total Tokens | 1849.2 | 1412.2 | 3913.4 |
| Misses | 0/100 | 0/100 | 0/100 |

---

### Tool-Calling Benchmark (All 15 Frameworks)

**Date:** April 11, 2026
**Model:** gpt-4.1-mini (OpenAI)
**Prompt:** _"What day of the week is it today?"_ — requires the agent to infer that `date_tool` should be called, then compute the day of the week from the returned date.
**Iterations:** 10 per framework, fresh agent each iteration (no memory)
**Detection method:** Content-based — checks if the response contains today's actual date or the correct day of the week.

#### Response Time & Tool-Call Reliability

| Framework | Response Time (mean +/- std) | Tool Calls | Errors |
|---|---|---|---|
| AG2 | 1.34 +/- 0.09s | 10/10 | 0 |
| Agno | 2.01 +/- 0.27s | 10/10 | 0 |
| Claude SDK | 5.61 +/- 0.45s | 10/10 | 0 |
| CrewAI | 1.64 +/- 0.17s | 10/10 | 0 |
| Google ADK | 1.35 +/- 0.25s | 10/10 | 0 |
| LangChain | 1.45 +/- 0.38s | 10/10 | 0 |
| LangGraph | 1.41 +/- 0.26s | 10/10 | 0 |
| LlamaIndex | 1.28 +/- 0.14s | 10/10 | 0 |
| LlamaIndex FC | 1.67 +/- 0.40s | 10/10 | 0 |
| Microsoft AF | 2.82 +/- 1.20s | 10/10 | 0 |
| OpenAI (Raw) | 1.50 +/- 0.25s | 10/10 | 0 |
| OpenAI Agents SDK | 1.80 +/- 0.38s | 10/10 | 0 |
| PydanticAI | 1.32 +/- 0.07s | 10/10 | 0 |
| Smolagents | 3.41 +/- 0.53s | 10/10 | 0 |
| Strands | 2.40 +/- 0.23s | 10/10 | 0 |

#### Token Usage (Average per Call)

After two rounds of targeted token-tracking research and fixes (see [Token Tracking Research](#token-tracking-research) below), **all 15 frameworks now report token usage**.

| Framework | Avg Prompt Tokens | Avg Completion Tokens | Avg Total Tokens | Notes |
|---|---|---|---|---|
| Smolagents | 5,088 | 208 | 5,296 | CodeAgent multi-step overhead |
| CrewAI | 1,158 | 50 | 1,207 | Task/crew prompt overhead |
| Agno | 843 | 18 | 861 | Via `RunMetrics` dataclass |
| Google ADK | 716 | 19 | 735 | Via `event.usage_metadata` |
| OpenAI Agents SDK | 696 | 24 | 720 | Via `RunResult.raw_responses[].usage` |
| Microsoft AF | 696 | 19 | 715 | Via `AgentResponse.usage_details` |
| PydanticAI | 699 | 16 | 715 | Via `result.usage()` |
| Strands | 693 | 18 | 711 | Via `result.metrics.get_summary()` |
| OpenAI (Raw) | 684 | 20 | 704 | Direct from API response |
| AG2 | 683 | 18 | 701 | Via `agent.get_total_usage()` |
| LangChain | 681 | 17 | 698 | Via `response_metadata` |
| LangGraph | 681 | 17 | 698 | Via `response_metadata` |
| LlamaIndex | 642 | 10 | 652 | Client-side tiktoken |
| LlamaIndex FC | 640 | 10 | 649 | Client-side tiktoken |
| Claude SDK | 3 | 38 | 41 | Partial — only `ResultMessage.usage` (output-heavy) |

#### Token Tracking Research

Two rounds of systematic investigation were conducted across all 8 frameworks that initially reported 0 tokens. Each framework's documentation (README files in their respective example folders) was searched for token usage tracking APIs, and the framework source code was inspected where documentation was lacking.

**Round 1 — Fixes applied (4 frameworks):**

| Framework | Has Token API? | Fix Applied | Result |
|---|---|---|---|
| **OpenAI Agents SDK** | Yes | Fixed field names: `input_tokens`/`output_tokens` (not `prompt_tokens`/`completion_tokens`) on `RunResult.raw_responses[].usage` | 0 → **721** tokens |
| **Strands** | Yes | Fixed access pattern: `result.metrics.get_summary()["accumulated_usage"]` with camelCase keys (`inputTokens`, `outputTokens`, `totalTokens`) instead of treating `metrics` as a dict | 0 → **712** tokens |
| **Smolagents** | Yes | Fixed extraction: iterate `agent.memory.steps` and sum `step.token_usage.input_tokens`/`output_tokens` from `ActionStep` objects (not `model.last_input_token_count`) | 0 → **5,776** tokens |
| **Claude SDK** | Yes (partial) | Fixed: switched to `asyncio.run()`, extract `usage` dict from `ResultMessage` (keys: `input_tokens`, `output_tokens`). Only reports output-heavy counts; input tokens are minimal because the SDK doesn't expose full prompt tokenization | 0 → **62** tokens |

**Round 2 — Fixes applied (4 frameworks):**

| Framework | Has Token API? | Fix Applied | Result |
|---|---|---|---|
| **AG2** | Yes | `ConversableAgent.get_total_usage()` returns a nested dict keyed by model name with `prompt_tokens`/`completion_tokens`/`total_tokens`. Previously tried non-existent `result.usage` | 0 → **701** tokens |
| **Agno** | Yes | `RunOutput.metrics` is a `RunMetrics` dataclass (not a dict) with `input_tokens`/`output_tokens`/`total_tokens` fields. Previously used old dict-based `response.metrics.get("prompt_tokens", [0])` pattern which returned 0 | 0 → **861** tokens |
| **Google ADK** | Yes | `Event.usage_metadata` is a `GenerateContentResponseUsageMetadata` with `prompt_token_count`/`candidates_token_count`/`total_token_count`. Previously checked non-existent `event.usage` | 0 → **735** tokens |
| **Microsoft AF** | Yes | `AgentResponse.usage_details` is a `UsageDetails` TypedDict with `input_token_count`/`output_token_count`/`total_token_count`. Fallback: scan `message.contents` for `Content` items with `type == "usage"`. Previously checked non-existent `result.usage` attribute | 0 → **715** tokens |

**Summary:** **15/15 frameworks now report token usage** (7 original + 4 fixed in round 1 + 4 fixed in round 2). The final 4 frameworks (AG2, Agno, Google ADK, Microsoft AF) all had token APIs, but they were not obvious from main documentation — they required source code inspection and runtime introspection to discover.

#### Key Observations

1. **100% tool-call reliability**: All 15 frameworks correctly identified the need to call `date_tool` and returned the correct day of the week in all 150 runs. This validates that `gpt-4.1-mini` with proper tool registration is highly reliable for single-tool invocations.

2. **Fastest frameworks (< 1.55s mean)**: LlamaIndex (1.28s), PydanticAI (1.32s), AG2 (1.34s), Google ADK (1.35s), and LangGraph (1.41s). These share a common trait: minimal framework overhead around the LLM call.

3. **Slowest frameworks (> 3s mean)**: Claude SDK (5.61s), Smolagents (3.41s), and Microsoft AF (2.82s).
   - Claude SDK's overhead is expected — it calls the Anthropic API (Claude model) rather than OpenAI, with a cross-provider round-trip.
   - Smolagents uses a CodeAgent that generates and executes Python code, adding parsing and execution overhead. This also explains its 8x token usage (5,296 vs ~700 for direct tool-calling frameworks).
   - Microsoft AF had intermittent high-latency runs (5.85s) causing a high std dev (1.20s).

4. **Most consistent (lowest std dev)**: PydanticAI (0.07s), AG2 (0.09s), and LlamaIndex (0.14s) showed the most predictable response times.

5. **Token efficiency tiers**:
   - **Efficient** (~650-720 tokens): LlamaIndex, LangChain, LangGraph, OpenAI Raw, AG2, PydanticAI, Strands, Microsoft AF, OpenAI Agents SDK, Google ADK — all use direct tool-calling with minimal prompt overhead.
   - **Moderate** (~860-1,200 tokens): Agno (~861 tokens, slightly higher prompt due to instructions formatting), CrewAI (~1,207 tokens, task/crew abstraction adds substantial prompt overhead).
   - **Heavy** (~5,300 tokens): Smolagents — CodeAgent generates Python code, often with retries on format errors, causing 8x more tokens than direct tool-calling frameworks.
   - **Partial** (~41 tokens): Claude SDK — only reports output tokens from `ResultMessage.usage`; input token count is not fully exposed by the SDK.

6. **Token tracking coverage**: After two rounds of research and fixes, **15/15 frameworks (100%) now report token usage**. All 8 frameworks that initially reported 0 tokens had viable token APIs — they were just not obvious from main documentation and required source code inspection to discover. The most obscure were AG2 (`get_total_usage()` on agent, not on result), Agno (API migration from dict to dataclass), Google ADK (`usage_metadata` field name differs from OpenAI convention), and Microsoft AF (`usage_details` TypedDict on response).

---

## Key Findings

### 1. Implementation Simplicity vs. Feature Richness

There is a clear trade-off between simplicity and capability:
- **Simplest** (< 150 LOC): OpenAI Agents SDK, Claude SDK, CrewAI, AG2, Microsoft AF
- **Most feature-rich** (> 200 LOC): LangGraph, LlamaIndex, OpenAI (raw), Agno
- The raw OpenAI API requires the most code but gives the most control

### 2. Async is Becoming the Norm

Five frameworks are async-native (Claude SDK, Google ADK, LlamaIndex, Microsoft AF, PydanticAI), requiring either `asyncio` wrappers or framework-provided sync helpers. PydanticAI and OpenAI Agents SDK offer the cleanest approach with built-in `run_sync()` methods. LlamaIndex requires the `handler = agent.run(...); await handler` pattern within `asyncio.run()`. Claude SDK, Google ADK, and Microsoft AF require `asyncio.run()` with an inner async function.

### 3. Tool Registration is Fragmented

Every framework has its own decorator or registration pattern. There is no standard for tool definition across frameworks. This makes tool portability between frameworks difficult. Google ADK and AG2 are the simplest (plain functions), while Smolagents and Strands require specific docstring formats.

### 4. Memory Management Varies Widely

- **Best**: Agno (built-in, configurable), LlamaIndex (token-limited buffer), LangGraph (checkpoint-based)
- **Adequate**: AG2, Smolagents, Strands (framework-managed lists)
- **Manual**: PydanticAI, Claude SDK, OpenAI raw (developer manages message history)
- **None**: OpenAI Agents SDK, Microsoft AF, CrewAI (task-based, no conversational memory)

### 5. RAG Integration Maturity

- **Agno** has the most natural RAG integration (native `knowledge` parameter)
- **LlamaIndex** provides the most control (separate `RetrieverTool` vs `QueryEngineTool`)
- **CrewAI** abstracts RAG almost entirely (`StringKnowledgeSource`)
- **LangChain/LangGraph** treats RAG as just another tool (retriever-as-tool pattern)

### 6. Token Tracking is Not Standardized

Every framework exposes tokens differently: five distinct extraction patterns across 20 agents (see [Token Tracking Accuracy Deep-Dive](#token-tracking-accuracy-deep-dive)). Field names vary between OpenAI-style (`prompt_tokens`), Anthropic-style (`input_tokens`), AWS-style (`inputTokens`), and PydanticAI-style (`request_tokens`). Safety levels range from inner try/except with multiple fallbacks to direct attribute access with no guards. Only LlamaIndex tracks embedding tokens; all other agents — including RAG agents that actively use embeddings — report 0 for embedding costs.

After two rounds of targeted research and fixes, **all 15 frameworks now report token usage** (previously only 7). The final 4 frameworks (AG2, Agno, Google ADK, Microsoft AF) all had token APIs that were not obvious from main documentation — they required source code inspection and runtime introspection to discover. Smolagents — previously capturing only the last LLM call's tokens — now correctly sums `ActionStep.token_usage` across all steps, revealing that its CodeAgent architecture uses 8x more tokens (~5,296) than direct tool-calling frameworks (~700).

### 7. Performance Across All 15 Frameworks

From the tool-calling benchmark (10 iterations x 15 agents, gpt-4.1-mini):
- **All 15 frameworks achieved 100% tool-call reliability** (150/150 successful invocations)
- **Fastest**: LlamaIndex (1.28s), PydanticAI (1.32s), AG2 (1.34s), Google ADK (1.35s), LangGraph (1.41s) — all under 1.45s mean
- **Slowest**: Claude SDK (5.61s, cross-provider), Smolagents (3.41s, code generation overhead), Microsoft AF (2.82s, intermittent latency spikes)
- **Most consistent**: PydanticAI (0.07s std dev), AG2 (0.09s), LlamaIndex (0.14s)
- **Most token-efficient**: LlamaIndex (~652 total), LlamaIndex FC (~649 total)
- **Least token-efficient**: Smolagents (~5,296 total, 8x more than average due to CodeAgent), CrewAI (~1,207 total, 2x more due to task/crew prompt overhead)
- **Token tracking**: **15/15 frameworks report tokens** (up from 7/15 after two rounds of targeted fixes)

### 8. Historical Performance Patterns

From the earlier benchmark data on the original four frameworks:
- **LangGraph** suffers from memory accumulation over many iterations
- **LlamaIndex** is the most token-efficient for RAG tasks
- **Agno** provides the most consistent results across iterations
- **OpenAI raw API** has competitive response times but lacks framework conveniences

---

## Recommendations

### For Getting Started Quickly

**PydanticAI** or **OpenAI Agents SDK** -- Both are concise, well-documented, and have clean APIs. PydanticAI is more flexible (multiple providers), while OpenAI Agents SDK has the simplest multi-agent handoff pattern.

### For Production RAG Systems

**LlamaIndex** or **Agno** -- LlamaIndex gives the most control over the retrieval pipeline. Agno provides the most seamless integration where RAG is a native concept, not a bolted-on tool.

### For Multi-Agent Workflows

**CrewAI** or **OpenAI Agents SDK** -- CrewAI excels at structured task workflows with role-based agents. OpenAI Agents SDK offers clean agent handoffs for dynamic routing.

### For Maximum Flexibility

**LangGraph** -- The graph-based architecture allows custom state management, complex control flows, and fine-grained agent behavior. Higher learning curve but maximum customization.

### For Enterprise/Cloud Integration

**Google ADK** (GCP), **Strands** (AWS), **Microsoft AF** (Azure) -- Each is designed for its respective cloud ecosystem with native integrations.

### For Lightweight/Experimental Use

**Smolagents** -- HuggingFace's lightweight agent library. Good for experimentation and prototyping with open-source models.

---

## Appendix: File Reference

### Basic Agent Files

| File | Framework | Lines |
|---|---|---|
| `ag2_agent.py` | AG2 | 167 |
| `agno_agent.py` | Agno | 198 |
| `claude_sdk_agent.py` | Claude Agent SDK | 176 |
| `crewai_agent.py` | CrewAI | 144 |
| `google_adk_agent.py` | Google ADK | 171 |
| `langchain_agent.py` | LangChain | 161 |
| `langgraph_agent.py` | LangGraph | 232 |
| `llama_index_agent.py` | LlamaIndex (FunctionAgent) | 206 |
| `llama_index_fc_agent.py` | LlamaIndex (FunctionAgent) | 205 |
| `microsoft_agent.py` | Microsoft Agent Framework | 167 |
| `openai_agent.py` | OpenAI (raw API) | 280 |
| `openai_agents_sdk_agent.py` | OpenAI Agents SDK | 137 |
| `pydantic_ai_agent.py` | PydanticAI | 186 |
| `smolagents_agent.py` | Smolagents | 177 |
| `strands_agent.py` | Strands Agents SDK | 158 |

### RAG+API Agent Files

| File | Framework | Lines |
|---|---|---|
| `agno_rag_api_agent.py` | Agno | 266 |
| `crewai_rag_api_agent.py` | CrewAI | 191 |
| `langchain_rag_api_agent.py` | LangChain | 209 |
| `langgraph_rag_api_agent.py` | LangGraph | 279 |
| `llama_index_rag_api_agent.py` | LlamaIndex | 238 |

### Benchmark Scripts

| File | Purpose | Lines |
|---|---|---|
| `benchmark_runner.py` | Standardized benchmark suite (basic/rag/api/target) | 486 |
| `benchmark_multi_agent.py` | Multi-agent system benchmark | 423 |
| `benchmark_tool_calling.py` | Tool-calling reliability benchmark (all 15 agents) | 281 |

### Infrastructure

| File | Purpose |
|---|---|
| `settings.py` | Environment configuration (pydantic-settings) |
| `prompts.py` | Shared system prompts |
| `utils.py` | CLI parsing, REPL execution, token utilities |
| `shared_functions/` | F1API, MetroAPI, Generic tools |
| `knowledge_base/` | Champions League match data for RAG |
| `agent-ui.py` | Streamlit chat UI |
