# LlamaIndex

- Repo: https://github.com/run-llama/llama_index
- Documentation: https://developers.llamaindex.ai/

## What is LlamaIndex?

LlamaIndex (formerly GPT Index) is a data framework for LLM applications. It provides tools to build agentic systems with a focus on Retrieval-Augmented Generation (RAG). 

This folder contains **simple and straightforward examples** that demonstrate LlamaIndex's core features. Each example is focused, minimal, and easy to understand.

Key strengths include:
- **RAG-first architecture**: Built for document-based reasoning and retrieval
- **Event-driven workflows**: Flexible, composable workflow orchestration
- **Agent tools**: Function calling and query engines as tools
- **Memory management**: Conversation history and context persistence
- **Streaming**: First-class streaming support for tokens and events
- **Community tools**: 40+ integrations available via LlamaHub

## LlamaIndex Examples

### How to setup

#### Create a virtual environment and install dependencies

Run the following command to create a virtual environment and install dependencies using UV:

```bash
uv sync
```

Alternatively, use UV to run files directly without manual activation:
```bash
uv run <example_name>.py
```

#### .env

See .env.example and create a .env (on the root of the repository).
You need to get an OpenAI endpoint and key and fill them in.

### Example Progression

**Core Fundamentals** (Required):
- `00_hello_world.py` - Basic RAG with document loading and querying
- `01_tools.py` - Function calling agents with custom tools
- `02_structured_outputs.py` - Enforcing response schemas with Pydantic
- `03_memory.py` - Conversation memory and context management
- `04_streaming.py` - Real-time event and token streaming
- `05_memory_advanced.py` - Memory with initial_messages for persistent context

**Agent Capabilities** (Recommended):
- `06_agent_types.py` - Different agent implementations (ReAct, Function Calling)
- `07_multi_modal_agents.py` - Vision agents for image understanding
- `08_manual_agents.py` - Manual agent control with step execution
- `09_agent_delegation.py` - Wrapping agents as tools for delegation
- `10_agentic_rag.py` - Multiple query engines with intelligent tool selection
- `11_agent_workflows.py` - Complete workflows combining RAG and reasoning

### Workflow Examples (agent_workflows/)

**Getting Started**:
- `01_gettings_started.py` - Basic workflow with events and steps
- `02_branches_and_loops.py` - Conditional branching and looping patterns

**State & Data Flow**:
- `03_managing_state.py` - Context state (typed/untyped, locking, persistence)
- `04_streaming.py` - Streaming progress events and handling termination
- `05_concurrent_execution.py` - Parallel step execution and event collection

**Advanced Patterns**:
- `06_human_in_the_loop.py` - InputRequiredEvent for human interaction
- `07_customizing_entry_exit_points.py` - Custom StartEvent/StopEvent for type safety
- `08_drawing_workflow.py` - Visualizing workflows (HTML export, debugger UI)
- `09_resource_objects.py` - Dependency injection with Resource/ResourceConfig
- `10_retry_steps_execution.py` - Retry policies for transient failures
- `11_workflow_as_a_server.py` - Exposing workflows via HTTP API
- `12_observability.py` - OpenTelemetry tracing and observability tools

### Key LlamaIndex Differentiators

| Aspect | LlamaIndex Specialty |
|--------|----------------------|
| **Document Processing** | Native RAG with indices, chunking, metadata |
| **Query Flexibility** | Router engines, multiple query strategies |
| **Memory System** | Sophisticated blocks with semantic memory |
| **Workflows** | Event-driven with full custom control |
| **Community** | 40+ tools via LlamaHub ecosystem |
| **Streaming** | First-class event + token streaming |
| **Enterprise** | LlamaParse for complex documents, multimodal support |

### Documentation References

- Main Docs: https://developers.llamaindex.ai/
- Agents Guide: https://developers.llamaindex.ai/python/framework/use_cases/agents/
- Workflows: https://developers.llamaindex.ai/python/llamaagents/workflows/
- Workflow API Reference: https://developers.llamaindex.ai/python/workflows-api-reference/
- Memory: https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/
- Streaming: https://developers.llamaindex.ai/python/framework/understanding/agent/streaming/
- LlamaHub: https://llamahub.ai/
