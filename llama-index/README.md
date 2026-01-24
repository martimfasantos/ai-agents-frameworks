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

#### Virtual environment

Create a simple virtual environment with:

```bash
python3 -m venv .venv
```

Then activate it with:
```bash
# On Linux/macOS
source .venv/bin/activate
# On Windows
.venv\Scripts\activate
```

And install the requirements with:
```bash
pip install -r requirements.txt
```

#### .env

See .env.example and create a .env (on the root of the repository).
You need to get an OpenAI endpoint and key and fill them in.

### Example Progression

**Core Fundamentals** (Required):
- `00_hello_world.py` - Basic RAG with document loading and querying
- `01_tools.py` - Function calling agents with custom tools
- `02_structured_outputs.py` - Enforcing response schemas with Pydantic

**Agent Capabilities** (Recommended):
- `03_memory.py` - Conversation memory and context management
- `04_streaming.py` - Real-time event and token streaming with custom workflow
- `05_memory_advanced.py` - Memory with initial_messages for persistent context
- `06_agentic_rag.py` - Multiple query engines (vector vs summary) with intelligent tool selection
- `07_async_patterns.py` - Async agent execution with achat
- `08_agent_delegation.py` - Wrapping agents as tools for delegation

**Advanced Patterns** (Optional):
- `09_router_engine.py` - RouterQueryEngine with LLMSingleSelector for automatic index selection
- `10_workflow_custom.py` - Custom Workflow with @step decorator, Events, and Context state

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
- Workflows: https://developers.llamaindex.ai/python/framework/understanding/workflows/
- Memory: https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/
- Streaming: https://developers.llamaindex.ai/python/framework/understanding/agent/streaming/
- LlamaHub: https://llamahub.ai/
