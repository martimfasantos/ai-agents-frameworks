# LlamaIndex Agent Workflows Examples

This directory contains examples demonstrating various features of LlamaIndex Workflows. Each example is self-contained and showcases a specific workflow capability.

## Overview

Workflows in LlamaIndex provide a flexible way to orchestrate complex multi-step operations with features like branching, looping, state management, concurrency, and more.

## Examples

### 1. Branches and Loops (`branches_and_loops.py`)
Demonstrates conditional branching and looping patterns in workflows.

**Features:**
- Random path selection between branches
- Dynamic loop iteration counts
- Conditional loop termination

**Documentation:** https://developers.llamaindex.ai/python/llamaagents/workflows/branches_and_loops/

---

### 2. Managing State (`managing_state.py`)
Shows how to maintain and share state across workflow steps.

**Features:**
- Context-based state management
- Typed state with Pydantic models
- Atomic state updates with locking
- Persistent state across multiple runs

**Documentation:** https://developers.llamaindex.ai/python/llamaagents/workflows/managing_state/

---

### 3. Streaming (`streaming.py`)
Demonstrates real-time event streaming for progress updates.

**Features:**
- Progress event emission with `write_event_to_stream()`
- Token-by-token LLM response streaming
- Real-time user feedback during execution

**Documentation:** https://developers.llamaindex.ai/python/llamaagents/workflows/streaming/

---

### 4. Concurrent Execution (`concurrent_execution.py`)
Shows how to run workflow steps in parallel.

**Features:**
- Emitting multiple events with `send_event()`
- Concurrent step execution with `num_workers`
- Collecting parallel results with `collect_events()`

**Documentation:** https://developers.llamaindex.ai/python/llamaagents/workflows/concurrent_execution/

---

### 5. Human in the Loop (`human_in_the_loop.py`)
Demonstrates interactive workflows that wait for user input.

**Features:**
- `InputRequiredEvent` for requesting user input
- `HumanResponseEvent` for handling responses
- Pausing and resuming workflow execution

**Documentation:** https://developers.llamaindex.ai/python/llamaagents/workflows/human_in_the_loop/

---

### 6. Customizing Entry/Exit Points (`customizing_entry_exit_points.py`)
Shows how to create custom Start and Stop events for type safety.

**Features:**
- Custom `StartEvent` with typed fields
- Custom `StopEvent` with structured returns
- Better IDE support and type checking

**Documentation:** https://developers.llamaindex.ai/python/llamaagents/workflows/customizing_entry_exit_points/

---

### 7. Drawing Workflows (`drawing.py`)
Demonstrates workflow visualization capabilities.

**Features:**
- Drawing all possible workflow paths
- Exporting diagrams to HTML
- Using WorkflowServer for interactive debugging

**Documentation:** https://developers.llamaindex.ai/python/llamaagents/workflows/drawing/

**Note:** Requires `pip install llama-index-utils-workflow`

---

### 8. Resources (`resources.py`)
Shows how to inject external dependencies into workflow steps.

**Features:**
- Resource injection with factory functions
- Sharing resources across steps
- Resource caching control

**Documentation:** https://developers.llamaindex.ai/python/llamaagents/workflows/resources/

---

### 9. Retry Steps (`retry_steps.py`)
Demonstrates automatic retry mechanisms for handling failures.

**Features:**
- `ConstantDelayRetryPolicy` for fixed intervals
- `ExponentialBackoffRetryPolicy` for rate limiting
- Custom retry policies

**Documentation:** https://developers.llamaindex.ai/python/llamaagents/workflows/retry_steps/

---

### 10. Observability (`observability.py`)
Shows how to integrate observability and tracing.

**Features:**
- OpenTelemetry integration
- Custom spans and events
- Third-party tools (Arize Phoenix, Langfuse)

**Documentation:** https://developers.llamaindex.ai/python/llamaagents/workflows/observability/

## Running the Examples

Each example can be run independently:

```bash
# Navigate to the llama-index directory
cd llama-index

# Run any example
python agent_workflows/branches_and_loops.py
python agent_workflows/managing_state.py
python agent_workflows/streaming.py
# ... etc
```

## Prerequisites

Make sure you have the required dependencies installed:

```bash
# Install LlamaIndex workflows
pip install llama-index-core workflows

# For OpenAI examples
pip install llama-index-llms-openai

# For visualization (optional)
pip install llama-index-utils-workflow

# For observability (optional)
pip install llama-index-observability-otel
```

## Environment Setup

Some examples require API keys. Set up your environment:

```bash
# Create a .env file or export directly
export OPENAI_API_KEY="your-api-key-here"
```

Or configure in `settings.py` as used throughout the llama-index examples.

## Learn More

- **Official Documentation:** https://developers.llamaindex.ai/python/llamaagents/workflows/
- **API Reference:** https://developers.llamaindex.ai/python/workflows-api-reference/
- **LlamaIndex Main Docs:** https://docs.llamaindex.ai/

## File Structure

```
agent_workflows/
├── README.md                          # This file
├── branches_and_loops.py             # Branching and looping
├── managing_state.py                  # State management
├── streaming.py                       # Event streaming
├── concurrent_execution.py            # Parallel execution
├── human_in_the_loop.py              # User interaction
├── customizing_entry_exit_points.py  # Custom events
├── drawing.py                         # Visualization
├── resources.py                       # Dependency injection
├── retry_steps.py                     # Retry policies
└── observability.py                   # Tracing and monitoring
```

## Contributing

These examples follow the structure established in the parent directory's examples (e.g., `00_hello_world.py`, `01_tools.py`). Each file includes:

- Clear docstring explaining the example
- Feature list with bullet points
- Link to official documentation
- Simple, runnable code
- Educational comments
