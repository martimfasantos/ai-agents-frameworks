import asyncio
from workflows import Workflow, step
from workflows.context import Context
from workflows.events import Event, StartEvent, StopEvent
from llama_agents.server import WorkflowServer


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Exposing workflows over HTTP with WorkflowServer
- Running the server programmatically or via CLI
- Workflow Debugger UI for visualization and debugging
- API endpoints: /run, /run-nowait, /events, /results, /handlers
- Streaming events via SSE or NDJSON
- Sending events to running workflows (human-in-the-loop via API)
- Canceling workflow runs
- Using WorkflowClient for programmatic server interaction

The WorkflowServer class exposes workflows over a RESTful HTTP API.
It includes a debugger UI at the root / path for visualizing, running,
and debugging workflows in real time. Workflows can be run synchronously
(/run) or asynchronously (/run-nowait), with events streamed via
/events/{handler_id}. The WorkflowClient provides a Python interface
for listing workflows, running them, streaming events, and sending
human-in-the-loop responses programmatically.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/deployment/
-------------------------------------------------------
"""


# --- 1. Define a streaming event and a simple workflow ---
class StreamEvent(Event):
    sequence: int


class GreetingWorkflow(Workflow):
    @step
    async def greet(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """Greet the user, streaming progress events"""
        for i in range(3):
            ctx.write_event_to_stream(StreamEvent(sequence=i))
            await asyncio.sleep(0.3)

        name = ev.get("name", "World")
        return StopEvent(result=f"Hello, {name}!")


# --- 2. Create a WorkflowServer and add workflows ---
greet_wf = GreetingWorkflow()

server = WorkflowServer()
server.add_workflow("greet", greet_wf)


# --- 3. Run the server programmatically ---
async def main():
    await server.serve(host="0.0.0.0", port=8080)

if __name__ == "__main__":
    asyncio.run(main())

# --- 4. Run the server via CLI ---
# Run the server with:
#   uv run 11_workflow_as_a_server.py
#
# The server starts on 0.0.0.0:8080 by default.
# Configure with WORKFLOWS_PY_SERVER_HOST and WORKFLOWS_PY_SERVER_PORT env vars.

# --- 5. Debugger UI ---
# Open http://localhost:8080 to see the Workflow Debugger UI.
# Features: workflow visualization, event logging, human-in-the-loop support,
# multiple runs tracking, and automatic schema detection.

# --- 6. API Endpoints ---
# GET  /health                    → {"status": "healthy"}
# GET  /workflows                 → list of registered workflow names
# POST /workflows/{name}/run      → run synchronously, returns result
# POST /workflows/{name}/run-nowait → run async, returns handler_id
# GET  /events/{handler_id}       → stream events (SSE or NDJSON)
# POST /events/{handler_id}       → send an event (human-in-the-loop)
# GET  /handlers/{handler_id}     → get workflow result (202 if still running)
# GET  /handlers                  → list all handlers (running + completed)
# POST /handlers/{handler_id}/cancel → cancel a running workflow