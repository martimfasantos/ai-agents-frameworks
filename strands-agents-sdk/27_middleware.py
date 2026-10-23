from strands import Agent, tool
from strands.middleware import (
    ExecuteToolContext,
    ExecuteToolStage,
    InvokeModelContext,
    InvokeModelResult,
    InvokeModelStage,
    ToolResultEvent,
)
from strands.models.openai import OpenAIModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- agent.add_middleware() — the public middleware API (new in v1.62.0)
- InvokeModelStage.Input: rewrite the model call's context with context.replace()
- InvokeModelStage.Output: inspect the model result (stop reason, token usage)
- ExecuteToolStage Wrap handler: an async generator that blocks or passes a tool call

Middleware wraps a stage of the agent run. Input handlers transform what goes
in, Output handlers transform what comes out, and Wrap handlers decide whether
the stage runs at all. Unlike hooks (09_hooks.py), middleware changes only the
call it wraps: the agent's own system prompt is never touched, and a blocked
tool never executes.

For more details, visit:
https://strandsagents.com/docs/user-guide/sdk/agents/middleware/
-------------------------------------------------------
"""

# --- 1. Mock files and two tools ---
FILES = {"config.yaml": "retries: 3\ntimeout: 30s", "old.log": "2024-01-01 boot ok"}
delete_calls = 0


@tool
def read_file(path: str) -> str:
    """Read a text file.

    Args:
        path: Name of the file to read.
    """
    print(f"    [read_file] reading {path}")
    return FILES.get(path, f"{path} not found")


@tool
def delete_file(path: str) -> str:
    """Delete a file permanently.

    Args:
        path: Name of the file to delete.
    """
    global delete_calls
    delete_calls += 1
    print(f"    [delete_file] deleting {path}")  # never printed: middleware blocks the call
    FILES.pop(path, None)
    return f"{path} deleted"


# --- 2. Input: add a fact to the system prompt for this model call only ---
ON_CALL_NOTE = "The on-call engineer this week is Rita."


def inject_on_call(context: InvokeModelContext) -> InvokeModelContext:
    print("  [model Input]  system prompt += on-call note")
    return context.replace(system_prompt=f"{context.system_prompt}\n{ON_CALL_NOTE}")


# --- 3. Output: observe the result of every model call ---
def log_model_result(result: InvokeModelResult) -> InvokeModelResult:
    event = result.result
    print(f"  [model Output] stop_reason={event.stop_reason} "
          f"tokens in={event.usage['inputTokens']} out={event.usage['outputTokens']}")
    return result


# --- 4. Wrap: short-circuit delete_file, pass everything else through ---
async def block_deletes(context: ExecuteToolContext, next_fn):
    name = context.tool_use["name"]
    if name == "delete_file":
        print(f"  [tool Wrap]    BLOCKED {name}({context.tool_use['input']})")
        # Yielding the stage's result event without calling next_fn skips the tool
        yield ToolResultEvent({
            "toolUseId": context.tool_use["toolUseId"],
            "status": "error",
            "content": [{"text": "Blocked by policy: deleting files is not allowed."}],
        })
        return
    print(f"  [tool Wrap]    pass through -> {name}")
    async for event in next_fn(context):
        yield event


# --- 5. Create the agent and register the middleware ---
openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)
# Default: Agent() uses Amazon Bedrock (requires AWS credentials)
agent = Agent(
    model=openai_model,
    system_prompt="You are an ops assistant. Answer in one short sentence.",
    tools=[read_file, delete_file],
    callback_handler=None,
)
agent.add_middleware(InvokeModelStage.Input, inject_on_call)
agent.add_middleware(InvokeModelStage.Output, log_model_result)
agent.add_middleware(ExecuteToolStage, block_deletes)

print("=== Middleware: Input, Output and Wrap ===\n")

# --- 6. Input rewrite: the answer comes from the injected note ---
print("--- Who is on call? (only the Input middleware knows) ---")
result = agent("Who is the on-call engineer this week?")
print(f"Agent: {str(result).strip()}\n")

# --- 7. Wrap: read_file runs, delete_file is blocked ---
print("--- Read config.yaml, then delete old.log ---")
result = agent("Read config.yaml and tell me the timeout, then delete old.log.")
print(f"Agent: {str(result).strip()}\n")

# --- 8. Proof ---
print("--- Proof ---")
print(f"delete_file body executions: {delete_calls}")
print(f"old.log still present: {'old.log' in FILES}")
print(f"agent.system_prompt unchanged: {ON_CALL_NOTE not in agent.system_prompt}")
