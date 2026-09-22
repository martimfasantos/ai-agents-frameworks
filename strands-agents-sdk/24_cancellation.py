import os
import threading
import time

from strands import Agent, tool
from strands.models.openai import OpenAIModel

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Strands Agents with the following features:
- Agent.cancel() to stop a run in flight, from any thread
- agent.cancel_signal, the event tools and hooks can check
- tool_context.cancel_signal for cooperative cancellation inside a tool

A long agent run sometimes has to be abandoned — the user closed the tab,
a deadline passed, a newer request superseded it. Agent.cancel() is
thread-safe and stops the agent at the next safe point: during model
streaming, before or after tool execution, or inside an SDK-built tool
that checks its cancel_signal. Long-running tools should poll that
signal so they stop promptly instead of running to completion.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/agents/agent-loop/
-------------------------------------------------------
"""

model = OpenAIModel(
    client_args={"api_key": settings.OPENAI_API_KEY.get_secret_value()},
    model_id=settings.OPENAI_MODEL_NAME,
)


# --- 1. A slow tool that cooperates with cancellation ---
@tool(context=True)
def slow_audit(records: int, tool_context) -> str:
    """Audit a number of records. Takes about a second per record."""
    for i in range(records):
        # Checking the signal is what makes cancellation prompt rather than
        # "whenever this tool happens to finish".
        if tool_context.cancel_signal.is_set():
            print(f"  [tool] cancelled after {i} of {records} records")
            return f"Audit cancelled after {i} records."
        time.sleep(1)
        print(f"  [tool] audited record {i + 1}/{records}")
    return f"Audited all {records} records."


agent = Agent(
    model=model,
    tools=[slow_audit],
    system_prompt="Use the slow_audit tool when asked to audit. Be brief.",
)

# --- 2. Cancel from another thread while the agent works ---
print("=== Cancelling a run in flight ===\n")


def cancel_after(seconds: float) -> None:
    time.sleep(seconds)
    print(f"  [main] calling agent.cancel() after {seconds}s")
    agent.cancel()


watcher = threading.Thread(target=cancel_after, args=(3.5,), daemon=True)
watcher.start()

started = time.monotonic()
result = agent("Audit 30 records.")
elapsed = time.monotonic() - started

print(f"\nStopped after {elapsed:.1f}s (the tool alone would need ~30s)")
# A cancelled run returns whatever the agent had produced, which for a
# cancellation during tool use is usually nothing.
print(f"Result text: {str(result).strip()!r}")
print(f"Stop reason: {result.stop_reason}")

# --- 3. The signal clears, so the agent stays reusable ---
print(f"\ncancel_signal set after the run: {agent.cancel_signal.is_set()}")
# callback_handler=None silences the default token stream so the reply is
# printed once, by us.
reused = Agent(model=model, callback_handler=None)
print(f"Reused agent: {reused('Reply with the single word: ready')}")
