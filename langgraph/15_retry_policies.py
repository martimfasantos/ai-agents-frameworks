import time
import random

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Configuring RetryPolicy on nodes for automatic retries
- Exponential backoff with jitter
- Handling transient failures gracefully
- Custom retry conditions

RetryPolicy automatically retries failed nodes with configurable
backoff. This is essential for nodes that call external APIs, which may
have transient failures (rate limits, network timeouts). You can control
max attempts, initial/max wait times, backoff multiplier, and which
exceptions trigger retries.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph
-----------------------------------------------------------------------
"""


# --- 1. Define state ---
class ProcessState(TypedDict):
    task: str
    attempt_log: list[str]
    result: str


# --- 2. Simulate a flaky external API ---
call_count = 0


def flaky_api_node(state: ProcessState) -> dict:
    """Simulates an unreliable API that fails on first 2 attempts."""
    global call_count
    call_count += 1

    log_entry = f"Attempt {call_count} at {time.strftime('%H:%M:%S')}"

    if call_count <= 2:
        log_entry += " - FAILED (simulated transient error)"
        print(f"  [!] {log_entry}")
        raise ConnectionError(f"Simulated API failure on attempt {call_count}")

    log_entry += " - SUCCESS"
    print(f"  [+] {log_entry}")
    return {
        "attempt_log": [log_entry],
        "result": f"API returned data for '{state['task']}' after {call_count} attempts",
    }


def process_result_node(state: ProcessState) -> dict:
    """Process the successful API result."""
    return {"result": f"Processed: {state['result']}"}


# --- 3. Build graph with RetryPolicy ---
builder = StateGraph(ProcessState)

# Add the flaky node with a retry policy
builder.add_node(
    "api_call",
    flaky_api_node,
    retry_policy=RetryPolicy(
        max_attempts=5,  # Up to 5 total attempts
        initial_interval=0.5,  # Start with 0.5s wait
        backoff_factor=2.0,  # Double the wait each retry
        max_interval=5.0,  # Cap wait at 5 seconds
        jitter=True,  # Add randomness to prevent thundering herd
    ),
)

# No retry on this node — it's reliable
builder.add_node("process", process_result_node)

builder.add_edge(START, "api_call")
builder.add_edge("api_call", "process")
builder.add_edge("process", END)

graph = builder.compile()

# --- 4. Run — the flaky node will fail twice then succeed ---
print("=== Retry Policy Demo ===\n")
print("Calling flaky API (will fail first 2 attempts)...\n")

result = graph.invoke(
    {
        "task": "fetch_user_data",
        "attempt_log": [],
        "result": "",
    }
)

print(f"\nFinal result: {result['result']}")


# --- 5. Demonstrate retry exhaustion ---
print("\n\n=== Retry Exhaustion Demo ===\n")


always_fails_count = 0


def always_fails_node(state: ProcessState) -> dict:
    """A node that always fails."""
    global always_fails_count
    always_fails_count += 1
    print(f"  [!] Attempt {always_fails_count} - FAILED")
    raise ConnectionError("Permanent failure")


builder2 = StateGraph(ProcessState)
builder2.add_node(
    "bad_api",
    always_fails_node,
    retry_policy=RetryPolicy(
        max_attempts=3,
        initial_interval=0.1,
        backoff_factor=1.5,
        max_interval=1.0,
    ),
)
builder2.add_edge(START, "bad_api")
builder2.add_edge("bad_api", END)

graph2 = builder2.compile()

print("Calling permanently broken API (max 3 attempts)...\n")

try:
    graph2.invoke({"task": "doomed_task", "attempt_log": [], "result": ""})
except Exception as e:
    print(f"\nRetries exhausted! Caught: {type(e).__name__}: {e}")
