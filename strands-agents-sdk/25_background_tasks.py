import os
import time

from strands import Agent, tool
from strands.background_tasks import BackgroundTasksConfig
from strands.models.openai import OpenAIModel

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Strands Agents with the following features:
- background_tasks to let slow tools run off the critical path
- BackgroundTasksConfig with always / never to pin execution mode
- max_concurrency to bound how many run at once

By default a tool call blocks the agent loop: nothing else happens until
it returns. With background_tasks the model can dispatch slow, independent
work and collect the results when they land, so several long calls overlap
instead of queueing. always/never override the model's choice per tool —
useful when you know a tool is slow and independent, or when it must stay
inline because later reasoning depends on it immediately.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/agents/agent-loop/
-------------------------------------------------------
"""

model = OpenAIModel(
    client_args={"api_key": settings.OPENAI_API_KEY.get_secret_value()},
    model_id=settings.OPENAI_MODEL_NAME,
)


# --- 1. Three independent, deliberately slow lookups ---
TOOL_SECONDS = 2.0
spans: list[tuple[float, float]] = []


@tool
def fetch_sales(region: str) -> str:
    """Fetch the quarterly sales figure for a region."""
    begin = time.monotonic()
    print(f"  [tool] fetch_sales({region}) starting")
    time.sleep(TOOL_SECONDS)
    figures = {"emea": "412k", "apac": "388k", "americas": "690k"}
    print(f"  [tool] fetch_sales({region}) done")
    spans.append((begin, time.monotonic()))
    return figures.get(region.lower(), "unknown")


agent = Agent(
    model=model,
    tools=[fetch_sales],
    callback_handler=None,
    system_prompt=(
        "Call fetch_sales once per region named. Then report each region and "
        "its figure on one line."
    ),
    # --- 2. Pin this tool to background execution ---
    background_tasks=BackgroundTasksConfig(
        always=[fetch_sales],  # never run inline, regardless of what the model picks
        max_concurrency=3,  # let all three overlap
        wait_for_completion=True,  # results are in hand before the call returns
    ),
)

print("=== Background tool execution ===\n")

started = time.monotonic()
result = agent("Get sales for EMEA, APAC and Americas.")
elapsed = time.monotonic() - started

print(f"\n{result}")

# --- 3. Measure the overlap rather than assert it ---
tool_time = sum(end - start for start, end in spans)
wall_time = max(end for _, end in spans) - min(start for start, _ in spans)
print(f"Tool work:      {tool_time:.1f}s across {len(spans)} calls")
print(f"Wall clock:     {wall_time:.1f}s — they overlapped instead of queueing")
print(f"Whole run:      {elapsed:.1f}s (the rest is model turns)")
