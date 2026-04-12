import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Run-level token metrics via RunOutput.metrics (RunMetrics dataclass)
- Per-tool-call timing metrics
- Per-model breakdown from metrics.details

Agno provides a RunMetrics dataclass on every RunOutput with
input_tokens, output_tokens, total_tokens, and optional per-model
detail breakdowns. Tool calls also carry their own timing metrics
(start_time, end_time, duration) for performance analysis.

For more details, visit:
https://github.com/agno-agi/agno/blob/main/cookbook/02_agents/14_advanced/tool_call_metrics.py
-------------------------------------------------------
"""


# --- 1. Define a custom tool ---
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name to get weather for.

    Returns:
        A weather report string.
    """
    weather_data = {
        "london": "Cloudy, 14°C",
        "tokyo": "Sunny, 28°C",
        "new york": "Partly cloudy, 22°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


# --- 2. Create the agent ---
agent = Agent(
    model=OpenAIChat(
        id=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    ),
    tools=[get_weather],
    instructions="You are a helpful weather assistant. Be concise (1-2 sentences).",
    markdown=False,
)

# --- 3. Run the agent ---
print("=== Agno Usage Metrics ===\n")
print("--- Running agent ---")
run_output = agent.run("What's the weather in London and Tokyo?")

print(f"Response: {run_output.content}\n")

# --- 4. Display run-level metrics ---
print("--- Run-Level Metrics ---")
metrics = run_output.metrics
if metrics is not None:
    print(f"  Input tokens:   {metrics.input_tokens}")
    print(f"  Output tokens:  {metrics.output_tokens}")
    print(f"  Total tokens:   {metrics.total_tokens}")
    if metrics.cache_read_tokens:
        print(f"  Cache read:     {metrics.cache_read_tokens}")
    if metrics.cache_write_tokens:
        print(f"  Cache write:    {metrics.cache_write_tokens}")
    if metrics.reasoning_tokens:
        print(f"  Reasoning:      {metrics.reasoning_tokens}")
    if metrics.cost is not None:
        print(f"  Estimated cost: ${metrics.cost:.6f}")
    if metrics.duration is not None:
        print(f"  Duration:       {metrics.duration:.3f}s")
    if metrics.time_to_first_token is not None:
        print(f"  TTFT:           {metrics.time_to_first_token:.3f}s")
else:
    print("  No metrics available")

# --- 5. Display tool call metrics ---
print("\n--- Tool Call Metrics ---")
if run_output.tools:
    for tc in run_output.tools:
        print(f"  Tool: {tc.tool_name}")
        if tc.metrics:
            print(f"    Duration: {tc.metrics}")
        print()
else:
    print("  No tool calls recorded")

# --- 6. Display per-model detail breakdown ---
print("--- Per-Model Details ---")
if metrics is not None and metrics.details:
    for model_type, model_metrics_list in metrics.details.items():
        print(f"\n  {model_type}:")
        for mm in model_metrics_list:
            print(f"    Input tokens:  {mm.input_tokens}")
            print(f"    Output tokens: {mm.output_tokens}")
            print(f"    Total tokens:  {mm.total_tokens}")
            if mm.cost is not None:
                print(f"    Cost:          ${mm.cost:.6f}")
            print(f"    Provider:      {mm.provider}")
else:
    print("  No per-model details available")

print("\n=== Usage Metrics Demo Complete ===")
