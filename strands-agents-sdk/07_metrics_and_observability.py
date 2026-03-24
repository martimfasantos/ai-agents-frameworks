import json

from strands import Agent, tool
from strands.models.openai import OpenAIModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- AgentResult metrics and summary
- Token usage tracking (input, output, total)
- Tool call statistics (count, success rate, timing)
- Execution timing (duration, cycles, average cycle time)

Every agent invocation returns an AgentResult with comprehensive
observability data. This is useful for monitoring costs, debugging
slow runs, and understanding how the agent uses its tools.

For more details, visit:
https://strandsagents.com/docs/user-guide/observability-evaluation/observability/
-------------------------------------------------------
"""

# --- 1. Define a tool for the agent to use ---


@tool
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number.

    Args:
        n: The position in the Fibonacci sequence (0-indexed)
    """
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# --- 2. Configure model, create agent, and run a task ---
openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)
# Default: Agent(tools=[...]) uses Amazon Bedrock (requires AWS credentials)
agent = Agent(
    model=openai_model,
    tools=[fibonacci],
    callback_handler=None,
)

print("=== Agent Metrics & Observability ===\n")
result = agent("Calculate the 10th and 20th Fibonacci numbers.")

# --- 3. Access the result ---
print(f"Response: {result.message}\n")
print(f"Stop reason: {result.stop_reason}")

# --- 4. Metrics summary ---
print("\n--- Metrics Summary ---")
metrics_summary = result.metrics.get_summary()
print(json.dumps(metrics_summary, indent=2, default=str))

# --- 5. Token usage ---
print("\n--- Token Usage ---")
usage = metrics_summary.get("accumulated_usage", {})
print(f"Input tokens: {usage.get('inputTokens', 'N/A')}")
print(f"Output tokens: {usage.get('outputTokens', 'N/A')}")
print(f"Total tokens: {usage.get('totalTokens', 'N/A')}")

# --- 6. Tool usage ---
print("\n--- Tool Usage ---")
tool_usage = metrics_summary.get("tool_usage", {})
for tool_name, tool_data in tool_usage.items():
    stats = tool_data.get("execution_stats", {})
    print(f"Tool: {tool_name}")
    print(f"  Calls: {stats.get('call_count', 0)}")
    print(f"  Success rate: {stats.get('success_rate', 0):.0%}")
    print(f"  Avg time: {stats.get('average_time', 0):.4f}s")

# --- 7. Execution timing ---
print("\n--- Execution Timing ---")
print(f"Total duration: {metrics_summary.get('total_duration', 'N/A'):.2f}s")
print(f"Total cycles: {metrics_summary.get('total_cycles', 'N/A')}")
print(f"Avg cycle time: {metrics_summary.get('average_cycle_time', 'N/A'):.4f}s")
