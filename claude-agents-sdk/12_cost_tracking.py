import asyncio

from dotenv import load_dotenv

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Cost tracking via ResultMessage.total_cost_usd and usage dict
- Limiting agent turns with max_turns
- Setting a budget cap with max_budget_usd
- Controlling thinking depth with the effort parameter

These controls help manage agent costs in production. max_turns limits
how many tool-call rounds the agent can take, max_budget_usd sets a
hard dollar cap, and effort (low/medium/high/max) controls how deeply
the model reasons before responding.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/cost-tracking
https://platform.claude.com/docs/en/agent-sdk/agent-loop
-------------------------------------------------------
"""

# --------------------------------------------------------------
# Example 1: Track Cost and Usage
# --------------------------------------------------------------
print("=== Example 1: Cost and Usage Tracking ===")


async def example_cost_tracking():
    async for message in query(
        prompt="Explain what a REST API is in two sentences.",
        options=ClaudeAgentOptions(),
    ):
        if isinstance(message, ResultMessage):
            print(f"Result: {message.result}")
            print(
                f"Cost: ${message.total_cost_usd:.6f}"
                if message.total_cost_usd
                else "Cost: N/A"
            )
            print(f"Turns: {message.num_turns}")
            print(
                f"Duration: {message.duration_ms}ms (API: {message.duration_api_ms}ms)"
            )
            if message.usage:
                print(f"Usage: {message.usage}")


asyncio.run(example_cost_tracking())

# --------------------------------------------------------------
# Example 2: Limit Turns
# --------------------------------------------------------------
print("\n=== Example 2: Limit Max Turns ===")


async def example_max_turns():
    options = ClaudeAgentOptions(
        max_turns=3,
        allowed_tools=["Read", "Glob"],
        permission_mode="bypassPermissions",
    )

    async for message in query(
        prompt="Find and summarize all Python files in the current directory.",
        options=options,
    ):
        if isinstance(message, ResultMessage):
            print(f"Stopped after {message.num_turns} turns")
            print(f"Stop reason: {message.stop_reason}")
            print(f"Result: {message.result}")


asyncio.run(example_max_turns())

# --------------------------------------------------------------
# Example 3: Budget Cap
# --------------------------------------------------------------
print("\n=== Example 3: Budget Cap ===")


async def example_budget_cap():
    options = ClaudeAgentOptions(
        max_budget_usd=0.05,  # 5 cents max
    )

    async for message in query(
        prompt="What is the capital of France?",
        options=options,
    ):
        if isinstance(message, ResultMessage):
            cost = message.total_cost_usd
            print(f"Result: {message.result}")
            print(f"Cost: ${cost:.6f}" if cost else "Cost: N/A")
            print(f"Budget limit: $0.05")


asyncio.run(example_budget_cap())

# --------------------------------------------------------------
# Example 4: Effort Level
# --------------------------------------------------------------
print("\n=== Example 4: Effort Level ===")


async def example_effort():
    # Low effort for quick, simple answers
    options = ClaudeAgentOptions(effort="low")

    async for message in query(
        prompt="What is 2 + 2?",
        options=options,
    ):
        if isinstance(message, ResultMessage):
            print(f"[Low effort] Result: {message.result}")
            print(f"Duration: {message.duration_ms}ms")


asyncio.run(example_effort())
