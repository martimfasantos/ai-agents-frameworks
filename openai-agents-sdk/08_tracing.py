import os
import asyncio

from agents import Agent, Runner, trace

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Tracing with the trace() context manager
- Grouping multiple runs under a single trace

Tracing captures LLM calls, handoffs, tool calls, and execution flow.
Wrapping multiple Runner.run calls in a single trace() groups them for
unified debugging on the OpenAI platform.

For more details, visit:
https://openai.github.io/openai-agents-python/tracing/
-------------------------------------------------------
"""

# --- 1. Define the agent ---
agent = Agent(
    name="Joke generator",
    instructions="Tell funny jokes.",
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    # --- 2. Run the agent with tracing ---
    with trace("08_tracing"):
        first_result = await Runner.run(agent, "Tell me a joke")
        second_result = await Runner.run(
            agent, f"Rate this joke: {first_result.final_output}"
        )
        print(f"Joke: {first_result.final_output}")
        print(f"Rating: {second_result.final_output}")

    print("\nCheck your traces at: https://platform.openai.com/traces")


if __name__ == "__main__":
    asyncio.run(main())
