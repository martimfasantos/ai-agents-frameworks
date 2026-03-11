import os
import asyncio
from pydantic import BaseModel

from agents import (
    Agent,
    MessageOutputItem,
    Runner,
    RunResult,
    ToolCallItem,
    ToolCallOutputItem,
    function_tool,
)

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Tool usage with @function_tool
- Output models for structured tool responses
- Internal messages inspection
- Usage metrics

This example registers a weather tool on an agent, runs a query,
then inspects the internal messages and token-usage metrics that
the SDK collects automatically.

For more details, visit:
https://openai.github.io/openai-agents-python/tools/
-------------------------------------------------------
"""


# --- 1. Define the output model for the weather data ---
class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str


# --- 2. Define a function tool to get weather information ---
@function_tool
def get_weather(city: str) -> Weather:
    """Retrieves the current weather for a given city."""
    print("[debug] get_weather called")
    return Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind.")


# --- 3. Define the agent with the function tool registered ---
agent = Agent(
    name="Weather Agent",
    instructions="You are a helpful agent.",
    tools=[get_weather],
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    # --- 4. Run the agent ---
    result: RunResult = await Runner.run(agent, input="What's the weather in Tokyo?")
    print("-" * 50)
    print(f"Final response:\n\t{result.final_output}")

    # --- 5. Print the internal messages (cleaned up) ---
    print("-" * 50)
    print("Internal messages:")
    for item in result.new_items:
        if isinstance(item, ToolCallItem):
            raw = item.raw_item
            name = getattr(raw, "name", "unknown")
            args = getattr(raw, "arguments", "")
            print(f"  [tool_call] {name}({args})")
        elif isinstance(item, ToolCallOutputItem):
            print(f"  [tool_output] {item.output}")
        elif isinstance(item, MessageOutputItem):
            text = item.raw_item.content[0].text if item.raw_item.content else ""
            print(f"  [message] {text[:200]}")
        else:
            print(f"  [{item.type}]")

    # --- 6. Print token-usage metrics ---
    print("-" * 50)
    print("Token usage:")
    for i, raw_response in enumerate(result.raw_responses, 1):
        usage = raw_response.usage
        print(
            f"  Response {i}: "
            f"input={usage.input_tokens}, output={usage.output_tokens}, "
            f"total={usage.total_tokens}"
        )
    print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
