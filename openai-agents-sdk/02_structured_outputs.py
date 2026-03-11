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
- Output models for tool responses
- Output types on agents for structured final output
- Filtering sensitive data via output types

This example shows how to use output_type on an Agent to constrain
the final response schema.  The tool returns a full user profile
(including location and IBAN), but the agent's output_type strips
those fields so only safe data reaches the caller.

For more details, visit:
https://openai.github.io/openai-agents-python/results/
-------------------------------------------------------
"""


# --- 1. Define the full user profile model ---
class UserProfile(BaseModel):
    id: int
    name: str
    age: int
    location: str
    iban: str


# --- 2. Define a filtered output model ---
class FilteredUserProfile(BaseModel):
    name: str
    age: int


# --- 3. Define a function tool to get user information ---
@function_tool
def get_user_profile(id: int) -> UserProfile:
    """Retrieves a user profile by ID."""
    print("[debug] get_user_profile called")
    return UserProfile(
        id=id,
        name="Martim Santos",
        age=24,
        location="Lisbon, Portugal",
        iban="DE89370400440532013000",
    )


# --- 4. Define the agent with the output type that filters sensitive data ---
agent = Agent(
    name="Profile Agent",
    instructions="You are a helpful agent.",
    tools=[get_user_profile],
    output_type=FilteredUserProfile,
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    # --- 5. Run the agent and display results ---
    result: RunResult = await Runner.run(
        agent, input="Who is user 1 and what's his location?"
    )

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
    print("-" * 50)

    # --- 6. Show that output_type filtered sensitive fields ---
    print(f"Final output type: {type(result.final_output).__name__}")
    print(f"Final output: {result.final_output}")
    print()
    print(
        "Note: The tool returned a full UserProfile (with location and IBAN),\n"
        "but output_type=FilteredUserProfile ensures only 'name' and 'age' are exposed."
    )


if __name__ == "__main__":
    asyncio.run(main())
