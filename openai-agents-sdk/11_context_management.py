import os
import asyncio
from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner, function_tool

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Typed context with RunContextWrapper
- Dependency injection via context
- Accessing context inside tools

Context is a dependency-injection mechanism: you create any Python
object, pass it to Runner.run(), and every tool, hook, and guardrail
receives a RunContextWrapper[T] that exposes your object via .context.
This keeps tools decoupled from global state.

For more details, visit:
https://openai.github.io/openai-agents-python/context/
-------------------------------------------------------
"""


# --- 1. Define a typed context object ---
@dataclass
class UserInfo:
    name: str
    uid: int
    is_pro_user: bool


# --- 2. Define a tool that reads from context ---
@function_tool
async def fetch_user_age(wrapper: RunContextWrapper[UserInfo]) -> str:
    """Fetch the age of the current user. Call this to get user age info."""
    # In a real app this would hit a database; here we simulate
    return f"User {wrapper.context.name} (uid={wrapper.context.uid}) is 47 years old"


# --- 3. Define a tool that checks context for feature flags ---
@function_tool
async def check_pro_status(wrapper: RunContextWrapper[UserInfo]) -> str:
    """Check whether the current user has a pro subscription."""
    if wrapper.context.is_pro_user:
        return f"{wrapper.context.name} is a Pro subscriber."
    return f"{wrapper.context.name} is on the Free plan."


# --- 4. Create the agent typed with UserInfo context ---
agent = Agent[UserInfo](
    name="Account Assistant",
    instructions="You help users check their account details. Use the tools provided.",
    tools=[fetch_user_age, check_pro_status],
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    # --- 5. Create the context and run ---
    user_info = UserInfo(name="Martim", uid=42, is_pro_user=True)
    print(f"Context injected: {user_info}")
    print()

    result = await Runner.run(
        starting_agent=agent,
        input="What is my age and am I a pro user?",
        context=user_info,
    )

    # --- 6. Show tool outputs and final result ---
    print("Tool outputs from context-aware tools:")
    for item in result.new_items:
        if hasattr(item, "output"):
            print(f"  -> {item.output}")
    print()
    print(f"Final output: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
