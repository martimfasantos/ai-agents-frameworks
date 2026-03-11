import os
import asyncio
from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Dynamic instructions via a function
- Personalizing agent behavior at runtime

Instead of a static string, the instructions parameter can be a
function that receives the RunContextWrapper and the Agent, and
returns a string.  This lets you inject per-request data (user name,
current date, feature flags, etc.) into the system prompt.

For more details, visit:
https://openai.github.io/openai-agents-python/agents/#dynamic-instructions
-------------------------------------------------------
"""


# --- 1. Define the context type ---
@dataclass
class UserContext:
    name: str
    language: str
    is_premium: bool


# --- 2. Define the dynamic instructions function ---
def build_instructions(
    context: RunContextWrapper[UserContext], agent: Agent[UserContext]
) -> str:
    """Generates a personalised system prompt based on the user's context."""
    ctx = context.context
    tier = "premium" if ctx.is_premium else "free-tier"
    return (
        f"You are a helpful assistant for {ctx.name} ({tier} user). "
        f"Always reply in {ctx.language}. Be concise."
    )


# --- 3. Create the agent with dynamic instructions ---
agent = Agent[UserContext](
    name="Personalised Assistant",
    instructions=build_instructions,
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    # --- 4. Run with a Portuguese premium user ---
    ctx_pt = UserContext(name="Martim", language="Portuguese", is_premium=True)
    result_pt = await Runner.run(agent, "What is the capital of Japan?", context=ctx_pt)
    print(f"Portuguese reply: {result_pt.final_output}")

    # --- 5. Run with an English free-tier user ---
    ctx_en = UserContext(name="Alice", language="English", is_premium=False)
    result_en = await Runner.run(agent, "What is the capital of Japan?", context=ctx_en)
    print(f"English reply:    {result_en.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
