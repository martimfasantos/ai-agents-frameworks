import os

from agents import Agent, Runner

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Creating a simple agent
- Running a synchronous request

A minimal "Hello World" showing how to define an Agent with a system prompt
and run it synchronously using Runner.run_sync.

For more details, visit:
https://openai.github.io/openai-agents-python/
-------------------------------------------------------
"""

# --- 1. Define the agent ---
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    model=settings.OPENAI_MODEL_NAME,
)

# --- 2. Run the agent with a user message ---
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
