import os
import asyncio

from google.adk.agents import LlmAgent

from settings import settings
from utils import call_agent_async

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- Creating an LlmAgent with a simple system instruction
- Running a single-turn conversation using Runner and InMemorySessionService
- Using async execution via asyncio

Google ADK (Agent Development Kit) is Google's framework for building
AI agents powered by Gemini models. This simplest example creates an
agent with a basic instruction and sends it a greeting to get a response.

For more details, visit:
https://google.github.io/adk-docs/get-started/quickstart/
-------------------------------------------------------
"""

# --- 1. Create the agent ---
hello_world_agent = LlmAgent(
    name="hello_world_agent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction="You are a helpful assistant. Respond in 1-2 sentences.",
    description="A simple AI assistant.",
)

# --- 2. Run the agent ---
print("Query: Hello, who are you?")
asyncio.run(call_agent_async(hello_world_agent, "Hello, who are you?"))
