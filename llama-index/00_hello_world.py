
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from settings import settings


"""
-------------------------------------------------------
In this example, we explore a simple Hello World agent
-------------------------------------------------------
"""

# --- 1. Define the agent ---
agent = FunctionAgent(
    name="hello_world_agent",
    description="A simple hello world agent.",
    llm=OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    ),
    system_prompt="You are a helpful assistant that greets the user.",
)

# --- 2. Run the agent with a user message ---
async def main():
    # Run the agent
    response = await agent.run("Hello World!")
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())