import asyncio

from dotenv import load_dotenv

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Built-in code interpreter tool
- Built-in web search tool
- Using OpenAIChatClient for hosted tool support

Built-in tools are hosted by the model provider and run
server-side. Code interpreter executes Python in a sandbox,
and web search retrieves live information.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/tools/?pivots=programming-language-python
-------------------------------------------------------
"""


async def main() -> None:
    # --- 1. Create the client ---
    client = OpenAIChatClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    # --------------------------------------------------------------
    # Example 1: Code Interpreter
    # --------------------------------------------------------------
    print("=== Example 1: Code Interpreter ===")

    # --- 2. Get the code interpreter tool ---
    code_interpreter = client.get_code_interpreter_tool()

    agent_code = client.as_agent(
        name="math-assistant",
        instructions="You are a math assistant. Use code interpreter to solve problems. Show your work.",
        tools=[code_interpreter],
    )

    result = await agent_code.run(
        "Calculate the first 10 Fibonacci numbers and their sum."
    )
    print(result.text)
    print()

    # --------------------------------------------------------------
    # Example 2: Web Search
    # --------------------------------------------------------------
    print("=== Example 2: Web Search ===")

    # --- 3. Get the web search tool ---
    web_search = client.get_web_search_tool()

    agent_search = client.as_agent(
        name="research-assistant",
        instructions="You are a research assistant. Use web search to find current information. Be concise.",
        tools=[web_search],
    )

    result = await agent_search.run("What is the Microsoft Agent Framework?")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
