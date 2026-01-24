import asyncio
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Asynchronous agent execution
- Using async/await with agents
- Non-blocking operations

LlamaIndex supports async operations throughout its API, allowing for better
performance in concurrent scenarios.

For more details, visit:
https://docs.llamaindex.ai/en/stable/module_guides/agent/agents/
-------------------------------------------------------
"""


def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


async def main():
    # --- 1. Configure the LLM ---
    llm = OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    )
    Settings.llm = llm

    print("LLM configured for async operations")
    print("-" * 50)

    # --- 2. Create tools ---
    add_tool = FunctionTool.from_defaults(fn=add_numbers)
    multiply_tool = FunctionTool.from_defaults(fn=multiply_numbers)

    # --- 3. Create agent ---
    agent = FunctionAgent(
        llm=llm,
        tools=[add_tool, multiply_tool],
        verbose=True
    )

    print("Agent created with math tools")
    print("-" * 50)

    # --- 4. Use async agent methods ---
    # Using achat (async chat) instead of chat
    query = "What is 15 plus 27 multiplied by 3?"
    response = await agent.achat(query)

    print(f"Query: {query}")
    print(f"Response: {response}")
    print("-" * 50)

    """
    Expected output:
    - Agent uses both tools in sequence
    - Demonstrates async execution
    - Returns correct calculation result
    """


if __name__ == "__main__":
    asyncio.run(main())
