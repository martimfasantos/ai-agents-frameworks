import asyncio
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Wrapping an agent as a tool
- Agent delegation pattern
- Specialized agents

This shows how one agent can use another agent as a tool, enabling delegation
of specialized tasks.

For more details, visit:
https://docs.llamaindex.ai/en/stable/module_guides/agent/agents/
-------------------------------------------------------
"""


def calculate(a: float, b: float, operation: str) -> float:
    """Perform basic math operations."""
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    elif operation == "subtract":
        return a - b
    elif operation == "divide":
        return a / b if b != 0 else 0
    return 0


async def main():
    # --- 1. Configure the LLM ---
    llm = OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    )
    Settings.llm = llm

    print("LLM configured for agent delegation")
    print("-" * 50)

    # --- 2. Create a specialized math agent ---
    math_tool = FunctionTool.from_defaults(fn=calculate)
    
    math_agent = FunctionAgent(
        llm=llm,
        tools=[math_tool],
        verbose=True,
        system_prompt="You are a math expert. Perform calculations accurately."
    )

    print("Created Math Agent")
    print("-" * 50)

    # --- 3. Wrap the math agent as a tool ---
    # This allows other agents to delegate math tasks to this specialist
    math_agent_tool = FunctionTool.from_defaults(
        fn=lambda query: math_agent.chat(query).response,
        name="math_expert",
        description="A math expert that can perform complex calculations"
    )

    print("Wrapped Math Agent as a tool")
    print("-" * 50)

    # --- 4. Create a master agent that uses the math agent ---
    master_agent = FunctionAgent(
        llm=llm,
        tools=[math_agent_tool],
        verbose=True,
        system_prompt="You are a helpful assistant. Delegate math questions to the math expert."
    )

    print("Created Master Agent")
    print("-" * 50)

    # --- 5. Test agent delegation ---
    query = "Calculate 25 multiplied by 4, then add 10"
    response = await master_agent.achat(query)

    print(f"Query: {query}")
    print(f"Response: {response}")
    print("-" * 50)

    """
    Expected output:
    - Master agent delegates calculation to math expert
    - Math agent performs the calculations
    - Master agent returns the result
    - Demonstrates agent delegation pattern
    """


if __name__ == "__main__":
    asyncio.run(main())
