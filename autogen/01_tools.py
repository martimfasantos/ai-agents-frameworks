import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from settings import settings

"""
------------------------------------------------------------------------
In this example, we explore Autogen agents with the following features:
- Tool usage with function tools
- Automatic tool schema generation
- Tool call execution and result handling

This example shows how to define Python functions as tools that agents
can invoke. Autogen automatically generates tool schemas from function
signatures and docstrings, and the AssistantAgent executes tools within
its run loop.

For more details, visit:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html
------------------------------------------------------------------------
"""


# --- 1. Define a tool that searches the web for information ---
# For simplicity, we use a mock function here that returns a static string.
async def web_search_func(query: str) -> str:
    """Find information on the web"""
    return "AutoGen is a programming framework for building multi-agent applications."


# NOTE: This step is automatically performed inside the AssistantAgent
# if the tool is a Python function.
web_search_function_tool = FunctionTool(
    web_search_func, description="Find information on the web"
)


async def main() -> None:
    # --- 2. Define the model client ---
    model_client = OpenAIChatCompletionClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    # --- 3. Define the agent ---
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[web_search_func],
        system_message="Use tools to solve tasks.",
    )

    # --- 4. Run the agent and stream the output ---
    result = await Console(
        agent.run_stream(task="Find information on AutoGen"),
        output_stats=True,
    )

    # --- 5. Print the final answer ---
    print("-" * 50)
    print("Final Answer:", result.messages[-1].content)

    # --- 6. Print the tool schema ---
    print("-" * 50)
    print("Tool Schema:", web_search_function_tool.schema)

    # --- 7. Close the model client ---
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
