import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Function tool definition with FunctionTool.from_defaults()
- Custom function tools with type annotations
- RAG-based tools with metadata filtering
- Tool calling with verbose output
- Multiple tool orchestration in a single LLM call

LlamaIndex provides native support for function calling, allowing you to define
custom tools that the LLM can invoke. Type annotations and docstrings are used
as prompts for the LLM to understand tool usage.

For more details, visit:
https://docs.llamaindex.ai/en/stable/module_guides/agent/tools/
-------------------------------------------------------
"""

# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

# Another tool
def divide(a: float, b: float) -> float:
    """Useful for dividing two numbers."""
    return a / b if b != 0 else 0
divide_tool = FunctionTool.from_defaults(divide)


# RAG-based tool example
from llama_index.core.tools import QueryEngineTool

tool = QueryEngineTool.from_defaults(
    query_engine, name="...", description="..."
)

# Google Gmail integration tool example
from llama_index.tools.google import GmailToolSpec

tool_spec = GmailToolSpec()
agent = FunctionAgent(llm=llm, tools=tool_spec.to_tool_list())

# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    name="multiply_agent",
    description="A simple multiply agent.",
    llm=OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    ),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
    tools=[multiply],
)


async def main():
    # Run the agent
    response = await agent.run("What is 1234 * 4567?")
    print(str(response))

if __name__ == "__main__":
    asyncio.run(main())
