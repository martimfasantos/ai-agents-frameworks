import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow.workflow_events import ToolCall
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Creation and definition of custom tools
- Integration of third-party tools (e.g., Google Gmail)
- Tool calling with verbose output

LlamaIndex provides native support for function calling, allowing you to define
custom tools that the LLM can invoke. Type annotations and docstrings are used
as prompts for the LLM to understand tool usage.

For more details, visit:
https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/tools
-------------------------------------------------------
"""

# --- 1. Define the tools ---
# 1.1 Simple function example
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

# 1.2 Constructing FunctionTool from function
from llama_index.core.tools import FunctionTool
from typing import Annotated

def divide(
    a: Annotated[float, "The numerator"],
    b: Annotated[float, "The denominator"]
) -> float:
    """Useful for dividing two numbers."""
    return a / b if b != 0 else 0

divide_tool = FunctionTool.from_defaults(
    divide, name="divide", description="Divide two numbers."
)

# 1.3 Google Gmail integration tool example
from llama_index.tools.google import GmailToolSpec

tool_spec = GmailToolSpec()

# --- 2. Create an agent with our tools ---
agent = FunctionAgent(
    name="multi_tool_agent",
    description="An agent that can multiply and divide numbers, and read Gmail emails.",
    llm=OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    ),
    system_prompt="You are a helpful assistant that uses tools.",
    tools=[multiply, divide_tool] + tool_spec.to_tool_list(),
)

# --- 3. Run the agent with a user message ---
async def main():
    # Test multiplying
    handler = agent.run("What is 12.5 * 4?")
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            print(f"Tool selected: {ev.tool_name}")
    result = await handler
    print(result)
    
    # Test dividing
    handler = agent.run("What is 20 / 4?")
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            print(f"Tool selected: {ev.tool_name}")
    result = await handler
    print(result)
    
    # Test Gmail tool (requires proper Google API setup)
    # response = await agent.run("Read my latest emails from Gmail.")
    # print(str(response))

if __name__ == "__main__":
    asyncio.run(main())
