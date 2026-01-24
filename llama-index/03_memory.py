import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Memory class for conversation history management
- Token-limited memory with automatic truncation
- Context object for state management across agent interactions
- Multi-turn conversations with memory persistence
- Session-based memory management

LlamaIndex provides sophisticated memory management capabilities that allow
agents to maintain conversation context while respecting token limits. This is
essential for building conversational AI applications.

For more details, visit:
https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/
-------------------------------------------------------
"""


memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


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

