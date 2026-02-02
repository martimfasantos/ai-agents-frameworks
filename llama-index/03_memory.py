import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow.workflow_events import ToolCall
from llama_index.llms.openai import OpenAI
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Memory class for conversation history management
- Token-limited memory with automatic truncation
- Multi-turn conversations with memory persistence

LlamaIndex provides a simple memory management capabilities that allow
agents to maintain conversation context while respecting token limits.

For more details, visit:
https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/
-------------------------------------------------------
"""

# --- 1. Define the memory ---
from llama_index.core.memory import Memory

memory = Memory.from_defaults(session_id="my_session", token_limit=40000)

# --- 2. Define the tool ---
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

# --- 3. Create an agent with the tool ---
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

# --- 4. Run the agent with memory ---
async def main():
    # 4.1 Run the agent
    handler = agent.run("What is 1234 * 4567?", memory=memory)
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            print(f"Tool selected: {ev.tool_name}")
    
    response = await handler
    print(response)
    
    # 4.2 Run again to see memory in action
    handler = agent.run("What is 1234 * 4567?", memory=memory)
    tool_call = False
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            tool_call = True
    
    response = await handler
    print("Was tool called again? ", tool_call)
    print("Response with memory: ", response)

if __name__ == "__main__":
    asyncio.run(main())

