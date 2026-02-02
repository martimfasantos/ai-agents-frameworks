import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionAgent
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Event-based streaming with async generators
- Real-time monitoring of agent operations
- Delta streaming for partial outputs

LlamaIndex provides first-class streaming support through its event-driven
workflow architecture. This enables real-time output and monitoring of agent
operations, tool calls, and intermediate steps.

For more details, visit:
https://developers.llamaindex.ai/python/framework/understanding/agent/streaming/
-------------------------------------------------------
"""

# --- 1. Define a simple tool ---
def get_status() -> str:
    """Useful for getting the current status."""
    return "All systems operational."

# --- 2. Create an agent with the tool ---
agent = FunctionAgent(
    name="status_agent",
    description="A simple status agent.",
    llm=OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    ),
    system_prompt="You are a helpful assistant that can get the current status.",
    tools=[get_status],
    streaming=True,
)

# --- 3. Run the agent ---
async def main():
    # 3.1 Define the handler
    handler = agent.run("What is the current status?")
    # 3.2 Stream events
    async for ev in handler.stream_events():
        print(f"Event: {ev}")
    # 3.3 Get final output
    output = await handler
    print("Final Output: ", output)
    print("-" * 50)
    
    # 3.4 Another example only text delta streaming
    from llama_index.core.agent.workflow.workflow_events import AgentStream
    
    handler = agent.run("Write a small poem about AI agents.")
    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)
    
    output = await handler

if __name__ == "__main__":
    asyncio.run(main())