import asyncio
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Memory with initial_messages for system context
- Persistent user profiles and preferences
- Multi-turn conversations with context retention

This shows LlamaIndex's memory system where you can inject initial context
that persists throughout the conversation. This is useful for maintaining
user profiles, system instructions, or background knowledge.

For more details, visit:
https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/
-------------------------------------------------------
"""


def get_current_time() -> str:
    """Return the current time."""
    from datetime import datetime
    return datetime.now().strftime("%I:%M %p")


async def main():
    # --- 1. Configure the LLM ---
    llm = OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    )
    Settings.llm = llm

    print("LLM configured with memory support")
    print("-" * 50)

    # --- 2. Create tools for the agent ---
    time_tool = FunctionTool.from_defaults(fn=get_current_time)

    # --- 3. Create memory with initial context ---
    # This context will persist throughout the conversation
    memory = Memory.from_defaults(
        token_limit=2000,
        initial_messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. The user's name is Alice and she prefers concise answers."
            }
        ]
    )

    print("Memory initialized with user context")
    print("-" * 50)

    # --- 4. Create agent with memory ---
    agent = FunctionAgent(
        llm=llm,
        tools=[time_tool],
        memory=memory,
        verbose=True
    )

    # --- 5. Have a multi-turn conversation ---
    # Notice how the agent remembers the user's name from initial memory
    
    ctx = Context()
    
    query1 = "Hi, what's your role?"
    response1 = await agent.achat(query1, ctx=ctx)
    print(f"User: {query1}")
    print(f"Agent: {response1}")
    print("-" * 50)

    query2 = "What's my name?"
    response2 = await agent.achat(query2, ctx=ctx)
    print(f"User: {query2}")
    print(f"Agent: {response2}")
    print("-" * 50)

    query3 = "What time is it?"
    response3 = await agent.achat(query3, ctx=ctx)
    print(f"User: {query3}")
    print(f"Agent: {response3}")
    print("-" * 50)

    """
    Expected output:
    - Agent remembers Alice's name from initial memory
    - Agent uses time tool when asked about time
    - Demonstrates persistent memory across conversation turns
    """


if __name__ == "__main__":
    asyncio.run(main())
