import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Short-term memory with InMemorySaver checkpointer
- Thread-based conversation persistence
- Multi-turn conversations that remember context

Short-term memory lets an agent remember previous messages within
a conversation thread. LangChain uses a checkpointer (like
InMemorySaver) to persist state. Each thread_id identifies a
separate conversation, so multiple users can chat independently.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/short-term-memory
-------------------------------------------------------
"""


# --- 1. Define a tool ---
@tool
def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    times = {
        "utc": "14:30 UTC",
        "est": "09:30 EST",
        "jst": "23:30 JST",
    }
    return times.get(timezone.lower(), f"Time not available for {timezone}")


# --- 2. Create agent with checkpointer for memory ---
agent = create_agent(
    model=init_chat_model(f"openai:{settings.OPENAI_MODEL_NAME}"),
    tools=[get_time],
    system_prompt="You are a helpful assistant. Be concise.",
    checkpointer=InMemorySaver(),
)

# --- 3. Have a multi-turn conversation on thread "abc" ---
config = {"configurable": {"thread_id": "abc"}}

print("=== Thread 'abc': Multi-turn conversation ===")

result_1 = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "My name is Alice. What time is it in UTC?"}
        ]
    },
    config=config,
)
print(f"Turn 1: {result_1['messages'][-1].content}\n")

result_2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What about EST?"}]},
    config=config,
)
print(f"Turn 2: {result_2['messages'][-1].content}\n")

# The agent should remember the user's name from turn 1
result_3 = agent.invoke(
    {"messages": [{"role": "user", "content": "Do you remember my name?"}]},
    config=config,
)
print(f"Turn 3: {result_3['messages'][-1].content}\n")

# --- 4. Show that a different thread has no memory ---
print("=== Thread 'xyz': Fresh conversation ===")

config_new = {"configurable": {"thread_id": "xyz"}}

result_4 = agent.invoke(
    {"messages": [{"role": "user", "content": "Do you know my name?"}]},
    config=config_new,
)
print(f"Turn 1: {result_4['messages'][-1].content}")
