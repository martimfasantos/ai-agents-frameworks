import os
from dataclasses import dataclass
from typing import Any, cast

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Long-term memory with InMemoryStore
- Reading from the store in tools via runtime.store
- Writing to the store to persist data across conversations

Long-term memory persists data across different conversations and
sessions. Unlike short-term memory (scoped to a thread), long-term
memory uses a store that organizes data by namespace and key. Tools
can read and write to the store, enabling features like user
preferences and cross-session personalization.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/long-term-memory
-------------------------------------------------------
"""


# --- 1. Define context schema ---
@dataclass
class UserContext:
    user_id: str


# --- 2. Create the store and seed some data ---
store = InMemoryStore()

# Pre-populate with a user's preferences
store.put(
    ("preferences",),
    "user_alice",
    {"tone": "casual", "topics_of_interest": ["AI", "cooking", "travel"]},
)


# --- 3. Define tools that read/write from the store ---
@tool
def get_preferences(runtime: ToolRuntime[UserContext]) -> str:
    """Look up the current user's preferences from long-term memory."""
    assert runtime.store is not None
    user_id = runtime.context.user_id
    prefs = runtime.store.get(("preferences",), user_id)
    if prefs:
        return f"Preferences for {user_id}: {prefs.value}"
    return f"No preferences found for {user_id}"


@tool
def save_preference(
    key: str,
    value: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """Save a preference for the current user to long-term memory."""
    assert runtime.store is not None
    user_id = runtime.context.user_id

    # Read existing preferences and merge
    existing = runtime.store.get(("preferences",), user_id)
    prefs: dict[str, Any] = existing.value if existing else {}
    prefs[key] = value

    # Write updated preferences
    runtime.store.put(("preferences",), user_id, prefs)
    return f"Saved preference: {key} = {value}"


# --- 4. Create the model ---
model = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

# --- 5. Create agent with store ---
agent = create_agent(
    model=model,
    tools=[get_preferences, save_preference],
    system_prompt="You are a personal assistant. Use tools to read and save user preferences.",
    context_schema=UserContext,
    store=store,
)

# --- 6. Read existing preferences ---
print("=== Reading Alice's Preferences ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What are my preferences?"}]},
    context=UserContext(user_id="user_alice"),
)
print(f"Response: {result['messages'][-1].content}\n")

# --- 7. Save a new preference ---
print("=== Saving a New Preference ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Save my favorite color as blue."}]},
    context=UserContext(user_id="user_alice"),
)
print(f"Response: {result['messages'][-1].content}\n")

# --- 8. Verify the preference persists (different invocation) ---
print("=== Verifying Persistence ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What are my preferences now?"}]},
    context=UserContext(user_id="user_alice"),
)
print(f"Response: {result['messages'][-1].content}\n")

# --- 9. Different user has no preferences ---
print("=== Different User (Bob) ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What are my preferences?"}]},
    context=UserContext(user_id="user_bob"),
)
print(f"Response: {result['messages'][-1].content}")
