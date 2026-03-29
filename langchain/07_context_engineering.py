import os
from dataclasses import dataclass
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    dynamic_prompt,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from langgraph.store.memory import InMemoryStore

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Context engineering: dynamic prompts from state, store, and runtime context
- Dynamic tool selection based on user permissions
- Message injection via wrap_model_call

Context engineering is the art of providing the right information
to the LLM at the right time. This example combines dynamic prompts
(personalized per user), context-aware tool filtering (role-based
access), and message injection (adding extra context before model
calls) to build a reliable, personalized agent.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/context-engineering
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = ChatOpenAI(
    model=settings.OPENAI_MODEL_NAME,
    temperature=0.1,
    max_tokens=1000,
    timeout=30,
)


# --- 2. Define context schema ---
@dataclass
class AppContext:
    user_id: str
    user_role: str  # "admin" or "viewer"


# --- 3. Create a store with user preferences ---
store = InMemoryStore()
store.put(
    ("preferences",),
    "user_admin",
    {"communication_style": "technical and detailed", "timezone": "UTC"},
)
store.put(
    ("preferences",),
    "user_viewer",
    {"communication_style": "simple and brief", "timezone": "EST"},
)


# --- 4. Dynamic prompt from store (user preferences) ---
@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id
    user_role = request.runtime.context.user_role

    # Read preferences from long-term memory
    prefs = request.runtime.store.get(("preferences",), user_id)
    style = prefs.value.get("communication_style", "balanced") if prefs else "balanced"

    base = f"You are a helpful assistant. Respond in a {style} style."
    if user_role == "admin":
        base += " The user has admin access and can perform all operations."
    else:
        base += " The user has read-only access."
    return base


# --- 5. Dynamic tool filtering based on role ---
@tool
def read_data(query: str) -> str:
    """Read data from the system."""
    data = {
        "users": "Total users: 1,234. Active today: 456.",
        "revenue": "Monthly revenue: $45,678. Growth: +12%.",
        "system": "CPU: 45%, Memory: 62%, Disk: 78%.",
    }
    for key, value in data.items():
        if key in query.lower():
            return value
    return f"Data query result for: {query}"


@tool
def write_data(key: str, value: str) -> str:
    """Write or update data in the system (admin only)."""
    return f"Updated {key} = {value}"


@tool
def delete_data(key: str) -> str:
    """Delete data from the system (admin only)."""
    return f"Deleted: {key}"


@wrap_model_call
def filter_tools_by_role(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Filter available tools based on user role."""
    user_role = request.runtime.context.user_role

    if user_role == "admin":
        # Admins get all tools
        pass
    else:
        # Viewers only get read tools
        tools = [t for t in request.tools if t.name == "read_data"]
        request = request.override(tools=tools)

    return handler(request)


# --- 6. Message injection: add context about recent activity ---
@wrap_model_call
def inject_activity_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Inject recent activity context into the conversation."""
    user_id = request.runtime.context.user_id

    # Simulate fetching recent activity
    activity = {
        "user_admin": "Last login: 2 hours ago. Recent actions: viewed dashboard, exported report.",
        "user_viewer": "Last login: yesterday. Recent actions: viewed user stats.",
    }

    user_activity = activity.get(user_id, "No recent activity.")

    # Inject as additional context (transient — not saved to state)
    messages = [
        *request.messages,
        {"role": "user", "content": f"[System context: {user_activity}]"},
    ]
    request = request.override(messages=messages)

    return handler(request)


# --- 7. Create the agent with all context engineering layers ---
agent = create_agent(
    model=model,
    tools=[read_data, write_data, delete_data],
    middleware=[personalized_prompt, filter_tools_by_role, inject_activity_context],
    context_schema=AppContext,
    store=store,
)

# --- 8. Test with admin user ---
print("=== Admin User ===")
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Show me the system stats and update the alert threshold to 90%.",
            }
        ]
    },
    context=AppContext(user_id="user_admin", user_role="admin"),
)
print(f"Response: {result['messages'][-1].content}\n")

# --- 9. Test with viewer user (should only have read access) ---
print("=== Viewer User ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Show me the revenue data."}]},
    context=AppContext(user_id="user_viewer", user_role="viewer"),
)
print(f"Response: {result['messages'][-1].content}")
