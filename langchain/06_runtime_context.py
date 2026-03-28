import os
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Runtime context with context_schema and dataclasses
- Dependency injection into tools via ToolRuntime[Context]
- Passing per-invocation configuration without exposing it to the LLM

Runtime context provides dependency injection for tools and
middleware. Instead of hardcoding user IDs or config values,
you pass them at invocation time. The context is available to
tools via the ToolRuntime parameter but is hidden from the LLM
tool schema, keeping the interface clean.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/runtime
-------------------------------------------------------
"""


# --- 1. Define the context schema ---
@dataclass
class UserContext:
    user_id: str
    user_name: str
    preferred_language: str


# --- 2. Define tools that access runtime context ---
@tool
def get_user_profile(runtime: ToolRuntime[UserContext]) -> str:
    """Look up the current user's profile information."""
    ctx = runtime.context
    return (
        f"User ID: {ctx.user_id}, "
        f"Name: {ctx.user_name}, "
        f"Preferred language: {ctx.preferred_language}"
    )


@tool
def get_user_orders(runtime: ToolRuntime[UserContext]) -> str:
    """Look up the current user's recent orders."""
    # Simulated order data keyed by user_id
    orders = {
        "user_42": [
            {"id": "ORD-001", "item": "Wireless Mouse", "status": "Delivered"},
            {"id": "ORD-002", "item": "USB-C Hub", "status": "Shipped"},
        ],
        "user_99": [
            {"id": "ORD-100", "item": "Keyboard", "status": "Processing"},
        ],
    }
    user_orders = orders.get(runtime.context.user_id, [])
    if not user_orders:
        return "No orders found."
    lines = [f"  - {o['id']}: {o['item']} ({o['status']})" for o in user_orders]
    return "Recent orders:\n" + "\n".join(lines)


# --- 3. Create the agent with a context schema ---
agent = create_agent(
    model=init_chat_model(f"openai:{settings.OPENAI_MODEL_NAME}"),
    tools=[get_user_profile, get_user_orders],
    system_prompt="You are a customer support assistant. Use the available tools to help the user.",
    context_schema=UserContext,
)

# --- 4. Invoke with context for user_42 ---
print("=== User 42: Alice ===")
result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Show me my profile and recent orders."}
        ]
    },
    context=UserContext(
        user_id="user_42", user_name="Alice", preferred_language="English"
    ),
)
print(result["messages"][-1].content)
print()

# --- 5. Invoke with context for a different user ---
print("=== User 99: Bob ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What orders do I have?"}]},
    context=UserContext(
        user_id="user_99", user_name="Bob", preferred_language="Portuguese"
    ),
)
print(result["messages"][-1].content)
