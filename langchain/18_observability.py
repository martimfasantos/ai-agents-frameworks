import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- LangSmith observability and tracing
- Adding metadata and tags to traces
- Selective tracing with tracing_context

LangSmith provides tracing for LangChain agents — every step from
user input to final response is recorded, including tool calls and
model interactions. Tracing is enabled by setting environment
variables. You can add metadata/tags for filtering and use
tracing_context for selective tracing.

Note: This example runs without a LangSmith account (tracing is
a no-op if LANGSMITH_TRACING is not set). Set the env vars below
to enable actual trace logging.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/observability
-------------------------------------------------------
"""

# --- 1. Configure LangSmith tracing (optional) ---
# Uncomment these lines and set your API key to enable tracing:
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_API_KEY"] = "<your-langsmith-api-key>"
# os.environ["LANGSMITH_PROJECT"] = "langchain-examples"

print("LangSmith tracing enabled:", os.environ.get("LANGSMITH_TRACING", "false"))
print()


# --- 2. Define tools ---
@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its ID."""
    orders = {
        "ORD-001": "Wireless Mouse — Delivered on Jan 15",
        "ORD-002": "USB-C Hub — Shipped, arriving Jan 20",
        "ORD-003": "Mechanical Keyboard — Processing",
    }
    return orders.get(order_id, f"Order {order_id} not found")


# --- 3. Create the model ---
model = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

# --- 4. Create the agent ---
agent = create_agent(
    model=model,
    tools=[lookup_order],
    system_prompt="You are a customer support assistant. Be concise.",
)

# --- 5. Invoke with metadata and tags ---
print("=== Traced Invocation with Metadata ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the status of order ORD-002?"}]},
    config={
        "tags": ["example", "observability-demo"],
        "metadata": {
            "user_id": "user_42",
            "session_id": "session_abc",
            "environment": "development",
        },
    },
)
print(f"Response: {result['messages'][-1].content}\n")

# --- 6. Selective tracing with tracing_context ---
print("=== Selective Tracing ===")
try:
    import langsmith as ls

    # This invocation WILL be traced (if LangSmith is configured)
    with ls.tracing_context(enabled=True, project_name="langchain-examples"):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Look up order ORD-001"}]}
        )
        print(f"Traced response: {result['messages'][-1].content}\n")

    # This invocation will NOT be traced
    with ls.tracing_context(enabled=False):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Look up order ORD-003"}]}
        )
        print(f"Untraced response: {result['messages'][-1].content}")

except ImportError:
    print("langsmith not installed — selective tracing skipped")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Look up order ORD-001"}]}
    )
    print(f"Response: {result['messages'][-1].content}")
