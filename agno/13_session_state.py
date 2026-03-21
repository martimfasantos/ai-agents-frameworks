from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.run.base import RunContext
from agno.tools import tool
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Session state for persisting data across tool calls
- run_context.session_state for reading/writing state in tools
- SqliteStorage backend to persist state between runs
- Initializing session_state with default values

Session state is a dictionary shared between the agent and its
tools within a session. Tools can read and write to it via
run_context.session_state, enabling stateful workflows like
shopping carts, counters, or multi-step forms. A storage
backend (e.g. SqliteStorage) is required for state to persist
across multiple agent.run() calls.

For more details, visit:
https://docs.agno.com/agents/state
-------------------------------------------------------
"""


# --- 1. Define tools that use session state ---
@tool
def add_to_cart(run_context: RunContext, item: str, price: float) -> str:
    """Add an item to the shopping cart.

    Args:
        run_context: Injected automatically by Agno with session state.
        item: The name of the item to add.
        price: The price of the item.

    Returns:
        Confirmation of the item added.
    """
    cart = run_context.session_state.get("cart", [])
    cart.append({"item": item, "price": price})
    run_context.session_state["cart"] = cart
    run_context.session_state["total"] = sum(i["price"] for i in cart)
    return f"Added '{item}' (${price:.2f}) to cart. Total: ${run_context.session_state['total']:.2f}"


@tool
def view_cart(run_context: RunContext) -> str:
    """View the current shopping cart contents.

    Args:
        run_context: Injected automatically by Agno with session state.

    Returns:
        A formatted string showing cart contents and total.
    """
    cart = run_context.session_state.get("cart", [])
    if not cart:
        return "Your cart is empty."
    lines = [f"  - {i['item']}: ${i['price']:.2f}" for i in cart]
    total = run_context.session_state.get("total", 0)
    return "Shopping Cart:\n" + "\n".join(lines) + f"\n  Total: ${total:.2f}"


@tool
def clear_cart(run_context: RunContext) -> str:
    """Clear all items from the shopping cart.

    Args:
        run_context: Injected automatically by Agno with session state.

    Returns:
        Confirmation that the cart was cleared.
    """
    run_context.session_state["cart"] = []
    run_context.session_state["total"] = 0.0
    return "Cart cleared."


# --- 2. Create the agent with session state ---
# A storage backend (db=) is required for session_state to persist across runs.
# Without storage, session_state is deepcopied fresh for every run() call.
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    tools=[add_to_cart, view_cart, clear_cart],
    session_state={"cart": [], "total": 0.0},
    db=SqliteDb(db_file="/tmp/agno_session_state_example.db"),
    instructions=[
        "You are a shopping assistant.",
        "Help the user manage their shopping cart.",
        "Always confirm what was added and show the running total.",
    ],
    markdown=True,
)

# --- 3. Simulate a shopping session ---
print("=== Step 1: Add items ===\n")
run_output = agent.run("Add a notebook for $12.99 and a pen for $3.50 to my cart.")
pprint_run_response(run_output)

print("\n=== Step 2: View cart ===\n")
run_output = agent.run("What's in my cart?")
pprint_run_response(run_output)

print("\n=== Step 3: Add more and check total ===\n")
run_output = agent.run("Add a coffee mug for $8.00. What's my total now?")
pprint_run_response(run_output)

# Show the raw session state from the last run output
print("\n=== Raw session state ===")
print(f"Cart: {run_output.session_state}")
