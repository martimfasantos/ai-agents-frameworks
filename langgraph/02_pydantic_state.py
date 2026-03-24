from dotenv import load_dotenv

from pydantic import BaseModel, Field, field_validator
from langgraph.graph import StateGraph, START, END

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- Using Pydantic BaseModel as graph state for runtime validation
- Field validators that enforce constraints on state transitions
- Default values and computed fields in state

While TypedDict is the standard way to define state, LangGraph also
supports Pydantic BaseModel for cases where you need runtime validation,
default values, or complex field logic. This ensures invalid state
is caught early rather than causing subtle bugs downstream.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/graph-api
-----------------------------------------------------------------------
"""


# --- 1. Define Pydantic state with validation ---
class OrderState(BaseModel):
    """State for an order processing pipeline with validation."""

    customer_name: str = Field(description="Name of the customer")
    items: list[str] = Field(default_factory=list, description="Items in the order")
    total_price: float = Field(
        default=0.0, ge=0, description="Total price (must be >= 0)"
    )
    discount_pct: float = Field(
        default=0.0, ge=0, le=100, description="Discount percentage 0-100"
    )
    status: str = Field(default="pending", description="Order status")

    @field_validator("customer_name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Customer name cannot be empty")
        return v.strip().title()


# --- 2. Define processing nodes ---
def add_items_node(state: OrderState) -> dict:
    """Add items to the order and calculate base price."""
    items = ["Widget A", "Gadget B", "Gizmo C"]
    prices = {"Widget A": 29.99, "Gadget B": 49.99, "Gizmo C": 19.99}
    total = sum(prices[item] for item in items)
    return {"items": items, "total_price": total, "status": "items_added"}


def apply_discount_node(state: OrderState) -> dict:
    """Apply loyalty discount based on customer status."""
    discount = 15.0  # 15% loyalty discount
    discounted_price = state.total_price * (1 - discount / 100)
    return {
        "discount_pct": discount,
        "total_price": discounted_price,
        "status": "discount_applied",
    }


def finalize_order_node(state: OrderState) -> dict:
    """Finalize the order."""
    return {"status": "confirmed"}


# --- 3. Build the graph ---
builder = StateGraph(OrderState)

builder.add_node("add_items", add_items_node)
builder.add_node("apply_discount", apply_discount_node)
builder.add_node("finalize", finalize_order_node)

builder.add_edge(START, "add_items")
builder.add_edge("add_items", "apply_discount")
builder.add_edge("apply_discount", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()

# --- 4. Run with valid input ---
print("=== Pydantic State Validation ===\n")

result = graph.invoke(OrderState(customer_name="  john doe  "))

print(f"Customer: {result['customer_name']}")  # Validated & title-cased
print(f"Items: {result['items']}")
print(f"Original Total: $99.97")
print(f"Discount: {result['discount_pct']}%")
print(f"Final Price: ${result['total_price']:.2f}")
print(f"Status: {result['status']}")

# --- 5. Demonstrate validation error ---
print("\n=== Validation Error Demo ===\n")
try:
    graph.invoke(OrderState(customer_name="", total_price=-10))
except Exception as e:
    print(f"Caught validation error: {type(e).__name__}: {e}")
