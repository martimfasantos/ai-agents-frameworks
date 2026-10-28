from dotenv import load_dotenv

from pydantic import BaseModel, Field, ValidationError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore LangGraph with the following features:
- interrupt(..., response_schema=Model) for typed resume values
- Interrupt.response_schema: the JSON Schema a client can render as a form
- pydantic.ValidationError when a Command(resume=...) does not match
- Retrying on the same thread; get_state().interrupts lists pending ones

A plain interrupt() hands back whatever the caller sends, so every node
has to re-check its resume payload by hand. With response_schema the
approval gate becomes a contract: the caller receives a JSON Schema to
build its input form, and interrupt() returns a validated model instance.
A malformed answer raises ValidationError without consuming the
interrupt, so the client can fix the payload and resume again.

For more details, visit:
https://docs.langchain.com/oss/python/langgraph/interrupts#define-an-interrupt-response-schema
-----------------------------------------------------------------------
"""


# --- 1. The typed answer the approver must send back ---
class Decision(BaseModel):
    approved: bool = Field(description="Approve the refund?")
    amount: float = Field(description="Amount to refund, in USD")
    note: str | None = Field(default=None, description="Optional reason")


class RefundState(TypedDict, total=False):
    order_id: str
    requested: float
    status: str


# --- 2. Approval gate: interrupt() validates the resume into a Decision ---
def approval_gate(state: RefundState) -> dict:
    decision = interrupt(
        {"order_id": state["order_id"], "requested": state["requested"]},
        response_schema=Decision,
    )
    print(f"  [node] interrupt() returned {decision!r}")
    if not decision.approved:
        return {"status": "refund declined"}
    return {"status": f"refunded ${decision.amount:.2f} on {state['order_id']}"}


builder = StateGraph(RefundState)
builder.add_node("approval_gate", approval_gate)
builder.add_edge(START, "approval_gate")
builder.add_edge("approval_gate", END)
graph = builder.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "refund-ORD-1042"}}

# --- 3. First run pauses; the caller receives the schema for the answer ---
print("=== 1. The interrupt carries a response schema ===\n")

result = graph.invoke({"order_id": "ORD-1042", "requested": 125.50}, config)
pending = result["__interrupt__"][0]
schema = pending.response_schema
print(f"  waiting on: {pending.value}")
print(f"  schema '{schema['title']}', required fields: {schema['required']}")
for name, field in schema["properties"].items():
    kind = field.get("type") or " | ".join(option["type"] for option in field["anyOf"])
    print(f"    {name:<9}{kind:<15}{field['description']}")

# --- 4. A malformed resume is rejected and the interrupt stays pending ---
print("\n=== 2. Bad resume -> pydantic.ValidationError ===\n")

bad_answer = {"approved": "maybe", "amount": "lots"}
print(f"  resume sent: {bad_answer}")
try:
    graph.invoke(Command(resume=bad_answer), config)
except ValidationError as exc:
    for error in exc.errors():
        print(f"  rejected {error['loc'][0]!r}: {error['msg']}")

still_pending = graph.get_state(config).interrupts
print(f"  pending interrupts: {len(still_pending)} (same id: {still_pending[0].id == pending.id})")

# --- 5. Retry on the same thread; string inputs are coerced into the model ---
print("\n=== 3. Corrected resume on the same thread ===\n")

good_answer = {"approved": "yes", "amount": "125.5"}
print(f"  resume sent: {good_answer}")
result = graph.invoke(Command(resume=good_answer), config)
print(f"  status: {result['status']}")
print(f"  pending interrupts: {len(graph.get_state(config).interrupts)}")
