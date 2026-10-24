import json
import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import Workflow, node, START
from google.genai import types
from pydantic import BaseModel

from utils import print_new_section

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- @node(require_confirmation=...): gate a function node on human approval
- A predicate (lambda amount: amount > 100) so only risky calls pause
- Pausing with an adk_request_input function call, resuming with a
  FunctionResponse carrying {"confirmed": True | False}

Some workflow steps — refunds, deletions, payments — should not run until a
person signs off. With require_confirmation the node checks its arguments
first: small refunds run straight through, large ones pause the run and wait
for the caller to approve or reject. A rejection never runs the function; the
node outputs an error instead. No LLM is involved, so no API key is needed.

For more details, visit:
https://github.com/google/adk-python/blob/v2.13.0/docs/guides/workflow/node/index.md
-------------------------------------------------------
"""

APP_NAME = "refund_demo"
USER_ID = "user"
APPROVAL_LIMIT = 100


# --- 1. A refund node that asks for approval above the limit ---
class RefundRequest(BaseModel):
    order_id: str
    amount: float


# The predicate receives the node's bound arguments by name. parameter_binding
# "node_input" unpacks the RefundRequest into order_id and amount.
@node(
    require_confirmation=lambda amount: amount > APPROVAL_LIMIT,
    parameter_binding="node_input",
)
def issue_refund(order_id: str, amount: float) -> str:
    """Sends the refund to the customer's original payment method."""
    return f"Refunded ${amount:.2f} on {order_id}."


refund_workflow = Workflow(
    name="refund_workflow",
    input_schema=RefundRequest,
    edges=[(START, issue_refund)],
)


# --- 2. Run a refund, answering the confirmation request if one arrives ---
async def process_refund(request: RefundRequest, approve: bool) -> None:
    session_id = request.order_id
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    runner = Runner(
        node=refund_workflow, app_name=APP_NAME, session_service=session_service
    )

    message = types.Content(
        role="user", parts=[types.Part(text=request.model_dump_json())]
    )
    pending_call = None
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=message
    ):
        for call in event.get_function_calls():
            pending_call = call
            print(f"  [workflow] paused with function call {call.name!r}")
            print(f"             id:      {call.id}")
            print(f"             message: {call.args['message']}")
            print(f"             args:    {call.args['payload']['args']}")
        if event.output is not None:
            print(f"  [workflow] {event.node_name} -> {event.output!r}")

    if pending_call is None:
        print(f"  No confirmation requested (${request.amount:.2f} <= ${APPROVAL_LIMIT}).")
        return

    # --- 3. Resume: answer the pending call by id with the decision ---
    decision = {"confirmed": approve}
    print(f"  [user]     responds {json.dumps(decision)}")
    resume = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=pending_call.id, name=pending_call.name, response=decision
                )
            )
        ],
    )
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=resume
    ):
        if event.output is not None:
            print(f"  [workflow] {event.node_name} -> {event.output!r}")


async def main() -> None:
    scenarios = [
        ("Small refund — runs without confirmation", "ORD-101", 40, True),
        ("Large refund — paused, then approved", "ORD-102", 250, True),
        ("Large refund — paused, then rejected", "ORD-103", 900, False),
    ]
    for title, order_id, amount, approve in scenarios:
        request = RefundRequest(order_id=order_id, amount=amount)
        print_new_section(title)
        print(f"  Request: {request.order_id}, ${request.amount:.2f}\n")
        await process_refund(request, approve)


if __name__ == "__main__":
    asyncio.run(main())
