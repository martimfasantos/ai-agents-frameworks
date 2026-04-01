import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowRunResult,
    handler,
    response_handler,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Human-in-the-loop workflows with request_info
- Pausing execution to collect user input
- Resuming workflows with user responses

Human-in-the-loop workflows pause execution to request
information from a user, then resume once the response
is provided. This is key for approval flows, interactive
wizards, and any workflow that needs human judgment.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop?pivots=programming-language-python
-------------------------------------------------------
"""


# --- 1. Define data types ---
@dataclass
class OrderRequest:
    item: str
    quantity: int
    price: float


@dataclass
class ApprovalRequest:
    message: str
    total: float


@dataclass
class ApprovalResponse:
    approved: bool
    reason: str


@dataclass
class OrderResult:
    status: str
    details: str


# --- 2. Define an executor that requests human approval ---
class OrderProcessor(Executor):
    """Processes orders with human approval for high-value items."""

    @handler
    async def handle_order(
        self, message: OrderRequest, ctx: WorkflowContext[OrderResult]
    ) -> None:
        total = message.quantity * message.price
        print(
            f"[OrderProcessor] Processing order: {message.quantity}x {message.item} = ${total:.2f}"
        )

        if total > 100:
            # Request human approval for high-value orders
            print(
                f"[OrderProcessor] High-value order (${total:.2f}) — requesting approval..."
            )
            await ctx.request_info(
                request_data=ApprovalRequest(
                    message=f"Approve order: {message.quantity}x {message.item} for ${total:.2f}?",
                    total=total,
                ),
                response_type=ApprovalResponse,
            )
        else:
            # Auto-approve low-value orders
            print(f"[OrderProcessor] Low-value order — auto-approved.")
            await ctx.yield_output(
                OrderResult(
                    status="approved",
                    details=f"Auto-approved: {message.quantity}x {message.item} for ${total:.2f}",
                )
            )

    @response_handler
    async def handle_approval(
        self,
        original_request: ApprovalRequest,
        response: ApprovalResponse,
        ctx: WorkflowContext[OrderResult],
    ) -> None:
        """Handles the human's approval/rejection response."""
        if response.approved:
            print(f"[OrderProcessor] Order APPROVED: {response.reason}")
            await ctx.yield_output(
                OrderResult(
                    status="approved",
                    details=f"Human-approved (${original_request.total:.2f}): {response.reason}",
                )
            )
        else:
            print(f"[OrderProcessor] Order REJECTED: {response.reason}")
            await ctx.yield_output(
                OrderResult(
                    status="rejected",
                    details=f"Rejected: {response.reason}",
                )
            )


async def main() -> None:
    # --- 3. Build the workflow ---
    processor = OrderProcessor(id="order-processor")
    workflow = WorkflowBuilder(start_executor=processor).build()

    # --------------------------------------------------------------
    # Example 1: Low-value order (auto-approved)
    # --------------------------------------------------------------
    print("=== Example 1: Low-Value Order (Auto-Approve) ===")
    result = await workflow.run(OrderRequest(item="Notebook", quantity=2, price=15.00))
    outputs = result.get_outputs()
    for output in outputs:
        print(f"Result: {output}\n")

    # --------------------------------------------------------------
    # Example 2: High-value order (requires human approval)
    # --------------------------------------------------------------
    print("=== Example 2: High-Value Order (Human Approval) ===")
    result: WorkflowRunResult = await workflow.run(
        OrderRequest(item="Laptop", quantity=3, price=999.00)
    )

    # Check for pending approval requests
    request_events = result.get_request_info_events()
    if request_events:
        for event in request_events:
            print(f"Approval requested: {event.request_type}")
            print(f"Request data: {event.request_type}")

        # Simulate providing a human response
        print("Simulating human approval...")
        responses = {
            request_events[0].request_id: ApprovalResponse(
                approved=True,
                reason="Budget approved by manager",
            )
        }

        # Resume the workflow with only the responses (no message)
        result = await workflow.run(
            responses=responses,
        )

        outputs = result.get_outputs()
        for output in outputs:
            print(f"Result: {output}")
    else:
        outputs = result.get_outputs()
        for output in outputs:
            print(f"Result: {output}")


if __name__ == "__main__":
    asyncio.run(main())
