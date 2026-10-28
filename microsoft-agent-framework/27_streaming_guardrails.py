import asyncio
import re

from dotenv import load_dotenv

from agent_framework import AgentContext, AgentResponse, agent_middleware
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Streaming output controls registered from agent middleware
- stream_buffer_updates to hold every chunk until the reply is checked
- stream_result_transforms to rewrite the finished reply
- stream_result_gates_after to block a reply before any chunk is released

A streamed reply normally reaches the user token by token, so a check on
the finished text runs too late, and a per-chunk check misses a pattern
split across two tokens. With buffering on, the transform and the gate
see the complete reply first; a clean reply is then replayed chunk by
chunk, a rewritten one is rebuilt from the new text, and a blocked one
releases nothing.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/concepts/agents/middleware/result-overrides?pivots=programming-language-python
-------------------------------------------------------
"""

INTERNAL_ID = re.compile(r"ACCT-\d+")
FORBIDDEN_MARKER = "INTERNAL-ONLY"


class GuardrailViolation(Exception):
    """Raised by the result gate to stop a reply from being released."""


# --- 1. A tool whose backend notes carry internal-only details ---
def get_order_status(order_id: str) -> str:
    """Looks up the status note for an order (simulated)."""
    notes = {
        "1041": "Delivered on 1 Oct to the front desk.",
        "1042": "Shipped on 3 Oct via DHL. Billing account ACCT-778812.",
        "1043": "On hold. INTERNAL-ONLY: flagged for manual fraud review.",
    }
    return notes.get(order_id, f"Order {order_id} not found.")


# --- 2. Result transform: return a replacement, or None to keep the reply ---
def mask_internal_ids(response: AgentResponse) -> AgentResponse | None:
    """Masks internal account IDs in the finished reply."""
    found = INTERNAL_ID.findall(response.text)
    if not found:
        return None
    for message in response.messages:
        for content in message.contents:
            if content.text:
                content.text = INTERNAL_ID.sub("ACCT-****", content.text)
    print(f"  [guardrail] transform masked {', '.join(found)}")
    return response


# --- 3. Result gate: return None to allow, raise to block ---
def block_internal_only(response: AgentResponse) -> None:
    """Refuses to release a reply that carries the internal-only marker."""
    if FORBIDDEN_MARKER in response.text:
        print(f"  [guardrail] gate found {FORBIDDEN_MARKER!r} in the reply")
        raise GuardrailViolation(f"reply contained {FORBIDDEN_MARKER}")


# --- 4. Middleware registers the controls before call_next() ---
@agent_middleware
async def output_guardrail(context: AgentContext, call_next) -> None:
    """Holds the stream until the transform and gate have seen the full reply."""
    context.stream_buffer_updates = True
    context.stream_result_transforms.append(mask_internal_ids)
    # A replaced reply can't reuse the buffered chunks, so rebuild them from it.
    context.stream_result_to_updates = AgentResponse.to_updates
    context.stream_result_gates_after.append(block_internal_only)
    await call_next()


async def main() -> None:
    # --- 5. Create the agent with the guardrail middleware ---
    client = OpenAIChatClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )
    agent = client.as_agent(
        name="order-support",
        instructions=(
            "You are an order-support assistant. Call get_order_status, then reply "
            "in one sentence that quotes the status note verbatim, including any codes."
        ),
        tools=[get_order_status],
        middleware=[output_guardrail],
    )

    # --- 6. Stream each reply and count what actually reaches the user ---
    for title, order_id in (
        ("Case 1: clean reply is replayed", "1041"),
        ("Case 2: internal ID is rewritten", "1042"),
        ("Case 3: INTERNAL-ONLY reply is blocked", "1043"),
    ):
        print(f"=== {title} ===")
        chunks: list[str] = []
        try:
            async for update in agent.run(f"What's the status of order {order_id}?", stream=True):
                if update.text:
                    chunks.append(update.text)
            outcome = f"Agent: {''.join(chunks)}"
        except GuardrailViolation as exc:
            outcome = f"Blocked: {exc}"
        print(f"Chunks that reached the user: {len(chunks)}")
        print(f"{outcome}\n")


if __name__ == "__main__":
    asyncio.run(main())
