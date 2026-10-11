import os
import asyncio

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from settings import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- Runner.run_async(abort_signal=...) to cancel a run from the caller
- ctx.is_aborted to tell a user abort apart from other cancellations
- Sealed history: the open tool call gets an INVOCATION_ABORTED response

A user pressing "Stop" or an SSE client disconnecting should halt the agent
without leaving a half-finished turn behind. Passing an asyncio.Event as
abort_signal lets the caller stop the run at any point: the runner cancels the
in-flight tool and answers its open function call with a synthetic error
event, so the next turn starts from a consistent history.

For more details, visit:
https://github.com/google/adk-python/blob/v2.11.0/docs/guides/runners/runner/abort.md
-------------------------------------------------------
"""

APP_NAME = "cancel_demo"
USER_ID = "user"
SESSION_ID = "s1"
TOTAL_BATCHES = 10
STOP_AFTER_SECONDS = 1.2


# --- 1. A slow tool the user will give up on ---
async def export_sales_report(region: str, ctx: Context) -> dict:
    """Exports the full sales report for a region. Slow: runs in 10 batches."""
    batch = 0
    try:
        for batch in range(1, TOTAL_BATCHES + 1):
            print(f"  [tool] exporting batch {batch}/{TOTAL_BATCHES}")
            await asyncio.sleep(0.5)
    except asyncio.CancelledError:
        # The runner cancels the tool's task; is_aborted says it was the user.
        print(f"  [tool] cancelled in batch {batch} (ctx.is_aborted={ctx.is_aborted})")
        raise
    return {"region": region, "rows": 48_210}


agent = LlmAgent(
    name="report_agent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "Call export_sales_report for the requested region, then reply in one sentence."
    ),
    tools=[export_sales_report],
)


# --- 2. Print one session event per line ---
def describe(event: Event) -> str:
    items = []
    parts = event.content.parts if event.content and event.content.parts else []
    for part in parts:
        if part.text:
            items.append(f'text "{part.text.strip()}"')
        elif part.function_call:
            call = part.function_call
            items.append(f"call {call.name}({dict(call.args)})")
        elif part.function_response:
            items.append(f"response {part.function_response.response}")
    if event.error_code:
        items.append(f"error_code={event.error_code}")
    return f"{event.author:<13} " + "; ".join(items)


async def main() -> None:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    abort_signal = asyncio.Event()

    def press_stop() -> None:
        print(f"  [user] Stop pressed after {STOP_AFTER_SECONDS}s -> abort_signal.set()")
        abort_signal.set()

    # --- 3. Run with an abort signal; "press Stop" once the export starts ---
    print("=== Running with abort_signal ===\n")
    message = types.Content(
        role="user", parts=[types.Part(text="Export the EMEA sales report.")]
    )
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=message,
        abort_signal=abort_signal,
    ):
        for call in event.get_function_calls():
            print(f"  [stream] model called {call.name}({dict(call.args)})")
            asyncio.get_running_loop().call_later(STOP_AFTER_SECONDS, press_stop)
        if event.error_code:
            print(f"  [stream] {event.error_code}: {event.error_message}")
    print("  [stream] generator finished — no exception reached the caller")

    # --- 4. Inspect what the aborted turn left in the session ---
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print("\n=== Session history after the abort ===\n")
    for event in session.events:
        print(f"  {describe(event)}")

    call_ids = {fc.id for e in session.events for fc in e.get_function_calls()}
    answered_ids = {fr.id for e in session.events for fr in e.get_function_responses()}
    print(f"\n  Function calls left without a response: {len(call_ids - answered_ids)}")


if __name__ == "__main__":
    asyncio.run(main())
