import sys
import asyncio
from pathlib import Path
from llama_index.llms.openai import OpenAI
from workflows import Workflow, Context, step
from workflows.events import (
    Event,
    StartEvent,
    StopEvent,
    WorkflowTimedOutEvent,
    WorkflowCancelledEvent,
    WorkflowFailedEvent,
)
sys.path.append(str(Path(__file__).parent.parent))
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Streaming progress events using ctx.write_event_to_stream()
- Consuming streamed events with handler.stream_events()
- Custom progress events for real-time feedback
- Handling workflow termination events (timeout, cancel, failure)

Workflows can take time to fully execute. To provide a good user
experience, you can stream events as they occur using the Context
object's write_event_to_stream() method. The consumer uses
handler.stream_events() to receive events in real-time. When a
workflow ends abnormally, specific StopEvent subclasses are published:
WorkflowTimedOutEvent, WorkflowCancelledEvent, WorkflowFailedEvent.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/streaming/
-------------------------------------------------------
"""


# --- 1. Define custom progress events ---
class ProgressEvent(Event):
    msg: str


class FirstEvent(Event):
    first_output: str


class SecondEvent(Event):
    second_output: str


# --- 2. Define a workflow that streams progress events ---
class StreamingWorkflow(Workflow):
    @step
    async def step_one(self, ctx: Context, ev: StartEvent) -> FirstEvent:
        """Stream a progress event, then continue to the next step"""
        # You need to manually write progress events to the stream
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is starting..."))
        await asyncio.sleep(0.5)
        ctx.write_event_to_stream(ProgressEvent(msg="Step one is complete!"))
        return FirstEvent(first_output="First step done")

    @step
    async def step_two(self, ctx: Context, ev: FirstEvent) -> SecondEvent:
        """Stream incremental progress for a long-running operation"""
        llm = OpenAI(
            model=settings.OPENAI_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY.get_secret_value()
        )
        generator = await llm.astream_complete(
            "Please give me the first 2 paragraphs of Moby Dick, a book in the public domain."
        )  # NOTE: Increase the number of paragraphs to see a timeout
        full_resp = ""
        async for response in generator:
            # Allow the workflow to stream this piece of response
            ctx.write_event_to_stream(ProgressEvent(msg=response.delta))
            full_resp += response.delta
        return SecondEvent(second_output="Second step done")

    @step
    async def step_three(self, ctx: Context, ev: SecondEvent) -> StopEvent:
        """Final step with a progress event"""
        ctx.write_event_to_stream(ProgressEvent(msg="Step three — finishing up!"))
        return StopEvent(result="Workflow complete.")


# --- 3. Consume streamed events and handle termination ---
async def main():
    w = StreamingWorkflow(timeout=30, verbose=False)
    # run() starts the workflow in the background
    handler = w.run()

    # Stream events as they occur
    async for ev in handler.stream_events():
        if isinstance(ev, ProgressEvent):
            print(f"[Progress] {ev.msg}")
        # Handle abnormal workflow termination events
        elif isinstance(ev, WorkflowTimedOutEvent):
            print(f"[Timeout] Workflow timed out after {ev.timeout}s")
        elif isinstance(ev, WorkflowCancelledEvent):
            print("[Cancelled] Workflow was cancelled")
        elif isinstance(ev, WorkflowFailedEvent):
            print(f"[Failed] Step '{ev.step_name}' failed: {ev.exception_message}")

    # Get the final result after streaming completes
    final_result = await handler
    print(f"\nFinal result: {final_result}")


if __name__ == "__main__":
    asyncio.run(main())
