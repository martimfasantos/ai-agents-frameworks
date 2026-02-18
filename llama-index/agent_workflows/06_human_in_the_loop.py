import asyncio
from workflows import Workflow, Context, step
from workflows.events import StartEvent, StopEvent, InputRequiredEvent, HumanResponseEvent


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Human-in-the-loop using InputRequiredEvent and HumanResponseEvent
- Streaming events to handle human interaction
- Stopping/resuming workflows between human responses
- Using ctx.wait_for_event() as an alternative approach

Workflows support human-in-the-loop patterns using InputRequiredEvent
to request input and HumanResponseEvent to receive it. The workflow
pauses until the human responds via handler.ctx.send_event(). For
async scenarios (e.g., web requests), you can serialize the context
with ctx.to_dict(), break out of the loop, and resume later with
Context.from_dict(). An alternative is ctx.wait_for_event(), which
waits for input within a single step (but code before it must be
idempotent since it replays on each trigger).

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/human_in_the_loop/
-------------------------------------------------------
"""


# --- 1. Basic human-in-the-loop with InputRequiredEvent ---
class HumanInTheLoopWorkflow(Workflow):
    @step
    async def ask_for_name(self, ev: StartEvent) -> InputRequiredEvent:
        """Request user's name"""
        return InputRequiredEvent(prefix="What is your name? ")

    @step
    async def greet_and_ask_age(self, ev: HumanResponseEvent) -> InputRequiredEvent:
        """Greet user, then request age"""
        name = ev.response
        return InputRequiredEvent(prefix=f"Hello {name}! How old are you? ")

    @step
    async def process_response(self, ev: HumanResponseEvent) -> StopEvent:
        """Process the final human response"""
        return StopEvent(result=f"You said: {ev.response}. Thanks for chatting!")


# --- 2. Stream events and handle human input ---
async def run_basic_example():
    workflow = HumanInTheLoopWorkflow(timeout=120, verbose=False)
    handler = workflow.run()

    # Stream events and respond to input requests
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            response = input(event.prefix)
            handler.ctx.send_event(HumanResponseEvent(response=response))

    final_result = await handler
    print(f"Result: {final_result}")


# --- 3. Stopping/resuming between human responses ---
async def run_stop_resume_example():
    workflow = HumanInTheLoopWorkflow(timeout=120, verbose=False)
    handler = workflow.run()

    # Stream until we need input, then break and serialize context
    saved_event = None
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            # Serialize the context for later (e.g., store in a database)
            ctx_dict = handler.ctx.to_dict()
            saved_event = event
            await handler.cancel_run()
            break

    if saved_event is not None:
        # Simulate async response (e.g., from a web request)
        response = input(saved_event.prefix)

        # Restore context and resume the workflow
        restored_ctx = Context.from_dict(workflow, ctx_dict)
        handler = workflow.run(ctx=restored_ctx)

        # Send the human response event to resume
        handler.ctx.send_event(HumanResponseEvent(response=response))

        # Continue streaming
        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                response = input(event.prefix)
                handler.ctx.send_event(HumanResponseEvent(response=response))

        final_result = await handler
        print(f"Result: {final_result}")


# --- 4. Run the examples ---
async def main():
    print("=== Basic Human-in-the-Loop ===")
    await run_basic_example()


if __name__ == "__main__":
    asyncio.run(main())
