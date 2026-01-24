import asyncio
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent, InputRequiredEvent, HumanResponseEvent


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Human-in-the-loop pattern using InputRequiredEvent
- Collecting human responses with HumanResponseEvent
- Pausing workflow execution for user input
- Event streaming to handle human interaction

Workflows support flexible human-in-the-loop patterns, allowing you to
pause execution and wait for human input before continuing.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/human_in_the_loop/
-------------------------------------------------------
"""


class HumanInTheLoopWorkflow(Workflow):
    @step
    async def ask_for_name(self, ev: StartEvent) -> InputRequiredEvent:
        """Request user's name"""
        return InputRequiredEvent(prefix="What is your name? ")

    @step
    async def ask_for_age(self, ev: HumanResponseEvent) -> InputRequiredEvent:
        """Store name and request age"""
        name = ev.response
        return InputRequiredEvent(prefix=f"{name}, what is your age? ")

    @step
    async def process_response(self, ev: HumanResponseEvent) -> StopEvent:
        """Process the final response"""
        age = ev.response
        try:
            age_int = int(age)
            message = f"You are {age_int} years old"
        except ValueError:
            message = f"You entered: {age}"
        return StopEvent(result=message)


async def main():
    workflow = HumanInTheLoopWorkflow(timeout=120, verbose=False)
    handler = workflow.run()
    
    # Stream events and handle human input
    async for event in handler.stream_events():
        if isinstance(event, InputRequiredEvent):
            response = input(event.prefix)
            handler.ctx.send_event(HumanResponseEvent(response=response))
    
    final_result = await handler
    print(f"\nResult: {final_result}")


if __name__ == "__main__":
    asyncio.run(main())
