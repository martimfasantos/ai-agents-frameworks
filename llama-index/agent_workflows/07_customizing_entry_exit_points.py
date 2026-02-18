import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from llama_index.llms.openai import OpenAI
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent, Event
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Custom StartEvent with typed fields for complex inputs
- Custom StopEvent with structured return values
- Passing custom start events via start_event= parameter
- Type-safe workflow entry and exit for better IDE support

By default, keyword arguments to run() become fields on a StartEvent,
and the result field of StopEvent can be any type. For complex data,
create custom subclasses of StartEvent and StopEvent with typed fields.
Pass custom start events via workflow.run(start_event=my_event). When
using a custom StopEvent, the result of run() is the event instance
itself (not result), so you access fields directly (e.g., result.jokes).

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/customizing_entry_exit_points/
-------------------------------------------------------
"""


# --- 1. Custom StartEvent with typed fields ---
class JokeRequestEvent(StartEvent):
    topic: str
    style: str
    num_jokes: int


# --- 2. Custom StopEvent with structured return ---
class JokeResponseEvent(StopEvent):
    jokes: list[str]
    topic: str
    total_generated: int


class JokeEvent(Event):
    joke: str


# --- 3. Workflow using custom entry and exit points ---
class CustomEntryExitWorkflow(Workflow):
    llm = OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    @step
    async def generate_jokes(self, ev: JokeRequestEvent) -> JokeResponseEvent:
        """Generate jokes based on custom start event parameters"""
        jokes = []
        for _ in range(ev.num_jokes):
            prompt = f"Tell me a {ev.style} joke about {ev.topic}. Keep it short (1-2 lines)."
            response = await self.llm.acomplete(prompt)
            jokes.append(str(response).strip())

        # Return a custom StopEvent with structured fields
        return JokeResponseEvent(
            jokes=jokes,
            topic=ev.topic,
            total_generated=len(jokes),
        )


# --- 4. Run the workflow with a custom start event ---
async def main():
    workflow = CustomEntryExitWorkflow(timeout=60, verbose=False)

    # Create and pass a custom start event
    custom_start = JokeRequestEvent(
        topic="programming",
        num_jokes=2,
        style="clever",
    )

    # NOTE: result is now a JokeResponseEvent instance, not a string!
    result = await workflow.run(start_event=custom_start)

    # Access typed fields directly on the result
    print(f"Topic: {result.topic}")
    print(f"Generated {result.total_generated} jokes:\n")
    for idx, joke in enumerate(result.jokes, 1):
        print(f"{idx}. {joke}\n")


if __name__ == "__main__":
    asyncio.run(main())
