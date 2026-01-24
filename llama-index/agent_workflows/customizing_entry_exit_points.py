import asyncio
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from workflows import Workflow, step
from workflows.events import StartEvent, StopEvent
from settings import settings


"""
-------------------------------------------------------
In this example, we explore LlamaIndex Workflows with the following features:
- Custom StartEvent with typed fields
- Custom StopEvent with structured return values
- Type-safe workflow entry and exit points
- Better IDE autocompletion and type checking

Custom events at entry and exit points provide better type safety and
make workflow interfaces more explicit and maintainable.

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/customizing_entry_exit_points/
-------------------------------------------------------
"""


# Define a custom StartEvent with typed fields
class CustomStartEvent(StartEvent):
    topic: str
    num_jokes: int
    style: str


# Define a custom StopEvent with structured return
class CustomStopEvent(StopEvent):
    jokes: list[str]
    topic: str
    total_generated: int


class CustomEntryExitWorkflow(Workflow):
    @step
    async def generate_jokes(self, ev: CustomStartEvent) -> CustomStopEvent:
        """Generate jokes based on custom start event parameters"""
        llm = OpenAI(
            model=settings.OPENAI_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY.get_secret_value()
        )
        
        jokes = []
        for i in range(ev.num_jokes):
            prompt = f"Tell me a {ev.style} joke about {ev.topic}. Keep it short."
            response = await llm.acomplete(prompt)
            jokes.append(str(response).strip())
        
        return CustomStopEvent(
            jokes=jokes,
            topic=ev.topic,
            total_generated=len(jokes)
        )


async def main():
    workflow = CustomEntryExitWorkflow(timeout=60, verbose=False)
    
    custom_start = CustomStartEvent(
        topic="programming",
        num_jokes=2,
        style="clever"
    )
    
    # WARNING: result is now a CustomStopEvent instance, not a string!
    result = await workflow.run(start_event=custom_start)
    
    print(f"Topic: {result.topic}")
    print(f"Generated {result.total_generated} jokes:\n")
    for idx, joke in enumerate(result.jokes, 1):
        print(f"{idx}. {joke}")


if __name__ == "__main__":
    asyncio.run(main())
