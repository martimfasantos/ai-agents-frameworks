import sys
import asyncio
from pathlib import Path
from llama_index.llms.openai import OpenAI
from workflows import Workflow, step
from workflows.events import (
    Event,
    StartEvent,
    StopEvent,
)
sys.path.append(str(Path(__file__).parent.parent))
from settings import settings


"""
-------------------------------------------------------
In this example, we explore a simple LlamaIndex Workflow

For more details, visit:
https://developers.llamaindex.ai/python/llamaagents/workflows/
-------------------------------------------------------
"""

# --- 1. Define the LLM ---
class JokeEvent(Event):
    joke: str


# --- 2. Create a workflow that generates and critiques a joke ---
class JokeFlow(Workflow):
    # 2.1 Define the LLM
    llm = OpenAI(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value()
    )

    # 2.2 Define the starting step of the workflow
    @step
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic

        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        return JokeEvent(joke=str(response))

    # 2.3 Define the critique step of the workflow
    @step
    async def critique_joke(self, ev: JokeEvent) -> StopEvent:
        joke = ev.joke

        prompt = f"Give a thorough analysis and critique of the following joke: {joke}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))


# --- 3. Define the main function to run the workflow ---
async def main():
    w = JokeFlow(timeout=60, verbose=False)
    result = await w.run(topic="pirates")
    print(str(result))

if __name__ == "__main__":
    asyncio.run(main())