
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from settings import settings



"""
-------------------------------------------------------
In this example, we explore LlamaIndex with the following features:
- Pydantic models for structured data validation
- Enforcing output schemas with agent output_schema parameter
- Type-safe responses from agents
- Tool output models with Pydantic

Structured outputs ensure that agent responses conform to a specific schema,
enabling type-safe integration with downstream systems. LlamaIndex uses Pydantic
models to define and validate these schemas.

For more details, visit:
https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/
-------------------------------------------------------
"""
from typing import List
from pydantic import BaseModel, Field


class Song(BaseModel):
    """Data model for a song."""

    title: str
    length_seconds: int


class Album(BaseModel):
    """Data model for an album."""

    name: str
    artist: str
    songs: List[Song]
    

from llama_index.core.llms import ChatMessage

llm = OpenAI(
    model=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)
sllm = llm.as_structured_llm(output_cls=Album)

# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    name="multiply_agent",
    description="A simple multiply agent.",
    llm=sllm,
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)


async def main():
    # Run the agent
    response = await agent.run("Generate an example album from The Shining")
    print(str(response))
    
    # Run the LLM directly for structured output
    output = await sllm.achat("Generate an example album from The Shining")
    # get actual object
    output_obj: Album = output.raw
    print("Is output of type Album? ", isinstance(output_obj, Album))
    print(str(output_obj))

if __name__ == "__main__":
    asyncio.run(main())
